import arxiv
import argparse
import os
import sys
from dotenv import load_dotenv
import yaml
load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, send_email
from tqdm import trange,tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from paper import ArxivPaper
from llm import set_global_llm
import feedparser
from datetime import datetime, timedelta, timezone

def get_zotero_corpus(id:str,key:str) -> list[dict]:
    zot = zotero.Zotero(id, 'user', key)
    collections = zot.everything(zot.collections())
    collections = {c['key']:c for c in collections}
    logger.info(f"Retrieved {len(collections)} collections from Zotero.")
    corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    corpus = [c for c in corpus if c['data']['abstractNote'] != '']
    def get_collection_path(col_key:str) -> str:
        if p := collections[col_key]['data']['parentCollection']:
            return get_collection_path(p) + '/' + collections[col_key]['data']['name']
        else:
            return collections[col_key]['data']['name']
    for c in corpus:
        paths = [get_collection_path(col) for col in c['data']['collections']]
        c['paths'] = paths
    return corpus

def filter_corpus(corpus:list[dict], pattern:str) -> list[dict]:
    _,filename = mkstemp()
    with open(filename,'w') as file:
        file.write(pattern)
    matcher = parse_gitignore(filename,base_dir='./')
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c['paths']]
        if not any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus

def filter_recent_papers(papers: list, days: int = 3) -> list:
    """过滤最近N天的论文
    
    Args:
        papers: 论文列表
        days: 天数
        
    Returns:
        过滤后的论文列表
    """
    recent_days = datetime.now(timezone.utc) - timedelta(days=days)
    return [p for p in papers if p.published >= recent_days]

def get_arxiv_paper_by_category(query:str, debug:bool=False, max_results:int=50) -> list[ArxivPaper]:
    client = arxiv.Client(num_retries=10,delay_seconds=10)
    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    if 'Feed error for query' in feed.feed.title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")
    
    # Search papers from Arxiv by category
    if not debug:
        papers = []
        all_paper_ids = [i.id.removeprefix("oai:arXiv.org:") for i in feed.entries if i.arxiv_announce_type == 'new']
        bar = tqdm(total=len(all_paper_ids),desc="Retrieving Arxiv papers")
        for i in range(0,len(all_paper_ids),max_results):
            search = arxiv.Search(id_list=all_paper_ids[i:i+max_results])
            results = list(client.results(search))
            filtered_results = filter_recent_papers(results)
            batch = [ArxivPaper(p) for p in filtered_results]
            bar.update(len(batch))
            papers.extend(batch)
        bar.close()
    else:
        logger.debug("Retrieve 5 arxiv papers regardless of the date.")
        search = arxiv.Search(query='cat:cs.AI', sort_by=arxiv.SortCriterion.SubmittedDate)
        papers = []
        for i in client.results(search):
            papers.append(ArxivPaper(i))
            if len(papers) == 5:
                break
    return papers

def get_arxiv_paper_by_keyword(query:str, debug:bool=False, max_results:int=10) -> list[ArxivPaper]:
    # Search papers from Arxiv by keywords and append to the list
    client = arxiv.Client(num_retries=10,delay_seconds=10)
    
    # 使用简单的关键词搜索，不在查询中限制日期
    search_query = f"all:{query.strip()}"
    
    logger.debug(f"Search query: {search_query}")
    search = arxiv.Search(query=search_query, max_results=max_results*3, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    # 获取结果并过滤
    all_results = list(client.results(search))
    filtered_results = filter_recent_papers(all_results)
    # sort by published date and truncate to max_results
    filtered_results.sort(key=lambda x: x.published, reverse=True)
    filtered_results = filtered_results[:max_results]
    
    return [ArxivPaper(p, keyword=query.strip()) for p in filtered_results]

parser = argparse.ArgumentParser(description='Recommender system for academic papers')

def add_argument(*args, **kwargs):
    def get_env(key:str,default=None):
        # handle environment variables generated at Workflow runtime
        # Unset environment variables are passed as '', we should treat them as None
        v = os.environ.get(key)
        if v == '' or v is None:
            return default
        return v
    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get('dest',args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)
    if env_value is not None:
        #convert env_value to the specified type
        if kwargs.get('type') == bool:
            env_value = env_value.lower() in ['true','1']
        else:
            env_value = kwargs.get('type')(env_value)
        parser.set_defaults(**{arg_full_name:env_value})


if __name__ == '__main__':
    
    add_argument('--zotero_id', type=str, help='Zotero user ID')
    add_argument('--zotero_key', type=str, help='Zotero API key')
    add_argument('--zotero_ignore',type=str,help='Zotero collection to ignore, using gitignore-style pattern.')
    add_argument('--send_empty', type=bool, help='If get no arxiv paper, send empty email',default=False)
    add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend',default=100)
    add_argument('--arxiv_query', type=str, help='Arxiv search query by category')
    add_argument('--arxiv_query_keyword', type=str, help='Arxiv search query by keyword', default=None)
    add_argument('--smtp_server', type=str, help='SMTP server')
    add_argument('--smtp_port', type=int, help='SMTP port')
    add_argument('--sender', type=str, help='Sender email address')
    add_argument('--receiver', type=str, help='Receiver email address')
    add_argument('--sender_password', type=str, help='Sender email password')
    add_argument('--research_interests', type=str, help='Research interests', default=None)
    add_argument(
        "--use_llm_api",
        type=bool,
        help="Use OpenAI API to generate TLDR",
        default=False,
    )
    add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        default=None,
    )
    add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://api.openai.com/v1",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="gpt-4o",
    )
    add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
        default="English",
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    def parse_research_interests(interests):
        """解析研究兴趣，支持字符串和列表格式"""
        if not interests:
            return None
        if isinstance(interests, str):
            return [interest.strip() for interest in interests.split(',')]
        return interests

    def load_config_from_yaml(config_file="private_config.yaml"):
        """从YAML文件加载配置"""
        if not os.path.exists(config_file):
            return {}
        
        logger.info(f"找到配置文件 {config_file}，正在读取...")
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    # 读取配置
    yaml_config = load_config_from_yaml()
    llm_config = yaml_config.get('LLM_RECOMMENDER', {})
    
    # 从YAML配置中覆盖参数值（YAML优先级高于命令行参数）
    if yaml_config:
        for key, value in yaml_config.items():
            arg_name = key.lower()
            if hasattr(args, arg_name):
                setattr(args, arg_name, value)
    
    # 研究兴趣优先级：YAML > 命令行参数
    research_interests = (
        parse_research_interests(llm_config.get('RESEARCH_INTERESTS')) or
        parse_research_interests(args.research_interests)
    )
    
    # 合并最终配置
    llm_recommender_config = {
        'research_interests': research_interests,
        'corpus_batch_size': llm_config.get('CORPUS_BATCH_SIZE', 20),
        'candidate_batch_size': llm_config.get('CANDIDATE_BATCH_SIZE', 8),
        'keyword_bonus': llm_config.get('KEYWORD_BONUS', 2.0),
        'default_score': llm_config.get('DEFAULT_SCORE', 5.0)
    }
    logger.info(f"读取LLM推荐配置: {llm_recommender_config}")

    assert (
        not args.use_llm_api or args.openai_api_key is not None
    )  # If use_llm_api is True, openai_api_key must be provided
    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    if args.zotero_ignore:
        logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")
        for c in corpus:
            logger.info(f"Paper: {c['data']['title']}, Paths: {c['paths']}")
    logger.info("Retrieving Arxiv papers...")
    papers = get_arxiv_paper_by_category(args.arxiv_query, args.debug)
    logger.info(f"Retrieved {len(papers)} papers from category search.")

    # parse arxiv_query_keyword and search papers by keyword
    if args.arxiv_query_keyword:
        logger.info("Searching papers by keywords...")
        keywords = [k.strip() for k in args.arxiv_query_keyword.split(',')]
        
        keyword_papers = []
        for arxiv_keyword in keywords:
            batch = get_arxiv_paper_by_keyword(arxiv_keyword, args.debug)
            logger.info(f"Found {len(batch)} papers for keyword '{arxiv_keyword}'")
            keyword_papers.extend(batch)
        
        # 去重：基于论文ID去重
        existing_ids = {paper.arxiv_id for paper in papers}
        unique_keyword_papers = [p for p in keyword_papers if p.arxiv_id not in existing_ids]
        logger.info(f"Added {len(unique_keyword_papers)} unique papers from keyword search (removed {len(keyword_papers) - len(unique_keyword_papers)} duplicates)")
        papers.extend(unique_keyword_papers)

    # 去重papers，基于arxiv_id, 并按发布时间排序
    papers = list(set(papers))
    papers.sort(key=lambda x: x.published, reverse=True)
    logger.info(f"Total papers retrieved: {len(papers)}")

    # 设置LLM（在rerank之前）
    if args.use_llm_api:
        logger.info("Using OpenAI API as global LLM.")
        set_global_llm(api_key=args.openai_api_key, base_url=args.openai_api_base, model=args.model_name, lang=args.language)
    else:
        logger.info("Using Local LLM as global LLM.")
        set_global_llm(lang=args.language)

    if len(papers) == 0:
        logger.info("No new papers found. Yesterday maybe a holiday and no one submit their work :). If this is not the case, please check the ARXIV_QUERY.")
        if not args.send_empty:
          exit(0)
    else:
        logger.info("Reranking papers...")
        # 使用LLM进行智能推荐，传递配置参数
        papers = rerank_paper(papers, corpus, use_llm=args.use_llm_api, llm_config=llm_recommender_config)
        if args.max_paper_num != -1:
            papers = papers[:args.max_paper_num]
    
    html = render_email(papers)
    logger.info("Sending email...")
    send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)
    logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")

    
    
