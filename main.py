import arxiv
import argparse
import os
import sys
from dotenv import load_dotenv
import yaml
load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from recommender import rerank_paper
from construct_email import render_email, send_email
from tqdm import trange,tqdm
from loguru import logger
from zotero_utils import get_zotero_corpus, filter_corpus
from paper import ArxivPaper
from llm import set_global_llm
import feedparser
from datetime import datetime, timedelta, timezone



def filter_recent_papers(papers: list, days: int = 3) -> list:
    """过滤最近N天的论文
    
    Args:
        papers: 论文列表
        days: 天数
        
    Returns:
        过滤后的论文列表
    """
    recent_days = datetime.now(timezone.utc) - timedelta(days=days)
    logger.debug(f"Filtering papers since: {recent_days} (UTC)")
    
    filtered_papers = []
    for p in papers:
        if p.published >= recent_days:
            filtered_papers.append(p)
        else:
            logger.debug(f"Filtered out paper published on {p.published}: {p.title[:50]}...")
    
    logger.debug(f"Date filter: {len(papers)} -> {len(filtered_papers)} papers")
    return filtered_papers

def get_arxiv_paper_by_category(query:str, debug:bool=False, max_results:int=50) -> list[ArxivPaper]:
    client = arxiv.Client(num_retries=10,delay_seconds=10)
    rss_url = f"https://rss.arxiv.org/atom/{query}"
    logger.info(f"Fetching RSS feed from: {rss_url}")
    
    feed = feedparser.parse(rss_url)
    
    # 添加更详细的feed信息调试
    logger.info(f"RSS feed status: {getattr(feed, 'status', 'N/A')}")
    logger.info(f"RSS feed title: {getattr(feed.feed, 'title', 'N/A')}")
    logger.info(f"RSS feed updated: {getattr(feed.feed, 'updated', 'N/A')}")
    logger.info(f"RSS feed retrieved. Total entries: {len(feed.entries)}")
    
    if 'Feed error for query' in feed.feed.title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")
    
    # Search papers from Arxiv by category
    if not debug:
        papers = []
        
        # 如果RSS feed有数据，使用RSS方式
        if len(feed.entries) > 0:
            all_paper_ids = [i.id.removeprefix("oai:arXiv.org:") for i in feed.entries if hasattr(i, 'arxiv_announce_type') and i.arxiv_announce_type == 'new']
            logger.info(f"Found {len(all_paper_ids)} new papers from RSS feed")
            
            # 调试：显示前几个论文ID和相关信息
            for i, entry in enumerate(feed.entries[:5]):
                announce_type = getattr(entry, 'arxiv_announce_type', 'N/A')
                published = getattr(entry, 'published', 'N/A')
                entry_id = entry.id.removeprefix('oai:arXiv.org:') if hasattr(entry, 'id') else 'N/A'
                logger.info(f"Entry {i}: ID={entry_id}, announce_type={announce_type}, published={published}")
            
            if len(all_paper_ids) > 0:
                bar = tqdm(total=len(all_paper_ids),desc="Retrieving Arxiv papers")
                for i in range(0,len(all_paper_ids),max_results):
                    search = arxiv.Search(id_list=all_paper_ids[i:i+max_results])
                    results = list(client.results(search))
                    logger.info(f"Batch {i//max_results + 1}: Retrieved {len(results)} papers from arxiv API")
                    
                    filtered_results = filter_recent_papers(results)
                    logger.info(f"Batch {i//max_results + 1}: After date filtering: {len(filtered_results)} papers")
                    
                    batch = [ArxivPaper(p) for p in filtered_results]
                    bar.update(len(batch))
                    papers.extend(batch)
                bar.close()
            else:
                logger.warning("RSS feed中没有找到'new'类型的论文，可能今天没有新论文发布")
        
        # 如果RSS feed为空或没有找到新论文，使用直接搜索作为回退
        if len(papers) == 0:
            logger.info("RSS feed为空或无新论文，使用直接搜索作为回退方案...")
            
            # 将查询字符串转换为搜索查询
            categories = query.split('+')
            search_queries = []
            for cat in categories:
                search_queries.append(f"cat:{cat}")
            
            combined_query = " OR ".join(search_queries)
            logger.info(f"使用搜索查询: {combined_query}")
            
            search = arxiv.Search(
                query=combined_query, 
                max_results=max_results*2,  # 获取更多结果以便过滤
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            all_results = list(client.results(search))
            logger.info(f"直接搜索获取到 {len(all_results)} 篇论文")
            
            # 过滤最近的论文
            filtered_results = filter_recent_papers(all_results)
            logger.info(f"日期过滤后剩余 {len(filtered_results)} 篇论文")
            
            # 限制数量
            filtered_results = filtered_results[:max_results]
            papers = [ArxivPaper(p) for p in filtered_results]
            logger.info(f"最终返回 {len(papers)} 篇论文")
            
    else:
        logger.debug("Debug模式：获取5篇cs.AI论文，不考虑日期限制")
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
        
        papers.extend(keyword_papers)

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
        
        # 调试：检查推荐后的论文分数分布
        scores = [getattr(p, 'score', None) for p in papers]
        logger.info(f"推荐后论文分数分布: 最高分={max(scores) if scores else 'N/A'}, 最低分={min(scores) if scores else 'N/A'}")
        logger.info(f"分数为None的论文数: {sum(1 for s in scores if s is None)}")
        logger.info(f"分数小于等于5的论文数: {sum(1 for s in scores if s is not None and s <= 5)}")
        
        # 打印前几篇论文的详细信息
        for i, p in enumerate(papers[:5]):
            logger.info(f"论文 {i+1}: {p.title[:50]}... | 分数: {getattr(p, 'score', 'None')}")
        
        if args.max_paper_num != -1:
            papers = papers[:args.max_paper_num]
    
    html = render_email(papers)
    logger.info("Sending email...")
    send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)
    logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")
    
    
    
