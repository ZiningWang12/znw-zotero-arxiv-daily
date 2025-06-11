import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from loguru import logger
from utils.zotero_utils import get_zotero_corpus, filter_corpus
from src.llm import set_global_llm
from src.recommender import rerank_paper
from utils.construct_email import render_email, send_email

# 导入重构后的模块
from config.config import create_argument_parser, merge_configs, validate_config
from src.arxiv_client import (
    get_arxiv_paper_by_category, 
    get_arxiv_papers_by_keywords, 
    deduplicate_and_sort_papers
)
from src.paper_processor import limit_papers_by_type, print_paper_statistics


def setup_logging(debug: bool):
    """设置日志配置"""
    if debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")


def get_zotero_papers(args):
    """获取Zotero论文库"""
    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    
    if args.zotero_ignore:
        logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")
        for c in corpus:
            logger.info(f"Paper: {c['data']['title']}, Paths: {c['paths']}")
    
    return corpus


def get_arxiv_papers(args):
    """获取arxiv论文"""
    logger.info("Retrieving Arxiv papers...")
    papers = get_arxiv_paper_by_category(args.arxiv_query, args.debug)
    logger.info(f"Retrieved {len(papers)} papers from category search.")
    
    # 添加关键词搜索的论文
    keyword_papers = get_arxiv_papers_by_keywords(args.arxiv_query_keyword, args.debug)
    papers.extend(keyword_papers)
    
    # 去重并排序
    papers = deduplicate_and_sort_papers(papers)
    logger.info(f"Total papers retrieved: {len(papers)}")
    
    return papers


def setup_llm(args, llm_recommender_config):
    """设置LLM"""
    if args.use_llm_api:
        logger.info("Using OpenAI API as global LLM.")
        set_global_llm(
            api_key=args.openai_api_key, 
            base_url=args.openai_api_base, 
            model=args.model_name, 
            lang=args.language,
            config=llm_recommender_config
        )
    else:
        logger.info("Using Local LLM as global LLM.")
        set_global_llm(lang=args.language, config=llm_recommender_config)


def process_papers(papers, corpus, args, llm_recommender_config):
    """处理论文：推荐排序和数量限制"""
    if len(papers) == 0:
        logger.info("No new papers found. Yesterday maybe a holiday and no one submit their work :). "
                   "If this is not the case, please check the ARXIV_QUERY.")
        if not args.send_empty:
            exit(0)
        return papers
    
    logger.info("Reranking papers...")
    # 使用LLM进行智能推荐，传递配置参数
    papers = rerank_paper(papers, corpus, use_llm=args.use_llm_api, llm_config=llm_recommender_config)
    
    # 打印统计信息
    print_paper_statistics(papers)
    
    # 限制论文数量
    papers = limit_papers_by_type(papers, args.max_paper_num)
    
    return papers


def main():
    """主函数"""
    # 解析配置
    parser = create_argument_parser()
    args = parser.parse_args()
    args, llm_recommender_config = merge_configs(args)
    validate_config(args)
    
    # 设置日志
    setup_logging(args.debug)
    
    # 获取Zotero论文库
    corpus = get_zotero_papers(args)
    
    # 获取arxiv论文
    papers = get_arxiv_papers(args)
    
    # 设置LLM
    setup_llm(args, llm_recommender_config)
    
    # 处理论文
    papers = process_papers(papers, corpus, args, llm_recommender_config)
    
    # 生成和发送邮件
    html = render_email(papers)
    logger.info("Sending email...")
    send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)
    logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")


if __name__ == '__main__':
    main()
