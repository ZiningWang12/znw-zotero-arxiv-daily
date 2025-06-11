import arxiv
from datetime import datetime, timedelta, timezone
from loguru import logger
from src.paper import ArxivPaper


def filter_recent_papers(papers: list, days: int = 2) -> list:
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


def get_arxiv_paper_by_category(query: str, debug: bool = False, max_results: int = 50) -> list[ArxivPaper]:
    """根据类别搜索arxiv论文"""
    client = arxiv.Client(num_retries=10, delay_seconds=10)
    
    if not debug:
        # 将查询字符串转换为搜索查询
        categories = query.split('+')
        search_queries = []
        for cat in categories:
            search_queries.append(f"cat:{cat}")
        
        combined_query = " OR ".join(search_queries)
        logger.info(f"使用搜索查询: {combined_query}")
        
        # 设置一个较大的搜索限制来确保获取足够多的结果
        # 然后通过日期过滤来筛选出真正需要的论文
        search_limit = 1000  # 设置一个合理的上限
        
        search = arxiv.Search(
            query=combined_query, 
            max_results=search_limit,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        all_results = []
        
        # 获取结果并逐步检查，直到找到足够多的最近论文或确认没有更多最近论文
        logger.info(f"开始获取论文，最大搜索数量: {search_limit}")
        batch_count = 0
        
        for result in client.results(search):
            all_results.append(result)
            
            # 每处理一定数量的论文就检查一次日期过滤结果
            if len(all_results) % max_results == 0:
                batch_count += 1
                logger.info(f"批次 {batch_count}: 已获取 {len(all_results)} 篇论文")
                
                # 检查最近一批论文是否还有符合日期条件的
                recent_batch = all_results[-max_results:]
                filtered_batch = filter_recent_papers(recent_batch)
                
                # 如果最近一批论文都不符合日期条件，可能已经超出时间范围
                if not filtered_batch:
                    logger.info(f"最近 {max_results} 篇论文都不符合日期条件，停止搜索")
                    break
        
        logger.info(f"总共获取到 {len(all_results)} 篇论文")
        
        # 对所有结果进行日期过滤
        filtered_results = filter_recent_papers(all_results)
        logger.info(f"日期过滤后剩余 {len(filtered_results)} 篇论文")
        
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


def get_arxiv_paper_by_keyword(query: str, debug: bool = False, max_results: int = 10) -> list[ArxivPaper]:
    """根据关键词搜索arxiv论文"""
    # Search papers from Arxiv by keywords and append to the list
    client = arxiv.Client(num_retries=10, delay_seconds=10)
    
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


def get_arxiv_papers_by_keywords(arxiv_query_keyword: str, debug: bool = False) -> list[ArxivPaper]:
    """根据关键词列表获取arxiv论文"""
    if not arxiv_query_keyword:
        return []
    
    logger.info("Searching papers by keywords...")
    keywords = [k.strip() for k in arxiv_query_keyword.split(',')]
    
    keyword_papers = []
    if not debug:
        for arxiv_keyword in keywords:
            batch = get_arxiv_paper_by_keyword(arxiv_keyword, debug)
            logger.info(f"Found {len(batch)} papers for keyword '{arxiv_keyword}'")
            keyword_papers.extend(batch)
    else:
        keyword_papers = get_arxiv_paper_by_keyword("robotics", debug, max_results=3)
        logger.info(f"Found {len(keyword_papers)} papers for keyword 'robotics'")
    
    return keyword_papers


def deduplicate_and_sort_papers(papers: list[ArxivPaper]) -> list[ArxivPaper]:
    """去重并按发布时间排序论文"""
    # 去重papers，基于arxiv_id, 并按发布时间排序
    papers = list(set(papers))
    papers.sort(key=lambda x: x.published, reverse=True)
    return papers 