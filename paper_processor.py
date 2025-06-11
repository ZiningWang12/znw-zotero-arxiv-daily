from loguru import logger
from paper import ArxivPaper


def limit_papers_by_type(papers: list[ArxivPaper], max_paper_num: int) -> list[ArxivPaper]:
    """分别限制关键作者论文和非关键作者论文的数量"""
    if max_paper_num == -1:
        return papers
    
    # 分离关键作者论文和普通论文
    key_author_papers = []
    other_papers = []
    
    for paper in papers:
        if hasattr(paper, 'key_authors') and paper.key_authors:
            key_author_papers.append(paper)
        else:
            other_papers.append(paper)
    
    # 分别限制数量
    limited_key_author_papers = key_author_papers[:max_paper_num]
    limited_other_papers = other_papers[:max_paper_num]
    
    logger.info(f"限制论文数量: 关键作者论文 {len(limited_key_author_papers)}/{len(key_author_papers)}, "
                f"其他论文 {len(limited_other_papers)}/{len(other_papers)}")
    
    # 重新合并，保持关键作者论文在前的顺序
    return limited_key_author_papers + limited_other_papers


def print_paper_statistics(papers: list[ArxivPaper]):
    """打印论文推荐统计信息"""
    # 调试：检查推荐后的论文分数分布
    scores = [getattr(p, 'score', None) for p in papers]
    logger.info(f"推荐后论文分数分布: 最高分={max(scores) if scores else 'N/A'}, "
                f"最低分={min(scores) if scores else 'N/A'}")
    logger.info(f"分数为None的论文数: {sum(1 for s in scores if s is None)}")
    logger.info(f"分数小于等于5的论文数: {sum(1 for s in scores if s is not None and s <= 5)}")
    
    # 打印前几篇论文的详细信息
    for i, p in enumerate(papers[:5]):
        logger.info(f"论文 {i+1}: {p.title[:50]}... | 分数: {getattr(p, 'score', 'None')}") 