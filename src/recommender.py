import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.paper import ArxivPaper
from datetime import datetime
from loguru import logger
from tqdm import tqdm
from src.llm import get_llm
import json
import os
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import re

class AuthorBasedRecommender:
    def __init__(self, author_data_file: str = "author_data.json"):
        """初始化基于作者的推荐器"""
        self.key_authors = set()
        self.load_author_data(author_data_file)
        
    def load_author_data(self, author_data_file: str):
        """加载作者数据 - 无脑信任JSON中的所有作者"""
        if not os.path.exists(author_data_file):
            logger.warning(f"作者数据文件不存在: {author_data_file}")
            return
            
        logger.info("加载作者数据...")
        with open(author_data_file, 'r', encoding='utf-8') as f:
            author_data = json.load(f)
        
        # 无脑加载所有作者
        if isinstance(author_data, list):
            # 新格式：数组
            for author in author_data:
                name = author.get('name', '')
                if name:
                    self.key_authors.add(name)
        else:
            # 旧格式：字典
            for author_name in author_data.keys():
                if author_name:
                    self.key_authors.add(author_name)
        
        logger.info(f"加载了 {len(self.key_authors)} 位关键作者")

def extract_authors_from_paper(paper: ArxivPaper) -> List[str]:
    """从论文中提取作者列表"""
    authors = []
    if hasattr(paper, 'authors') and paper.authors:
        for author in paper.authors:
            # 处理作者名字格式
            if hasattr(author, 'name'):
                full_name = author.name
            else:
                full_name = str(author)
            authors.append(full_name.strip())
    return authors

def author_name_match(paper_author: str, key_author: str) -> bool:
    """简单的作者名字匹配"""
    paper_author = paper_author.lower().strip()
    key_author = key_author.lower().strip()
    
    # 完全匹配
    if paper_author == key_author:
        return True
        
    # 简单的部分匹配（姓氏匹配）
    paper_parts = paper_author.split()
    key_parts = key_author.split()
    
    if len(paper_parts) >= 2 and len(key_parts) >= 2:
        # 姓氏匹配且名字首字母匹配
        if paper_parts[-1] == key_parts[-1] and paper_parts[0][0] == key_parts[0][0]:
            return True
    
    return False

def is_paper_from_key_author(paper: ArxivPaper, author_recommender: AuthorBasedRecommender) -> Tuple[bool, List[str]]:
    """检查论文是否来自关键作者"""
    paper_authors = extract_authors_from_paper(paper)
    matched_key_authors = []
    
    for paper_author in paper_authors:
        for key_author in author_recommender.key_authors:
            if author_name_match(paper_author, key_author):
                matched_key_authors.append(key_author)
                break  # 找到匹配就跳出
    
    if matched_key_authors:
        logger.info(f"关键作者论文: {paper.title[:50]}... -> {matched_key_authors[0]}")
        return True, matched_key_authors
    
    return False, []

def llm_based_rerank_paper(candidate: list[ArxivPaper], corpus: list[dict], 
                           config: dict = None) -> list[ArxivPaper]:
    """
    使用LLM基于用户研究兴趣进行智能推荐
    
    Args:
        candidate: 候选论文列表
        corpus: 用户历史论文库
        config: 推荐配置字典，包含所有必要参数
    """
    logger.info("开始使用LLM进行智能推荐...")
    
    # 使用配置参数，提供默认值
    if config is None:
        config = {}
    
    research_interests = config.get('research_interests', ["embodied AI"])
    corpus_batch_size = config.get('corpus_batch_size', 20)
    candidate_batch_size = config.get('candidate_batch_size', 8)
    keyword_bonus = config.get('keyword_bonus', 2.0)
    default_score = config.get('default_score', 5.0)
    abstract_max_length = config.get('abstract_max_length', 500)
    
    # 处理研究兴趣参数
    if isinstance(research_interests, str):
        research_interests = [interest.strip() for interest in research_interests.split(',')]
    
    logger.info(f"配置参数: corpus_batch_size={corpus_batch_size}, candidate_batch_size={candidate_batch_size}")
    logger.info(f"研究兴趣领域: {', '.join(research_interests)}")
    
    # 构建corpus摘要信息
    corpus_sorted = sorted(corpus, key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'), reverse=True)
    recent_corpus = corpus_sorted[:corpus_batch_size]
    
    corpus_info = []
    for i, paper in enumerate(recent_corpus):
        corpus_info.append({
            "id": i + 1,
            "title": paper['data']['title'],
            "abstract": paper['data']['abstractNote'][:abstract_max_length]  # 使用配置的长度限制
        })
    
    # 分批处理候选论文
    scored_candidates = []
    
    for i in range(0, len(candidate), candidate_batch_size):
        batch = candidate[i:i+candidate_batch_size]
        logger.info(f"处理第 {i//candidate_batch_size + 1} 批候选论文 ({len(batch)} 篇)")
        
        # 构建批次候选论文信息
        candidate_info = []
        for j, paper in enumerate(batch):
            candidate_info.append({
                "id": j + 1,
                "title": paper.title,
                "abstract": paper.summary[:abstract_max_length]  # 使用配置的长度限制
            })
        
        # 构建prompt
        prompt = f"""
你是一位AI研究领域的专家。请根据用户的研究兴趣和历史阅读偏好，和候选论文的学术贡献，为候选论文打分。

用户的主要研究兴趣包括：{', '.join(research_interests)}

用户最近阅读的论文示例：
{json.dumps(corpus_info, ensure_ascii=False, indent=2)}

请为以下候选论文打分（1-10分，10分最相关）：
{json.dumps(candidate_info, ensure_ascii=False, indent=2)}

评分标准：
- 9-10分：与用户核心研究兴趣高度相关，且论文学术贡献高，具有重要学术价值
- 7-8分：与用户研究兴趣相关，且论文学术贡献较高，值得关注
- 5-6分：部分相关，可能有一定参考价值，或论文学术贡献一般
- 3-4分：相关性较低，但在相关领域，或论文学术贡献较低
- 0-2分：基本不相关，或论文学术贡献很低

请以JSON格式返回评分结果，格式如下：
{{
  "scores": [
    {{"id": 1, "score": 8.5, "reason": "简短评分理由"}},
    {{"id": 2, "score": 6.0, "reason": "简短评分理由"}},
    ...
  ]
}}

请确保返回的JSON格式正确，并为每篇论文提供合理的评分和简短理由。
"""

        try:
            llm = get_llm()
            messages = [
                {"role": "system", "content": "你是一位专业的AI研究领域专家，擅长评估学术论文的相关性和重要性。"},
                {"role": "user", "content": prompt}
            ]
            
            response = llm.generate(messages)
            logger.debug(f"LLM响应: {response}")
            
            # 解析响应
            try:
                # 提取JSON部分
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    json_str = response.split("```")[1].strip()
                else:
                    json_str = response.strip()
                
                result = json.loads(json_str)
                scores_data = result.get("scores", [])
                
                # 应用评分
                for score_info in scores_data:
                    paper_id = score_info["id"] - 1
                    if 0 <= paper_id < len(batch):
                        score = float(score_info["score"])
                        reason = score_info.get("reason", "")
                        batch[paper_id].score = score
                        batch[paper_id].llm_reason = reason
                        logger.info(f"论文: {batch[paper_id].title[:50]}... | 评分: {score} | 理由: {reason}")
                
                scored_candidates.extend(batch)
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"解析LLM响应失败: {e}，使用默认评分")
                for paper in batch:
                    paper.score = default_score
                scored_candidates.extend(batch)
                
        except Exception as e:
            logger.error(f"LLM评分失败: {e}，使用默认评分")
            for paper in batch:
                paper.score = default_score
            scored_candidates.extend(batch)
    
    # 关键词加分
    scored_candidates = keyword_score_update(scored_candidates, keyword_bonus, config)
    
    # 排序
    scored_candidates = sorted(scored_candidates, key=lambda x: x.score, reverse=True)
    
    logger.info("LLM评分完成，论文已按分数排序")
    return scored_candidates

def rerank_with_author_priority(candidate: List[ArxivPaper], corpus: List[dict], 
                               model: str = 'avsolatorio/GIST-small-Embedding-v0', 
                               use_llm: bool = True, 
                               llm_config: dict = None) -> List[ArxivPaper]:
    """
    两阶段推荐：先按相关性排序，然后将关键作者的论文提到前面
    """
    logger.info("开始推荐：相关性排序 + 关键作者优先")
    
    # 第一阶段：按相关性排序
    if use_llm:
        try:
            if llm_config:
                ranked_papers = llm_based_rerank_paper(
                    candidate, corpus, config=llm_config
                )
            else:
                ranked_papers = llm_based_rerank_paper(candidate, corpus, config={})
        except Exception as e:
            logger.error(f"LLM推荐失败，使用传统方法: {e}")
            ranked_papers = traditional_rerank_paper(candidate, corpus, model, llm_config)
    else:
        ranked_papers = traditional_rerank_paper(candidate, corpus, model, llm_config)

    # 过滤低分论文
    score_threshold = llm_config.get('score_filter_threshold', 5.0) if llm_config else 5.0
    ranked_papers = [paper for paper in ranked_papers if paper.score > score_threshold]
    
    # 第二阶段：关键作者优先
    author_recommender = AuthorBasedRecommender()
    
    if not author_recommender.key_authors:
        logger.warning("没有关键作者数据")
        return ranked_papers
    
    # 分离论文
    key_author_papers = []
    other_papers = []
    
    for paper in ranked_papers:
        is_key_author, matched_authors = is_paper_from_key_author(paper, author_recommender)
        if is_key_author:
            paper.key_authors = matched_authors
            key_author_papers.append(paper)
        else:
            other_papers.append(paper)
    
    # 合并结果：关键作者论文在前
    final_result = key_author_papers + other_papers
    
    logger.info(f"推荐完成: {len(key_author_papers)} 篇关键作者论文在前，{len(other_papers)} 篇其他论文")
    
    return final_result

def traditional_rerank_paper(candidate: List[ArxivPaper], corpus: List[dict], 
                           model: str = None, config: dict = None) -> List[ArxivPaper]:
    """传统的基于嵌入相似度的推荐方法"""
    logger.info("使用传统嵌入相似度方法进行推荐...")
    
    # 使用配置参数
    if config is None:
        config = {}
    
    if model is None:
        model = config.get('embedding_model', 'avsolatorio/GIST-small-Embedding-v0')
    
    use_time_decay = config.get('use_time_decay', True)
    score_scale_factor = config.get('score_scale_factor', 10.0)
    
    encoder = SentenceTransformer(model)

    # 按日期排序corpus
    corpus = sorted(corpus, key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'), reverse=True)
    
    if use_time_decay:
        time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        time_decay_weight = time_decay_weight / time_decay_weight.sum()
    else:
        time_decay_weight = np.ones(len(corpus)) / len(corpus)
    
    logger.info("Encoding corpus abstracts...")
    corpus_feature = encoder.encode([paper['data']['abstractNote'] for paper in tqdm(corpus, desc="Corpus")])
    logger.info("Encoding candidate papers...")
    candidate_feature = encoder.encode([paper.summary for paper in tqdm(candidate, desc="Candidates")])
    
    sim = encoder.similarity(candidate_feature, corpus_feature)
    scores = (sim * time_decay_weight).sum(axis=1) * score_scale_factor
    for s, c in zip(scores, candidate):
        c.score = s.item()

    keyword_bonus = config.get('keyword_bonus', 0.5)
    candidate = keyword_score_update(candidate, keyword_bonus, config)
    candidate = sorted(candidate, key=lambda x: x.score, reverse=True)
    return candidate

def keyword_score_update(candidate: List[ArxivPaper], keyword_bonus: float = 0.5, config: dict = None) -> List[ArxivPaper]:
    """为关键词匹配的论文添加额外分数"""
    if config is None:
        config = {}
    
    max_score_limit = config.get('max_score_limit', 10.0)
    
    for c in candidate:
        if hasattr(c, 'search_keyword') and c.search_keyword:
            c.score = min(c.score + keyword_bonus, max_score_limit)
            logger.debug(f"关键词匹配加分: {c.title[:50]}... (+{keyword_bonus}分)")
    return candidate

def rerank_paper(candidate: List[ArxivPaper], corpus: List[dict], 
                 model: str = 'avsolatorio/GIST-small-Embedding-v0', 
                 use_llm: bool = True, 
                 llm_config: dict = None) -> List[ArxivPaper]:
    """主推荐函数"""
    return rerank_with_author_priority(candidate, corpus, model, use_llm, llm_config)


