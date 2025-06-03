import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from paper import ArxivPaper
from datetime import datetime
from loguru import logger
from tqdm import tqdm
from llm import get_llm
import json

def llm_based_rerank_paper(candidate: list[ArxivPaper], corpus: list[dict], 
                           research_interests: list[str] = None,
                           corpus_batch_size: int = 20, 
                           candidate_batch_size: int = 8,
                           keyword_bonus: float = 2.0,
                           default_score: float = 5.0) -> list[ArxivPaper]:
    """
    使用LLM（Gemini API）基于用户研究兴趣进行智能推荐
    
    Args:
        candidate: 候选论文列表
        corpus: 用户历史论文库
        research_interests: 用户研究兴趣领域列表（可以是字符串或列表）
        corpus_batch_size: 用于参考的历史论文数量
        candidate_batch_size: 每批处理的候选论文数量
        keyword_bonus: 关键词匹配论文的额外加分
        default_score: API调用失败时的默认分数
    """
    logger.info("开始使用LLM进行智能推荐...")
    
    # 处理研究兴趣参数，支持字符串和列表格式
    if research_interests is None:
        research_interests = ["embodied AI"]
    elif isinstance(research_interests, str):
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
            "abstract": paper['data']['abstractNote'][:500]  # 限制长度
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
                "abstract": paper.summary[:500]  # 限制长度
            })
        
        # 构建prompt
        prompt = f"""
你是一位AI研究领域的专家。请根据用户的研究兴趣和历史阅读偏好，为候选论文打分。

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
- 1-2分：基本不相关，或论文学术贡献很低

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
                    paper_id = score_info["id"] - 1  # 转换为0-based索引
                    if 0 <= paper_id < len(batch):
                        score = float(score_info["score"])
                        reason = score_info.get("reason", "")
                        batch[paper_id].score = score
                        batch[paper_id].llm_reason = reason
                        logger.info(f"论文: {batch[paper_id].title[:50]}... | 评分: {score} | 理由: {reason}")
                
                scored_candidates.extend(batch)
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"解析LLM响应失败: {e}，使用默认评分")
                # 如果解析失败，使用默认评分
                for paper in batch:
                    paper.score = default_score
                scored_candidates.extend(batch)
                
        except Exception as e:
            logger.error(f"LLM评分失败: {e}，使用默认评分")
            # 如果LLM调用失败，使用默认评分
            for paper in batch:
                paper.score = default_score
            scored_candidates.extend(batch)
    
    # 对关键词匹配的论文给予额外加分
    # scored_candidates = keyword_score_update(scored_candidates, keyword_bonus)
    
    # 排序
    scored_candidates = sorted(scored_candidates, key=lambda x: x.score, reverse=True)
    
    logger.info("LLM评分完成，论文已按分数排序")
    return scored_candidates

def rerank_paper(candidate:list[ArxivPaper], corpus:list[dict], 
                 model:str='avsolatorio/GIST-small-Embedding-v0', 
                 use_llm:bool=True, llm_config:dict=None) -> list[ArxivPaper]:
    """
    重新排名候选论文
    
    Args:
        candidate: 候选论文列表
        corpus: 用户历史论文库
        model: 嵌入模型名称（当use_llm=False时使用）
        use_llm: 是否使用LLM进行评分
        llm_config: LLM推荐算法配置字典
    """
    if use_llm:
        try:
            if llm_config:
                return llm_based_rerank_paper(
                    candidate, 
                    corpus,
                    research_interests=llm_config.get('research_interests'),
                    corpus_batch_size=llm_config.get('corpus_batch_size', 20),
                    candidate_batch_size=llm_config.get('candidate_batch_size', 8),
                    keyword_bonus=llm_config.get('keyword_bonus', 2.0),
                    default_score=llm_config.get('default_score', 5.0)
                )
            else:
                return llm_based_rerank_paper(candidate, corpus)
        except Exception as e:
            logger.error(f"LLM推荐失败，回退到传统方法: {e}")
    
    # 传统的基于嵌入相似度的方法
    logger.info("使用传统嵌入相似度方法进行推荐...")
    encoder = SentenceTransformer(model)

    #sort corpus by date, from newest to oldest
    corpus = sorted(corpus,key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),reverse=True)
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
    logger.info("Encoding corpus abstracts...")
    logger.info("Corpus feature sample: {}".format(corpus[0]['data']['abstractNote']))
    corpus_feature = encoder.encode([paper['data']['abstractNote'] for paper in tqdm(corpus, desc="Corpus")])
    logger.info("Encoding candidate papers...")
    candidate_feature = encoder.encode([paper.summary for paper in tqdm(candidate, desc="Candidates")])
    
    sim = encoder.similarity(candidate_feature,corpus_feature) # [n_candidate, n_corpus]
    scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
    for s,c in zip(scores,candidate):
        c.score = s.item()

    # debug score
    for i,c in enumerate(corpus):
        logger.info(f"Corpus: {c['data']['title']}, Score: {sim[i].max()}, candidate: {candidate[sim[i].argmax()].title}")
    for i, c in enumerate(candidate):
        logger.info(f"Paper: {c.title}, Score: {c.score}, maxScore: {max(sim[i])}, zotero_paper: {corpus[sim[i].argmax()]['data']['title']}")

    candidate = keyword_score_update(candidate)
    candidate = sorted(candidate,key=lambda x: x.score,reverse=True)
    return candidate

def keyword_score_update(candidate:list[ArxivPaper], keyword_bonus:float=2.0) -> list[ArxivPaper]:
    """
    为关键词匹配的论文添加额外分数
    
    Args:
        candidate: 候选论文列表  
        keyword_bonus: 关键词匹配的额外加分
    """
    for c in candidate:
        if c.search_keyword:
            c.score = max(c.score + keyword_bonus, 10)
            logger.debug(f"关键词匹配加分: {c.title[:50]}... (+{keyword_bonus}分)")
    return candidate


