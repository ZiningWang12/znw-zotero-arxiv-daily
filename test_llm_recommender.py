#!/usr/bin/env python3
"""
测试LLM推荐算法的脚本 - 使用Gemini API（配置化版本）
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv(override=True)

from recommender import llm_based_rerank_paper
from paper import ArxivPaper
from llm import set_global_llm
import arxiv
from loguru import logger
import yaml

def create_mock_corpus():
    """创建模拟的corpus数据"""
    return [
        {
            'data': {
                'title': 'Embodied AI: A Survey on Integrating Perception and Action',
                'abstractNote': 'This paper presents a comprehensive survey of embodied artificial intelligence, focusing on the integration of perception and action in robotic systems. We discuss recent advances in vision-language models, world models, and their applications in robotics.',
                'dateAdded': '2024-01-15T10:00:00Z'
            }
        },
        {
            'data': {
                'title': 'Learning World Models for Robotic Manipulation',
                'abstractNote': 'We propose a novel approach for learning world models that can predict the effects of robotic actions in complex environments. Our method combines transformer architectures with reinforcement learning.',
                'dateAdded': '2024-01-10T09:00:00Z'
            }
        },
        {
            'data': {
                'title': 'Vision-Language Models for Robot Navigation',
                'abstractNote': 'This work explores the use of large vision-language models for robot navigation tasks. We demonstrate how natural language instructions can be grounded in visual observations.',
                'dateAdded': '2024-01-05T08:00:00Z'
            }
        }
    ]

def create_mock_candidates():
    """创建模拟的候选论文"""
    # 创建一些模拟的arxiv.Result对象
    class MockArxivResult:
        def __init__(self, title, summary, arxiv_id):
            self.title = title
            self.summary = summary
            self._arxiv_id = arxiv_id
            self.authors = []
            self.pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        def get_short_id(self):
            return self._arxiv_id
    
    mock_results = [
        MockArxivResult(
            "Multimodal Foundation Models for Robotic Manipulation",
            "We present a multimodal foundation model that can understand both visual and textual instructions for robotic manipulation tasks. Our approach leverages large-scale pretraining on internet data to develop generalizable robotic skills.",
            "2024.0001"
        ),
        MockArxivResult(
            "Deep Reinforcement Learning for Autonomous Driving",
            "This paper proposes a deep reinforcement learning approach for autonomous driving in urban environments. We use a transformer-based world model to predict future states and plan optimal trajectories.",
            "2024.0002"  
        ),
        MockArxivResult(
            "Attention Mechanisms in Natural Language Processing",
            "We study various attention mechanisms used in transformer models for natural language processing tasks. Our analysis focuses on computational efficiency and model interpretability.",
            "2024.0003"
        ),
        MockArxivResult(
            "Embodied Question Answering with Vision-Language Models", 
            "We introduce a new framework for embodied question answering that combines vision-language models with robotic exploration. Our agent can navigate environments and answer questions about observed objects.",
            "2024.0004"
        ),
        MockArxivResult(
            "Graph Neural Networks for Social Network Analysis",
            "This work applies graph neural networks to social network analysis problems. We demonstrate improved performance on node classification and link prediction tasks.",
            "2024.0005"
        )
    ]
    
    return [ArxivPaper(result) for result in mock_results]

def load_test_config():
    """从配置文件加载测试参数"""
    config_file = "private_config.yaml"
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
            if 'LLM_RECOMMENDER' in config:
                return {
                    'research_interests': config['LLM_RECOMMENDER'].get('RESEARCH_INTERESTS'),
                    'corpus_batch_size': config['LLM_RECOMMENDER'].get('CORPUS_BATCH_SIZE', 20),
                    'candidate_batch_size': config['LLM_RECOMMENDER'].get('CANDIDATE_BATCH_SIZE', 8),
                    'keyword_bonus': config['LLM_RECOMMENDER'].get('KEYWORD_BONUS', 2.0),
                    'default_score': config['LLM_RECOMMENDER'].get('DEFAULT_SCORE', 5.0)
                }
    
    # 默认配置
    return {
        'research_interests': [
            "embodied AI", "robotics", "world model", 
            "multimodal learning", "vision-language models"
        ],
        'corpus_batch_size': 10,
        'candidate_batch_size': 3,
        'keyword_bonus': 2.0,
        'default_score': 5.0
    }

def main():
    # 设置日志级别
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    print("🚀 开始测试LLM推荐算法（使用Gemini API + 配置化版本）...")
    
    # 配置Gemini API参数
    gemini_api_key = "AIzaSyDijBhdh2Hj-D6C-iaTatB_5zvX9C5YRi0"
    gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model = "gemini-2.0-flash"
    
    # 也可以从环境变量获取，如果设置了的话
    api_key = os.getenv('GEMINI_API_KEY', gemini_api_key)
    base_url = os.getenv('GEMINI_BASE_URL', gemini_base_url)
    model_name = os.getenv('GEMINI_MODEL', gemini_model)
    
    if not api_key or api_key == "your-api-key-here":
        print("❌ 请配置有效的Gemini API密钥")
        print("可以在脚本中直接设置，或者设置环境变量GEMINI_API_KEY")
        return
    
    # 设置LLM为Gemini
    try:
        set_global_llm(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            lang='Chinese'
        )
        print(f"✅ Gemini LLM设置成功")
        print(f"   API Base URL: {base_url}")
        print(f"   Model: {model_name}")
    except Exception as e:
        print(f"❌ Gemini LLM设置失败: {e}")
        return
    
    # 加载测试配置
    test_config = load_test_config()
    print(f"\n⚙️  测试配置:")
    print(f"   研究兴趣领域: {len(test_config['research_interests'])} 个")
    print(f"   Corpus批大小: {test_config['corpus_batch_size']}")
    print(f"   候选论文批大小: {test_config['candidate_batch_size']}")
    print(f"   关键词加分: {test_config['keyword_bonus']}")
    print(f"   默认分数: {test_config['default_score']}")
    
    # 创建测试数据
    corpus = create_mock_corpus()
    candidates = create_mock_candidates()
    
    # 模拟一篇有关键词的论文
    candidates[0].search_keyword = "robot manipulation"
    
    print(f"\n📚 Corpus包含 {len(corpus)} 篇论文")
    print(f"📄 候选论文包含 {len(candidates)} 篇")
    print(f"🔑 其中 1 篇论文匹配关键词（将获得 +{test_config['keyword_bonus']} 分加成）")
    
    # 运行LLM推荐
    try:
        print("\n🤖 开始使用Gemini进行配置化LLM推荐...")
        ranked_papers = llm_based_rerank_paper(
            candidates, 
            corpus,
            research_interests=test_config['research_interests'],
            corpus_batch_size=test_config['corpus_batch_size'],
            candidate_batch_size=test_config['candidate_batch_size'],
            keyword_bonus=test_config['keyword_bonus'],
            default_score=test_config['default_score']
        )
        
        print("\n📊 推荐结果：")
        print("=" * 80)
        for i, paper in enumerate(ranked_papers, 1):
            score = getattr(paper, 'score', 0)
            reason = getattr(paper, 'llm_reason', '无理由')
            keyword_mark = " 🔑" if paper.search_keyword else ""
            print(f"{i}. 【评分: {score:.1f}】{keyword_mark}")
            print(f"   标题: {paper.title}")
            print(f"   理由: {reason}")
            if paper.search_keyword:
                print(f"   关键词: {paper.search_keyword}")
            print(f"   摘要: {paper.summary[:100]}...")
            print("-" * 80)
        
        print("✅ Gemini配置化推荐测试完成！")
        
        # 分析结果
        scores = [getattr(p, 'score', 0) for p in ranked_papers]
        print(f"\n📈 评分分析:")
        print(f"   最高分: {max(scores):.1f}")
        print(f"   最低分: {min(scores):.1f}")
        print(f"   平均分: {sum(scores)/len(scores):.1f}")
        print(f"   评分范围: {max(scores) - min(scores):.1f}")
        
        # 检查关键词加分是否生效
        keyword_papers = [p for p in ranked_papers if p.search_keyword]
        if keyword_papers:
            print(f"   关键词论文分数: {[p.score for p in keyword_papers]}")
        
        # 检查是否有好的区分度
        if max(scores) - min(scores) > 2:
            print("✅ 评分区分度良好，推荐算法工作正常")
        else:
            print("⚠️  评分区分度较小，可能需要调整算法参数")
        
    except Exception as e:
        print(f"❌ 推荐过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 