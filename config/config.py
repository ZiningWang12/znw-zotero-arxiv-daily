import argparse
import os
import yaml
from loguru import logger


def get_env(key: str, default=None):
    """处理环境变量，将空字符串视为None"""
    v = os.environ.get(key)
    if v == '' or v is None:
        return default
    return v


def create_argument_parser():
    """创建并配置参数解析器"""
    parser = argparse.ArgumentParser(description='Recommender system for academic papers')
    
    def add_argument(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
        arg_full_name = kwargs.get('dest', args[-1][2:])
        env_name = arg_full_name.upper()
        env_value = get_env(env_name)
        if env_value is not None:
            # 转换环境变量值到指定类型
            if kwargs.get('type') == bool:
                env_value = env_value.lower() in ['true', '1']
            else:
                env_value = kwargs.get('type')(env_value)
            parser.set_defaults(**{arg_full_name: env_value})

    # 添加所有参数
    add_argument('--zotero_id', type=str, help='Zotero user ID')
    add_argument('--zotero_key', type=str, help='Zotero API key')
    add_argument('--zotero_ignore', type=str, help='Zotero collection to ignore, using gitignore-style pattern.')
    add_argument('--send_empty', type=bool, help='If get no arxiv paper, send empty email', default=False)
    add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend', default=100)
    add_argument('--arxiv_query', type=str, help='Arxiv search query by category')
    add_argument('--arxiv_query_keyword', type=str, help='Arxiv search query by keyword', default=None)
    add_argument('--smtp_server', type=str, help='SMTP server')
    add_argument('--smtp_port', type=int, help='SMTP port')
    add_argument('--sender', type=str, help='Sender email address')
    add_argument('--receiver', type=str, help='Receiver email address')
    add_argument('--sender_password', type=str, help='Sender email password')
    add_argument('--research_interests', type=str, help='Research interests', default=None)
    add_argument('--use_llm_api', type=bool, help='Use OpenAI API to generate TLDR', default=False)
    add_argument('--openai_api_key', type=str, help='OpenAI API key', default=None)
    add_argument('--openai_api_base', type=str, help='OpenAI API base URL', default='https://api.openai.com/v1')
    add_argument('--model_name', type=str, help='LLM Model Name', default='gpt-4o')
    add_argument('--language', type=str, help='Language of TLDR', default='English')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    return parser


def parse_research_interests(interests):
    """解析研究兴趣，支持字符串和列表格式"""
    if not interests:
        return None
    if isinstance(interests, str):
        return [interest.strip() for interest in interests.split(',')]
    return interests


def load_config_from_yaml(config_file="config/private_config.yaml", public_config_file="config/config.yaml"):
    """从YAML文件加载配置"""
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    elif os.path.exists(public_config_file):
        with open(public_config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    else:
        raise FileNotFoundError(f"找不到配置文件 {config_file} 或 {public_config_file}")


def merge_configs(args):
    """合并各种配置源的参数"""
    # 读取YAML配置
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
        # 核心推荐参数
        'research_interests': research_interests,
        'corpus_batch_size': llm_config.get('CORPUS_BATCH_SIZE', 20),
        'candidate_batch_size': llm_config.get('CANDIDATE_BATCH_SIZE', 8),
        'keyword_bonus': llm_config.get('KEYWORD_BONUS', 2.0),
        'default_score': llm_config.get('DEFAULT_SCORE', 5.0),
        
        # 文本处理参数
        'abstract_max_length': llm_config.get('ABSTRACT_MAX_LENGTH', 500),
        'score_filter_threshold': llm_config.get('SCORE_FILTER_THRESHOLD', 5.0),
        'max_score_limit': llm_config.get('MAX_SCORE_LIMIT', 10.0),
        'score_scale_factor': llm_config.get('SCORE_SCALE_FACTOR', 10.0),
        
        # API调用和限流参数
        'max_requests_per_minute': llm_config.get('MAX_REQUESTS_PER_MINUTE', 9),
        'api_retry_attempts': llm_config.get('API_RETRY_ATTEMPTS', 3),
        'api_retry_delay': llm_config.get('API_RETRY_DELAY', 3.0),
        'rate_limit_buffer': llm_config.get('RATE_LIMIT_BUFFER', 1.0),  # 额外等待时间
        
        # 传统推荐相关参数
        'embedding_model': llm_config.get('EMBEDDING_MODEL', 'avsolatorio/GIST-small-Embedding-v0'),
        'use_time_decay': llm_config.get('USE_TIME_DECAY', True)
    }
    
    logger.info(f"读取LLM推荐配置: {llm_recommender_config}")
    
    return args, llm_recommender_config


def validate_config(args):
    """验证配置的完整性"""
    assert (
        not args.use_llm_api or args.openai_api_key is not None
    ), "If use_llm_api is True, openai_api_key must be provided" 