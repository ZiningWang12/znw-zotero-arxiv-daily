#!/usr/bin/env python3
"""
Zotero工具模块 - 提供通用的Zotero数据处理功能
"""

import os
from tempfile import mkstemp
from gitignore_parser import parse_gitignore
from pyzotero import zotero
from loguru import logger


def get_zotero_corpus(zotero_id: str, zotero_key: str) -> list[dict]:
    """从Zotero获取论文库
    
    Args:
        zotero_id: Zotero用户ID
        zotero_key: Zotero API密钥
        
    Returns:
        论文列表，每篇论文包含paths字段
    """
    zot = zotero.Zotero(zotero_id, 'user', zotero_key)
    
    # 获取集合信息
    collections = zot.everything(zot.collections())
    collections_dict = {c['key']: c for c in collections}
    logger.info(f"Retrieved {len(collections_dict)} collections from Zotero.")
    
    # 获取论文数据
    corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    corpus = [c for c in corpus if c['data'].get('abstractNote', '') != '' or c['data'].get('title', '') != '']
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    
    # 添加集合路径信息
    def get_collection_path(col_key: str) -> str:
        if col_key not in collections_dict:
            return "Unknown"
        col = collections_dict[col_key]
        if p := col['data'].get('parentCollection'):
            return get_collection_path(p) + '/' + col['data']['name']
        else:
            return col['data']['name']
    
    for paper in corpus:
        paths = [get_collection_path(col) for col in paper['data'].get('collections', [])]
        paper['paths'] = paths
    
    return corpus


def filter_corpus(corpus: list[dict], pattern: str) -> list[dict]:
    """使用gitignore样式的模式过滤corpus
    
    Args:
        corpus: 论文列表
        pattern: gitignore样式的过滤模式
        
    Returns:
        过滤后的论文列表
    """
    if not pattern:
        return corpus
        
    _, filename = mkstemp()
    with open(filename, 'w') as file:
        file.write(pattern)
    matcher = parse_gitignore(filename, base_dir='./')
    
    new_corpus = []
    for c in corpus:
        # 使用paths字段进行匹配
        match_results = [matcher(p) for p in c.get('paths', [])]
        if not any(match_results):
            new_corpus.append(c)
    
    os.remove(filename)
    return new_corpus 