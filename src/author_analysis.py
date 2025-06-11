#!/usr/bin/env python3
"""
Author analysis functionality for the arXiv daily email system.

作者分析程序 - 分析Zotero数据库中的关键科研人员
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
import yaml
from dotenv import load_dotenv
from loguru import logger
from typing import Dict, List, Set, Tuple
import re
import json
from datetime import datetime
from utils.zotero_utils import get_zotero_corpus, filter_corpus

load_dotenv(override=True)

class AuthorAnalyzer:
    def __init__(self, zotero_id: str, zotero_key: str):
        """初始化作者分析器"""
        self.zotero_id = zotero_id
        self.zotero_key = zotero_key
        self.corpus = []
        self.authors_info = defaultdict(lambda: {
            'papers': [],
            'total_papers': 0,
            'collections': set(),
            'years': set()
        })
        
    def load_zotero_corpus(self) -> List[dict]:
        """从Zotero加载论文数据"""
        logger.info("正在从Zotero加载论文数据...")
        self.corpus = get_zotero_corpus(self.zotero_id, self.zotero_key)
        return self.corpus
    
    def extract_author_info(self):
        """提取作者信息 - 只关注一作、二作和通讯作者"""
        logger.info("正在分析作者信息（只关注一作、二作和通讯作者）...")
        
        for paper in self.corpus:
            data = paper['data']
            title = data.get('title', '')
            year = data.get('date', '')[:4] if data.get('date') else 'Unknown'
            collections = paper.get('paths', [])  # 使用paths字段而不是collection_paths
            
            # 提取作者信息
            creators = data.get('creators', [])
            authors = [c for c in creators if c.get('creatorType') == 'author']
            
            if not authors:
                continue
                
            # 只关注一作、二作和通讯作者
            target_authors = []
            
            # 一作
            if len(authors) >= 1:
                first_author = authors[0]
                first_name = first_author.get('firstName', '')
                last_name = first_author.get('lastName', '')
                full_name = f"{first_name} {last_name}".strip()
                if full_name:
                    target_authors.append((full_name, '一作'))
            
            # 二作
            if len(authors) >= 2:
                second_author = authors[1]
                first_name = second_author.get('firstName', '')
                last_name = second_author.get('lastName', '')
                full_name = f"{first_name} {last_name}".strip()
                if full_name:
                    target_authors.append((full_name, '二作'))
            
            # 通讯作者（假设为最后一位作者，且不是一作）
            if len(authors) >= 2:
                last_author = authors[-1]
                first_name = last_author.get('firstName', '')
                last_name = last_author.get('lastName', '')
                full_name = f"{first_name} {last_name}".strip()
                if full_name and full_name != target_authors[0][0]:  # 不重复计算一作
                    target_authors.append((full_name, '通讯作者'))
            
            # 记录目标作者信息
            for author_name, author_type in target_authors:
                author_info = self.authors_info[author_name]
                author_info['total_papers'] += 1
                author_info['years'].add(year)
                
                # 简化的论文信息
                paper_info = {
                    'title': title,
                    'year': year,
                    'role': author_type,
                    'key': paper['key']
                }
                author_info['papers'].append(paper_info)
                
                # 添加主要集合信息（只保留第一个集合）
                if collections:
                    author_info['collections'].add(collections[0])
    
    def get_author_statistics(self) -> Dict:
        """获取作者统计信息"""
        logger.info("生成作者统计信息...")
        
        stats = {
            'total_authors': len(self.authors_info),
            'top_authors_by_papers': [],
            'collaboration_network': {},
            'authors_by_collection': defaultdict(list),
            'authors_by_year': defaultdict(list),
            'prolific_authors': []  # 高产作者 (>= 3篇论文)
        }
        
        # 按论文数量排序作者
        authors_by_papers = sorted(
            self.authors_info.items(),
            key=lambda x: x[1]['total_papers'],
            reverse=True
        )
        
        stats['top_authors_by_papers'] = [
            {
                'name': name,
                'papers_count': info['total_papers'],
                'years': sorted(list(info['years'])),
                'collections': list(info['collections'])
            }
            for name, info in authors_by_papers[:50]  # Top 50
        ]
        
        # 高产作者
        stats['prolific_authors'] = [
            {
                'name': name,
                'papers_count': info['total_papers'],
                'recent_papers': [p['title'] for p in info['papers'][-3:]]  # 最近3篇
            }
            for name, info in authors_by_papers if info['total_papers'] >= 3
        ]
        
        # 按集合统计作者
        for name, info in self.authors_info.items():
            for collection in info['collections']:
                stats['authors_by_collection'][collection].append({
                    'name': name,
                    'papers_count': info['total_papers']
                })
        
        # 按年份统计作者
        for name, info in self.authors_info.items():
            for year in info['years']:
                if year != 'Unknown':
                    stats['authors_by_year'][year].append({
                        'name': name,
                        'papers_count': info['total_papers']
                    })
        
        return stats
    
    def find_key_authors(self, min_papers: int = 2) -> List[Dict]:
        """识别关键作者"""
        logger.info(f"识别关键作者（最少 {min_papers} 篇论文）...")
        
        key_authors = []
        for name, info in self.authors_info.items():
            if info['total_papers'] >= min_papers:
                # 计算影响力指标
                collection_diversity = len(info['collections'])
                year_span = len(info['years'])
                
                # 统计作者角色分布
                role_stats = {}
                for paper in info['papers']:
                    role = paper['role']
                    role_stats[role] = role_stats.get(role, 0) + 1
                
                # 一作权重更高
                first_author_bonus = role_stats.get('一作', 0) * 0.5
                corresponding_author_bonus = role_stats.get('通讯作者', 0) * 0.3
                
                key_authors.append({
                    'name': name,
                    'papers_count': info['total_papers'],
                    'collection_diversity': collection_diversity,
                    'year_span': year_span,
                    'role_distribution': role_stats,
                    'influence_score': info['total_papers'] * 0.4 + collection_diversity * 0.3 + year_span * 0.2 + first_author_bonus + corresponding_author_bonus,
                    'collections': list(info['collections']),
                    'years': sorted(list(info['years']))
                })
        
        # 按影响力排序
        key_authors.sort(key=lambda x: x['influence_score'], reverse=True)
        return key_authors
    
    def generate_report(self, output_file: str = None) -> str:
        """生成分析报告"""
        logger.info("生成分析报告...")
        
        stats = self.get_author_statistics()
        key_authors = self.find_key_authors()
        
        report = []
        report.append("# Zotero作者分析报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"总论文数: {len(self.corpus)}")
        report.append(f"总作者数: {stats['total_authors']}")
        report.append("")
        
        # 高产作者
        report.append("## 高产作者 (≥3篇论文)")
        if stats['prolific_authors']:
            for i, author in enumerate(stats['prolific_authors'][:20], 1):
                report.append(f"{i}. **{author['name']}** - {author['papers_count']}篇")
                if author['recent_papers']:
                    report.append(f"   最近论文: {', '.join(author['recent_papers'][:2])}")
                report.append("")
        else:
            report.append("暂无高产作者")
        
        # Top作者详细信息
        report.append("## Top 20 作者详细信息")
        for i, author in enumerate(stats['top_authors_by_papers'][:20], 1):
            # 获取作者的角色分布
            author_info = self.authors_info[author['name']]
            role_stats = {}
            for paper in author_info['papers']:
                role = paper['role']
                role_stats[role] = role_stats.get(role, 0) + 1
            
            role_display = []
            if '一作' in role_stats:
                role_display.append(f"一作{role_stats['一作']}篇")
            if '二作' in role_stats:
                role_display.append(f"二作{role_stats['二作']}篇")
            if '通讯作者' in role_stats:
                role_display.append(f"通讯{role_stats['通讯作者']}篇")
                
            report.append(f"### {i}. {author['name']}")
            report.append(f"- 论文数量: {author['papers_count']}")
            report.append(f"- 作者角色: {', '.join(role_display)}")
            report.append(f"- 发表年份: {', '.join(author['years'])}")
            report.append(f"- 研究领域: {', '.join(author['collections'][:3])}")
            report.append("")
        
        # 关键作者
        report.append("## 关键作者 (综合影响力排序)")
        for i, author in enumerate(key_authors[:15], 1):
            role_display = []
            if '一作' in author['role_distribution']:
                role_display.append(f"一作{author['role_distribution']['一作']}篇")
            if '二作' in author['role_distribution']:
                role_display.append(f"二作{author['role_distribution']['二作']}篇")
            if '通讯作者' in author['role_distribution']:
                role_display.append(f"通讯{author['role_distribution']['通讯作者']}篇")
                
            report.append(f"### {i}. {author['name']}")
            report.append(f"- 影响力得分: {author['influence_score']:.2f}")
            report.append(f"- 论文数量: {author['papers_count']}")
            report.append(f"- 作者角色: {', '.join(role_display)}")
            report.append(f"- 研究领域覆盖: {author['collection_diversity']}个领域")
            report.append(f"- 活跃年限: {author['year_span']}年")
            report.append("")
        
        # 按领域统计
        report.append("## 各研究领域Top作者")
        for collection, authors in stats['authors_by_collection'].items():
            if len(authors) >= 2:  # 只显示有多位作者的领域
                top_authors_in_field = sorted(authors, key=lambda x: x['papers_count'], reverse=True)[:5]
                report.append(f"### {collection}")
                for author in top_authors_in_field:
                    report.append(f"- {author['name']} ({author['papers_count']}篇)")
                report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"报告已保存到: {output_file}")
        
        return report_text
    
    def calculate_author_score(self, name: str, info: dict) -> float:
        """计算作者综合得分"""
        # 统计作者角色
        roles = {}
        for paper in info['papers']:
            role = paper['role']
            roles[role] = roles.get(role, 0) + 1
        
        # 角色权重：通讯作者 > 一作 > 二作
        role_weights = {
            '通讯作者': 3.0,
            '一作': 2.0,
            '二作': 1.0
        }
        
        # 计算角色加权得分
        role_score = 0
        for role, count in roles.items():
            weight = role_weights.get(role, 0.5)
            role_score += count * weight
        
        # 年份新颖度得分（越新的年份权重越高）
        current_year = datetime.now().year
        year_score = 0
        valid_years = [y for y in info['years'] if y != 'Unknown' and y.isdigit()]
        
        if valid_years:
            for year_str in valid_years:
                year = int(year_str)
                # 年份越新，权重越高（最近5年内的权重递减）
                if year >= current_year - 4:  # 2021-2025
                    year_weight = 1.0 - (current_year - year) * 0.1
                    year_score += year_weight
                else:
                    year_score += 0.3  # 较老的年份给较低权重
        
        # 综合得分 = 角色加权得分 * 0.6 + 论文总数 * 0.3 + 年份新颖度 * 0.1
        total_score = role_score * 0.6 + info['total_papers'] * 0.3 + year_score * 0.1
        
        return total_score

    def export_author_data(self, output_file: str, min_papers_for_export: int = 2, max_authors: int = 15):
        """导出作者数据为JSON格式 - 智能排序，限制人数"""
        logger.info(f"导出top作者数据到: {output_file} (最少{min_papers_for_export}篇论文，最多{max_authors}人)")
        
        # 计算所有符合条件作者的综合得分
        author_scores = []
        for name, info in self.authors_info.items():
            if info['total_papers'] >= min_papers_for_export:
                score = self.calculate_author_score(name, info)
                author_scores.append((name, info, score))
        
        # 按综合得分排序
        author_scores.sort(key=lambda x: x[2], reverse=True)
        
        # 限制输出人数
        top_authors = author_scores[:max_authors]
        
        export_data = []
        for name, info, score in top_authors:
            # 统计作者角色
            roles = {}
            for paper in info['papers']:
                role = paper['role']
                roles[role] = roles.get(role, 0) + 1
            
            # 构建简化的作者信息
            author_entry = {
                'name': name,
                'roles': roles,
                'field': list(info['collections'])[0] if info['collections'] else 'Unknown'
            }
            
            export_data.append(author_entry)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已导出 {len(export_data)} 位top作者的智能排序信息（原始作者总数: {len(self.authors_info)}）")
        logger.info("排序规则: 通讯作者>一作>二作，越新的文章权重越高")
    
    def get_author_keywords(self, min_papers: int = 2) -> Dict[str, List[str]]:
        """从作者的论文标题中提取关键词"""
        logger.info("提取作者关键词...")
        
        author_keywords = {}
        for name, info in self.authors_info.items():
            if info['total_papers'] >= min_papers:
                # 从论文信息中提取标题
                titles = [paper['title'] for paper in info['papers']]
                all_titles = ' '.join(titles).lower()
                
                # 简单的关键词提取（可以改进）
                words = re.findall(r'\b[a-zA-Z]{4,}\b', all_titles)
                word_freq = Counter(words)
                
                # 过滤常见词
                stop_words = {'with', 'using', 'based', 'approach', 'method', 'analysis', 
                             'study', 'research', 'paper', 'towards', 'from', 'learning'}
                keywords = [word for word, freq in word_freq.most_common(10) 
                           if word not in stop_words and freq >= 2]
                
                author_keywords[name] = keywords
        
        return author_keywords

def main():
    parser = argparse.ArgumentParser(description='Zotero作者分析工具')
    parser.add_argument('--zotero_id', type=str, help='Zotero用户ID', 
                       default=os.getenv('ZOTERO_ID'))
    parser.add_argument('--zotero_key', type=str, help='Zotero API密钥',
                       default=os.getenv('ZOTERO_KEY'))
    parser.add_argument('--output_report', type=str, help='输出报告文件路径',
                       default='author_analysis_report.md')
    parser.add_argument('--output_data', type=str, help='输出数据文件路径',
                       default='data/author_data.json')
    parser.add_argument('--min_papers', type=int, help='关键作者最少论文数',
                       default=2)
    parser.add_argument('--min_export_papers', type=int, help='导出作者最少论文数',
                       default=2)
    parser.add_argument('--max_authors', type=int, help='导出作者最大数量',
                       default=15)
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 从YAML配置文件读取配置
    def load_config_from_yaml(config_file="config/private_config.yaml"):
        """从YAML文件加载配置"""
        if not os.path.exists(config_file):
            return {}
        
        logger.info(f"找到配置文件 {config_file}，正在读取...")
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    # 读取配置
    yaml_config = load_config_from_yaml()
    
    # 从YAML配置中获取Zotero凭据（优先级高于命令行参数）
    zotero_ignore = None
    if yaml_config:
        args.zotero_id = yaml_config.get('ZOTERO_ID') or args.zotero_id
        args.zotero_key = yaml_config.get('ZOTERO_KEY') or args.zotero_key
        zotero_ignore = yaml_config.get('ZOTERO_IGNORE')
    
    if not args.zotero_id or not args.zotero_key:
        logger.error("请提供Zotero ID和API密钥")
        sys.exit(1)
    
    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")
    
    # 创建分析器
    analyzer = AuthorAnalyzer(args.zotero_id, args.zotero_key)
    
    # 加载数据
    analyzer.load_zotero_corpus()
    
    # 过滤corpus（根据ZOTERO_IGNORE配置）
    if zotero_ignore:
        logger.info(f"正在过滤corpus，忽略模式: {zotero_ignore}")
        original_count = len(analyzer.corpus)
        analyzer.corpus = filter_corpus(analyzer.corpus, zotero_ignore)
        filtered_count = len(analyzer.corpus)
        logger.info(f"过滤完成，从 {original_count} 篇论文减少到 {filtered_count} 篇论文")
    
    # 分析作者信息
    analyzer.extract_author_info()
    
    # 生成报告
    report = analyzer.generate_report(args.output_report)
    
    # 导出数据（可配置人数限制）
    analyzer.export_author_data(args.output_data, min_papers_for_export=args.min_export_papers, max_authors=args.max_authors)
    
    # 生成关键词分析
    keywords = analyzer.get_author_keywords(args.min_papers)
    logger.info("作者关键词分析（Top 10作者）:")
    for i, (author, words) in enumerate(list(keywords.items())[:10], 1):
        logger.info(f"{i}. {author}: {', '.join(words[:5])}")
    
    # 显示简要统计
    stats = analyzer.get_author_statistics()
    logger.success(f"分析完成！")
    logger.info(f"总作者数: {stats['total_authors']}")
    logger.info(f"高产作者数: {len(stats['prolific_authors'])}")
    eligible_authors_count = len([name for name, info in analyzer.authors_info.items() if info['total_papers'] >= args.min_export_papers])
    actual_export_count = min(eligible_authors_count, args.max_authors)
    logger.info(f"符合条件的作者数: {eligible_authors_count}")
    logger.info(f"实际导出作者数: {actual_export_count} (最大限制: {args.max_authors})")
    logger.info(f"报告已保存: {args.output_report}")
    logger.info(f"数据已导出: {args.output_data}")
    
    # 显示JSON格式示例
    if analyzer.authors_info:
        logger.info("配置参数:")
        logger.info(f"  最少论文数: {args.min_export_papers}")
        logger.info(f"  最大作者数: {args.max_authors}")
        logger.info("JSON格式说明:")
        logger.info("  每位作者包含: name(姓名), papers(论文数), roles(角色统计), years(年份), field(主要领域)")
        logger.info("  角色包括: 一作、二作、通讯作者")
        logger.info("  排序规则: 通讯作者>一作>二作，越新的文章权重越高")

if __name__ == '__main__':
    main() 