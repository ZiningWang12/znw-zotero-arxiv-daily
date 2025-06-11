from typing import Optional
from functools import cached_property
from tempfile import TemporaryDirectory
import arxiv
import tarfile
import re
import json
from src.llm import get_llm
import requests
from requests.adapters import HTTPAdapter, Retry
from loguru import logger
import tiktoken
from contextlib import ExitStack
from urllib3.util.retry import Retry



class ArxivPaper:
    def __init__(self,paper:arxiv.Result,keyword:str=None):
        self._paper = paper
        self.score = None
        self.search_keyword = keyword
        self.llm_reason = None  # 存储LLM评分理由
        self.key_authors = []  # 匹配的关键作者列表
        self.author_importance = 0.0  # 作者重要性分数
    
    @property
    def title(self) -> str:
        return self._paper.title
    
    @property
    def summary(self) -> str:
        return self._paper.summary
    
    @property
    def authors(self) -> list[str]:
        return self._paper.authors
    
    @property
    def published(self):
        return self._paper.published
    
    @cached_property
    def arxiv_id(self) -> str:
        return re.sub(r'v\d+$', '', self._paper.get_short_id())
    
    @property
    def pdf_url(self) -> str:
        return self._paper.pdf_url
    
    @cached_property
    def code_url(self) -> Optional[str]:
        s = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1)
        s.mount('https://', HTTPAdapter(max_retries=retries))
        try:
            paper_list = s.get(f'https://paperswithcode.com/api/v1/papers/?arxiv_id={self.arxiv_id}').json()
        except Exception as e:
            logger.debug(f'Error when searching {self.arxiv_id}: {e}')
            return None

        if paper_list.get('count',0) == 0:
            return None
        paper_id = paper_list['results'][0]['id']

        try:
            repo_list = s.get(f'https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/').json()
        except Exception as e:
            logger.debug(f'Error when searching {self.arxiv_id}: {e}')
            return None
        if repo_list.get('count',0) == 0:
            return None
        return repo_list['results'][0]['url']
    
    @cached_property
    def tex(self) -> dict[str,str]:
        try:
            with ExitStack() as stack:
                tmpdirname = stack.enter_context(TemporaryDirectory())
                file = self._paper.download_source(dirpath=tmpdirname)
                try:
                    tar = stack.enter_context(tarfile.open(file))
                except tarfile.ReadError:
                    logger.debug(f"Failed to find main tex file of {self.arxiv_id}: Not a tar file.")
                    return None
     
                tex_files = [f for f in tar.getnames() if f.endswith('.tex')]
                if len(tex_files) == 0:
                    logger.debug(f"Failed to find main tex file of {self.arxiv_id}: No tex file.")
                    return None
                
                bbl_file = [f for f in tar.getnames() if f.endswith('.bbl')]
                match len(bbl_file) :
                    case 0:
                        if len(tex_files) > 1:
                            logger.debug(f"Cannot find main tex file of {self.arxiv_id} from bbl: There are multiple tex files while no bbl file.")
                            main_tex = None
                        else:
                            main_tex = tex_files[0]
                    case 1:
                        main_name = bbl_file[0].replace('.bbl','')
                        main_tex = f"{main_name}.tex"
                        if main_tex not in tex_files:
                            logger.debug(f"Cannot find main tex file of {self.arxiv_id} from bbl: The bbl file does not match any tex file.")
                            main_tex = None
                    case _:
                        logger.debug(f"Cannot find main tex file of {self.arxiv_id} from bbl: There are multiple bbl files.")
                        main_tex = None
                if main_tex is None:
                    logger.debug(f"Trying to choose tex file containing the document block as main tex file of {self.arxiv_id}")
                #read all tex files
                file_contents = {}
                for t in tex_files:
                    f = tar.extractfile(t)
                    content = f.read().decode('utf-8',errors='ignore')
                    #remove comments
                    content = re.sub(r'%.*\n', '\n', content)
                    content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)
                    content = re.sub(r'\\iffalse.*?\\fi', '', content, flags=re.DOTALL)
                    #remove redundant \n
                    content = re.sub(r'\n+', '\n', content)
                    content = re.sub(r'\\\\', '', content)
                    #remove consecutive spaces
                    content = re.sub(r'[ \t\r\f]{3,}', ' ', content)
                    if main_tex is None and re.search(r'\\begin\{document\}', content):
                        main_tex = t
                        logger.debug(f"Choose {t} as main tex file of {self.arxiv_id}")
                    file_contents[t] = content
                
                if main_tex is not None:
                    main_source:str = file_contents[main_tex]
                    #find and replace all included sub-files
                    include_files = re.findall(r'\\input\{(.+?)\}', main_source) + re.findall(r'\\include\{(.+?)\}', main_source)
                    for f in include_files:
                        if not f.endswith('.tex'):
                            file_name = f + '.tex'
                        else:
                            file_name = f
                        main_source = main_source.replace(f'\\input{{{f}}}', file_contents.get(file_name, ''))
                    file_contents["all"] = main_source
                else:
                    logger.debug(f"Failed to find main tex file of {self.arxiv_id}: No tex file containing the document block.")
                    file_contents["all"] = None
            return file_contents
        except Exception as e:
            logger.warning(f"Failed to download or parse tex file for {self.arxiv_id}: {e}")
            return None
    
    @cached_property
    def tldr(self) -> str:
        """获取论文的TLDR摘要，通过合并的LLM调用获取"""
        return self.llm_extracted_info.get("tldr", "Summary unavailable")

    @cached_property
    def affiliations(self) -> Optional[list[str]]:
        """获取论文的机构信息，通过合并的LLM调用获取，并在失败时回退到Semantic Scholar"""
        # 首选方法：从 .tex 文件解析
        affs = self.llm_extracted_info.get("affiliations", [])
        
        if affs:
            return affs
            
        # 备用方法：如果首选方法失败，则从 Semantic Scholar 获取
        logger.debug(f"首选方法提取机构信息失败 for {self.arxiv_id}，尝试从 Semantic Scholar 获取。")
        affs_fallback = self._fetch_affiliations_from_semantic_scholar()
        
        return affs_fallback if affs_fallback else None

    def _fetch_affiliations_from_semantic_scholar(self) -> list[str]:
        """
        备用方法：从 Semantic Scholar API 获取作者机构信息。
        """
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{self.arxiv_id}?fields=authors.affiliations"
        try:
            s = requests.Session()
            # 设置重试策略，增加网络请求的稳定性
            retries = Retry(total=3, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504])
            s.mount('https://', HTTPAdapter(max_retries=retries))
            
            response = s.get(api_url, timeout=15)
            response.raise_for_status()  # 如果请求失败 (4xx or 5xx), 则抛出异常
            
            data = response.json()
            
            if not data.get('authors'):
                logger.info(f"Semantic Scholar API for {self.arxiv_id} 未返回作者信息。")
                return []
                
            all_affs = []
            for author in data['authors']:
                if author.get('affiliations'):
                    # `affiliations` 是一个字符串列表
                    all_affs.extend(author['affiliations'])
            
            if not all_affs:
                logger.info(f"Semantic Scholar 数据中未找到机构信息 for {self.arxiv_id}。")
                return []

            # 清理和去重，同时保持顺序
            seen = set()
            unique_affs = [x for x in all_affs if not (x in seen or seen.add(x))]
            
            logger.info(f"成功从 Semantic Scholar 获取到 {self.arxiv_id} 的机构信息: {unique_affs}")
            return unique_affs

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.info(f"在 Semantic Scholar 上未找到论文 {self.arxiv_id}。")
            else:
                logger.warning(f"从 Semantic Scholar 获取信息时出现HTTP错误 for {self.arxiv_id}: {e}")
            return []
        except Exception as e:
            logger.warning(f"获取或解析 Semantic Scholar 数据时出错 for {self.arxiv_id}: {e}")
            return []

    @cached_property
    def llm_extracted_info(self) -> dict:
        """
        使用一次LLM调用同时提取TLDR和机构信息，提高效率
        Returns: {"tldr": str, "affiliations": list[str]}
        """
        if self.tex is None:
            logger.debug(f"无tex内容 for {self.arxiv_id}, 仅使用摘要生成TLDR")
            # 如果没有tex文件，只生成基于摘要的TLDR
            llm = get_llm()
            prompt = f"""Based on this paper's title and abstract, provide:
1. A one-sentence TLDR summary in {llm.lang}
2. Extract affiliations (return empty list if not available)

Title: {self.title}
Abstract: {self.summary}

Please respond in JSON format:
{{
    "tldr": "one sentence summary",
    "affiliations": []
}}"""
            
            try:
                response = llm.generate([
                    {"role": "system", "content": "You are an expert at summarizing academic papers and extracting affiliations. Always respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ])
                
                # 解析JSON响应
                json_match = re.search(r'\{.*\}', response, flags=re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                    return {
                        "tldr": result.get("tldr", "Summary unavailable"),
                        "affiliations": result.get("affiliations", [])
                    }
            except Exception as e:
                logger.warning(f"LLM合并调用失败 for {self.arxiv_id}: {e}")
            
            return {"tldr": "Summary unavailable", "affiliations": []}
        
        # 有tex文件的情况，提取详细信息
        content = self.tex.get("all")
        if content is None:
            content = "\n".join(self.tex.values())
        
        # 提取introduction和conclusion（用于TLDR）
        introduction = ""
        conclusion = ""
        # 清理content
        clean_content = re.sub(r'~?\\cite.?\{.*?\}', '', content)
        clean_content = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', clean_content, flags=re.DOTALL)
        clean_content = re.sub(r'\\begin\{table\}.*?\\end\{table\}', '', clean_content, flags=re.DOTALL)
        
        # 查找introduction和conclusion
        intro_match = re.search(r'\\section\{Introduction\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', clean_content, flags=re.DOTALL)
        if intro_match:
            introduction = intro_match.group(0)
        
        concl_match = re.search(r'\\section\{Conclusion\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', clean_content, flags=re.DOTALL)
        if concl_match:
            conclusion = concl_match.group(0)
        
        # 提取作者信息区域（用于机构信息）
        author_info = self._extract_author_region(content)
        
        # 构建合并的prompt
        llm = get_llm()
        prompt = f"""Analyze this academic paper and provide both a summary and author affiliations.

Paper Information:
Title: {self.title}
Abstract: {self.summary}
Introduction: {introduction[:2000]}
Conclusion: {conclusion[:1000]}

Author Information Section:
{author_info[:2000] if author_info else "Not available"}

Please provide:
1. A one-sentence TLDR summary in {llm.lang}
2. Extract the main institutional affiliations (universities, companies, research institutes)

Respond in JSON format:
{{
    "tldr": "one sentence summary of the paper's main contribution",
    "affiliations": ["Institution 1", "Institution 2", ...]
}}

For affiliations:
- Extract only main institution names (e.g., "Stanford University", not "Department of CS, Stanford University")
- Look for universities, companies, research institutes
- Handle footnote formats like "X and Y are with Institution Name"
- Return empty list if no clear affiliations found
- Remove duplicates"""

        # 使用token限制
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            if len(prompt_tokens) > 4000:
                prompt_tokens = prompt_tokens[:4000]
                prompt = enc.decode(prompt_tokens)
        except:
            # 如果tiktoken失败，简单截断
            prompt = prompt[:8000]
        
        try:
            response = llm.generate([
                {"role": "system", "content": "You are an expert at analyzing academic papers. You can summarize papers concisely and extract institutional affiliations accurately. Always respond in valid JSON format."},
                {"role": "user", "content": prompt}
            ])
            
            # 解析JSON响应
            json_match = re.search(r'\{.*\}', response, flags=re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                affiliations = result.get("affiliations", [])
                
                # 清理和验证机构信息
                cleaned_affiliations = []
                if isinstance(affiliations, list):
                    for aff in affiliations:
                        if isinstance(aff, str) and len(aff.strip()) > 2:
                            cleaned_aff = aff.strip()
                            # 过滤明显的邮箱域名
                            if not cleaned_aff.endswith(('.com', '.org', '.net', '.edu')) or \
                               any(inst in cleaned_aff.lower() for inst in ['university', 'institute', 'college']):
                                cleaned_affiliations.append(cleaned_aff)
                
                # 去重
                unique_affiliations = list(dict.fromkeys(cleaned_affiliations))
                
                logger.debug(f"LLM合并提取成功 for {self.arxiv_id}: TLDR={result.get('tldr', '')[:50]}..., 机构={unique_affiliations}")
                
                return {
                    "tldr": result.get("tldr", "Summary unavailable"),
                    "affiliations": unique_affiliations
                }
                
        except Exception as e:
            logger.warning(f"LLM合并调用失败 for {self.arxiv_id}: {e}")
        
        return {"tldr": "Summary unavailable", "affiliations": []}
    
    def _extract_author_region(self, content: str) -> str:
        """提取作者信息区域的辅助方法"""
        possible_regions = [
            r'\\author.*?\\maketitle',
            r'\\begin{document}.*?\\begin{abstract}', 
            r'\\title.*?(?=\\section|\\begin{abstract})',
            r'\\author.*?\\date',
            r'\\title.*?\\author.*?(?=\\section|\\begin{abstract}|\\maketitle)',
            r'\\author.*?(?=\\section)',
            r'\\footnote.*?(?=\\section|\\begin{abstract}|\\maketitle|$)',
            r'\\footnotetext.*?(?=\\section|\\begin{abstract}|\\maketitle|$)',
            r'.*?(?:are with|is with|affiliated with).*?(?=\\section|\\begin{abstract}|\\maketitle|$)',
            r'(?:^|\\title).*?(?:university|institute|college|lab|department|shanghai|beijing|tsinghua|stanford|mit|google|microsoft|openai|deepmind).*?(?=\\section|\\begin{abstract})',
            r'^.{0,3000}',
            r'.*?@.*?\..*?(?=\\section|\\begin{abstract})',
            r'.*?(?:university|institute|college|lab|department).*?(?=\\section|\\begin{abstract})',
            r'.*?(?:footnote|thanks).*?(?:university|institute|college|department).*?(?=\\section|\\begin{abstract}|$)',
        ]
        
        affiliation_keywords = [
            'university', 'institute', 'college', 'lab', 'department', 'school', 'center', 'centre', 
            'academy', '@', 'tech', 'polytechnic', 'are with', 'is with', 'affiliated with', 
            'research', 'laboratory', 'faculty', 'division', 'shanghai', 'beijing', 'china', 
            'tsinghua', 'stanford', 'mit', 'google', 'microsoft', 'openai', 'deepmind'
        ]
        
        for pattern in possible_regions:
            try:
                match = re.search(pattern, content, flags=re.DOTALL | re.IGNORECASE)
                if match:
                    candidate_region = match.group(0)
                    if any(keyword in candidate_region.lower() for keyword in affiliation_keywords):
                        # 清理文本
                        clean_region = re.sub(r'\\(?:section|subsection|subsubsection)\{.*?\}', ' ', candidate_region)
                        clean_region = re.sub(r'\\(?:cite|ref|label)\{.*?\}', ' ', clean_region)
                        clean_region = re.sub(r'\\(?:textbf|textit|emph)\{(.*?)\}', r'\1', clean_region)
                        clean_region = re.sub(r'\\[a-zA-Z]+\*?(\[.*?\])?\{([^{}]*)\}', r'\2', clean_region)
                        clean_region = re.sub(r'\{|\}', ' ', clean_region)
                        clean_region = re.sub(r'\\\\|\n+', ' ', clean_region)
                        clean_region = re.sub(r'\s+', ' ', clean_region).strip()
                        return clean_region
            except Exception:
                continue
        
        return ""

    def __hash__(self):
        """基于arxiv_id生成哈希值，使对象可用于set"""
        return hash(self.arxiv_id)
    
    def __eq__(self, other):
        """基于arxiv_id判断对象相等性"""
        if not isinstance(other, ArxivPaper):
            return False
        return self.arxiv_id == other.arxiv_id