from typing import Optional
from functools import cached_property
from tempfile import TemporaryDirectory
import arxiv
import tarfile
import re
from llm import get_llm
import requests
from requests.adapters import HTTPAdapter, Retry
from loguru import logger
import tiktoken
from contextlib import ExitStack



class ArxivPaper:
    def __init__(self,paper:arxiv.Result,keyword:str=None):
        self._paper = paper
        self.score = None
        self.search_keyword = keyword
        self.llm_reason = None  # 存储LLM评分理由
    
    @property
    def title(self) -> str:
        return self._paper.title
    
    @property
    def summary(self) -> str:
        return self._paper.summary
    
    @property
    def authors(self) -> list[str]:
        return self._paper.authors
    
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
    
    @cached_property
    def tldr(self) -> str:
        introduction = ""
        conclusion = ""
        if self.tex is not None:
            content = self.tex.get("all")
            if content is None:
                content = "\n".join(self.tex.values())
            #remove cite
            content = re.sub(r'~?\\cite.?\{.*?\}', '', content)
            #remove figure
            content = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', content, flags=re.DOTALL)
            #remove table
            content = re.sub(r'\\begin\{table\}.*?\\end\{table\}', '', content, flags=re.DOTALL)
            #find introduction and conclusion
            # end word can be \section or \end{document} or \bibliography or \appendix
            match = re.search(r'\\section\{Introduction\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
            if match:
                introduction = match.group(0)
            match = re.search(r'\\section\{Conclusion\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
            if match:
                conclusion = match.group(0)
        llm = get_llm()
        prompt = """Given the title, abstract, introduction and the conclusion (if any) of a paper in latex format, generate a one-sentence TLDR summary in __LANG__:
        
        \\title{__TITLE__}
        \\begin{abstract}__ABSTRACT__\\end{abstract}
        __INTRODUCTION__
        __CONCLUSION__
        """
        prompt = prompt.replace('__LANG__', llm.lang)
        prompt = prompt.replace('__TITLE__', self.title)
        prompt = prompt.replace('__ABSTRACT__', self.summary)
        prompt = prompt.replace('__INTRODUCTION__', introduction)
        prompt = prompt.replace('__CONCLUSION__', conclusion)

        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)
        
        tldr = llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        return tldr

    @cached_property
    def affiliations(self) -> Optional[list[str]]:
        if self.tex is not None:
            content = self.tex.get("all")
            if content is None:
                content = "\n".join(self.tex.values())
                
            # 扩展搜索模式，涵盖更多常见的LaTeX格式
            possible_regions = [
                # 标准格式：author到maketitle
                r'\\author.*?\\maketitle',
                # 文档开始到摘要（扩大搜索范围）
                r'\\begin{document}.*?\\begin{abstract}', 
                # 标题页区域：从title到section或abstract
                r'\\title.*?(?=\\section|\\begin{abstract})',
                # author到date之间
                r'\\author.*?\\date',
                # author到title之间（有些论文author在title后面）
                r'\\title.*?\\author.*?(?=\\section|\\begin{abstract}|\\maketitle)',
                # author到section之间
                r'\\author.*?(?=\\section)',
                # 脚注区域（特别针对MagicGripper这种格式）
                r'\\footnote.*?(?=\\section|\\begin{abstract}|\\maketitle|$)',
                # 查找脚注命令后的内容
                r'\\footnotetext.*?(?=\\section|\\begin{abstract}|\\maketitle|$)',
                # 查找包含"are with"/"is with"模式的区域（作者机构常用表达）
                r'.*?(?:are with|is with|affiliated with).*?(?=\\section|\\begin{abstract}|\\maketitle|$)',
                # 查找标题页作者信息块（新增：处理A4Bench这种格式）
                r'(?:^|\\title).*?(?:university|institute|college|lab|department|shanghai|beijing|tsinghua|stanford|mit|google|microsoft|openai|deepmind).*?(?=\\section|\\begin{abstract})',
                # 整个文档前3000字符（进一步扩大搜索范围）
                r'^.{0,3000}',
                # 查找包含email地址的区域（通常在作者信息附近）
                r'.*?@.*?\..*?(?=\\section|\\begin{abstract})',
                # 查找包含university/institute关键词的区域
                r'.*?(?:university|institute|college|lab|department).*?(?=\\section|\\begin{abstract})',
                # 查找页面底部的脚注区域
                r'.*?(?:footnote|thanks).*?(?:university|institute|college|department).*?(?=\\section|\\begin{abstract}|$)',
            ]
            
            information_region = None
            
            # 按优先级尝试匹配
            for pattern in possible_regions:
                try:
                    match = re.search(pattern, content, flags=re.DOTALL | re.IGNORECASE)
                    if match:
                        candidate_region = match.group(0)
                        # 检查是否包含可能的机构信息关键词
                        affiliation_keywords = [
                            'university', 'institute', 'college', 'lab', 'department', 
                            'school', 'center', 'centre', 'academy', '@', 'tech', 'polytechnic',
                            # 增加脚注中常见的表达方式
                            'are with', 'is with', 'affiliated with', 'belong to',
                            # 增加更多机构类型
                            'research', 'laboratory', 'faculty', 'division',
                            # 增加常见地理位置和机构名称
                            'shanghai', 'beijing', 'china', 'usa', 'uk', 'japan', 'singapore',
                            'tsinghua', 'peking', 'fudan', 'sjtu', 'stanford', 'mit', 'harvard', 'berkeley',
                            'google', 'microsoft', 'openai', 'deepmind', 'meta', 'nvidia', 'apple',
                            'carnegie', 'mellon', 'caltech', 'princeton', 'yale', 'columbia',
                            # 欧洲常见机构关键词  
                            'cambridge', 'oxford', 'london', 'edinburgh', 'eth', 'epfl'
                        ]
                        
                        if any(keyword in candidate_region.lower() for keyword in affiliation_keywords):
                            information_region = candidate_region
                            logger.debug(f"找到作者信息区域 for {self.arxiv_id}, 使用模式: {pattern[:30]}...")
                            break
                except Exception as e:
                    logger.debug(f"模式匹配失败 {pattern}: {e}")
                    continue
            
            if not information_region:
                logger.debug(f"Failed to extract affiliations of {self.arxiv_id}: No author information found.")
                return None
                
            # 清理和预处理文本
            # 更温和的清理方式，保留更多有用信息
            information_region = re.sub(r'\\(?:section|subsection|subsubsection)\{.*?\}', ' ', information_region)  # 移除章节标题
            information_region = re.sub(r'\\(?:cite|ref|label)\{.*?\}', ' ', information_region)  # 移除引用标签
            information_region = re.sub(r'\\(?:textbf|textit|emph)\{(.*?)\}', r'\1', information_region)  # 保留格式化文本内容
            information_region = re.sub(r'\\[a-zA-Z]+\*?(\[.*?\])?\{([^{}]*)\}', r'\2', information_region)  # 移除LaTeX命令但保留内容
            information_region = re.sub(r'\{|\}', ' ', information_region)  # 移除大括号
            information_region = re.sub(r'\\\\|\n+', ' ', information_region)  # 移除换行符和\\
            information_region = re.sub(r'\s+', ' ', information_region).strip()  # 标准化空格
            
            prompt = f"""Given the author information from a research paper, extract the affiliations of the authors.

The author information may be in different formats:
1. Standard LaTeX author blocks
2. Footnotes with author abbreviations (e.g., "W. F and D. Zhang are with...")
3. Mixed formats with affiliations in footnotes
4. Direct listing format (e.g., "Author Name\\nemail@domain.com\\nUniversity Name\\nCity, Country")

Return a Python list of unique affiliations, like ['Stanford University', 'MIT', 'Google Research'].

Rules:
1. Extract only the main institution name (e.g., 'Stanford University' not 'Department of CS, Stanford University')
2. Remove duplicates
3. If no affiliations found, return []
4. Focus on universities, companies, research institutions
5. Ignore personal email domains and email addresses
6. Handle abbreviations in footnotes (e.g., "W. F and D. Zhang are with Imperial College London")
7. Look for patterns like "are with", "is with", "affiliated with"
8. In direct listing format, institutions usually appear after email addresses
9. Common institution types: University, Institute, College, Lab, AI Lab, Research Center, Company

Author information:
{information_region[:4000]}"""  # 限制长度避免token过多
            
            llm = get_llm()
            try:
                affiliations = llm.generate(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at extracting institutional affiliations from academic paper author information. You can handle various formats including standard author blocks, footnotes with abbreviations, and mixed formats. Focus on identifying university names, research institutes, and companies. Return only a Python list format, like ['University A', 'Company B']. Be concise and accurate."
                        },
                        {"role": "user", "content": prompt},
                    ]
                )

                # 更robust的解析方式
                affiliations_text = affiliations.strip()
                
                # 尝试提取列表格式
                list_match = re.search(r'\[.*?\]', affiliations_text, flags=re.DOTALL)
                if list_match:
                    try:
                        affiliations_list = eval(list_match.group(0))
                        if isinstance(affiliations_list, list):
                            # 清理和去重
                            cleaned_affiliations = []
                            for aff in affiliations_list:
                                if isinstance(aff, str) and len(aff.strip()) > 2:
                                    cleaned_aff = aff.strip()
                                    # 移除明显的个人邮箱域名
                                    if not cleaned_aff.endswith(('.com', '.org', '.net', '.edu')) or \
                                       any(inst in cleaned_aff.lower() for inst in ['university', 'institute', 'college']):
                                        cleaned_affiliations.append(cleaned_aff)
                            
                            # 去重并返回
                            unique_affiliations = list(dict.fromkeys(cleaned_affiliations))  # 保持顺序的去重
                            if unique_affiliations:
                                logger.debug(f"成功提取机构信息 for {self.arxiv_id}: {unique_affiliations}")
                                return unique_affiliations
                    except Exception as e:
                        logger.debug(f"解析机构列表失败 for {self.arxiv_id}: {e}")
                
                logger.debug(f"未能解析机构信息 for {self.arxiv_id}, LLM输出: {affiliations_text[:100]}...")
                return None
                
            except Exception as e:
                logger.debug(f"LLM调用失败 for {self.arxiv_id}: {e}")
                return None
        else:
            logger.debug(f"无tex内容 for {self.arxiv_id}")
            return None

    def __hash__(self):
        """基于arxiv_id生成哈希值，使对象可用于set"""
        return hash(self.arxiv_id)
    
    def __eq__(self, other):
        """基于arxiv_id判断对象相等性"""
        if not isinstance(other, ArxivPaper):
            return False
        return self.arxiv_id == other.arxiv_id