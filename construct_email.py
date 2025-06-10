from paper import ArxivPaper
import math
from tqdm import tqdm
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import smtplib
import datetime
from loguru import logger

framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
  return block_template

def get_block_html(title:str, authors:str, rate:str,arxiv_id:str, reason:str, abstract:str, pdf_url:str, code_url:str=None, affiliations:str=None, is_key_author:bool=False):
    code = f'<a href="{code_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #5bc0de; padding: 8px 16px; border-radius: 4px; margin-left: 8px;">Code</a>' if code_url else ''
    
    # 根据是否是关键作者论文选择不同的样式
    if is_key_author:
        # 关键作者论文使用紫色渐变边框和特殊背景
        border_style = "border: 3px solid; border-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%) 1; background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);"
        title_color = "#4a5568"
    else:
        # 普通论文使用默认样式
        border_style = "border: 1px solid #ddd;"
        title_color = "#333"
    
    block_template = f"""
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; {border_style} border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: {title_color};">
            {{title}}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {{authors}}
            <br>
            <i>{{affiliations}}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {{rate}}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>arXiv ID:</strong> {{arxiv_id}}
        </td>
    </tr>
    
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Recommendation Reason:</strong> {{reason}}
        </td>
    </tr>

    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {{abstract}}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            <a href="{{pdf_url}}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
            {{code}}
        </td>
    </tr>
</table>
"""
    return block_template.format(title=title, authors=authors,rate=rate,arxiv_id=arxiv_id,reason=reason,abstract=abstract, pdf_url=pdf_url, code=code, affiliations=affiliations)

def get_stars(score:float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 0.5
    high = 10
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'


def render_email(papers:list[ArxivPaper]):
    if len(papers) == 0:
        return framework.replace('__CONTENT__', get_empty_html())
    
    # 分离关键作者推荐的论文和普通推荐的论文
    key_author_papers = []
    regular_papers = []
    
    for paper in papers:
        if hasattr(paper, 'key_authors') and paper.key_authors:
            key_author_papers.append(paper)
        else:
            regular_papers.append(paper)
    
    parts = []
    
    # 如果有关键作者推荐的论文，先显示这一部分
    if key_author_papers:
        # 添加关键作者推荐部分的标题
        key_author_title = '''
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 20px 0; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <h2 style="margin: 0; font-size: 24px; font-weight: bold;">🌟 关键作者推荐论文</h2>
            <p style="margin: 10px 0 0 0; font-size: 16px; opacity: 0.9;">来自您关注的重要研究者的最新工作</p>
        </div>
        '''
        parts.append(key_author_title)
        
        # 渲染关键作者推荐的论文
        for p in tqdm(key_author_papers, desc='Rendering Key Author Papers'):
            rate = get_stars(p.score)
            # 保留前3位和最后2位作者
            if len(p.authors) > 5:
                authors = [a.name for a in p.authors[:3]]
                authors.extend([a.name for a in p.authors[-2:]])
                authors = ', '.join(authors)
                authors += ', ...'
            else:
                authors = [a.name for a in p.authors]
                authors = ', '.join(authors)

            if p.affiliations is not None:
                affiliations = p.affiliations[:5]
                affiliations = ', '.join(affiliations)
                if len(p.affiliations) > 5:
                    affiliations += ', ...'
            else:
                affiliations = 'Unknown Affiliation'
            
            # 为关键作者论文添加特殊标记
            key_author_info = f"📌 关键作者: {', '.join(p.key_authors[:3])}"
            if len(p.key_authors) > 3:
                key_author_info += "..."
            
            parts.append(get_block_html(
                p.title, authors, rate, p.arxiv_id, 
                f"{key_author_info}<br>{p.llm_reason}", 
                p.tldr, p.pdf_url, p.code_url, affiliations,
                is_key_author=True
            ))
    
    # 如果有普通推荐的论文，显示这一部分
    if regular_papers:
        # 如果前面有关键作者推荐，则添加普通推荐部分的标题
        if key_author_papers:
            regular_title = '''
            <div style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); color: white; padding: 15px; border-radius: 8px; margin: 20px 0; text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; font-size: 20px; font-weight: bold;">📚 相关性推荐论文</h3>
                <p style="margin: 8px 0 0 0; font-size: 14px; opacity: 0.9;">基于您的研究兴趣推荐的相关论文</p>
            </div>
            '''
            parts.append(regular_title)
        
        # 渲染普通推荐的论文
        for p in tqdm(regular_papers, desc='Rendering Regular Papers'):
            rate = get_stars(p.score)
            # 保留前3位和最后2位作者
            if len(p.authors) > 5:
                authors = [a.name for a in p.authors[:3]]
                authors.extend([a.name for a in p.authors[-2:]])
                authors = ', '.join(authors)
                authors += ', ...'
            else:
                authors = [a.name for a in p.authors]
                authors = ', '.join(authors)

            if p.affiliations is not None:
                affiliations = p.affiliations[:5]
                affiliations = ', '.join(affiliations)
                if len(p.affiliations) > 5:
                    affiliations += ', ...'
            else:
                affiliations = 'Unknown Affiliation'
            
            parts.append(get_block_html(
                p.title, authors, rate, p.arxiv_id, p.llm_reason, 
                p.tldr, p.pdf_url, p.code_url, affiliations,
                is_key_author=False
            ))

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    return framework.replace('__CONTENT__', content)

def send_email(sender:str, receiver:str, password:str,smtp_server:str,smtp_port:int, html:str,):
    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    msg = MIMEText(html, 'html', 'utf-8')
    msg['From'] = _format_addr('Github Action <%s>' % sender)
    msg['To'] = _format_addr('You <%s>' % receiver)
    today = datetime.datetime.now().strftime('%Y/%m/%d')
    msg['Subject'] = Header(f'Daily arXiv {today}', 'utf-8').encode()

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
    except Exception as e:
        logger.warning(f"Failed to use TLS. {e}")
        logger.warning(f"Try to use SSL.")
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)

    server.login(sender, password)
    server.sendmail(sender, [receiver], msg.as_string())
    server.quit()
