from llama_cpp import Llama
from openai import OpenAI
from loguru import logger
from time import sleep
import time
from collections import deque

GLOBAL_LLM = None

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None, lang: str = "English", config: dict = None):
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=5_000,
                n_threads=4,
                verbose=False,
            )
        self.model = model
        self.lang = lang
        
        # 从配置中读取限流参数，提供默认值
        self.config = config or {}
        self.max_requests_per_minute = self.config.get('max_requests_per_minute', 9)
        self.api_retry_attempts = self.config.get('api_retry_attempts', 3)
        self.api_retry_delay = self.config.get('api_retry_delay', 3.0)
        self.rate_limit_buffer = self.config.get('rate_limit_buffer', 1.0)
        
        # 频率限制相关属性
        self.request_times = deque()  # 存储请求时间戳
        self.is_gemini = model and "gemini" in model.lower() if model else False

    def _check_rate_limit(self):
        """检查并执行频率限制"""
        current_time = time.time()
        
        # 只对gemini模型或OpenAI API进行频率限制
        if not (self.is_gemini or isinstance(self.llm, OpenAI)):
            return
            
        # 移除超过1分钟的请求记录
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()
        
        # 如果请求数达到限制，等待
        if len(self.request_times) >= self.max_requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0]) + self.rate_limit_buffer
            logger.info(f"达到频率限制，等待 {sleep_time:.1f} 秒...")
            sleep(sleep_time)
            # 清理过期的请求记录
            current_time = time.time()
            while self.request_times and current_time - self.request_times[0] > 60:
                self.request_times.popleft()
        
        # 记录当前请求时间
        self.request_times.append(current_time)

    def generate(self, messages: list[dict]) -> str:
        # 执行频率限制检查
        self._check_rate_limit()
        
        if isinstance(self.llm, OpenAI):
            for attempt in range(self.api_retry_attempts):
                try:
                    response = self.llm.chat.completions.create(messages=messages, temperature=0, model=self.model)
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == self.api_retry_attempts - 1:
                        raise
                    sleep(self.api_retry_delay)
            return response.choices[0].message.content
        else:
            response = self.llm.create_chat_completion(messages=messages,temperature=0)
            return response["choices"][0]["message"]["content"]

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English", config: dict = None):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang, config=config)
    model_str_decrypt = [letter for letter in model]
    logger.info(f"Global LLM set to {model_str_decrypt} with lang {lang}")
    if config:
        logger.info(f"LLM配置: RPM限制={config.get('max_requests_per_minute', 9)}, "
                   f"重试次数={config.get('api_retry_attempts', 3)}, "
                   f"重试延迟={config.get('api_retry_delay', 3.0)}s")

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM