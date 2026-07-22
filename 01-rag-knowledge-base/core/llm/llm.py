"""
LLM 调用模块
支持多种 LLM 提供商：OpenAI、智谱AI、阿里云百炼
提供同步和流式生成接口
"""
from abc import ABC, abstractmethod
from typing import Generator, List, Optional, Dict, Any

from loguru import logger
from core.config import settings
from tenacity import retry, stop_after_attempt, wait_exponential


class BaseLLM(ABC):
    """LLM 抽象基类"""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """同步生成回答"""
        ...

    @abstractmethod
    def stream_generate(self, prompt: str, system_prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """流式生成回答"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称"""
        ...


class OpenAILLM(BaseLLM):
    """
    OpenAI API 调用

    支持 OpenAI 官方 API 及兼容接口（如 Azure OpenAI、本地 Ollama 等）
    """

    def __init__(self, api_key: str = None, api_base: str = None, model: str = None):
        self._api_key = api_key or settings.openai_api_key
        self._api_base = api_base or settings.openai_api_base
        self._model = model or settings.llm_model
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """初始化 OpenAI 客户端"""
        from openai import OpenAI
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._api_base,
        )
        logger.info(f"OpenAI LLM 客户端初始化成功: {self._model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """同步生成回答"""
        messages = self._build_messages(prompt, system_prompt)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=kwargs.get("temperature", settings.llm_temperature),
                max_tokens=kwargs.get("max_tokens", settings.llm_max_tokens),
                top_p=kwargs.get("top_p", settings.llm_top_p),
            )
            answer = response.choices[0].message.content
            logger.debug(f"OpenAI 生成完成，tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
            return answer
        except Exception as e:
            logger.error(f"OpenAI 生成失败: {e}")
            raise

    def stream_generate(self, prompt: str, system_prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """流式生成回答"""
        messages = self._build_messages(prompt, system_prompt)
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=kwargs.get("temperature", settings.llm_temperature),
                max_tokens=kwargs.get("max_tokens", settings.llm_max_tokens),
                top_p=kwargs.get("top_p", settings.llm_top_p),
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI 流式生成失败: {e}")
            raise

    def _build_messages(self, prompt: str, system_prompt: str = None) -> List[Dict]:
        """构造消息列表"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    @property
    def model_name(self) -> str:
        return self._model


class ZhipuLLM(BaseLLM):
    """
    智谱AI (GLM) 调用

    使用 zhipuai SDK 调用智谱 GLM 系列模型
    """

    def __init__(self, api_key: str = None, model: str = None):
        self._api_key = api_key or settings.zhipu_api_key
        self._model = model or settings.zhipu_model
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """初始化智谱AI客户端"""
        from zhipuai import ZhipuAI
        self._client = ZhipuAI(api_key=self._api_key)
        logger.info(f"智谱AI 客户端初始化成功: {self._model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """同步生成回答"""
        messages = self._build_messages(prompt, system_prompt)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=kwargs.get("temperature", settings.llm_temperature),
                max_tokens=kwargs.get("max_tokens", settings.llm_max_tokens),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"智谱AI 生成失败: {e}")
            raise

    def stream_generate(self, prompt: str, system_prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """流式生成回答"""
        messages = self._build_messages(prompt, system_prompt)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=kwargs.get("temperature", settings.llm_temperature),
                max_tokens=kwargs.get("max_tokens", settings.llm_max_tokens),
                stream=True,
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"智谱AI 流式生成失败: {e}")
            raise

    def _build_messages(self, prompt: str, system_prompt: str = None) -> List[Dict]:
        """构造消息列表"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    @property
    def model_name(self) -> str:
        return self._model


class DashscopeLLM(BaseLLM):
    """
    阿里云百炼 (DashScope) 调用

    使用 dashscope SDK 调用通义千问系列模型
    """

    def __init__(self, api_key: str = None, model: str = None):
        self._api_key = api_key or settings.dashscope_api_key
        self._model = model or settings.dashscope_model
        self._init_client()

    def _init_client(self) -> None:
        """初始化 DashScope"""
        import dashscope
        dashscope.api_key = self._api_key
        logger.info(f"DashScope 初始化成功: {self._model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """同步生成回答"""
        from dashscope import Generation
        messages = self._build_messages(prompt, system_prompt)
        try:
            response = Generation.call(
                model=self._model,
                messages=messages,
                temperature=kwargs.get("temperature", settings.llm_temperature),
                max_tokens=kwargs.get("max_tokens", settings.llm_max_tokens),
                result_format="message",
            )
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                raise RuntimeError(f"DashScope 调用失败: {response.code} - {response.message}")
        except Exception as e:
            logger.error(f"DashScope 生成失败: {e}")
            raise

    def stream_generate(self, prompt: str, system_prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """流式生成回答"""
        from dashscope import Generation
        messages = self._build_messages(prompt, system_prompt)
        try:
            responses = Generation.call(
                model=self._model,
                messages=messages,
                temperature=kwargs.get("temperature", settings.llm_temperature),
                max_tokens=kwargs.get("max_tokens", settings.llm_max_tokens),
                result_format="message",
                stream=True,
            )
            for response in responses:
                if response.status_code == 200 and response.output.choices:
                    content = response.output.choices[0].message.content
                    if content:
                        yield content
                else:
                    logger.warning(f"DashScope 流式响应异常: {response.code}")
        except Exception as e:
            logger.error(f"DashScope 流式生成失败: {e}")
            raise

    def _build_messages(self, prompt: str, system_prompt: str = None) -> List[Dict]:
        """构造消息列表"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    @property
    def model_name(self) -> str:
        return self._model


class LLMManager:
    """
    LLM 管理器

    统一管理不同 LLM 提供商，提供单例模式

    使用示例:
        manager = LLMManager()
        answer = manager.generate("你好")
        for token in manager.stream_generate("你好"):
            print(token, end="")
    """

    PROVIDER_MAP = {
        "openai": OpenAILLM,
        "zhipu": ZhipuLLM,
        "dashscope": DashscopeLLM,
    }

    def __init__(self, provider: str = None):
        """
        初始化 LLM 管理器

        Args:
            provider: LLM 提供商 (openai/zhipu/dashscope)
        """
        self._provider_name = provider or settings.llm_provider
        self._llm: Optional[BaseLLM] = None
        self._init_llm()

    def _init_llm(self) -> None:
        """根据配置初始化 LLM"""
        provider_class = self.PROVIDER_MAP.get(self._provider_name)
        if provider_class is None:
            raise ValueError(
                f"不支持的 LLM 提供商: {self._provider_name}，"
                f"可选: {list(self.PROVIDER_MAP.keys())}"
            )

        try:
            self._llm = provider_class()
            logger.info(f"LLM 管理器初始化成功: 提供商={self._provider_name}, 模型={self._llm.model_name}")
        except Exception as e:
            logger.error(f"LLM 初始化失败: {e}")
            raise

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """同步生成回答"""
        return self._llm.generate(prompt, system_prompt, **kwargs)

    def stream_generate(self, prompt: str, system_prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """流式生成回答"""
        return self._llm.stream_generate(prompt, system_prompt, **kwargs)

    @property
    def model_name(self) -> str:
        """获取当前模型名称"""
        return self._llm.model_name

    @property
    def provider(self) -> str:
        """获取当前提供商名称"""
        return self._provider_name


# 全局单例（延迟初始化）
_llm_manager: Optional[LLMManager] = None


def get_llm_manager(force_reload: bool = False) -> LLMManager:
    """
    获取全局 LLM 管理器单例

    Args:
        force_reload: 是否强制重新加载

    Returns:
        LLMManager: LLM 管理器实例
    """
    global _llm_manager
    if _llm_manager is None or force_reload:
        _llm_manager = LLMManager()
    return _llm_manager