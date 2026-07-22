"""
适配器管理器 - 注册和管理所有模型适配器
提供统一的调用接口和适配器查找功能
"""
import logging
from typing import Optional

from app.config import AppConfig, ModelConfig, get_config
from app.models import ChatCompletionRequest, ChatCompletionResponse
from routers.base import BaseAdapter
from routers.openai_adapter import OpenAIAdapter
from routers.deepseek_adapter import DeepSeekAdapter
from routers.qwen_adapter import QwenAdapter
from routers.claude_adapter import ClaudeAdapter

logger = logging.getLogger(__name__)


class AdapterManager:
    """
    适配器管理器
    负责创建和管理所有模型适配器实例，提供按名称查找适配器的功能
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self._config = config or get_config()
        self._adapters: dict[str, BaseAdapter] = {}
        self._init_adapters()

    def _init_adapters(self):
        """根据配置初始化所有模型适配器"""
        # 适配器类映射：模型 key -> 适配器类
        adapter_classes = {
            "gpt-4o": OpenAIAdapter,
            "deepseek": DeepSeekAdapter,
            "qwen": QwenAdapter,
            "claude": ClaudeAdapter,
        }

        for model_key, adapter_cls in adapter_classes.items():
            model_config = self._config.models.get(model_key)
            if model_config is None:
                logger.warning(f"模型 {model_key} 未配置，跳过初始化")
                continue

            # 检查是否配置了 API Key
            if not model_config.api_key or model_config.api_key.startswith("sk-your-"):
                logger.warning(
                    f"模型 {model_key} 的 API Key 未配置或为占位符，"
                    f"该模型将不可用（可作为降级备用但会调用失败）"
                )

            self._adapters[model_key] = adapter_cls(model_config)
            logger.info(f"已注册模型适配器: {model_key} -> {model_config.name}")

    def get_adapter(self, model_key: str) -> Optional[BaseAdapter]:
        """根据模型 key 获取对应的适配器"""
        return self._adapters.get(model_key)

    def get_available_models(self) -> list[str]:
        """获取所有已注册的模型 key 列表"""
        return list(self._adapters.keys())

    def get_all_adapters(self) -> dict[str, BaseAdapter]:
        """获取所有适配器的字典"""
        return self._adapters.copy()

    async def call_model(
        self,
        model_key: str,
        request: ChatCompletionRequest,
        timeout: float = 60.0,
        api_key: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """
        调用指定模型

        参数:
            model_key: 模型标识（如 gpt-4o, deepseek 等）
            request: 聊天请求
            timeout: 超时时间
            api_key: 指定 API Key（不指定则自动轮询）

        返回:
            ChatCompletionResponse

        异常:
            ValueError: 模型不存在
            httpx.HTTPStatusError: HTTP 错误
        """
        adapter = self.get_adapter(model_key)
        if adapter is None:
            available = self.get_available_models()
            raise ValueError(
                f"模型 '{model_key}' 不存在，可用模型: {available}"
            )
        return await adapter.chat_completion(request, timeout=timeout, api_key=api_key)


# 全局适配器管理器单例
_adapter_manager: Optional[AdapterManager] = None


def get_adapter_manager() -> AdapterManager:
    """获取全局适配器管理器实例"""
    global _adapter_manager
    if _adapter_manager is None:
        _adapter_manager = AdapterManager()
    return _adapter_manager


def reset_adapter_manager():
    """重置全局适配器管理器（主要用于测试）"""
    global _adapter_manager
    _adapter_manager = None