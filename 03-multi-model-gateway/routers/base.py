"""
模型适配器基类 - 定义统一接口，所有模型适配器必须继承此类
"""
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from app.config import ModelConfig
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    UsageInfo,
    ChatCompletionChoice,
    ChatMessage,
)


class BaseAdapter(ABC):
    """
    模型适配器基类
    每个模型（OpenAI / DeepSeek / Qwen / Claude）需要实现此基类的抽象方法
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._key_index = 0  # 用于多 Key 轮询

    @property
    def model_key(self) -> str:
        """模型在网关中的标识 key"""
        return self.config.name

    @property
    def provider(self) -> str:
        """服务商标识"""
        return self.config.provider

    def get_next_api_key(self) -> str:
        """
        获取下一个 API Key（轮询策略）
        如果配置了多个 Key，则依次轮询，避免单个 Key 触发速率限制
        """
        keys = self.config.all_keys
        if not keys:
            return ""
        key = keys[self._key_index % len(keys)]
        self._key_index += 1
        return key

    @property
    def input_price_per_1k(self) -> float:
        """输入价格（美元/千 token）"""
        return self.config.input_price_per_1k

    @property
    def output_price_per_1k(self) -> float:
        """输出价格（美元/千 token）"""
        return self.config.output_price_per_1k

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """根据 Token 用量计算成本（美元）"""
        input_cost = (prompt_tokens / 1000) * self.input_price_per_1k
        output_cost = (completion_tokens / 1000) * self.output_price_per_1k
        return input_cost + output_cost

    @abstractmethod
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        timeout: float = 60.0,
        api_key: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """
        发送聊天补全请求

        参数:
            request: 聊天补全请求
            timeout: 超时时间（秒）
            api_key: 指定使用的 API Key（不指定则自动轮询）

        返回:
            ChatCompletionResponse 统一格式的响应

        异常:
            httpx.HTTPStatusError: HTTP 错误
            httpx.TimeoutException: 超时
            Exception: 其他错误
        """
        ...

    def _build_default_response(
        self,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: str = "stop",
    ) -> ChatCompletionResponse:
        """构建默认格式的响应对象"""
        total_tokens = prompt_tokens + completion_tokens
        cost = self.calculate_cost(prompt_tokens, completion_tokens)
        return ChatCompletionResponse(
            model=self.config.model_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            routed_model=self.model_key,
            cost_usd=cost,
        )

    def _build_error_response(self, error_msg: str) -> ChatCompletionResponse:
        """构建错误响应"""
        resp = ChatCompletionResponse(
            model=self.config.model_id,
            routed_model=self.model_key,
        )
        return resp