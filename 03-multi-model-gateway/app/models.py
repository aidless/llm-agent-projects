"""
统一数据模型 - 兼容 OpenAI API 格式的请求/响应模型
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatMessage:
    """单条聊天消息"""
    role: str        # system / user / assistant
    content: str     # 消息内容

    def to_openai_dict(self) -> dict:
        """转换为 OpenAI API 格式的字典"""
        return {"role": self.role, "content": self.content}


@dataclass
class ChatCompletionRequest:
    """聊天补全请求 - 兼容 OpenAI 格式"""
    model: Optional[str] = None          # 指定模型（可选，不指定则走智能路由）
    messages: list[ChatMessage] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    stream: bool = False                 # 是否流式输出
    # 自定义路由参数
    strategy: Optional[str] = None       # 路由策略: task_type / cost / latency / load_balance
    user_hint: Optional[str] = None      # 用户提示的任务类型

    @classmethod
    def from_openai_dict(cls, data: dict) -> "ChatCompletionRequest":
        """从 OpenAI API 格式的字典构建请求"""
        messages = []
        for msg in data.get("messages", []):
            messages.append(ChatMessage(
                role=msg["role"],
                content=msg.get("content", ""),
            ))
        return cls(
            model=data.get("model"),
            messages=messages,
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 2048),
            top_p=data.get("top_p", 1.0),
            stream=data.get("stream", False),
            strategy=data.get("strategy"),
            user_hint=data.get("user_hint"),
        )

    def get_last_user_message(self) -> str:
        """获取最后一条用户消息，用于任务类型判断"""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return ""


@dataclass
class ChatCompletionChoice:
    """单个回复选项"""
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


@dataclass
class UsageInfo:
    """Token 使用信息"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ChatCompletionResponse:
    """聊天补全响应 - 兼容 OpenAI 格式"""
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[ChatCompletionChoice] = field(default_factory=list)
    usage: UsageInfo = field(default_factory=UsageInfo)
    # 网关附加信息
    routed_model: str = ""       # 实际路由到的模型名称
    strategy_used: str = ""      # 使用的路由策略
    latency_ms: float = 0.0      # 请求延迟（毫秒）
    cost_usd: float = 0.0        # 本次请求成本（美元）

    def __post_init__(self):
        if not self.id:
            self.id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        if not self.created:
            self.created = int(time.time())

    def to_openai_dict(self) -> dict:
        """转换为 OpenAI API 格式的响应字典"""
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": c.index,
                    "message": c.message.to_openai_dict(),
                    "finish_reason": c.finish_reason,
                }
                for c in self.choices
            ],
            "usage": self.usage.to_dict(),
            # 网关自定义字段
            "routed_model": self.routed_model,
            "strategy_used": self.strategy_used,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
        }


@dataclass
class ModelErrorResponse:
    """模型调用错误响应"""
    error: str
    model: str = ""
    status_code: int = 500

    def to_dict(self) -> dict:
        return {
            "error": {
                "message": self.error,
                "type": "gateway_error",
                "model": self.model,
            },
            "status_code": self.status_code,
        }