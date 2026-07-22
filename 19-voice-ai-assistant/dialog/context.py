"""对话上下文管理"""

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MessageRole(str, Enum):
    """消息角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """对话消息"""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: dict = field(default_factory=dict)


@dataclass
class IntentResult:
    """意图识别结果"""
    intent: str
    confidence: float
    entities: dict = field(default_factory=dict)
    raw_text: str = ""


class DialogContext:
    """对话上下文

    管理多轮对话的历史消息、意图、实体等信息。
    """

    def __init__(self, max_history: int = 50, max_turns: int = 20):
        """
        Args:
            max_history: 最大保留消息数
            max_turns: 最大对话轮数 (一轮 = 一次用户输入 + 一次助手回复)
        """
        self.session_id: str = str(uuid.uuid4())[:12]
        self.messages: deque[Message] = deque(maxlen=max_history)
        self.max_turns = max_turns
        self._turn_count = 0
        self._intent_history: list[IntentResult] = []
        self._entities: dict = {}  # 对话级实体
        self._created_at: float = time.time()
        self._last_active: float = time.time()

    def add_message(self, role: MessageRole, content: str, **metadata) -> Message:
        """添加消息到上下文

        Args:
            role: 消息角色
            content: 消息内容
            **metadata: 附加元数据

        Returns:
            Message: 创建的消息对象
        """
        message = Message(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        self._last_active = time.time()

        if role == MessageRole.USER:
            self._turn_count += 1

        return message

    def get_messages(self, role: Optional[MessageRole] = None, limit: Optional[int] = None) -> list[Message]:
        """获取消息列表

        Args:
            role: 按角色过滤
            limit: 最大返回数量

        Returns:
            消息列表
        """
        messages = list(self.messages)
        if role is not None:
            messages = [m for m in messages if m.role == role]
        if limit is not None:
            messages = messages[-limit:]
        return messages

    def get_recent_messages(self, n: int = 10) -> list[Message]:
        """获取最近 N 条消息"""
        return list(self.messages)[-n:]

    def get_context_for_llm(self, system_prompt: str = "", max_tokens: int = 2000) -> list[dict]:
        """构建用于 LLM 的上下文

        Args:
            system_prompt: 系统提示
            max_tokens: 最大 token 数（粗略估计）

        Returns:
            消息字典列表，格式兼容 OpenAI API
        """
        result = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        # 估算 tokens (中文约 1.5 字/token，英文约 0.25 词/token)
        current_tokens = 0
        messages = list(self.messages)

        for msg in reversed(messages):
            estimated_tokens = len(msg.content) * 1.5
            if current_tokens + estimated_tokens > max_tokens:
                break
            result.insert(1 if system_prompt else 0, {
                "role": msg.role.value,
                "content": msg.content,
            })
            current_tokens += estimated_tokens

        return result

    def add_intent(self, intent_result: IntentResult):
        """记录意图识别结果"""
        self._intent_history.append(intent_result)
        # 更新实体
        for key, value in intent_result.entities.items():
            self._entities[key] = value

    def get_entities(self) -> dict:
        """获取当前对话中提取的实体"""
        return dict(self._entities)

    def update_entity(self, key: str, value: Any):
        """更新实体"""
        self._entities[key] = value

    def clear_entities(self):
        """清空实体"""
        self._entities.clear()

    def get_turn_count(self) -> int:
        """获取对话轮数"""
        return self._turn_count

    def is_max_turns(self) -> bool:
        """是否达到最大轮数"""
        return self._turn_count >= self.max_turns

    def is_expired(self, timeout_seconds: float = 300) -> bool:
        """是否超时"""
        return (time.time() - self._last_active) > timeout_seconds

    def get_idle_seconds(self) -> float:
        """获取空闲时间（秒）"""
        return time.time() - self._last_active

    def clear(self):
        """清空上下文"""
        self.messages.clear()
        self._intent_history.clear()
        self._entities.clear()
        self._turn_count = 0
        self._last_active = time.time()