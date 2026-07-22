"""对话历史管理模块 - 滑动窗口 + 摘要"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Optional

from app.models import Message, MessageRole


class DialogHistory:
    """对话历史管理器"""

    def __init__(
        self,
        max_window_size: int = 20,
        max_summary_length: int = 200,
    ) -> None:
        self._histories: dict[str, deque[Message]] = {}
        self._summaries: dict[str, str] = {}
        self._max_window_size = max_window_size
        self._max_summary_length = max_summary_length
        self._llm_summarize_func: Optional[callable] = None

    def set_llm_summarizer(self, func: callable) -> None:
        """设置 LLM 摘要回调"""
        self._llm_summarize_func = func

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Message:
        """添加消息到历史"""
        msg = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        if session_id not in self._histories:
            self._histories[session_id] = deque(maxlen=self._max_window_size)
        self._histories[session_id].append(msg)
        return msg

    def get_history(
        self,
        session_id: str,
        last_n: Optional[int] = None,
    ) -> list[Message]:
        """获取对话历史"""
        history = self._histories.get(session_id)
        if not history:
            return []
        messages = list(history)
        if last_n:
            messages = messages[-last_n:]
        return messages

    def get_history_dict(
        self,
        session_id: str,
        last_n: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """获取对话历史的字典表示"""
        messages = self.get_history(session_id, last_n)
        return [
            {
                "id": m.id,
                "role": m.role.value,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if isinstance(m.timestamp, datetime) else str(m.timestamp),
                "metadata": m.metadata,
            }
            for m in messages
        ]

    def get_context_window(
        self,
        session_id: str,
        max_tokens: int = 1000,
    ) -> list[Message]:
        """获取上下文窗口（简单字符数近似）"""
        messages = self.get_history(session_id)
        if not messages:
            return []

        result: list[Message] = []
        total_chars = 0
        for msg in reversed(messages):
            msg_chars = len(msg.content)
            if total_chars + msg_chars > max_tokens:
                break
            result.insert(0, msg)
            total_chars += msg_chars

        return result

    def get_summary(self, session_id: str) -> str:
        """获取对话摘要"""
        return self._summaries.get(session_id, "")

    def generate_summary(self, session_id: str) -> str:
        """生成对话摘要"""
        history = self.get_history(session_id)
        if not history:
            return ""

        # 如果有 LLM 摘要
        if self._llm_summarize_func:
            try:
                summary = self._llm_summarize_func(
                    [m.model_dump() for m in history]
                )
                if summary:
                    self._summaries[session_id] = summary[:self._max_summary_length]
                    return self._summaries[session_id]
            except Exception:
                pass

        # 简单摘要：提取关键信息
        summary_parts = []
        user_msgs = [m for m in history if m.role == MessageRole.USER]
        if user_msgs:
            summary_parts.append(f"用户共发送 {len(user_msgs)} 条消息")

        summary = "；".join(summary_parts) if summary_parts else "暂无对话记录"
        self._summaries[session_id] = summary
        return summary

    def get_message_count(self, session_id: str) -> int:
        """获取消息数量"""
        history = self._histories.get(session_id)
        return len(history) if history else 0

    def clear_history(self, session_id: str) -> None:
        """清除对话历史"""
        self._histories.pop(session_id, None)
        self._summaries.pop(session_id, None)

    def get_all_session_ids(self) -> list[str]:
        """获取所有会话ID"""
        return list(self._histories.keys())
