# ============================================
# 短期对话记忆
# 维护最近 N 轮的对话历史，用于保持上下文连贯性
# ============================================

import json
from collections import deque
from typing import Optional
from loguru import logger


class ShortTermMemory:
    """短期记忆：基于 deque 的有限窗口对话历史管理"""

    def __init__(self, max_turns: int = 20):
        """
        初始化短期记忆

        Args:
            max_turns: 最大保留轮数（一轮 = 一问一答）
        """
        self.max_turns = max_turns
        # 每个 entry 是一个 dict: {"role": str, "content": str}
        self._history: deque = deque(maxlen=max_turns * 2)
        self._metadata: dict = {}  # 会话元数据

    def add_message(self, role: str, content: str) -> None:
        """
        添加一条消息到对话历史

        Args:
            role: 消息角色（user / assistant / system / tool）
            content: 消息内容
        """
        entry = {"role": role, "content": content}
        self._history.append(entry)
        logger.debug(f"[短期记忆] 添加消息 role={role}, 当前共 {len(self._history)} 条")

    def add_user_message(self, content: str) -> None:
        """添加用户消息的快捷方法"""
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """添加助手消息的快捷方法"""
        self.add_message("assistant", content)

    def add_tool_message(self, content: str) -> None:
        """添加工具调用结果的快捷方法"""
        self.add_message("tool", content)

    def get_history(self, last_n: Optional[int] = None) -> list:
        """
        获取对话历史列表

        Args:
            last_n: 只获取最近 N 条，None 表示全部

        Returns:
            list[dict]: 对话历史列表
        """
        history_list = list(self._history)
        if last_n is not None and last_n > 0:
            history_list = history_list[-last_n:]
        return history_list

    def get_langchain_messages(self) -> list:
        """
        获取适合 LangChain 的消息格式

        Returns:
            list[BaseMessage]: HumanMessage / AIMessage 列表
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

        messages = []
        role_map = {
            "user": HumanMessage,
            "assistant": AIMessage,
            "system": SystemMessage,
            "tool": ToolMessage,
        }
        for entry in self._history:
            msg_class = role_map.get(entry["role"], HumanMessage)
            messages.append(msg_class(content=entry["content"]))
        return messages

    def clear(self) -> None:
        """清空对话历史"""
        self._history.clear()
        logger.info("[短期记忆] 已清空对话历史")

    def get_turn_count(self) -> int:
        """获取当前对话轮数（每两条消息为一轮）"""
        return len(self._history)

    def set_metadata(self, key: str, value: str) -> None:
        """设置会话元数据"""
        self._metadata[key] = value

    def get_metadata(self, key: str) -> Optional[str]:
        """获取会话元数据"""
        return self._metadata.get(key)

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "type": "short_term",
            "max_turns": self.max_turns,
            "history": list(self._history),
            "metadata": self._metadata,
            "turn_count": self.get_turn_count(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ShortTermMemory":
        """从字典反序列化"""
        mem = cls(max_turns=data.get("max_turns", 20))
        mem._metadata = data.get("metadata", {})
        for entry in data.get("history", []):
            mem.add_message(entry["role"], entry["content"])
        return mem

    def __repr__(self) -> str:
        return f"ShortTermMemory(turns={self.get_turn_count()}, max={self.max_turns})"
