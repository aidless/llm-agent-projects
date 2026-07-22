"""
对话管理模块 - ConversationManager

管理多轮对话的上下文、会话历史和持久化存储。

功能：
1. 会话创建和管理（基于唯一 session_id）
2. 多轮对话上下文维护
3. 内存存储 + JSON 文件持久化
4. 会话历史查询和清理
"""

import json
import os
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class Message:
    """单条消息"""
    role: str                                    # 角色：user / assistant / system
    content: str                                 # 消息内容
    timestamp: str = ""                          # 时间戳
    safety_score: float = 100.0                  # 安全评分
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "safety_score": self.safety_score,
            "metadata": self.metadata,
        }


@dataclass
class Session:
    """会话对象"""
    session_id: str                              # 会话唯一 ID
    created_at: str = ""                         # 创建时间
    updated_at: str = ""                         # 最后更新时间
    messages: List[Message] = field(default_factory=list)  # 消息历史
    metadata: Dict[str, Any] = field(default_factory=dict)  # 会话元数据

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 JSON 序列化）"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata,
        }

    def get_context_messages(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """
        获取最近的消息作为上下文（用于 LLM API 调用）

        Args:
            max_messages: 最多返回的消息数量

        Returns:
            List[Dict[str, str]]: 格式化的消息列表 [{role, content}]
        """
        recent = self.messages[-max_messages:]
        return [{"role": msg.role, "content": msg.content} for msg in recent]

    def add_message(self, role: str, content: str, safety_score: float = 100.0, metadata: Optional[Dict] = None) -> Message:
        """
        添加一条消息到会话

        Args:
            role: 消息角色
            content: 消息内容
            safety_score: 安全评分
            metadata: 元数据

        Returns:
            Message: 添加的消息对象
        """
        message = Message(
            role=role,
            content=content,
            safety_score=safety_score,
            metadata=metadata or {},
        )
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()
        return message


class ConversationManager:
    """
    对话管理器

    管理所有会话的生命周期，支持内存存储和文件持久化。
    """

    def __init__(self, persist_dir: str = "./data/sessions", max_history: int = 50):
        """
        初始化对话管理器

        Args:
            persist_dir: 会话持久化存储目录
            max_history: 每个会话最大保存的消息数量
        """
        self.persist_dir = persist_dir
        self.max_history = max_history

        # 内存中的会话存储 {session_id: Session}
        self._sessions: Dict[str, Session] = {}

        # 确保持久化目录存在
        os.makedirs(persist_dir, exist_ok=True)

        # 加载已有的持久化会话
        self._load_persisted_sessions()

    def create_session(self, session_id: Optional[str] = None, metadata: Optional[Dict] = None) -> Session:
        """
        创建一个新会话

        Args:
            session_id: 自定义会话 ID（可选，默认自动生成）
            metadata: 会话元数据

        Returns:
            Session: 新创建的会话对象
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        session = Session(
            session_id=session_id,
            metadata=metadata or {},
        )
        self._sessions[session_id] = session
        self._persist_session(session)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        获取指定会话

        Args:
            session_id: 会话 ID

        Returns:
            Optional[Session]: 会话对象，如果不存在返回 None
        """
        return self._sessions.get(session_id)

    def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        """
        获取或创建会话

        如果 session_id 存在则返回已有会话，否则创建新会话。

        Args:
            session_id: 会话 ID（可选）

        Returns:
            Session: 会话对象
        """
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        return self.create_session(session_id)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        safety_score: float = 100.0,
        metadata: Optional[Dict] = None,
    ) -> Optional[Message]:
        """
        向指定会话添加消息

        Args:
            session_id: 会话 ID
            role: 消息角色（user / assistant / system）
            content: 消息内容
            safety_score: 安全评分
            metadata: 元数据

        Returns:
            Optional[Message]: 添加的消息，会话不存在返回 None
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None

        message = session.add_message(
            role=role,
            content=content,
            safety_score=safety_score,
            metadata=metadata,
        )

        # 限制历史消息数量
        if len(session.messages) > self.max_history:
            session.messages = session.messages[-self.max_history:]

        # 持久化
        self._persist_session(session)
        return message

    def get_context(self, session_id: str, max_messages: int = 10) -> List[Dict[str, str]]:
        """
        获取会话的最近上下文消息

        Args:
            session_id: 会话 ID
            max_messages: 最多返回的消息数量

        Returns:
            List[Dict[str, str]]: 上下文消息列表
        """
        session = self._sessions.get(session_id)
        if session is None:
            return []
        return session.get_context_messages(max_messages)

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取会话的完整历史记录

        Args:
            session_id: 会话 ID

        Returns:
            List[Dict[str, Any]]: 所有消息的列表
        """
        session = self._sessions.get(session_id)
        if session is None:
            return []
        return [msg.to_dict() for msg in session.messages]

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        列出所有会话的摘要信息

        Returns:
            List[Dict[str, Any]]: 会话摘要列表
        """
        summaries = []
        for session_id, session in self._sessions.items():
            summaries.append({
                "session_id": session_id,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "message_count": len(session.messages),
                "metadata": session.metadata,
            })
        return summaries

    def delete_session(self, session_id: str) -> bool:
        """
        删除指定会话

        Args:
            session_id: 会话 ID

        Returns:
            bool: 是否成功删除
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            # 删除持久化文件
            file_path = os.path.join(self.persist_dir, f"{session_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        return False

    def clear_session_history(self, session_id: str) -> bool:
        """
        清除指定会话的消息历史（保留会话本身）

        Args:
            session_id: 会话 ID

        Returns:
            bool: 是否成功清除
        """
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session.messages = []
        session.updated_at = datetime.now().isoformat()
        self._persist_session(session)
        return True

    def _persist_session(self, session: Session) -> None:
        """
        将会话持久化到 JSON 文件

        Args:
            session: 会话对象
        """
        file_path = os.path.join(self.persist_dir, f"{session.session_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # 持久化失败不应影响主流程

    def _load_persisted_sessions(self) -> None:
        """
        加载持久化的会话文件

        从持久化目录中读取所有 .json 文件并恢复到内存中。
        """
        if not os.path.exists(self.persist_dir):
            return

        for filename in os.listdir(self.persist_dir):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(self.persist_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                session_id = data.get("session_id")
                if not session_id:
                    continue

                # 重建 Session 对象
                messages = []
                for msg_data in data.get("messages", []):
                    messages.append(Message(**msg_data))

                session = Session(
                    session_id=session_id,
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                    messages=messages,
                    metadata=data.get("metadata", {}),
                )
                self._sessions[session_id] = session

            except (json.JSONDecodeError, TypeError, KeyError):
                continue
