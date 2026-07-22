"""WebSocket 会话管理"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from fastapi import WebSocket


class SessionState(str, Enum):
    """会话状态"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"


@dataclass
class SessionInfo:
    """会话信息"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    state: SessionState = SessionState.CONNECTING
    websocket: Optional[WebSocket] = None
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def touch(self):
        """更新最后活跃时间"""
        self.last_active = time.time()

    def is_expired(self, timeout: float = 600) -> bool:
        """是否超时"""
        return (time.time() - self.last_active) > timeout


class SessionManager:
    """WebSocket 会话管理器"""

    def __init__(self, max_sessions: int = 1000, session_timeout: float = 600):
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self._sessions: dict[str, SessionInfo] = {}

    def create_session(self, websocket: Optional[WebSocket] = None) -> SessionInfo:
        """创建新会话"""
        # 清理超时会话
        self.cleanup_expired()

        if len(self._sessions) >= self.max_sessions:
            raise RuntimeError("Maximum number of sessions reached")

        session = SessionInfo(websocket=websocket)
        session.state = SessionState.CONNECTED
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """获取会话"""
        return self._sessions.get(session_id)

    def update_state(self, session_id: str, state: SessionState):
        """更新会话状态"""
        session = self._sessions.get(session_id)
        if session:
            session.state = state
            session.touch()

    def remove_session(self, session_id: str):
        """移除会话"""
        self._sessions.pop(session_id, None)

    def get_active_count(self) -> int:
        """获取活跃会话数"""
        return len(self._sessions)

    def cleanup_expired(self) -> int:
        """清理超时会话"""
        expired = [
            sid for sid, s in self._sessions.items()
            if s.is_expired(self.session_timeout)
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
        return len(expired)

    def get_all_session_ids(self) -> list[str]:
        """获取所有会话 ID"""
        return list(self._sessions.keys())