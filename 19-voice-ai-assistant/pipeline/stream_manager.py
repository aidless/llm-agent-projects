"""流管理器 - 管理音频和文本流的中间缓存与转发"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class StreamType(str, Enum):
    """流类型"""
    AUDIO_INPUT = "audio_input"        # 输入音频流
    ASR_PARTIAL = "asr_partial"          # ASR 中间结果
    ASR_FINAL = "asr_final"              # ASR 最终结果
    LLM_CHUNK = "llm_chunk"              # LLM 生成的文本块
    LLM_COMPLETE = "llm_complete"        # LLM 生成完成
    TTS_AUDIO_CHUNK = "tts_audio_chunk"  # TTS 音频块
    TTS_COMPLETE = "tts_complete"        # TTS 合成完成
    ERROR = "error"                      # 错误


@dataclass
class StreamEvent:
    """流事件"""
    type: StreamType
    data: Any = None
    session_id: str = ""
    timestamp: float = 0.0
    metadata: dict = field(default_factory=dict)


class StreamManager:
    """流管理器

    管理 Pipeline 各环节之间的数据流转，
    支持事件缓存和异步等待。
    """

    def __init__(self, buffer_size: int = 100):
        self._buffer_size = buffer_size
        self._events: dict[str, deque[StreamEvent]] = {}
        self._waiters: dict[str, list[asyncio.Future]] = {}
        self._closed_sessions: set[str] = set()

    def _get_buffer(self, session_id: str) -> deque[StreamEvent]:
        """获取会话的事件缓冲区"""
        if session_id not in self._events:
            self._events[session_id] = deque(maxlen=self._buffer_size)
        return self._events[session_id]

    def push(self, event: StreamEvent):
        """推送事件

        Args:
            event: 流事件
        """
        if event.timestamp == 0:
            event.timestamp = time.time()

        if event.session_id in self._closed_sessions:
            return

        buffer = self._get_buffer(event.session_id)
        buffer.append(event)

        # 通知等待者
        if event.session_id in self._waiters:
            for future in self._waiters[event.session_id]:
                if not future.done():
                    future.set_result(None)
            self._waiters[event.session_id].clear()

    async def wait_for_event(self, session_id: str, timeout: float = 30.0) -> Optional[StreamEvent]:
        """等待下一个事件

        Args:
            session_id: 会话 ID
            timeout: 超时时间（秒）

        Returns:
            StreamEvent 或 None
        """
        buffer = self._get_buffer(session_id)
        if buffer:
            return buffer.popleft()

        # 创建等待
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        if session_id not in self._waiters:
            self._waiters[session_id] = []
        self._waiters[session_id].append(future)

        try:
            await asyncio.wait_for(future, timeout=timeout)
            buffer = self._get_buffer(session_id)
            if buffer:
                return buffer.popleft()
        except asyncio.TimeoutError:
            pass
        finally:
            if future in self._waiters.get(session_id, []):
                self._waiters[session_id].remove(future)

        return None

    def get_pending_events(self, session_id: str) -> list[StreamEvent]:
        """获取所有待处理事件"""
        buffer = self._get_buffer(session_id)
        events = list(buffer)
        buffer.clear()
        return events

    def close_session(self, session_id: str):
        """关闭会话流"""
        self._closed_sessions.add(session_id)
        self._events.pop(session_id, None)
        self._waiters.pop(session_id, None)

    def is_session_closed(self, session_id: str) -> bool:
        """会话是否已关闭"""
        return session_id in self._closed_sessions

    def reset(self):
        """重置所有状态"""
        self._events.clear()
        self._waiters.clear()
        self._closed_sessions.clear()