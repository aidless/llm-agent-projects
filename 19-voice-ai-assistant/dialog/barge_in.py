"""打断处理 (Barge-in)"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Awaitable


class BargeInState(str, Enum):
    """打断状态"""
    IDLE = "idle"                   # 空闲
    SPEAKING = "speaking"           # 助手正在说话
    LISTENING = "listening"         # 助手正在听用户说话
    BARGE_IN_DETECTED = "barge_in_detected"  # 检测到打断
    PROCESSING = "processing"       # 处理打断中


@dataclass
class BargeInConfig:
    """打断检测配置"""
    # 是否启用打断
    enabled: bool = True
    # 检测到语音活动后的确认帧数（避免误触发）
    confirm_frames: int = 3
    # 能量阈值（用于确认用户确实在说话）
    energy_threshold: float = 500.0
    # 最大打断次数（防止频繁打断）
    max_barge_in_count: int = 5
    # 打断冷却时间（秒）
    cooldown_seconds: float = 1.0
    # 在一段话的哪个比例后允许打断（0.0 - 1.0）
    min_progress: float = 0.1


@dataclass
class BargeInEvent:
    """打断事件"""
    session_id: str
    state: BargeInState
    interrupted_text: str = ""
    new_audio_data: bytes = b""
    timestamp: float = 0.0


class BargeInHandler:
    """打断处理器

    处理用户在 TTS 播放期间的打断行为。
    """

    def __init__(self, config: Optional[BargeInConfig] = None):
        self.config = config or BargeInConfig()
        self._state = BargeInState.IDLE
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._barge_in_count = 0
        self._last_barge_in_time: float = 0
        self._current_response_text: str = ""
        self._response_progress: float = 0.0
        self._on_barge_in_callbacks: list[Callable[[BargeInEvent], Awaitable[None]]] = []

    def set_state(self, state: BargeInState):
        """设置打断状态"""
        self._state = state
        if state == BargeInState.LISTENING:
            self._speech_frame_count = 0
            self._silence_frame_count = 0

    def get_state(self) -> BargeInState:
        """获取当前状态"""
        return self._state

    def set_current_response(self, text: str, progress: float = 0.0):
        """设置当前正在播放的回复文本

        Args:
            text: 回复文本
            progress: 播放进度 (0.0 - 1.0)
        """
        self._current_response_text = text
        self._response_progress = progress

    def on_barge_in(self, callback: Callable[[BargeInEvent], Awaitable[None]]):
        """注册打断回调"""
        self._on_barge_in_callbacks.append(callback)

    def detect(self, energy: float) -> bool:
        """检测是否应该触发打断

        Args:
            energy: 当前帧能量

        Returns:
            bool: 是否触发打断
        """
        if not self.config.enabled:
            return False
        if self._state != BargeInState.SPEAKING:
            return False

        import time
        now = time.time()

        # 检查冷却时间
        if now - self._last_barge_in_time < self.config.cooldown_seconds:
            return False

        # 检查最大打断次数
        if self._barge_in_count >= self.config.max_barge_in_count:
            return False

        # 检查最小播放进度
        if self._response_progress < self.config.min_progress:
            return False

        if energy > self.config.energy_threshold:
            self._speech_frame_count += 1
            self._silence_frame_count = 0
        else:
            self._silence_frame_count += 1
            if self._silence_frame_count > 2:
                self._speech_frame_count = 0

        if self._speech_frame_count >= self.config.confirm_frames:
            self._state = BargeInState.BARGE_IN_DETECTED
            self._last_barge_in_time = now
            self._barge_in_count += 1
            return True

        return False

    async def handle_barge_in(self, session_id: str, new_audio: bytes = b"") -> BargeInEvent:
        """处理打断事件

        Args:
            session_id: 会话 ID
            new_audio: 用户的打断音频

        Returns:
            BargeInEvent: 打断事件
        """
        import time
        event = BargeInEvent(
            session_id=session_id,
            state=BargeInState.BARGE_IN_DETECTED,
            interrupted_text=self._current_response_text,
            new_audio_data=new_audio,
            timestamp=time.time(),
        )

        self._state = BargeInState.PROCESSING

        # 触发回调
        for callback in self._on_barge_in_callbacks:
            try:
                await callback(event)
            except Exception:
                pass  # 回调错误不影响主流程

        return event

    def reset(self):
        """重置打断状态"""
        self._state = BargeInState.IDLE
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._current_response_text = ""
        self._response_progress = 0.0

    def reset_count(self):
        """重置打断计数"""
        self._barge_in_count = 0