"""TTS 引擎抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional


class TTSAudioFormat(str, Enum):
    """TTS 输出音频格式"""
    WAV = "wav"
    MP3 = "mp3"


@dataclass
class TTSVoice:
    """TTS 音色"""
    voice_id: str
    name: str
    language: str = "zh"
    gender: str = "female"
    description: str = ""


@dataclass
class TTSConfig:
    """TTS 配置"""
    voice_id: str = "default"
    sample_rate: int = 16000
    speed: float = 1.0        # 语速 0.5 - 2.0
    pitch: float = 1.0        # 音调 0.5 - 2.0
    volume: float = 1.0       # 音量 0.0 - 1.0
    audio_format: TTSAudioFormat = TTSAudioFormat.WAV
    enable_ssml: bool = True  # 是否支持 SSML


@dataclass
class TTSResult:
    """TTS 合成结果"""
    audio_data: bytes
    audio_format: TTSAudioFormat
    sample_rate: int
    duration: float = 0.0
    text: str = ""
    is_final: bool = True


class TTSEngine(ABC):
    """TTS 引擎抽象基类"""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()

    @abstractmethod
    async def synthesize(self, text: str) -> TTSResult:
        """将文本转换为语音

        Args:
            text: 要合成的文本（可包含 SSML 标记）

        Returns:
            TTSResult: 合成结果
        """
        ...

    @abstractmethod
    async def stream_synthesize(self, text: str) -> AsyncIterator[TTSResult]:
        """流式合成语音

        Args:
            text: 要合成的文本

        Yields:
            TTSResult: 分块的音频数据
        """
        ...

    def list_voices(self) -> list[TTSVoice]:
        """列出可用音色"""
        return []

    def set_voice(self, voice_id: str):
        """设置音色"""
        self.config.voice_id = voice_id

    def set_speed(self, speed: float):
        """设置语速"""
        self.config.speed = max(0.5, min(2.0, speed))

    def set_pitch(self, pitch: float):
        """设置音调"""
        self.config.pitch = max(0.5, min(2.0, pitch))

    def set_volume(self, volume: float):
        """设置音量"""
        self.config.volume = max(0.0, min(1.0, volume))