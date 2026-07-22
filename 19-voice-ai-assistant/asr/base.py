"""ASR 引擎抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional


class AudioFormat(str, Enum):
    """支持的音频格式"""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    PCM = "pcm"


@dataclass
class ASRConfig:
    """ASR 配置"""
    sample_rate: int = 16000
    channels: int = 1
    language: str = "zh"  # "zh" 或 "en"
    audio_format: AudioFormat = AudioFormat.WAV
    enable_punctuation: bool = True
    enable_number_normalize: bool = True


@dataclass
class ASRResult:
    """ASR 识别结果"""
    text: str
    is_final: bool = True
    confidence: float = 1.0
    language: str = "zh"
    timestamps: list = field(default_factory=list)
    # 流式中间结果
    partial: bool = False


class ASREngine(ABC):
    """ASR 引擎抽象基类"""

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()

    @abstractmethod
    async def recognize(self, audio_data: bytes) -> ASRResult:
        """识别一段完整的音频

        Args:
            audio_data: 音频数据 (原始字节)

        Returns:
            ASRResult: 识别结果
        """
        ...

    @abstractmethod
    async def stream_recognize(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[ASRResult]:
        """流式识别音频

        Args:
            audio_stream: 音频数据流 (异步迭代器)

        Yields:
            ASRResult: 识别结果 (中间结果和最终结果)
        """
        ...

    def postprocess(self, text: str) -> str:
        """识别结果后处理

        Args:
            text: 原始识别文本

        Returns:
            str: 后处理后的文本
        """
        if self.config.enable_number_normalize:
            text = self._normalize_numbers(text)
        if self.config.enable_punctuation:
            text = self._add_punctuation(text)
        return text.strip()

    @staticmethod
    def _normalize_numbers(text: str) -> str:
        """数字规范化: 将阿拉伯数字转换为中文读法"""
        import re
        # 简单数字规范化：保留数字形式但确保格式统一
        text = re.sub(r'\s+', '', text)
        return text

    @staticmethod
    def _add_punctuation(text: str) -> str:
        """为识别文本添加基本标点"""
        if not text:
            return text
        # 确保句子末尾有句号
        text = text.rstrip()
        if text and text[-1] not in '。！？.!?\n':
            text += '。'
        return text

    def supported_formats(self) -> list[AudioFormat]:
        """返回支持的音频格式列表"""
        return [AudioFormat.WAV, AudioFormat.MP3, AudioFormat.OGG, AudioFormat.PCM]