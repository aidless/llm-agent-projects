"""Edge-TTS 引擎 - 基于 Microsoft Edge TTS 的免费真实 TTS 实现。

使用 edge-tts Python 库调用微软神经 TTS 接口，无需 API Key。
"""

import asyncio
import logging
import os
import re
from typing import AsyncIterator, Optional

from tts.base import TTSAudioFormat, TTSConfig, TTSEngine, TTSResult, TTSVoice

logger = logging.getLogger(__name__)


class EdgeTTS(TTSEngine):
    """基于 Edge TTS 的真实 TTS 实现。

    特点：
    - 完全免费，无需 API Key
    - 中文声音自然度高（微软神经 TTS）
    - 支持 SSML 控制语速/音调
    - 支持流式输出

    环境变量：
    - TTS_VOICE: 默认声音，默认 zh-CN-XiaoxiaoNeural
    - TTS_RATE: 语速调整，如 +10% 或 -10%
    - TTS_VOLUME: 音量调整，如 +5% 或 -5%

    注意：edge-tts 调用的是 Edge 浏览器 TTS 公共接口，
    非官方 API，不适合对 SLA 要求严格的生产环境。
    """

    # 常用声音列表
    AVAILABLE_VOICES = [
        TTSVoice(voice_id="zh-CN-XiaoxiaoNeural", name="晓晓 (女声·中文)", language="zh", gender="female"),
        TTSVoice(voice_id="zh-CN-YunxiNeural", name="云希 (男声·中文)", language="zh", gender="male"),
        TTSVoice(voice_id="zh-CN-YunjianNeural", name="云健 (男声·中文·体育)", language="zh", gender="male"),
        TTSVoice(voice_id="zh-CN-XiaoyiNeural", name="晓伊 (女声·中文)", language="zh", gender="female"),
        TTSVoice(voice_id="zh-CN-YunyangNeural", name="云扬 (男声·中文·新闻)", language="zh", gender="male"),
        TTSVoice(voice_id="en-US-AriaNeural", name="Aria (English Female)", language="en", gender="female"),
        TTSVoice(voice_id="en-US-GuyNeural", name="Guy (English Male)", language="en", gender="male"),
        TTSVoice(voice_id="en-US-JennyNeural", name="Jenny (English Female)", language="en", gender="female"),
    ]

    def __init__(self, config: Optional[TTSConfig] = None):
        super().__init__(config)
        # 默认声音从环境变量读取
        if self.config.voice_id == "default":
            self.config.voice_id = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
        self.rate = os.getenv("TTS_RATE", "+0%")
        self.volume = os.getenv("TTS_VOLUME", "+0%")
        logger.info(
            f"EdgeTTS 初始化: voice={self.config.voice_id}, rate={self.rate}, volume={self.volume}"
        )

    def list_voices(self) -> list[TTSVoice]:
        """列出可用音色。"""
        return list(self.AVAILABLE_VOICES)

    def _build_ssml(self, text: str) -> str:
        """构造 SSML，控制语速/音调/音量。"""
        # 简单的 SSML 包装
        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">'
            f'<prosody rate="{self.rate}" volume="{self.volume}">'
            f'{text}'
            f'</prosody>'
            f'</speak>'
        )

    async def synthesize(self, text: str) -> TTSResult:
        """合成完整文本为音频。"""
        try:
            import edge_tts
        except ImportError:
            raise ImportError("edge-tts 未安装。请运行: pip install edge-tts")

        try:
            communicate = edge_tts.Communicate(
                text,
                voice=self.config.voice_id,
                rate=self.rate,
                volume=self.volume,
            )

            # 收集所有音频数据
            audio_buffer = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.extend(chunk["data"])

            duration = len(text) * 0.1 / max(0.5, self.config.speed)

            return TTSResult(
                audio_data=bytes(audio_buffer),
                audio_format=self.config.audio_format,
                sample_rate=self.config.sample_rate,
                duration=duration,
                text=text,
                is_final=True,
            )
        except Exception as e:
            logger.error(f"EdgeTTS 合成失败: {e}")
            return TTSResult(
                audio_data=b"",
                audio_format=self.config.audio_format,
                sample_rate=self.config.sample_rate,
                duration=0.0,
                text=text,
                is_final=True,
            )

    async def stream_synthesize(self, text: str) -> AsyncIterator[TTSResult]:
        """流式合成，按句子分块。"""
        try:
            import edge_tts
        except ImportError:
            raise ImportError("edge-tts 未安装。请运行: pip install edge-tts")

        # 按标点分句
        sentences = re.split(r"([。！？!?\n])", text)
        combined = []
        i = 0
        while i < len(sentences):
            s = sentences[i]
            if i + 1 < len(sentences) and re.match(r"[。！？!?]", sentences[i + 1]):
                s += sentences[i + 1]
                i += 2
            else:
                i += 1
            if s.strip():
                combined.append(s)

        for idx, sentence in enumerate(combined):
            is_final = idx == len(combined) - 1
            try:
                communicate = edge_tts.Communicate(
                    sentence,
                    voice=self.config.voice_id,
                    rate=self.rate,
                    volume=self.volume,
                )
                audio_buffer = bytearray()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_buffer.extend(chunk["data"])

                yield TTSResult(
                    audio_data=bytes(audio_buffer),
                    audio_format=self.config.audio_format,
                    sample_rate=self.config.sample_rate,
                    duration=len(sentence) * 0.1 / max(0.5, self.config.speed),
                    text=sentence,
                    is_final=is_final,
                )
            except Exception as e:
                logger.warning(f"句子 '{sentence[:30]}...' 合成失败: {e}")
                yield TTSResult(
                    audio_data=b"",
                    audio_format=self.config.audio_format,
                    sample_rate=self.config.sample_rate,
                    duration=0.0,
                    text=sentence,
                    is_final=is_final,
                )