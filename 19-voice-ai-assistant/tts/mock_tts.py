"""Mock TTS 引擎 - 用于测试"""

import asyncio
import math
import struct
from typing import AsyncIterator, Optional

from tts.base import TTSAudioFormat, TTSConfig, TTSEngine, TTSResult, TTSVoice


class MockTTS(TTSEngine):
    """Mock TTS 引擎，生成简单的正弦波音频"""

    AVAILABLE_VOICES = [
        TTSVoice(voice_id="default", name="默认女声", language="zh", gender="female"),
        TTSVoice(voice_id="male", name="默认男声", language="zh", gender="male"),
        TTSVoice(voice_id="en_female", name="English Female", language="en", gender="female"),
        TTSVoice(voice_id="en_male", name="English Male", language="en", gender="male"),
    ]

    def __init__(self, config: Optional[TTSConfig] = None, fixed_audio: Optional[bytes] = None):
        """初始化 Mock TTS

        Args:
            config: TTS 配置
            fixed_audio: 固定返回的音频数据（用于确定性测试）
        """
        super().__init__(config)
        self.fixed_audio = fixed_audio

    def _generate_sine_audio(self, text: str, duration: Optional[float] = None) -> bytes:
        """生成正弦波音频数据

        Args:
            text: 输入文本（用于决定音频时长）
            duration: 固定时长（秒），不指定则根据文本长度计算

        Returns:
            bytes: WAV 格式音频数据
        """
        if self.fixed_audio:
            return self.fixed_audio

        if duration is None:
            # 每个字符约 0.1 秒
            duration = max(0.1, len(text) * 0.1 / self.config.speed)

        sample_rate = self.config.sample_rate
        num_samples = int(sample_rate * duration)
        frequency = 440.0 * self.config.pitch  # 基频 * 音调
        amplitude = self.config.volume

        samples = []
        for i in range(num_samples):
            t = i / sample_rate
            # 生成有包络的正弦波（模拟语音的起伏）
            envelope = 1.0
            # 淡入淡出
            fade_samples = int(0.01 * sample_rate)
            if i < fade_samples:
                envelope = i / fade_samples
            elif i > num_samples - fade_samples:
                envelope = (num_samples - i) / fade_samples

            value = int(amplitude * envelope * 32767 * math.sin(2 * math.pi * frequency * t))
            value = max(-32768, min(32767, value))
            samples.append(struct.pack('<h', value))

        pcm_data = b''.join(samples)

        # 创建 WAV 头
        byte_rate = sample_rate * 1 * 2  # 16-bit mono
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + len(pcm_data),
            b'WAVE',
            b'fmt ',
            16,
            1,  # PCM
            1,  # mono
            sample_rate,
            byte_rate,
            2,  # block align
            16, # bits per sample
            b'data',
            len(pcm_data),
        )

        return header + pcm_data

    async def synthesize(self, text: str) -> TTSResult:
        """合成完整文本为音频"""
        await asyncio.sleep(0.01)  # 模拟处理延迟

        audio_data = self._generate_sine_audio(text)
        duration = len(text) * 0.1 / self.config.speed

        return TTSResult(
            audio_data=audio_data,
            audio_format=self.config.audio_format,
            sample_rate=self.config.sample_rate,
            duration=duration,
            text=text,
            is_final=True,
        )

    async def stream_synthesize(self, text: str) -> AsyncIterator[TTSResult]:
        """流式合成，按句子分块"""
        # 按标点分句
        import re
        sentences = re.split(r'([。！？!?\n])', text)
        # 重新组合（把标点附到句子后面）
        combined = []
        i = 0
        while i < len(sentences):
            s = sentences[i]
            if i + 1 < len(sentences) and re.match(r'[。！？!?]', sentences[i + 1]):
                s += sentences[i + 1]
                i += 2
            else:
                i += 1
            if s.strip():
                combined.append(s)

        for idx, sentence in enumerate(combined):
            await asyncio.sleep(0.005)  # 模拟流式延迟
            is_final = (idx == len(combined) - 1)
            audio_data = self._generate_sine_audio(sentence)
            duration = len(sentence) * 0.1 / self.config.speed

            yield TTSResult(
                audio_data=audio_data,
                audio_format=self.config.audio_format,
                sample_rate=self.config.sample_rate,
                duration=duration,
                text=sentence,
                is_final=is_final,
            )

    def list_voices(self) -> list[TTSVoice]:
        """列出可用音色"""
        return list(self.AVAILABLE_VOICES)