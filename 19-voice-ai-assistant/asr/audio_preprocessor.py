"""音频预处理器 - 音频格式转换和基本处理"""

import asyncio
import io
import struct
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class AudioCodec(str, Enum):
    """音频编码格式"""
    PCM_S16LE = "pcm_s16le"
    PCM_U8 = "pcm_u8"
    PCM_F32LE = "pcm_f32le"


@dataclass
class AudioInfo:
    """音频信息"""
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # bytes per sample (2 = 16-bit)
    codec: AudioCodec = AudioCodec.PCM_S16LE
    duration: float = 0.0  # seconds


class AudioPreprocessor:
    """音频预处理器

    支持音频格式检测、重采样（简单实现）、音量归一化等。
    真实环境可用 pydub / soundfile 替代。
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_channels: int = 1,
        target_sample_width: int = 2,
    ):
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.target_sample_width = target_sample_width

    def detect_wav_info(self, audio_data: bytes) -> Optional[AudioInfo]:
        """检测 WAV 文件的基本信息

        Args:
            audio_data: WAV 格式的音频数据

        Returns:
            AudioInfo 或 None（如果不是有效的 WAV）
        """
        if len(audio_data) < 44:
            return None

        # 检查 RIFF 头
        if audio_data[:4] != b'RIFF' or audio_data[8:12] != b'WAVE':
            return None

        try:
            fmt_chunk_pos = audio_data.find(b'fmt ')
            if fmt_chunk_pos < 0:
                return None

            audio_format = struct.unpack_from('<H', audio_data, fmt_chunk_pos + 8)[0]
            channels = struct.unpack_from('<H', audio_data, fmt_chunk_pos + 10)[0]
            sample_rate = struct.unpack_from('<I', audio_data, fmt_chunk_pos + 12)[0]
            bits_per_sample = struct.unpack_from('<H', audio_data, fmt_chunk_pos + 22)[0]

            # 计算时长
            data_chunk_pos = audio_data.find(b'data')
            if data_chunk_pos < 0:
                return None
            data_size = struct.unpack_from('<I', audio_data, data_chunk_pos + 4)[0]
            byte_rate = sample_rate * channels * (bits_per_sample // 8)
            duration = data_size / byte_rate if byte_rate > 0 else 0.0

            codec_map = {1: AudioCodec.PCM_S16LE, 3: AudioCodec.PCM_F32LE}
            codec = codec_map.get(audio_format, AudioCodec.PCM_S16LE)

            return AudioInfo(
                sample_rate=sample_rate,
                channels=channels,
                sample_width=bits_per_sample // 8,
                codec=codec,
                duration=duration,
            )
        except (struct.error, IndexError):
            return None

    def extract_pcm_from_wav(self, audio_data: bytes) -> bytes:
        """从 WAV 数据中提取纯 PCM 数据

        Args:
            audio_data: WAV 格式音频数据

        Returns:
            bytes: 纯 PCM 数据
        """
        data_chunk_pos = audio_data.find(b'data')
        if data_chunk_pos < 0:
            return audio_data  # 可能已经是纯 PCM

        data_size = struct.unpack_from('<I', audio_data, data_chunk_pos + 4)[0]
        pcm_start = data_chunk_pos + 8
        return audio_data[pcm_start:pcm_start + data_size]

    def create_wav_header(
        self,
        sample_rate: int,
        channels: int,
        bits_per_sample: int,
        data_size: int,
    ) -> bytes:
        """创建 WAV 文件头

        Args:
            sample_rate: 采样率
            channels: 声道数
            bits_per_sample: 每样本位数
            data_size: PCM 数据大小

        Returns:
            bytes: 44 字节 WAV 文件头
        """
        byte_rate = sample_rate * channels * (bits_per_sample // 8)
        block_align = channels * (bits_per_sample // 8)

        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_size,
            b'WAVE',
            b'fmt ',
            16,  # fmt chunk size
            1,   # PCM format
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b'data',
            data_size,
        )
        return header

    def normalize_volume(self, pcm_data: bytes, target_peak: float = 0.8) -> bytes:
        """音量归一化

        Args:
            pcm_data: 16-bit PCM 数据
            target_peak: 目标峰值 (0.0 - 1.0)

        Returns:
            bytes: 归一化后的 PCM 数据
        """
        if len(pcm_data) < 2:
            return pcm_data

        import array
        samples = array.array('h')
        samples.frombytes(pcm_data[:len(pcm_data) - (len(pcm_data) % 2)])

        if not samples:
            return pcm_data

        max_val = max(abs(s) for s in samples)
        if max_val == 0:
            return pcm_data

        target_int = int(32767 * target_peak)
        scale = target_int / max_val

        normalized = array.array('h', (int(s * scale) for s in samples))
        # 裁剪到 16-bit 范围
        for i in range(len(normalized)):
            normalized[i] = max(-32768, min(32767, normalized[i]))

        return normalized.tobytes()

    async def preprocess(self, audio_data: bytes, source_format: str = "wav") -> tuple[bytes, AudioInfo]:
        """音频预处理主入口

        Args:
            audio_data: 原始音频数据
            source_format: 源格式

        Returns:
            tuple: (处理后的 PCM 数据, 音频信息)
        """
        if source_format == "wav":
            info = self.detect_wav_info(audio_data)
            if info is None:
                info = AudioInfo()
            pcm = self.extract_pcm_from_wav(audio_data)
        else:
            # 非 WAV 格式，假设已提供 PCM 数据
            info = AudioInfo(
                sample_rate=self.target_sample_rate,
                channels=self.target_channels,
                sample_width=self.target_sample_width,
            )
            pcm = audio_data

        return pcm, info

    @staticmethod
    def generate_sine_wave(
        frequency: float = 440.0,
        duration: float = 0.1,
        sample_rate: int = 16000,
        amplitude: float = 0.5,
    ) -> bytes:
        """生成正弦波 WAV 数据（用于测试）

        Args:
            frequency: 频率 (Hz)
            duration: 时长 (秒)
            sample_rate: 采样率
            amplitude: 振幅 (0.0 - 1.0)

        Returns:
            bytes: WAV 格式音频数据
        """
        import math
        num_samples = int(sample_rate * duration)
        samples = []
        for i in range(num_samples):
            t = i / sample_rate
            value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * t))
            value = max(-32768, min(32767, value))
            samples.append(struct.pack('<h', value))

        pcm_data = b''.join(samples)

        preprocessor = AudioPreprocessor()
        header = preprocessor.create_wav_header(
            sample_rate=sample_rate,
            channels=1,
            bits_per_sample=16,
            data_size=len(pcm_data),
        )
        return header + pcm_data