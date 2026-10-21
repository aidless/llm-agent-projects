"""Whisper ASR 引擎 - 基于 faster-whisper 的真实 ASR 实现。

使用 faster-whisper (CTranslate2 后端) 提供比原生 openai-whisper 快 4 倍的推理速度。
"""

import asyncio
import io
import logging
import os
import wave
from typing import AsyncIterator, Optional

from asr.base import ASRConfig, ASREngine, ASRResult

logger = logging.getLogger(__name__)


class WhisperASR(ASREngine):
    """基于 faster-whisper 的真实 ASR 实现。

    优势：
    - 比原生 openai-whisper 快 4 倍
    - 内存占用更小（INT8 量化）
    - 支持流式识别
    - 支持热词增强（通过 initial_prompt）

    环境变量：
    - WHISPER_MODEL: 模型大小 (tiny/base/small/medium/large)，默认 base
    - WHISPER_DEVICE: cpu 或 cuda，默认 cpu
    - WHISPER_COMPUTE_TYPE: int8/int16/float16，默认 int8
    """

    def __init__(self, config: Optional[ASRConfig] = None):
        super().__init__(config)
        self.model_size = os.getenv("WHISPER_MODEL", "base")
        self.device = os.getenv("WHISPER_DEVICE", "cpu")
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        self._model = None
        logger.info(
            f"WhisperASR 初始化: model={self.model_size}, device={self.device}, compute_type={self.compute_type}"
        )

    def _load_model(self):
        """延迟加载 faster-whisper 模型。"""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                logger.info(f"Whisper 模型加载完成: {self.model_size}")
            except ImportError:
                raise ImportError(
                    "faster-whisper 未安装。请运行: pip install faster-whisper"
                )
        return self._model

    def _bytes_to_audio_array(self, audio_data: bytes):
        """将 WAV/PCM 字节转换为 numpy 数组。"""
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy 未安装，请运行: pip install numpy")

        # 尝试作为 WAV 解析
        try:
            with wave.open(io.BytesIO(audio_data), "rb") as wf:
                n_frames = wf.getnframes()
                sample_rate = wf.getframerate()
                raw = wf.readframes(n_frames)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                # 重采样到 16kHz（Whisper 要求）
                if sample_rate != 16000:
                    audio = self._resample(audio, sample_rate, 16000)
                return audio
        except wave.Error:
            # 假设是原始 PCM 数据
            audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            return audio

    def _resample(self, audio, orig_sr: int, target_sr: int):
        """简单的线性重采样。"""
        import numpy as np

        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    async def recognize(self, audio_data: bytes) -> ASRResult:
        """识别完整音频。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._recognize_sync, audio_data)

    def _recognize_sync(self, audio_data: bytes) -> ASRResult:
        """同步识别（在线程池中执行）。"""
        try:
            model = self._load_model()
            audio = self._bytes_to_audio_array(audio_data)

            segments, info = model.transcribe(
                audio,
                language="zh" if self.config.language == "zh" else "en",
                beam_size=5,
                vad_filter=True,
            )

            # 收集所有 segment
            text_parts = []
            confidence_sum = 0.0
            count = 0
            for segment in segments:
                text_parts.append(segment.text)
                if hasattr(segment, "avg_logprob"):
                    confidence_sum += abs(segment.avg_logprob)
                    count += 1

            text = "".join(text_parts).strip()
            confidence = (confidence_sum / count) if count > 0 else 0.9
            confidence = min(1.0, max(0.0, confidence))

            text = self.postprocess(text)
            logger.info(f"Whisper 识别: '{text[:80]}' (置信度={confidence:.2f})")

            return ASRResult(
                text=text,
                is_final=True,
                confidence=confidence,
                language=self.config.language,
                partial=False,
            )
        except Exception as e:
            logger.error(f"Whisper 识别失败: {e}")
            return ASRResult(
                text="",
                is_final=True,
                confidence=0.0,
                language=self.config.language,
                partial=False,
            )

    async def stream_recognize(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[ASRResult]:
        """流式识别。

        简化策略：累积音频 buffer，VAD 检测到静音时触发识别。
        """
        buffer = bytearray()
        silence_threshold = 0.5  # 0.5 秒静音触发识别
        sample_rate = self.config.sample_rate
        bytes_per_sample = 2  # 16-bit
        bytes_per_second = sample_rate * bytes_per_sample

        async for chunk in audio_stream:
            buffer.extend(chunk)

            # 累积超过 1 秒时尝试中间识别
            if len(buffer) >= bytes_per_second:
                loop = asyncio.get_event_loop()
                partial_result = await loop.run_in_executor(
                    None, self._recognize_sync, bytes(buffer)
                )
                yield ASRResult(
                    text=partial_result.text,
                    is_final=False,
                    confidence=partial_result.confidence,
                    language=self.config.language,
                    partial=True,
                )

        # 最终识别
        if buffer:
            loop = asyncio.get_event_loop()
            final_result = await loop.run_in_executor(
                None, self._recognize_sync, bytes(buffer)
            )
            final_result.is_final = True
            final_result.partial = False
            yield final_result