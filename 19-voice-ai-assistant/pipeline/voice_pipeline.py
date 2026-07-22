"""语音处理流水线 - ASR -> LLM -> TTS 全链路异步处理"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from asr.base import ASREngine, ASRResult
from asr.audio_preprocessor import AudioPreprocessor, AudioInfo
from asr.vad import VADDetector, VADConfig, VADState
from tts.base import TTSEngine, TTSResult
from tts.ssml_parser import SSMLParser
from dialog.manager import DialogManager
from dialog.barge_in import BargeInHandler, BargeInState
from pipeline.stream_manager import StreamManager, StreamEvent, StreamType

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Pipeline 配置"""
    # VAD 配置
    vad_enabled: bool = True
    vad_energy_threshold: float = 500.0
    vad_silence_end_frames: int = 15
    # 超时
    asr_timeout: float = 30.0
    llm_timeout: float = 30.0
    tts_timeout: float = 30.0
    # 流式
    enable_streaming: bool = True
    tts_sentence_buffer_size: int = 3  # 累积多少句子后再触发 TTS


@dataclass
class PipelineResult:
    """Pipeline 处理结果"""
    session_id: str
    asr_text: str = ""
    llm_response: str = ""
    tts_audio: bytes = b""
    intent: str = ""
    error: Optional[str] = None


class VoicePipeline:
    """语音处理流水线

    实现 音频 -> ASR -> 文本 -> LLM -> 回复文本 -> TTS -> 音频 的完整链路。
    支持异步并行：TTS 可以在 LLM 生成时就开始。
    """

    def __init__(
        self,
        asr_engine: ASREngine,
        tts_engine: TTSEngine,
        dialog_manager: DialogManager,
        config: Optional[PipelineConfig] = None,
    ):
        self.asr = asr_engine
        self.tts = tts_engine
        self.dialog = dialog_manager
        self.config = config or PipelineConfig()
        self.stream_manager = StreamManager()
        self.preprocessor = AudioPreprocessor()
        self.ssml_parser = SSMLParser()

        self._vad = VADDetector(VADConfig(
            energy_threshold=self.config.vad_energy_threshold,
            silence_end_frames=self.config.vad_silence_end_frames,
        )) if self.config.vad_enabled else None

        self._audio_buffers: dict[str, deque[bytes]] = {}
        self._processing: dict[str, bool] = {}

    async def process_audio(
        self,
        session_id: str,
        audio_data: bytes,
        audio_format: str = "wav",
    ) -> AsyncIterator[StreamEvent]:
        """处理音频输入（完整音频）

        Args:
            session_id: 会话 ID
            audio_data: 音频数据
            audio_format: 音频格式

        Yields:
            StreamEvent: 各环节的事件
        """
        try:
            # 1. 音频预处理
            pcm_data, audio_info = await self.preprocessor.preprocess(audio_data, audio_format)

            # 2. VAD 检测
            if self._vad:
                segments = self._vad.process(pcm_data)
                # 如果没有检测到语音，直接返回
                if not segments:
                    yield StreamEvent(
                        type=StreamType.ASR_FINAL,
                        data=ASRResult(text="", is_final=True, confidence=0.0),
                        session_id=session_id,
                    )
                    return

            # 3. ASR 识别
            yield StreamEvent(
                type=StreamType.ASR_PARTIAL,
                data={"status": "processing"},
                session_id=session_id,
            )

            asr_result = await asyncio.wait_for(
                self.asr.recognize(audio_data),
                timeout=self.config.asr_timeout,
            )

            yield StreamEvent(
                type=StreamType.ASR_FINAL,
                data=asr_result,
                session_id=session_id,
            )

            # 4. LLM 对话
            async for event in self._process_text_stream(session_id, asr_result.text):
                yield event

        except asyncio.TimeoutError:
            yield StreamEvent(
                type=StreamType.ERROR,
                data={"error": "processing timeout", "stage": "asr"},
                session_id=session_id,
            )
        except Exception as e:
            yield StreamEvent(
                type=StreamType.ERROR,
                data={"error": str(e)},
                session_id=session_id,
            )

    async def process_audio_chunk(self, session_id: str, chunk: bytes):
        """处理音频块（流式输入）

        Args:
            session_id: 会话 ID
            chunk: 音频数据块
        """
        if session_id not in self._audio_buffers:
            self._audio_buffers[session_id] = deque()

        self._audio_buffers[session_id].append(chunk)

        # VAD 逐帧检测
        if self._vad:
            self._vad.process_frame(chunk)

    async def finalize_audio_stream(self, session_id: str) -> AsyncIterator[StreamEvent]:
        """结束音频流输入，触发完整处理

        Args:
            session_id: 会话 ID

        Yields:
            StreamEvent: 各环节的事件
        """
        # 合并所有音频块
        chunks = self._audio_buffers.pop(session_id, deque())
        if not chunks:
            return

        full_audio = b''.join(chunks)
        async for event in self.process_audio(session_id, full_audio):
            yield event

    async def _process_text_stream(
        self,
        session_id: str,
        text: str,
    ) -> AsyncIterator[StreamEvent]:
        """处理文本流 (LLM + TTS)

        ASR 完成后调用，执行 LLM 生成和 TTS 合成。
        LLM 流式生成的同时，积累足够的文本就启动 TTS。
        """
        if not text.strip():
            return

        full_response = ""
        sentence_buffer = []
        tts_tasks: list[asyncio.Task] = []

        # LLM 流式生成
        try:
            async for chunk_result in self.dialog.process_text_stream(
                session_id, text
            ):
                if chunk_result["type"] == "stream_chunk":
                    chunk_text = chunk_result["text"]
                    full_response += chunk_text

                    yield StreamEvent(
                        type=StreamType.LLM_CHUNK,
                        data={"text": chunk_text, "full_text": full_response},
                        session_id=session_id,
                    )

                    # 检查是否积累了足够文本启动 TTS
                    import re
                    sentences = re.split(r'(?<=[。！？!?])', chunk_text)
                    for s in sentences:
                        if s.strip():
                            sentence_buffer.append(s.strip())

                    if len(sentence_buffer) >= self.config.tts_sentence_buffer_size:
                        tts_text = ''.join(sentence_buffer)
                        sentence_buffer.clear()
                        # 异步启动 TTS
                        task = asyncio.create_task(
                            self._tts_and_emit(session_id, tts_text)
                        )
                        tts_tasks.append(task)

                elif chunk_result["type"] == "stream_end":
                    full_response = chunk_result["text"]

                    yield StreamEvent(
                        type=StreamType.LLM_COMPLETE,
                        data={"text": full_response},
                        session_id=session_id,
                    )

        except Exception as e:
            yield StreamEvent(
                type=StreamType.ERROR,
                data={"error": str(e), "stage": "llm"},
                session_id=session_id,
            )
            return

        # 处理剩余的句子
        if sentence_buffer:
            tts_text = ''.join(sentence_buffer)
            sentence_buffer.clear()
            task = asyncio.create_task(
                self._tts_and_emit(session_id, tts_text)
            )
            tts_tasks.append(task)

        # 等待所有 TTS 任务完成
        for task in tts_tasks:
            try:
                result = await task
                if result:
                    yield result
            except Exception as e:
                yield StreamEvent(
                    type=StreamType.ERROR,
                    data={"error": str(e), "stage": "tts"},
                    session_id=session_id,
                )

        yield StreamEvent(
            type=StreamType.TTS_COMPLETE,
            data={"text": full_response},
            session_id=session_id,
        )

    async def _tts_and_emit(self, session_id: str, text: str) -> Optional[StreamEvent]:
        """TTS 合成并返回事件"""
        try:
            # 处理 SSML
            if self.tts.config.enable_ssml and self.ssml_parser.is_ssml(text):
                parse_result = self.ssml_parser.parse(text)
                if parse_result.segments:
                    # 使用第一个段落的文本
                    text = parse_result.segments[0].text

            result = await asyncio.wait_for(
                self.tts.synthesize(text),
                timeout=self.config.tts_timeout,
            )

            return StreamEvent(
                type=StreamType.TTS_AUDIO_CHUNK,
                data=result,
                session_id=session_id,
            )
        except Exception as e:
            return StreamEvent(
                type=StreamType.ERROR,
                data={"error": str(e), "stage": "tts"},
                session_id=session_id,
            )

    def cancel_session(self, session_id: str):
        """取消会话处理"""
        self.stream_manager.close_session(session_id)
        self._audio_buffers.pop(session_id, None)
        self._processing.pop(session_id, None)
        if self._vad:
            self._vad.reset()