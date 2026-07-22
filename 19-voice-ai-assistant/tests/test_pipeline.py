"""Pipeline 模块测试"""

import asyncio
import pytest

from asr.base import ASRConfig, ASRResult
from asr.mock_asr import MockASR
from asr.audio_preprocessor import AudioPreprocessor
from tts.base import TTSConfig
from tts.mock_tts import MockTTS
from dialog.manager import DialogManager
from pipeline.voice_pipeline import VoicePipeline, PipelineConfig
from pipeline.stream_manager import StreamManager, StreamEvent, StreamType


def make_wav_data(frequency=440.0, duration=0.2):
    return AudioPreprocessor.generate_sine_wave(frequency=frequency, duration=duration)


class TestStreamManager:
    def test_push_and_get(self):
        """测试推送和获取事件"""
        mgr = StreamManager()
        event = StreamEvent(type=StreamType.ASR_FINAL, session_id="test")
        mgr.push(event)
        pending = mgr.get_pending_events("test")
        assert len(pending) == 1
        assert pending[0].type == StreamType.ASR_FINAL

    def test_multiple_events(self):
        """测试多个事件"""
        mgr = StreamManager()
        for i in range(5):
            mgr.push(StreamEvent(
                type=StreamType.LLM_CHUNK,
                session_id="test",
                data={"chunk": i},
            ))
        events = mgr.get_pending_events("test")
        assert len(events) == 5

    def test_close_session(self):
        """测试关闭会话"""
        mgr = StreamManager()
        mgr.push(StreamEvent(type=StreamType.ASR_FINAL, session_id="test"))
        mgr.close_session("test")
        assert mgr.is_session_closed("test") is True
        # 关闭后不应再能推送
        mgr.push(StreamEvent(type=StreamType.ASR_FINAL, session_id="test"))
        events = mgr.get_pending_events("test")
        assert len(events) == 0

    def test_buffer_limit(self):
        """测试缓冲区限制"""
        mgr = StreamManager(buffer_size=3)
        for i in range(5):
            mgr.push(StreamEvent(type=StreamType.LLM_CHUNK, session_id="test"))
        events = mgr.get_pending_events("test")
        # 应只保留最后 3 个
        assert len(events) == 3

    def test_reset(self):
        """测试重置"""
        mgr = StreamManager()
        mgr.push(StreamEvent(type=StreamType.ASR_FINAL, session_id="s1"))
        mgr.close_session("s1")
        mgr.reset()
        assert mgr.is_session_closed("s1") is False

    @pytest.mark.asyncio
    async def test_wait_for_event_timeout(self):
        """测试等待事件超时"""
        mgr = StreamManager()
        event = await mgr.wait_for_event("nonexistent", timeout=0.1)
        assert event is None

    @pytest.mark.asyncio
    async def test_wait_for_event_immediate(self):
        """测试等待已有事件立即返回"""
        mgr = StreamManager()
        mgr.push(StreamEvent(type=StreamType.ASR_FINAL, session_id="test"))
        event = await mgr.wait_for_event("test", timeout=1.0)
        assert event is not None
        assert event.type == StreamType.ASR_FINAL


class TestVoicePipeline:
    @pytest.mark.asyncio
    async def test_process_audio(self):
        """测试完整音频处理流程"""
        asr = MockASR(ASRConfig(language="zh"), fixed_text="你好")
        tts = MockTTS(TTSConfig())
        dialog = DialogManager()
        pipeline = VoicePipeline(asr, tts, dialog)

        audio = make_wav_data(duration=0.2)
        events = []
        async for event in pipeline.process_audio("test_session", audio):
            events.append(event)

        # 应有 ASR 和 LLM/TTS 事件
        types = [e.type for e in events]
        assert StreamType.ASR_PARTIAL in types
        assert StreamType.ASR_FINAL in types

    @pytest.mark.asyncio
    async def test_process_text_stream(self):
        """测试文本流处理"""
        asr = MockASR(ASRConfig(language="zh"), fixed_text="测试")
        tts = MockTTS(TTSConfig())
        dialog = DialogManager()
        pipeline = VoicePipeline(
            asr, tts, dialog,
            PipelineConfig(tts_sentence_buffer_size=1),
        )

        events = []
        async for event in pipeline._process_text_stream("test", "你好"):
            events.append(event)

        types = [e.type for e in events]
        assert StreamType.LLM_CHUNK in types
        assert StreamType.LLM_COMPLETE in types

    @pytest.mark.asyncio
    async def test_audio_chunk_and_finalize(self):
        """测试音频块处理和结束"""
        asr = MockASR(ASRConfig(language="zh"), fixed_text="测试")
        tts = MockTTS(TTSConfig())
        dialog = DialogManager()
        pipeline = VoicePipeline(asr, tts, dialog)

        # 发送音频块
        await pipeline.process_audio_chunk("test2", make_wav_data(duration=0.05))
        await pipeline.process_audio_chunk("test2", make_wav_data(duration=0.05))

        # 结束流
        events = []
        async for event in pipeline.finalize_audio_stream("test2"):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_cancel_session(self):
        """测试取消会话"""
        asr = MockASR(ASRConfig(language="zh"), fixed_text="测试")
        tts = MockTTS(TTSConfig())
        dialog = DialogManager()
        pipeline = VoicePipeline(asr, tts, dialog)

        await pipeline.process_audio_chunk("cancel_test", make_wav_data(duration=0.05))
        pipeline.cancel_session("cancel_test")

        # stream_manager 会话应关闭
        assert pipeline.stream_manager.is_session_closed("cancel_test")