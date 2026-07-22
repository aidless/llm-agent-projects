"""ASR 模块测试"""

import asyncio
import struct
import pytest
import math

from asr.base import ASRConfig, ASREngine, ASRResult, AudioFormat
from asr.mock_asr import MockASR
from asr.audio_preprocessor import AudioPreprocessor, AudioInfo, AudioCodec
from asr.vad import VADDetector, VADConfig, VADState, VADSegment


# ---- 辅助函数 ----

def make_wav_data(frequency: float = 440.0, duration: float = 0.2, amplitude: float = 0.8, sample_rate: int = 16000):
    """生成 WAV 格式正弦波音频数据"""
    return AudioPreprocessor.generate_sine_wave(
        frequency=frequency, duration=duration, amplitude=amplitude, sample_rate=sample_rate
    )


def make_pcm_data(frequency: float = 440.0, duration: float = 0.2, amplitude: float = 0.8, sample_rate: int = 16000):
    """生成纯 PCM 16-bit 数据"""
    num_samples = int(sample_rate * duration)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * t))
        value = max(-32768, min(32767, value))
        samples.append(struct.pack('<h', value))
    return b''.join(samples)


async def async_iter(items):
    """将列表转为异步迭代器"""
    for item in items:
        yield item


# ---- MockASR 测试 ----

class TestMockASR:
    @pytest.mark.asyncio
    async def test_recognize_returns_result(self):
        """测试 MockASR 识别返回结果"""
        asr = MockASR(fixed_text="你好世界")
        audio = make_wav_data()
        result = await asr.recognize(audio)
        assert isinstance(result, ASRResult)
        assert "你好世界" in result.text
        assert result.is_final is True
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_recognize_with_empty_audio(self):
        """测试空音频识别"""
        asr = MockASR(fixed_text="测试")
        result = await asr.recognize(b"")
        assert result.text  # 固定文本仍应返回

    @pytest.mark.asyncio
    async def test_stream_recognize(self):
        """测试流式识别"""
        asr = MockASR(fixed_text="流式测试")
        chunks = [make_wav_data(duration=0.05) for _ in range(3)]
        results = []
        async for result in asr.stream_recognize(async_iter(chunks)):
            results.append(result)
        assert len(results) > 0
        # 最后一个应该是最终结果
        assert results[-1].is_final is True
        # 中间结果应有 partial 标记
        partials = [r for r in results[:-1] if r.partial]
        assert len(partials) > 0

    @pytest.mark.asyncio
    async def test_recognize_language_zh(self):
        """测试中文识别"""
        asr = MockASR(ASRConfig(language="zh"), fixed_text="中文测试")
        audio = make_wav_data()
        result = await asr.recognize(audio)
        assert result.language == "zh"

    @pytest.mark.asyncio
    async def test_recognize_language_en(self):
        """测试英文识别"""
        asr = MockASR(ASRConfig(language="en"), fixed_text="hello world")
        audio = make_wav_data()
        result = await asr.recognize(audio)
        assert result.language == "en"

    def test_reset(self):
        """测试重置"""
        asr = MockASR()
        asr._chunk_count = 10
        asr._buffer.append(b"test")
        asr.reset()
        assert asr._chunk_count == 0
        assert len(asr._buffer) == 0


# ---- AudioPreprocessor 测试 ----

class TestAudioPreprocessor:
    def test_generate_sine_wave(self):
        """测试正弦波生成"""
        wav = make_wav_data(frequency=440, duration=0.1)
        assert len(wav) > 44  # WAV header + data
        assert wav[:4] == b'RIFF'
        assert wav[8:12] == b'WAVE'

    def test_detect_wav_info(self):
        """测试 WAV 信息检测"""
        wav = make_wav_data(sample_rate=16000, duration=0.1)
        preprocessor = AudioPreprocessor()
        info = preprocessor.detect_wav_info(wav)
        assert info is not None
        assert info.sample_rate == 16000
        assert info.channels == 1
        assert info.duration > 0

    def test_detect_wav_info_invalid(self):
        """测试无效 WAV 检测"""
        preprocessor = AudioPreprocessor()
        assert preprocessor.detect_wav_info(b"invalid") is None
        assert preprocessor.detect_wav_info(b"") is None
        assert preprocessor.detect_wav_info(b"short") is None

    def test_extract_pcm_from_wav(self):
        """测试 PCM 提取"""
        wav = make_wav_data(duration=0.1)
        preprocessor = AudioPreprocessor()
        pcm = preprocessor.extract_pcm_from_wav(wav)
        assert len(pcm) > 0
        assert len(pcm) < len(wav)  # 去掉了头

    def test_create_wav_header(self):
        """测试 WAV 头创建"""
        preprocessor = AudioPreprocessor()
        header = preprocessor.create_wav_header(16000, 1, 16, 3200)
        assert len(header) == 44
        assert header[:4] == b'RIFF'

    def test_normalize_volume(self):
        """测试音量归一化"""
        preprocessor = AudioPreprocessor()
        # 生成低音量 PCM
        pcm = struct.pack('<hh', 100, -100)
        normalized = preprocessor.normalize_volume(pcm, target_peak=0.8)
        assert len(normalized) == len(pcm)
        values = struct.unpack('<hh', normalized)
        # 归一化后值应该更大
        assert abs(values[0]) >= 100

    @pytest.mark.asyncio
    async def test_preprocess_wav(self):
        """测试 WAV 预处理"""
        wav = make_wav_data(duration=0.1)
        preprocessor = AudioPreprocessor()
        pcm, info = await preprocessor.preprocess(wav, "wav")
        assert len(pcm) > 0
        assert info.sample_rate == 16000

    def test_supported_formats(self):
        """测试 ASR 支持的格式"""
        asr = MockASR()
        formats = asr.supported_formats()
        assert AudioFormat.WAV in formats
        assert AudioFormat.MP3 in formats


# ---- VAD 测试 ----

class TestVAD:
    def test_vad_detects_silence(self):
        """测试 VAD 检测静音"""
        vad = VADDetector(VADConfig(
            energy_threshold=500.0,
            speech_start_frames=3,
            silence_end_frames=15,
        ))
        # 生成静音 PCM（全零）
        silence = b'\x00\x00' * 1600  # 100ms of silence at 16kHz 16-bit
        segments = vad.process(silence)
        assert len(segments) == 0  # 静音不应产生语音段

    def test_vad_detects_speech(self):
        """测试 VAD 检测语音信号"""
        vad = VADDetector(VADConfig(
            energy_threshold=100.0,  # 低阈值
            speech_start_frames=2,
            silence_end_frames=30,   # 高阈值避免过早结束
        ))
        # 生成有信号的音频
        pcm = make_pcm_data(frequency=440, duration=0.5, amplitude=0.9, sample_rate=16000)
        segments = vad.process(pcm)
        # 正弦波应被检测为语音
        assert len(segments) > 0
        assert segments[0].end_ms > segments[0].start_ms

    def test_vad_state_transitions(self):
        """测试 VAD 状态转换"""
        vad = VADDetector(VADConfig(
            energy_threshold=100.0,
            speech_start_frames=2,
            silence_end_frames=5,
        ))
        assert vad.get_state() == VADState.SILENCE
        assert vad.is_speaking() is False

    def test_vad_reset(self):
        """测试 VAD 重置"""
        vad = VADDetector()
        vad._speech_frame_count = 10
        vad.reset()
        assert vad.get_state() == VADState.SILENCE
        assert vad._speech_frame_count == 0
        assert len(vad.get_segments()) == 0

    def test_vad_segments_have_energy(self):
        """测试语音段有平均能量"""
        vad = VADDetector(VADConfig(
            energy_threshold=100.0,
            speech_start_frames=2,
            silence_end_frames=30,
        ))
        pcm = make_pcm_data(frequency=440, duration=0.3, amplitude=0.8, sample_rate=16000)
        segments = vad.process(pcm)
        if segments:
            assert segments[0].avg_energy > 0