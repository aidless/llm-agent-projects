"""TTS 模块测试"""

import asyncio
import pytest

from tts.base import TTSConfig, TTSEngine, TTSResult, TTSAudioFormat, TTSVoice
from tts.mock_tts import MockTTS
from tts.ssml_parser import SSMLParser, SSMLSegment, SSMLParseResult


# ---- MockTTS 测试 ----

class TestMockTTS:
    @pytest.mark.asyncio
    async def test_synthesize(self):
        """测试文本合成"""
        tts = MockTTS()
        result = await tts.synthesize("你好世界")
        assert isinstance(result, TTSResult)
        assert result.audio_data  # 有音频数据
        assert result.audio_format == TTSAudioFormat.WAV
        assert result.is_final is True
        assert len(result.audio_data) > 44  # WAV header + data

    @pytest.mark.asyncio
    async def test_synthesize_duration(self):
        """测试合成时长与文本长度相关"""
        tts = MockTTS()
        short_result = await tts.synthesize("短")
        long_result = await tts.synthesize("这是一段很长的测试文本")
        # 长文本的音频应该更大
        assert len(long_result.audio_data) > len(short_result.audio_data)

    @pytest.mark.asyncio
    async def test_stream_synthesize(self):
        """测试流式合成"""
        tts = MockTTS()
        text = "你好。世界。"
        results = []
        async for result in tts.stream_synthesize(text):
            results.append(result)

        assert len(results) > 0
        # 最后一个应该是最终块
        assert results[-1].is_final is True
        # 所有块都有音频数据
        for r in results:
            assert r.audio_data

    @pytest.mark.asyncio
    async def test_stream_synthesize_single_sentence(self):
        """测试单句流式合成"""
        tts = MockTTS()
        results = []
        async for result in tts.stream_synthesize("你好"):
            results.append(result)
        assert len(results) >= 1
        assert results[0].text == "你好"

    def test_list_voices(self):
        """测试列出音色"""
        tts = MockTTS()
        voices = tts.list_voices()
        assert len(voices) >= 2  # 至少有 default 和 male
        assert any(v.voice_id == "default" for v in voices)

    def test_set_speed(self):
        """测试设置语速"""
        tts = MockTTS()
        tts.set_speed(1.5)
        assert tts.config.speed == 1.5
        tts.set_speed(3.0)  # 超出范围应被裁剪
        assert tts.config.speed == 2.0

    def test_set_pitch(self):
        """测试设置音调"""
        tts = MockTTS()
        tts.set_pitch(0.8)
        assert tts.config.pitch == 0.8

    def test_set_volume(self):
        """测试设置音量"""
        tts = MockTTS()
        tts.set_volume(0.5)
        assert tts.config.volume == 0.5
        tts.set_volume(1.5)
        assert tts.config.volume == 1.0

    @pytest.mark.asyncio
    async def test_fixed_audio(self):
        """测试固定音频返回"""
        fixed = b'\x00' * 100
        tts = MockTTS(fixed_audio=fixed)
        result = await tts.synthesize("任何文本")
        assert result.audio_data == fixed


# ---- SSMLParser 测试 ----

class TestSSMLParser:
    def test_plain_text(self):
        """测试纯文本（非 SSML）"""
        parser = SSMLParser()
        assert parser.is_ssml("你好世界") is False
        result = parser.parse("你好世界")
        assert result.has_ssml is False
        assert len(result.segments) == 1
        assert result.segments[0].text == "你好世界"

    def test_ssml_speak(self):
        """测试简单 SSML"""
        parser = SSMLParser()
        ssml = "<speak>你好世界</speak>"
        assert parser.is_ssml(ssml) is True
        result = parser.parse(ssml)
        assert result.has_ssml is True

    def test_ssml_with_break(self):
        """测试 SSML break 标签"""
        parser = SSMLParser()
        ssml = "<speak>你好<break time=\"500ms\"/>世界</speak>"
        result = parser.parse(ssml)
        assert result.has_ssml is True

    def test_ssml_invalid_xml(self):
        """测试无效 XML 退回纯文本"""
        parser = SSMLParser()
        ssml = "<speak>未关闭标签"
        result = parser.parse(ssml)
        # 解析失败，退回纯文本
        assert len(result.segments) >= 1

    def test_ssml_with_prosody(self):
        """测试 SSML prosody 标签"""
        parser = SSMLParser()
        ssml = "<speak><prosody rate=\"fast\" volume=\"loud\">快速说话</prosody></speak>"
        result = parser.parse(ssml)
        assert result.has_ssml is True

    def test_rate_parsing(self):
        """测试 rate 解析"""
        parser = SSMLParser()
        assert parser._parse_rate("fast", 1.0) == 1.25
        assert parser._parse_rate("slow", 1.0) == 0.75
        assert parser._parse_rate("x-fast", 1.0) == 1.5
        assert parser._parse_rate("normal", 1.0) == 1.0
        assert parser._parse_rate("", 1.0) == 1.0

    def test_volume_parsing(self):
        """测试 volume 解析"""
        parser = SSMLParser()
        assert parser._parse_volume("loud", 0.8) == 1.0
        assert parser._parse_volume("soft", 0.8) == 0.5
        assert parser._parse_volume("silent", 0.8) == 0.0
        assert parser._parse_volume("", 0.8) == 0.8

    def test_break_parsing(self):
        """测试 break 解析"""
        import xml.etree.ElementTree as ET
        parser = SSMLParser()
        # _parse_break 接收 ET.Element
        elem = ET.fromstring('<break time="500ms"/>')
        assert parser._parse_break(elem) == 500
        elem2 = ET.fromstring('<break time="0.5s"/>')
        assert parser._parse_break(elem2) == 500
        elem3 = ET.fromstring('<break time="100"/>')
        assert parser._parse_break(elem3) == 100