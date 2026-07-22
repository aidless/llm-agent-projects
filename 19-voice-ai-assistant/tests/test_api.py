"""API 测试"""

import asyncio
import io
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app
from app.models import ChatRequest, TTSRequest, ASRRequest
from asr.base import ASRConfig
from asr.mock_asr import MockASR
from asr.audio_preprocessor import AudioPreprocessor
from tts.base import TTSConfig
from tts.mock_tts import MockTTS
from dialog.manager import DialogManager
from app.api import asr as asr_api
from app.api import tts as tts_api
from app.api import chat as chat_api


@pytest.fixture(autouse=True)
def _init_engines():
    """为 API 测试注入引擎"""
    asr_api.set_asr_engine(MockASR(ASRConfig(language="zh")))
    tts_api.set_tts_engine(MockTTS(TTSConfig()))
    chat_api.set_dialog_manager(DialogManager())


class TestHealthAPI:
    def test_health_check(self):
        """测试健康检查接口"""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestASRAPI:
    def test_asr_formats(self):
        """测试获取支持的音频格式"""
        client = TestClient(app)
        response = client.get("/asr/formats")
        assert response.status_code == 200
        data = response.json()
        assert "formats" in data
        assert "wav" in data["formats"]

    def test_asr_recognize_upload(self):
        """测试上传音频识别"""
        client = TestClient(app)
        wav_data = AudioPreprocessor.generate_sine_wave(duration=0.2)

        response = client.post(
            "/asr/recognize",
            files={"file": ("test.wav", wav_data, "audio/wav")},
            data={"language": "zh", "audio_format": "wav"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert data["is_final"] is True
        assert data["confidence"] > 0

    def test_asr_recognize_empty(self):
        """测试空音频识别"""
        client = TestClient(app)
        response = client.post(
            "/asr/recognize",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert response.status_code == 400


class TestTTSAPI:
    def test_list_voices(self):
        """测试列出音色"""
        client = TestClient(app)
        response = client.get("/tts/voices")
        assert response.status_code == 200
        data = response.json()
        assert "voices" in data
        assert len(data["voices"]) > 0

    def test_synthesize_info(self):
        """测试合成（元数据）"""
        client = TestClient(app)
        response = client.post(
            "/tts/synthesize/info",
            json={"text": "你好世界", "voice_id": "default", "speed": 1.0},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "你好世界"
        assert data["audio_format"] == "wav"
        assert data["sample_rate"] == 16000

    def test_synthesize_audio(self):
        """测试合成（音频）"""
        client = TestClient(app)
        response = client.post(
            "/tts/synthesize",
            json={"text": "你好", "voice_id": "default"},
        )
        assert response.status_code == 200
        assert len(response.content) > 44  # WAV header + data

    def test_tts_speed_clamping(self):
        """测试 TTS 语速范围限制 - speed 在范围内应正常工作"""
        client = TestClient(app)
        response = client.post(
            "/tts/synthesize/info",
            json={"text": "测试", "speed": 2.0},
        )
        assert response.status_code == 200


class TestChatAPI:
    def test_chat_message(self):
        """测试文本聊天"""
        client = TestClient(app)
        response = client.post(
            "/chat/message",
            json={"message": "你好", "stream": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["response"]
        assert data["intent"] == "greeting"
        assert data["session_id"]

    def test_chat_multi_turn(self):
        """测试多轮对话"""
        client = TestClient(app)

        # 第一轮
        r1 = client.post(
            "/chat/message",
            json={"message": "你好", "session_id": "multi_test"},
        )
        assert r1.status_code == 200
        data1 = r1.json()

        # 第二轮
        r2 = client.post(
            "/chat/message",
            json={"message": "谢谢", "session_id": "multi_test"},
        )
        assert r2.status_code == 200
        data2 = r2.json()
        assert data2["turn_count"] == 2

    def test_chat_stream(self):
        """测试流式聊天"""
        client = TestClient(app)
        response = client.post(
            "/chat/message/stream",
            json={"message": "你好"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        content = response.text
        assert "data:" in content

    def test_chat_empty_message(self):
        """测试空消息"""
        client = TestClient(app)
        response = client.post(
            "/chat/message",
            json={"message": ""},
        )
        assert response.status_code == 422  # validation error

    def test_delete_session(self):
        """测试删除会话"""
        client = TestClient(app)
        response = client.delete("/chat/session/test_session")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"