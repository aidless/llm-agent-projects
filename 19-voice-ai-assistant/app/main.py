"""FastAPI 应用入口

⚠️ **2026-07-22 安全加固**：原服务默认用 MockASR + MockTTS，听起来像
"语音 AI 助手"，实际是返回固定字符串的哑服务。本版本启动时显式检测
provider 配置，未配置真实现则启动失败。

启用 Mock（仅测试）：
    export ALLOW_MOCK_PROVIDERS=true
    uvicorn app.main:app

启用真实 ASR/TTS：
    export ASR_PROVIDER=whisper  # 或 funasr / azure
    export TTS_PROVIDER=edge-tts  # 或 azure / elevenlabs
    export OPENAI_API_KEY=sk-xxx  # LLM 必填
    uvicorn app.main:app
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

# 在导入前检测 mock 配置，因为 mock_asr/mock_tts 模块本身也会被加载
_MOCK_ALLOWED = os.getenv("ALLOW_MOCK_PROVIDERS", "").lower() in ("true", "1", "yes")
_ASR_PROVIDER = os.getenv("ASR_PROVIDER", "mock").lower()
_TTS_PROVIDER = os.getenv("TTS_PROVIDER", "mock").lower()


def _enforce_mock_policy():
    """启动时守卫：禁止隐式使用 mock。"""
    using_mock = _ASR_PROVIDER == "mock" or _TTS_PROVIDER == "mock"
    if using_mock and not _MOCK_ALLOWED:
        raise RuntimeError(
            "Voice AI Assistant 默认使用 Mock ASR/TTS（仅返回固定字符串）。\n"
            "生产部署必须配置真实 provider：\n"
            "  export ASR_PROVIDER=whisper  TTS_PROVIDER=edge-tts\n"
            "  export OPENAI_API_KEY=sk-xxx\n"
            "或显式允许 mock（仅用于本地测试）：\n"
            "  export ALLOW_MOCK_PROVIDERS=true"
        )


# 必须在所有 import 后执行
from asr.base import ASRConfig
from asr.mock_asr import MockASR
from tts.base import TTSConfig
from tts.mock_tts import MockTTS
from dialog.manager import DialogManager
from pipeline.voice_pipeline import VoicePipeline, PipelineConfig
from websocket.handler import WebSocketHandler
from websocket.session import SessionManager
from app.models import HealthResponse

# 全局实例
asr_engine = None
tts_engine = None
dialog_manager = None
voice_pipeline = None
ws_handler = None
session_manager = None


def _build_asr():
    """根据 ASR_PROVIDER 选择 ASR 引擎；mock 时必须显式 ALLOW。

    ⚠️ 仓库目前只实现了 MockASR。`whisper` / `azure` / `funasr` 等真实现
    需要后续实现（见 README "Roadmap"）。
    """
    if _ASR_PROVIDER == "mock":
        return MockASR(ASRConfig(language="zh"))
    raise NotImplementedError(
        f"ASR_PROVIDER={_ASR_PROVIDER!r} 未实现。"
        "当前只支持 'mock'（仅测试用）。请实现对应引擎或使用 mock。"
    )


def _build_tts():
    if _TTS_PROVIDER == "mock":
        return MockTTS(TTSConfig())
    raise NotImplementedError(
        f"TTS_PROVIDER={_TTS_PROVIDER!r} 未实现。"
        "当前只支持 'mock'（仅测试用）。请实现对应引擎或使用 mock。"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global asr_engine, tts_engine, dialog_manager, voice_pipeline, ws_handler, session_manager

    # 启动守卫：禁止隐式 mock
    _enforce_mock_policy()

    logger.info("Initializing Voice AI Assistant...")
    logger.info(f"  ASR provider: {_ASR_PROVIDER}")
    logger.info(f"  TTS provider: {_TTS_PROVIDER}")
    if _MOCK_ALLOWED:
        logger.warning("  ⚠️  ALLOW_MOCK_PROVIDERS=true — 使用 Mock 引擎")

    # 初始化 ASR / TTS
    asr_engine = _build_asr()
    tts_engine = _build_tts()

    # 初始化对话管理器
    dialog_manager = DialogManager()

    # 初始化 Pipeline
    voice_pipeline = VoicePipeline(
        asr_engine=asr_engine,
        tts_engine=tts_engine,
        dialog_manager=dialog_manager,
        config=PipelineConfig(),
    )

    # 初始化 WebSocket
    session_manager = SessionManager()
    ws_handler = WebSocketHandler(
        pipeline=voice_pipeline,
        session_manager=session_manager,
    )

    yield

    # 关闭清理
    logger.info("Shutting down Voice AI Assistant...")