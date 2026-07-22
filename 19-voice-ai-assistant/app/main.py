"""FastAPI 应用入口"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from asr.base import ASRConfig
from asr.mock_asr import MockASR
from tts.base import TTSConfig
from tts.mock_tts import MockTTS
from dialog.manager import DialogManager
from pipeline.voice_pipeline import VoicePipeline, PipelineConfig
from websocket.handler import WebSocketHandler
from websocket.session import SessionManager
from app.models import HealthResponse

logger = logging.getLogger(__name__)

# 全局实例
asr_engine = None
tts_engine = None
dialog_manager = None
voice_pipeline = None
ws_handler = None
session_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global asr_engine, tts_engine, dialog_manager, voice_pipeline, ws_handler, session_manager

    logger.info("Initializing Voice AI Assistant...")

    # 初始化 ASR
    asr_engine = MockASR(ASRConfig(language="zh"))

    # 初始化 TTS
    tts_engine = MockTTS(TTSConfig())

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

    # 注入到 API 路由
    from app.api import asr as asr_api
    from app.api import tts as tts_api
    from app.api import chat as chat_api
    asr_api.set_asr_engine(asr_engine)
    tts_api.set_tts_engine(tts_engine)
    chat_api.set_dialog_manager(dialog_manager)

    logger.info("Voice AI Assistant initialized successfully")
    yield

    logger.info("Shutting down Voice AI Assistant...")
    session_manager.cleanup_expired()


app = FastAPI(
    title="Voice AI Assistant",
    description="ASR + LLM + TTS 全链路语音对话系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
from app.api import asr as asr_api
from app.api import tts as tts_api
from app.api import chat as chat_api

app.include_router(asr_api.router)
app.include_router(tts_api.router)
app.include_router(chat_api.router)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="ok",
        active_sessions=session_manager.get_active_count() if session_manager else 0,
    )


@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """WebSocket 语音对话端点"""
    if ws_handler is None:
        await websocket.close(code=1013, reason="Service not ready")
        return
    await ws_handler.handle_connection(websocket)