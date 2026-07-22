"""ASR API 路由"""

import base64
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.models import ASRRequest, ASRResponse, ErrorResponse
from asr.base import ASRConfig, AudioFormat

router = APIRouter(prefix="/asr", tags=["ASR"])

# 全局 ASR 引擎（由 main.py 注入）
_asr_engine = None


def set_asr_engine(engine):
    """设置 ASR 引擎"""
    global _asr_engine
    _asr_engine = engine


def get_asr_engine():
    """获取 ASR 引擎"""
    if _asr_engine is None:
        raise HTTPException(status_code=503, detail="ASR engine not initialized")
    return _asr_engine


@router.post("/recognize", response_model=ASRResponse)
async def recognize_audio(
    file: UploadFile = File(..., description="音频文件"),
    language: str = Form(default="zh", description="语言"),
    audio_format: str = Form(default="wav", description="音频格式"),
):
    """识别上传的音频文件

    支持的音频格式: WAV, MP3, OGG, PCM
    """
    engine = get_asr_engine()

    # 读取音频数据
    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(status_code=400, detail="Empty audio data")

    # 更新语言配置
    original_language = engine.config.language
    engine.config.language = language

    try:
        result = await engine.recognize(audio_data)
        return ASRResponse(
            text=result.text,
            is_final=result.is_final,
            confidence=result.confidence,
            language=result.language,
        )
    finally:
        engine.config.language = original_language


@router.post("/recognize/base64", response_model=ASRResponse)
async def recognize_base64(request: ASRRequest, audio_b64: str):
    """通过 Base64 编码识别音频"""
    engine = get_asr_engine()

    try:
        audio_data = base64.b64decode(audio_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")

    original_language = engine.config.language
    engine.config.language = request.language
    engine.config.enable_punctuation = request.enable_punctuation
    engine.config.enable_number_normalize = request.enable_number_normalize

    try:
        result = await engine.recognize(audio_data)
        return ASRResponse(
            text=result.text,
            is_final=result.is_final,
            confidence=result.confidence,
            language=result.language,
        )
    finally:
        engine.config.language = original_language


@router.get("/formats")
async def list_formats():
    """列出支持的音频格式"""
    engine = get_asr_engine()
    formats = engine.supported_formats()
    return {"formats": [f.value for f in formats]}