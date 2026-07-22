"""TTS API 路由"""

import io
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.models import TTSRequest, TTSResponse, VoiceListResponse, ErrorResponse
from tts.base import TTSAudioFormat, TTSConfig

router = APIRouter(prefix="/tts", tags=["TTS"])

# 全局 TTS 引擎（由 main.py 注入）
_tts_engine = None


def set_tts_engine(engine):
    """设置 TTS 引擎"""
    global _tts_engine
    _tts_engine = engine


def get_tts_engine():
    """获取 TTS 引擎"""
    if _tts_engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not initialized")
    return _tts_engine


@router.post("/synthesize")
async def synthesize(request: TTSRequest):
    """文本转语音

    返回音频二进制数据。
    """
    engine = get_tts_engine()

    # 配置参数
    engine.set_voice(request.voice_id)
    engine.set_speed(request.speed)
    engine.set_pitch(request.pitch)
    engine.set_volume(request.volume)

    try:
        result = await engine.synthesize(request.text)

        media_type = "audio/wav" if result.audio_format == TTSAudioFormat.WAV else "audio/mpeg"
        return Response(
            content=result.audio_data,
            media_type=media_type,
            headers={
                "X-Audio-Format": result.audio_format.value,
                "X-Sample-Rate": str(result.sample_rate),
                "X-Duration": str(result.duration),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthesize/info", response_model=TTSResponse)
async def synthesize_info(request: TTSRequest):
    """文本转语音（仅返回元数据）"""
    engine = get_tts_engine()

    engine.set_voice(request.voice_id)
    engine.set_speed(request.speed)
    engine.set_pitch(request.pitch)
    engine.set_volume(request.volume)

    try:
        result = await engine.synthesize(request.text)
        return TTSResponse(
            text=result.text,
            audio_format=result.audio_format.value,
            sample_rate=result.sample_rate,
            duration=result.duration,
            voice_id=request.voice_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voices", response_model=VoiceListResponse)
async def list_voices():
    """列出可用音色"""
    engine = get_tts_engine()
    voices = engine.list_voices()
    return VoiceListResponse(
        voices=[
            {
                "voice_id": v.voice_id,
                "name": v.name,
                "language": v.language,
                "gender": v.gender,
            }
            for v in voices
        ]
    )