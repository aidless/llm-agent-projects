"""Pydantic 数据模型"""

from typing import Optional
from pydantic import BaseModel, Field


# ---- 请求模型 ----

class ASRRequest(BaseModel):
    """ASR 识别请求"""
    audio_format: str = Field(default="wav", description="音频格式: wav/mp3/ogg/pcm")
    language: str = Field(default="zh", description="语言: zh/en")
    enable_punctuation: bool = Field(default=True, description="是否添加标点")
    enable_number_normalize: bool = Field(default=True, description="是否规范化数字")


class TTSRequest(BaseModel):
    """TTS 合成请求"""
    text: str = Field(..., min_length=1, description="要合成的文本")
    voice_id: str = Field(default="default", description="音色 ID")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="语速")
    pitch: float = Field(default=1.0, ge=0.5, le=2.0, description="音调")
    volume: float = Field(default=1.0, ge=0.0, le=1.0, description="音量")
    audio_format: str = Field(default="wav", description="输出格式: wav/mp3")


class ChatRequest(BaseModel):
    """文本聊天请求"""
    message: str = Field(..., min_length=1, description="用户消息")
    session_id: Optional[str] = Field(default=None, description="会话 ID")
    stream: bool = Field(default=False, description="是否流式返回")


# ---- 响应模型 ----

class ASRResponse(BaseModel):
    """ASR 识别响应"""
    text: str
    is_final: bool = True
    confidence: float = 1.0
    language: str = "zh"


class TTSResponse(BaseModel):
    """TTS 合成响应（元数据，音频通过二进制返回）"""
    text: str
    audio_format: str = "wav"
    sample_rate: int = 16000
    duration: float = 0.0
    voice_id: str = "default"


class ChatResponse(BaseModel):
    """文本聊天响应"""
    session_id: str
    response: str
    intent: str = ""
    confidence: float = 0.0
    turn_count: int = 0


class ChatStreamChunk(BaseModel):
    """聊天流式响应块"""
    session_id: str
    text: str
    intent: str = ""
    is_final: bool = False


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    active_sessions: int = 0
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: str = ""


class VoiceListResponse(BaseModel):
    """音色列表响应"""
    voices: list[dict] = []


class SessionInfoResponse(BaseModel):
    """会话信息响应"""
    session_id: str
    state: str
    created_at: float
    last_active: float