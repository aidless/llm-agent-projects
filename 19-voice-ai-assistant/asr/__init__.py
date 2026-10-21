"""ASR 模块 - 语音识别引擎。"""

from asr.base import ASRConfig, ASREngine, ASRResult, AudioFormat
from asr.mock_asr import MockASR

__all__ = ["ASRConfig", "ASREngine", "ASRResult", "AudioFormat", "MockASR"]

# WhisperASR 是可选依赖（需要 faster-whisper）
try:
    from asr.whisper_asr import WhisperASR
    __all__.append("WhisperASR")
except ImportError:
    pass