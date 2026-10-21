"""TTS 模块 - 语音合成引擎。"""

from tts.base import TTSConfig, TTSEngine, TTSResult, TTSVoice, TTSAudioFormat
from tts.mock_tts import MockTTS
from tts.ssml_parser import SSMLParser

__all__ = ["TTSConfig", "TTSEngine", "TTSResult", "TTSVoice", "TTSAudioFormat", "MockTTS", "SSMLParser"]

# EdgeTTS 是可选依赖（需要 edge-tts）
try:
    from tts.edge_tts import EdgeTTS
    __all__.append("EdgeTTS")
except ImportError:
    pass