"""Vision 模块 - 提供图片描述、视觉问答和图表理解功能。"""

# OpenAIVisionClient 是可选依赖（需要 openai 包）
try:
    from vision.openai_client import OpenAIVisionClient
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

__all__ = []
if _OPENAI_AVAILABLE:
    __all__.append("OpenAIVisionClient")