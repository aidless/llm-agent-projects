"""FastAPI 应用入口。

⚠️ **2026-07-22 重要提示**：原服务 `ImageDescriber` 默认使用
`MockVisionClient`，返回固定字符串 `'这是一张 WxH 像素的图片 (mock 描述)'`。
本版本启动时检查 `VISION_PROVIDER` 环境变量，未配置真实现则启动失败。

启动示例：
    # 真实现（推荐）
    export VISION_PROVIDER=openai
    export OPENAI_API_KEY=sk-xxx
    uvicorn app.main:app

    # 仅本地测试（mock）
    export VISION_PROVIDER=mock
    export ALLOW_MOCK_PROVIDERS=true
    uvicorn app.main:app
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _enforce_vision_provider():
    """启动守卫：禁止默认 mock vision。"""
    provider = os.getenv("VISION_PROVIDER", "mock").lower()
    mock_allowed = os.getenv("ALLOW_MOCK_PROVIDERS", "").lower() in ("true", "1", "yes")

    if provider == "mock" and not mock_allowed:
        raise RuntimeError(
            "09 multimodal 默认使用 MockVisionClient（仅返回固定字符串）。\n"
            "请设置：\n"
            "  export VISION_PROVIDER=openai\n"
            "  export OPENAI_API_KEY=sk-xxx\n"
            "或显式允许 mock（仅本地测试）：\n"
            "  export VISION_PROVIDER=mock ALLOW_MOCK_PROVIDERS=true"
        )
    return provider


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    provider = _enforce_vision_provider()
    if provider == "mock":
        logger.warning("=" * 60)
        logger.warning("  ⚠️  VISION_PROVIDER=mock — 启动失败已绕过")
        logger.warning("  Vision API 调用将返回固定字符串")
        logger.warning("=" * 60)
    else:
        logger.info(f"VISION_PROVIDER={provider}")
    yield


app = FastAPI(
    title="多模态文档理解系统",
    description="支持图片/OCR/表格解析/版面分析的多模态文档处理系统，结合RAG实现文档问答。",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS（2026-07-22: 收紧 — 默认 allow_origins=* 是 P0 风险）
_cors_origins = os.getenv("MM_CORS_ORIGINS", "").split(",")
_cors_origins = [o.strip() for o in _cors_origins if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
else:
    logger.warning("MM_CORS_ORIGINS 未配置，CORS 中间件未启用")

# 注册路由
from app.api import parse, ocr, table, qa  # noqa: E402

app.include_router(parse.router)
app.include_router(ocr.router)
app.include_router(table.router)
app.include_router(qa.router)


@app.get("/")
async def root():
    return {
        "message": "多模态文档理解系统 API",
        "version": "1.0.0",
        "vision_provider": os.getenv("VISION_PROVIDER", "mock"),
        "auth_warning": (
            "no API routes are protected — see README"
            if os.getenv("VISION_PROVIDER", "mock") == "mock"
            else "configure bearer auth before production"
        ),
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "vision_provider": os.getenv("VISION_PROVIDER", "mock"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)