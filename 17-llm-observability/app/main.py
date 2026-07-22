"""FastAPI 入口 - LLM 应用全链路可观测性平台。"""

import sys
import os
from contextlib import asynccontextmanager
from typing import Optional

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.traces import router as traces_router
from app.api.metrics import router as metrics_router
from app.api.logs import router as logs_router
from app.api.feedback import router as feedback_router
from obs_logging.logger import LLMObservabilityLogger
from storage.memory_store import reset_store

logger = LLMObservabilityLogger("llm-observability")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    logger.info("LLM Observability Platform starting...")
    yield
    logger.info("LLM Observability Platform shutting down...")


app = FastAPI(
    title="LLM Observability Platform",
    description="LLM 应用全链路可观测性平台 - 模拟 OpenTelemetry + LangSmith/LangFuse 的核心功能",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(traces_router)
app.include_router(metrics_router)
app.include_router(logs_router)
app.include_router(feedback_router)


@app.get("/")
def root():
    """健康检查。"""
    return {
        "service": "LLM Observability Platform",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health():
    """健康检查端点。"""
    return {"status": "healthy"}