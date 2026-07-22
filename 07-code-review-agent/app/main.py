"""FastAPI 应用入口。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.review import router as review_router
from app.api.report import router as report_router
from app.models import HealthResponse
from app.store import review_store

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期管理。"""
    logger.info("AI Agent 代码审查系统启动...")
    yield
    logger.info("AI Agent 代码审查系统关闭...")


app = FastAPI(
    title="AI Agent 代码审查系统",
    description="基于 LangGraph 的多 Agent 协作代码审查，支持 AST 分析、GitHub API 集成",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(review_router, prefix="/api/v1", tags=["代码审查"])
app.include_router(report_router, prefix="/api/v1", tags=["审查报告"])


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """健康检查接口。"""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        agents_available=[
            "SecurityAgent",
            "PerformanceAgent",
            "StyleAgent",
            "LogicAgent",
            "SummaryAgent",
        ],
    )
