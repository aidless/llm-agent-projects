"""
app/main.py - FastAPI 应用入口

AI 模型评估基准测试平台 - 支持主流评测基准的自动化评测。
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models import HealthResponse
from benchmarks import BENCHMARK_REGISTRY

app = FastAPI(
    title="LLM Eval Benchmark Platform",
    description="""
    AI 模型评估基准测试平台 - 支持主流评测基准的自动化评测。

    ## 功能特性

    - **多基准评测**: MMLU、GSM8K、HumanEval、MT-Bench
    - **并发执行**: 异步并发评测，支持进度追踪
    - **评分系统**: 准确率、Pass@k、F1、置信区间、显著性检验
    - **模型管理**: 模型注册、版本管理
    - **排行榜**: 综合排名、分基准排名、模型卡片
    - **报告生成**: JSON/Markdown/CSV 格式，matplotlib 图表
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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
from app.api.benchmark import router as benchmark_router
from app.api.model import router as model_router
from app.api.leaderboard import router as leaderboard_router
from app.api.report import router as report_router

app.include_router(benchmark_router)
app.include_router(model_router)
app.include_router(leaderboard_router)
app.include_router(report_router)


@app.get("/", response_model=HealthResponse, tags=["system"])
async def health_check():
    """健康检查."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        benchmarks_available=list(BENCHMARK_REGISTRY.keys()),
    )


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """健康检查端点."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        benchmarks_available=list(BENCHMARK_REGISTRY.keys()),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)