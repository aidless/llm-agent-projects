"""
app/api/benchmark.py - 评测 API 路由
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

from benchmarks import get_benchmark, BENCHMARK_REGISTRY
from benchmarks.base import BenchmarkResult
from engine import Evaluator, EvalConfig, MockLLMClient, Scorer
from app.models import (
    EvalRequest,
    EvalResponse,
    EvalProgressResponse,
    BenchmarkResultResponse,
    ScoreDetail,
)

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])

# 全局状态 (生产环境应使用数据库)
_evaluator_cache: Dict[str, Evaluator] = {}
_latest_progress: Dict[str, EvalProgressResponse] = {}
_latest_results: Dict[str, list] = {}


@router.get("/list")
async def list_benchmarks():
    """列出所有可用的评测基准."""
    benchmarks = []
    for name, cls in BENCHMARK_REGISTRY.items():
        try:
            instance = cls()
            benchmarks.append({
                "name": name,
                "description": instance.description,
                "total_questions": instance.total_questions,
            })
        except Exception:
            benchmarks.append({
                "name": name,
                "description": "Failed to load",
                "total_questions": 0,
            })
    return {"benchmarks": benchmarks}


@router.get("/{benchmark_name}/info")
async def benchmark_info(benchmark_name: str):
    """获取基准详细信息."""
    try:
        benchmark = get_benchmark(benchmark_name)
        info = {
            "name": benchmark.name,
            "description": benchmark.description,
            "total_questions": benchmark.total_questions,
        }
        # 获取额外信息
        if hasattr(benchmark, "get_categories"):
            info["categories"] = benchmark.get_categories()
        if hasattr(benchmark, "get_difficulty_distribution"):
            info["difficulty_distribution"] = benchmark.get_difficulty_distribution()
        return info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/evaluate", response_model=EvalResponse)
async def evaluate_model(request: EvalRequest):
    """执行模型评测."""
    try:
        benchmark = get_benchmark(request.benchmark)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    config = EvalConfig(
        max_concurrent=request.max_concurrent,
        timeout_seconds=request.timeout_seconds,
        max_retries=request.max_retries,
        enable_cache=request.enable_cache,
        question_limit=request.question_limit,
    )

    # 使用 Mock 客户端 (测试环境)
    llm_client = MockLLMClient(accuracy_rate=0.8)
    evaluator = Evaluator(llm_client=llm_client, config=config)

    start_time = time.time()

    # 追踪进度
    def on_progress(progress):
        _latest_progress[request.model_name] = EvalProgressResponse(
            total=progress.total,
            completed=progress.completed,
            correct=progress.correct,
            errors=progress.errors,
            accuracy=progress.accuracy,
            progress_pct=progress.progress_pct,
            elapsed_seconds=progress.elapsed_seconds,
        )

    results = await evaluator.evaluate(
        benchmark=benchmark,
        model_name=request.model_name,
        on_progress=on_progress,
    )

    eval_time = time.time() - start_time

    # 计算评分
    scores = Scorer.full_report(results)

    # 存储结果
    _latest_results[f"{request.model_name}:{request.benchmark}"] = [
        BenchmarkResultResponse(
            question_id=r.question_id,
            question=r.question,
            expected=r.expected,
            predicted=r.predicted,
            correct=r.correct,
            score=r.score,
            error=r.error,
        )
        for r in results
    ]

    return EvalResponse(
        model_name=request.model_name,
        benchmark=request.benchmark,
        total_questions=len(results),
        correct=sum(1 for r in results if r.correct),
        accuracy=sum(1 for r in results if r.correct) / len(results) if results else 0,
        eval_time_seconds=round(eval_time, 2),
        scores={
            k: ScoreDetail(
                metric=v.metric_name,
                value=v.value,
                confidence_interval=(
                    {"low": v.confidence_low, "high": v.confidence_high}
                    if v.confidence_low is not None else None
                ),
                details=v.details if v.details else None,
            )
            for k, v in scores.items()
        },
        results=_latest_results[f"{request.model_name}:{request.benchmark}"],
    )


@router.get("/progress/{model_name}", response_model=EvalProgressResponse)
async def get_progress(model_name: str):
    """获取评测进度."""
    if model_name in _latest_progress:
        return _latest_progress[model_name]
    return EvalProgressResponse()


@router.get("/results/{model_name}/{benchmark_name}")
async def get_results(model_name: str, benchmark_name: str):
    """获取评测结果."""
    key = f"{model_name}:{benchmark_name}"
    if key in _latest_results:
        return {"model_name": model_name, "benchmark": benchmark_name, "results": _latest_results[key]}
    raise HTTPException(status_code=404, detail="No evaluation results found for this model and benchmark")