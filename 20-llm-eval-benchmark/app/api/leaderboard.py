"""
app/api/leaderboard.py - 排行榜 API 路由
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from leaderboard import Leaderboard, Visualizer
from engine.comparator import ModelEvalResult
from app.models import LeaderboardResponse, LeaderboardEntryResponse
from app.api.benchmark import _latest_results, _latest_progress

router = APIRouter(prefix="/api/leaderboard", tags=["leaderboard"])

# 全局排行榜
_global_leaderboard = Leaderboard()


def _rebuild_leaderboard():
    """从最新结果重建排行榜."""
    leaderboard = Leaderboard()

    # 遍历所有存储的结果
    for key, result_responses in _latest_results.items():
        if ":" not in key:
            continue
        model_name, benchmark_name = key.split(":", 1)

        # 从结果响应重建 BenchmarkResult
        from benchmarks.base import BenchmarkResult as BResult
        results = [
            BResult(
                question_id=r.question_id,
                question=r.question,
                expected=r.expected,
                predicted=r.predicted,
                correct=r.correct,
                score=r.score,
            )
            for r in result_responses
        ]

        model_eval = ModelEvalResult(
            model_name=model_name,
            benchmark_name=benchmark_name,
            results=results,
        )
        leaderboard.add_model_result(model_eval)

    return leaderboard


@router.get("/overall", response_model=LeaderboardResponse)
async def get_overall_leaderboard():
    """获取综合排行榜."""
    leaderboard = _rebuild_leaderboard()
    entries = leaderboard.get_overall_ranking()

    return LeaderboardResponse(
        total_models=len(entries),
        available_benchmarks=leaderboard.get_available_benchmarks(),
        overall_ranking=[
            LeaderboardEntryResponse(
                rank=e.rank,
                model_name=e.model_name,
                overall_score=round(e.overall_score, 4),
                overall_score_pct=f"{e.overall_score:.2%}",
                benchmark_scores=e.benchmark_scores,
            )
            for e in entries
        ],
    )


@router.get("/benchmark/{benchmark_name}")
async def get_benchmark_leaderboard(benchmark_name: str):
    """获取指定基准的排行榜."""
    leaderboard = _rebuild_leaderboard()
    ranking = leaderboard.get_benchmark_ranking(benchmark_name)

    if not ranking:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for benchmark '{benchmark_name}'"
        )

    return {
        "benchmark": benchmark_name,
        "ranking": ranking,
    }


@router.get("/model/{model_name}")
async def get_model_card(model_name: str):
    """获取模型卡片."""
    leaderboard = _rebuild_leaderboard()
    card = leaderboard.get_model_card(model_name)

    if not card:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found in leaderboard"
        )

    return card


@router.get("/charts")
async def generate_charts():
    """生成排行榜图表."""
    leaderboard = _rebuild_leaderboard()

    if not leaderboard.get_overall_ranking():
        raise HTTPException(status_code=404, detail="No evaluation data available")

    visualizer = Visualizer(output_dir="reports")
    charts = visualizer.generate_all_charts(leaderboard)

    return {
        "status": "generated",
        "charts": {name: path for name, path in charts.items() if path},
    }