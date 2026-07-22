"""
app/api/report.py - 报告 API 路由
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from engine.comparator import Comparator, ModelEvalResult
from engine.scorer import Scorer
from benchmarks.base import BenchmarkResult
from app.models import ReportRequest, ComparisonRequest, ComparisonResponse
from app.api.benchmark import _latest_results

router = APIRouter(prefix="/api/report", tags=["report"])


def _rebuild_comparator() -> Comparator:
    """从最新结果重建比较器."""
    comparator = Comparator()

    for key, result_responses in _latest_results.items():
        if ":" not in key:
            continue
        model_name, benchmark_name = key.split(":", 1)

        results = [
            BenchmarkResult(
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
        comparator.add_result(model_eval)

    return comparator


@router.post("/generate")
async def generate_report(request: ReportRequest):
    """生成评测报告."""
    comparator = _rebuild_comparator()

    if not comparator._model_results:
        raise HTTPException(status_code=404, detail="No evaluation results available")

    report_format = request.format.lower()
    summary = comparator.get_all_results()

    if report_format == "markdown":
        return _generate_markdown_report(summary, request, comparator)
    elif report_format == "csv":
        return _generate_csv_report(summary, request)
    else:
        return JSONResponse(content=summary)


def _generate_markdown_report(summary: dict, request: ReportRequest, comparator: Comparator):
    """生成 Markdown 格式报告."""
    lines = [
        "# AI Model Evaluation Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Overview",
        f"\n- **Total Models Evaluated**: {len(summary.get('models', []))}",
        f"- **Models**: {', '.join(summary.get('models', []))}",
        "",
    ]

    # 排名
    lines.append("## Overall Ranking")
    lines.append("")
    lines.append("| Rank | Model | Score |")
    lines.append("|------|-------|-------|")
    for entry in summary.get("rankings", []):
        lines.append(f"| {entry['rank']} | {entry['model_name']} | {entry['score']:.2%} |")
    lines.append("")

    # 详细结果
    if request.include_details:
        lines.append("## Detailed Results by Benchmark")
        lines.append("")
        for model_name, benchmarks in summary.get("per_benchmark", {}).items():
            lines.append(f"### {model_name}")
            lines.append("")
            for bname, bdata in benchmarks.items():
                lines.append(f"**{bname}**: {bdata.get('accuracy', 0):.2%} "
                             f"({bdata.get('correct', 0)}/{bdata.get('total_questions', 0)})")
            lines.append("")

    return PlainTextResponse(content="\n".join(lines), media_type="text/markdown")


def _generate_csv_report(summary: dict, request: ReportRequest):
    """生成 CSV 格式报告."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Rank", "Model", "Benchmark", "Score", "Correct", "Total"])

    for model_name, benchmarks in summary.get("per_benchmark", {}).items():
        for bname, bdata in benchmarks.items():
            writer.writerow([
                "",  # Rank filled later
                model_name,
                bname,
                f"{bdata.get('accuracy', 0):.4f}",
                bdata.get("correct", 0),
                bdata.get("total_questions", 0),
            ])

    return PlainTextResponse(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=eval_report.csv"},
    )


@router.post("/compare", response_model=ComparisonResponse)
async def compare_models(request: ComparisonRequest):
    """对比两个模型."""
    comparator = _rebuild_comparator()

    try:
        comparisons = comparator.compare_models(
            request.model_a,
            request.model_b,
            request.benchmark_name,
        )
        ranking = comparator.get_ranking(request.benchmark_name)

        return ComparisonResponse(
            model_a=request.model_a,
            model_b=request.model_b,
            comparisons=[c.to_dict() for c in comparisons],
            ranking=ranking,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))