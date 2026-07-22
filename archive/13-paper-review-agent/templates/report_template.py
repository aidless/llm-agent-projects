"""Report generation templates -- Markdown and JSON."""

from __future__ import annotations

import json
from typing import Any

from app.models import Recommendation, ReviewReport, ReviewDimension

from templates.review_criteria import get_all_criteria

# Dimension display names
_DIM_NAMES: dict[ReviewDimension, str] = {
    ReviewDimension.RELEVANCE: "相关性 (Relevance)",
    ReviewDimension.METHODOLOGY: "方法论 (Methodology)",
    ReviewDimension.NOVELTY: "创新性 (Novelty)",
    ReviewDimension.CLARITY: "可读性 (Clarity)",
}


def render_markdown(report: ReviewReport) -> str:
    """Render a ReviewReport as a Markdown string."""
    lines: list[str] = []

    # Header
    lines.append(f"# 论文评审报告")
    lines.append("")
    lines.append(f"**论文标题**: {report.paper_title}")
    lines.append(f"**目标会议/期刊**: {report.metadata.get('venue', 'N/A')}")
    lines.append(f"**使用模型**: {', '.join(report.metadata.get('models_used', []))}")
    lines.append(f"**总体评分**: {report.overall_score}/10")
    lines.append(f"**最终推荐**: {report.recommendation.value}")
    lines.append("")

    # Meta summary
    lines.append("## 综合评述")
    lines.append("")
    lines.append(report.meta_summary)
    lines.append("")

    # Cross-validation overview
    lines.append("## 交叉验证概览")
    lines.append("")
    lines.append("| 评审维度 | 模型A评分 | 模型B评分 | 一致性评分 | 一致? |")
    lines.append("| --- | --- | --- | --- | --- |")
    for cv in report.cross_validation_results:
        a_score = cv.reviews[0].score
        b_score = cv.reviews[1].score
        agree_mark = "Yes" if cv.is_consistent else "No"
        dim_name = _DIM_NAMES.get(cv.dimension, cv.dimension.value)
        lines.append(
            f"| {dim_name} | {a_score} | {b_score} | {cv.agreement_score:.2f} | {agree_mark} |"
        )
    lines.append("")

    # Per-dimension detail
    lines.append("## 分维度评审详情")
    lines.append("")

    for cv in report.cross_validation_results:
        review = cv.arbitrated_review if cv.arbitrated_review else cv.reviews[0]
        dim_name = _DIM_NAMES.get(cv.dimension, cv.dimension.value)

        lines.append(f"### {dim_name}  (评分: {review.score}/10)")
        lines.append("")
        lines.append(f"**总结**: {review.summary}")
        lines.append("")

        if review.strengths:
            lines.append("**优点**:")
            for s in review.strengths:
                lines.append(f"- {s}")
            lines.append("")

        if review.weaknesses:
            lines.append("**缺点**:")
            for w in review.weaknesses:
                lines.append(f"- {w}")
            lines.append("")

        if review.suggestions:
            lines.append("**建议**:")
            for s in review.suggestions:
                lines.append(f"- {s}")
            lines.append("")

        if not cv.is_consistent:
            lines.append(f"> **注意**: 两个模型存在分歧 -- {cv.disagreement_reason}")
            lines.append("")

    # Overall strengths/weaknesses/suggestions
    if report.meta_strengths:
        lines.append("## 综合优点")
        lines.append("")
        for s in report.meta_strengths:
            lines.append(f"- {s}")
        lines.append("")

    if report.meta_weaknesses:
        lines.append("## 综合缺点")
        lines.append("")
        for w in report.meta_weaknesses:
            lines.append(f"- {w}")
        lines.append("")

    if report.meta_suggestions:
        lines.append("## 综合建议")
        lines.append("")
        for s in report.meta_suggestions:
            lines.append(f"- {s}")
        lines.append("")

    return "\n".join(lines)


def render_json(report: ReviewReport) -> dict[str, Any]:
    """Render a ReviewReport as a JSON-serialisable dict."""
    data = report.model_dump()
    # Convert enums to strings
    data["recommendation"] = report.recommendation.value
    for dr in data.get("dimension_reviews", []):
        dr["dimension"] = dr["dimension"] if isinstance(dr["dimension"], str) else dr["dimension"].value
    for cv in data.get("cross_validation_results", []):
        cv["dimension"] = cv["dimension"] if isinstance(cv["dimension"], str) else cv["dimension"].value
    return data


def render_report(report: ReviewReport, fmt: str = "markdown") -> str:
    """Render report in the requested format.  Always returns a string.

    For JSON format, returns json.dumps(..., ensure_ascii=False, indent=2).
    """
    if fmt == "json":
        return json.dumps(render_json(report), ensure_ascii=False, indent=2)
    return render_markdown(report)