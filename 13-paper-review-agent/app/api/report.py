"""Report retrieval and formatting API."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional, Any
from app.models import ReviewReport, Recommendation
from templates.report_template import render_markdown, render_json

router = APIRouter()


# In-memory store for demo purposes
_report_store: dict[str, ReviewReport] = {}


class RenderRequest(BaseModel):
    report: ReviewReport
    format: str = "markdown"


@router.post("/report/render")
async def render_report_endpoint(req: RenderRequest):
    """Render an existing ReviewReport into markdown or JSON."""
    try:
        if req.format == "json":
            return {"success": True, "content": render_json(req.report)}
        return {"success": True, "content": render_markdown(req.report)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/report/sample")
async def sample_report():
    """Return a sample report for testing/demonstration."""
    from app.models import (
        DimensionReview,
        CrossValidationResult,
    )
    from app.models import ReviewDimension

    dr = DimensionReview(
        dimension=ReviewDimension.RELEVANCE,
        score=7.5,
        confidence=0.85,
        strengths=["论文主题与会议高度匹配"],
        weaknesses=["相关性论述可以更深入"],
        suggestions=["补充与会议Call for Papers的对应说明"],
        summary="论文与NeurIPS主题高度相关",
        model_name="gpt-4o",
    )
    dr2 = DimensionReview(
        dimension=ReviewDimension.RELEVANCE,
        score=7.0,
        confidence=0.80,
        strengths=["研究方向属于ML主流"],
        weaknesses=["应用场景与会议稍有偏差"],
        suggestions=["强调理论贡献"],
        summary="论文方向匹配",
        model_name="deepseek-chat",
    )
    cv = CrossValidationResult(
        dimension=ReviewDimension.RELEVANCE,
        reviews=[dr, dr2],
        agreement_score=0.75,
        is_consistent=True,
    )

    report = ReviewReport(
        paper_title="Sample Paper Title",
        paper_abstract="This is a sample abstract.",
        cross_validation_results=[cv],
        overall_score=7.2,
        recommendation=Recommendation.ACCEPT,
        meta_summary="这是一篇具有较好质量的论文，建议接收。",
        meta_strengths=["主题相关", "方法合理"],
        meta_weaknesses=["部分实验不足"],
        meta_suggestions=["补充消融实验"],
        metadata={"venue": "NeurIPS", "models_used": ["gpt-4o", "deepseek-chat"]},
    )

    return {"success": True, "report": report.model_dump()}
