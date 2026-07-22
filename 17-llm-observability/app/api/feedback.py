"""反馈管理 API - 提供反馈的创建、查询和统计接口。"""

from typing import Optional

from fastapi import APIRouter, HTTPException

from app.models import (
    FeedbackCreateRequest,
    FeedbackListResponse,
    FeedbackModel,
    FeedbackStatsResponse,
)
from evaluation.feedback import FeedbackCollector, FeedbackType
from evaluation.exporter import DataExporter
from storage.memory_store import get_store

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])

_collector = FeedbackCollector()
_exporter = DataExporter()


@router.post("", response_model=FeedbackModel)
def create_feedback(req: FeedbackCreateRequest):
    """创建反馈。"""
    fb_map = {
        "thumbs_up": lambda: _collector.add_thumbs_up(
            trace_id=req.trace_id, user_id=req.user_id, comment=req.comment,
        ),
        "thumbs_down": lambda: _collector.add_thumbs_down(
            trace_id=req.trace_id, user_id=req.user_id, comment=req.comment,
        ),
        "rating": lambda: _collector.add_rating(
            trace_id=req.trace_id, rating=req.value or 0,
            user_id=req.user_id, comment=req.comment,
        ),
        "correction": lambda: _collector.add_correction(
            trace_id=req.trace_id, correction=req.comment, user_id=req.user_id,
        ),
        "flag": lambda: _collector.add_thumbs_down(
            trace_id=req.trace_id, user_id=req.user_id, comment=f"FLAG: {req.comment}",
        ),
    }

    action = fb_map.get(req.feedback_type.value)
    if not action:
        raise HTTPException(status_code=400, detail=f"不支持的反馈类型: {req.feedback_type}")

    fb = action()
    return FeedbackModel(**fb.to_dict())


@router.get("", response_model=FeedbackListResponse)
def list_feedbacks(
    trace_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
):
    """查询反馈列表。"""
    store = get_store()
    feedbacks = store.list_feedbacks(trace_id=trace_id, limit=limit, offset=offset)
    total = store.feedback_count(trace_id=trace_id)

    return FeedbackListResponse(
        total=total,
        feedbacks=[FeedbackModel(**fb) for fb in feedbacks],
    )


@router.get("/stats", response_model=FeedbackStatsResponse)
def feedback_stats():
    """获取反馈统计信息。"""
    stats = _collector.get_stats()
    return FeedbackStatsResponse(**stats)


@router.get("/export/finetuning")
def export_for_finetuning(
    min_rating: Optional[float] = None,
    span_type: str = "llm",
):
    """导出用于微调的数据集。"""
    dataset = _exporter.export_for_finetuning(
        span_type=span_type,
        min_rating=min_rating,
    )
    return {
        "count": len(dataset),
        "data": dataset,
    }