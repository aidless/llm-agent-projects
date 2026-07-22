"""审查报告查询 API 路由。"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.store import review_store
from app.models import ReviewResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/reports/{review_id}", response_model=ReviewResponse)
async def get_report(review_id: str) -> ReviewResponse:
    """根据 review_id 查询审查报告。"""
    if review_id not in review_store:
        raise HTTPException(status_code=404, detail=f"未找到审查报告: {review_id}")

    report_data = review_store[review_id]
    return ReviewResponse(**report_data)


@router.get("/reports")
async def list_reports(
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """列出所有审查报告。"""
    all_ids = list(review_store.keys())
    total = len(all_ids)
    paginated_ids = all_ids[offset : offset + limit]

    reports = []
    for rid in paginated_ids:
        data = review_store[rid]
        reports.append({
            "review_id": rid,
            "status": data.get("status"),
            "created_at": data.get("created_at"),
            "total_findings": data.get("total_findings"),
            "filename": data.get("filename"),
            "language": data.get("language"),
        })

    return {"total": total, "offset": offset, "limit": limit, "reports": reports}
