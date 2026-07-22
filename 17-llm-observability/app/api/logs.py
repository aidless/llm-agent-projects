"""日志查询 API - 提供结构化日志的检索和过滤接口。"""

from typing import Optional

from fastapi import APIRouter, Query

from app.models import LogEntryModel, LogListResponse
from storage.memory_store import get_store

router = APIRouter(prefix="/api/v1/logs", tags=["logs"])


@router.get("", response_model=LogListResponse)
def query_logs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    trace_id: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    message_contains: Optional[str] = Query(None, alias="message"),
):
    """查询结构化日志，支持按 trace_id、级别和消息内容过滤。"""
    store = get_store()
    logs = store.query_logs(
        limit=limit,
        offset=offset,
        trace_id=trace_id,
        level=level,
        message_contains=message_contains,
    )
    total = store.log_count()

    return LogListResponse(
        total=total,
        logs=[
            LogEntryModel(
                timestamp=l.get("timestamp", 0),
                level=l.get("level", "INFO"),
                logger=l.get("logger", ""),
                message=l.get("message", ""),
                module=l.get("module", ""),
                function=l.get("function", ""),
                line=l.get("line", 0),
                trace_id=l.get("trace_id"),
                span_id=l.get("span_id"),
                exception=l.get("exception"),
            )
            for l in logs
        ],
        limit=limit,
        offset=offset,
    )