"""审计日志 API"""

from fastapi import APIRouter, HTTPException, Query

from app.models import ErrorResponse
from audit.logger import AuditLogger
from audit.event_tracker import EventTracker
from audit.reporter import AuditReporter

router = APIRouter(prefix="/api/v1", tags=["audit"])

# 使用与 execute API 相同的全局实例
from app.api.execute import audit_logger, event_tracker

reporter = AuditReporter(audit_logger, event_tracker)


@router.get(
    "/audit/logs",
    summary="查询审计日志",
    description="查询审计日志，支持按执行ID、事件类型和级别过滤",
)
async def query_logs(
    execution_id: str = Query(default=None, description="执行ID"),
    event_type: str = Query(default=None, description="事件类型"),
    level: str = Query(default=None, description="日志级别"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """查询审计日志"""
    logs = audit_logger.get_logs(
        execution_id=execution_id,
        event_type=event_type,
        level=level,
        limit=limit,
        offset=offset,
    )
    return {"logs": logs, "count": len(logs)}


@router.get(
    "/audit/snapshots",
    summary="获取执行快照",
    description="获取所有执行快照",
)
async def get_snapshots():
    """获取所有执行快照"""
    snapshots = audit_logger.get_all_snapshots()
    return {"snapshots": snapshots, "count": len(snapshots)}


@router.get(
    "/audit/snapshots/{execution_id}",
    summary="获取执行详情",
    description="获取指定执行的完整快照",
)
async def get_snapshot(execution_id: str):
    """获取执行快照"""
    snapshot = audit_logger.get_snapshot(execution_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail=f"Execution '{execution_id}' not found")
    return snapshot


@router.get(
    "/audit/events",
    summary="查询追踪事件",
    description="查询系统调用追踪事件",
)
async def query_events(
    execution_id: str = Query(default=None, description="执行ID"),
    event_type: str = Query(default=None, description="事件类型"),
    severity: str = Query(default=None, description="严重程度"),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """查询追踪事件"""
    events = event_tracker.get_events(
        execution_id=execution_id,
        event_type=event_type,
        severity=severity,
        limit=limit,
    )
    return {"events": events, "count": len(events)}


@router.get(
    "/audit/alerts",
    summary="获取安全告警",
    description="获取所有安全告警",
)
async def get_alerts(limit: int = Query(default=100, ge=1, le=1000)):
    """获取安全告警"""
    alerts = event_tracker.get_alerts(limit=limit)
    return {"alerts": alerts, "count": len(alerts)}


@router.get(
    "/audit/reports/summary",
    summary="汇总报告",
    description="生成审计汇总报告",
)
async def summary_report():
    """生成汇总报告"""
    return reporter.generate_summary_report()


@router.get(
    "/audit/reports/security",
    summary="安全报告",
    description="生成安全专项报告",
)
async def security_report():
    """生成安全专项报告"""
    return reporter.generate_security_report()


@router.get(
    "/audit/reports/execution/{execution_id}",
    summary="执行报告",
    description="生成单次执行的详细报告",
)
async def execution_report(execution_id: str):
    """生成执行详细报告"""
    report = reporter.generate_execution_report(execution_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Execution '{execution_id}' not found")
    return report


@router.get(
    "/audit/stats",
    summary="审计统计",
    description="获取日志和事件统计",
)
async def audit_stats():
    """获取审计统计"""
    return {
        "log_stats": audit_logger.get_stats(),
        "event_stats": event_tracker.get_stats(),
    }


@router.delete(
    "/audit/logs",
    summary="清除审计日志",
    description="清除所有审计日志和事件",
)
async def clear_audit_logs():
    """清除审计日志"""
    audit_logger.clear()
    event_tracker.clear()
    return {"message": "Audit logs cleared"}