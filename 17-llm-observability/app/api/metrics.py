"""指标查询 API - 提供指标列表、详情和聚合查询接口。"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.models import (
    AggregationResponse,
    MetricDataPoint,
    MetricDetailResponse,
    MetricInfo,
    MetricListResponse,
    MetricTypeEnum,
)
from metrics.aggregation import get_aggregator, WINDOW_CONFIGS
from metrics.registry import get_registry
from storage.memory_store import get_store

router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])


@router.get("", response_model=MetricListResponse)
def list_metrics():
    """列出所有已注册的指标。"""
    registry = get_registry()
    items = registry.list_metrics()
    metrics = [
        MetricInfo(
            name=m["name"],
            type=MetricTypeEnum(m["type"]),
            description=m["description"],
        )
        for m in items
    ]
    return MetricListResponse(metrics=metrics)


@router.get("/{metric_name}", response_model=MetricDetailResponse)
def get_metric(
    metric_name: str,
    window: Optional[str] = Query(None, description="时间窗口: 1m/5m/15m/1h"),
    limit: int = Query(100, ge=1, le=1000),
):
    """获取指定指标的详细数据。"""
    store = get_store()
    window_seconds = None
    if window:
        if window not in WINDOW_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail=f"无效的时间窗口: {window}，可选: {list(WINDOW_CONFIGS.keys())}",
            )
        window_seconds = WINDOW_CONFIGS[window]

    points = store.get_metrics(metric_name, window_seconds=window_seconds)
    data_points = [
        MetricDataPoint(value=p["value"], timestamp=p["timestamp"], tags=p.get("tags", {}))
        for p in points[-limit:]
    ]

    # 尝试获取 summary (来自 Histogram)
    summary = None
    registry = get_registry()
    all_metrics = registry.get_all_metrics()
    if metric_name in all_metrics:
        m = all_metrics[metric_name]
        if hasattr(m, "to_dict"):
            summary = m.to_dict()

    return MetricDetailResponse(
        name=metric_name,
        points=data_points,
        summary=summary,
    )


@router.get("/aggregate/{window}", response_model=AggregationResponse)
def aggregate_metrics(
    window: str = "5m",
    metric_names: Optional[str] = Query(None, description="逗号分隔的指标名，为空则聚合全部"),
):
    """按时间窗口聚合指标。"""
    if window not in WINDOW_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"无效的时间窗口: {window}，可选: {list(WINDOW_CONFIGS.keys())}",
        )

    aggregator = get_aggregator()

    if metric_names:
        names = [n.strip() for n in metric_names.split(",")]
        result = [aggregator.aggregate(name, window) for name in names]
    else:
        result = aggregator.aggregate_all(window)

    return AggregationResponse(aggregations=result, window=window)