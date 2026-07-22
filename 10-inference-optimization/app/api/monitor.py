"""监控 API - 性能监控指标接口"""

from __future__ import annotations

from fastapi import APIRouter, Response

from ...monitoring.metrics import MetricsCollector
from ...monitoring.prometheus_exporter import PrometheusExporter
from ...monitoring.dashboard import DashboardAPI

router = APIRouter(prefix="/v1", tags=["monitoring"])

metrics: MetricsCollector = None  # type: ignore
exporter: PrometheusExporter = None  # type: ignore
dashboard: DashboardAPI = None  # type: ignore


def init_monitoring(
    met: MetricsCollector,
    exp: PrometheusExporter,
    dash: DashboardAPI,
) -> None:
    global metrics, exporter, dashboard
    metrics = met
    exporter = exp
    dashboard = dash


@router.get("/metrics")
async def get_metrics():
    """获取所有指标"""
    if metrics is None:
        return {"error": "Metrics not initialized"}
    return metrics.get_all_metrics()


@router.get("/metrics/prometheus")
async def get_prometheus_metrics(response: Response):
    """Prometheus 格式指标 (text/plain)"""
    if exporter is None:
        response.status_code = 503
        return "Metrics not initialized"
    response.headers["Content-Type"] = "text/plain; version=0.0.4; charset=utf-8"
    return exporter.generate_metrics_for_scrape()


@router.get("/metrics/summary")
async def get_summary():
    """获取指标摘要"""
    if metrics is None:
        return {"error": "Metrics not initialized"}
    return metrics.get_summary()


@router.get("/dashboard/realtime")
async def get_realtime_stats():
    """获取实时统计"""
    if dashboard is None:
        return {"error": "Dashboard not initialized"}
    return dashboard.get_realtime_stats()


@router.get("/dashboard/gpu")
async def get_gpu_report():
    """获取 GPU 报告"""
    if dashboard is None:
        return {"error": "Dashboard not initialized"}
    return dashboard.get_gpu_report()


@router.get("/dashboard/inference")
async def get_inference_report():
    """获取推理报告"""
    if dashboard is None:
        return {"error": "Dashboard not initialized"}
    return dashboard.get_inference_report()


@router.get("/dashboard/overview")
async def get_overview():
    """获取指标概览"""
    if dashboard is None:
        return {"error": "Dashboard not initialized"}
    return dashboard.get_metrics_overview()
