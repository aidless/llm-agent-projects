"""监控数据 API - 提供监控数据查询接口"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .metrics import MetricsCollector


@dataclass
class TimeSeriesPoint:
    """时序数据点"""
    timestamp: float
    value: float


@dataclass
class DashboardConfig:
    """Dashboard 配置"""
    refresh_interval_ms: int = 1000
    history_window_seconds: int = 3600
    max_points: int = 500


class DashboardAPI:
    """监控数据查询 API"""

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        config: Optional[DashboardConfig] = None,
    ):
        self.metrics = metrics_collector
        self.config = config or DashboardConfig()

        # 历史数据存储
        self._history: Dict[str, List[TimeSeriesPoint]] = {}
        self._start_time = time.time()
        self._sample_interval = 1.0  # 每秒采样
        self._last_sample_time = 0.0

    def sample(self) -> None:
        """采样当前指标"""
        now = time.time()
        if now - self._last_sample_time < self._sample_interval:
            return
        self._last_sample_time = now

        all_metrics = self.metrics.get_all_metrics()
        for name, data in all_metrics.items():
            if name not in self._history:
                self._history[name] = []
            self._history[name].append(TimeSeriesPoint(
                timestamp=now,
                value=data["value"],
            ))
            # 限制历史长度
            if len(self._history[name]) > self.config.max_points:
                self._history[name] = self._history[name][-self.config.max_points:]

    def get_realtime_stats(self) -> Dict[str, Any]:
        """获取实时统计"""
        self.sample()
        summary = self.metrics.get_summary()
        return {
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self._start_time,
            **summary,
        }

    def get_metrics_overview(self) -> Dict[str, Any]:
        """获取指标概览"""
        return {
            "timestamp": time.time(),
            "metrics": self.metrics.get_all_metrics(),
            "summary": self.metrics.get_summary(),
        }

    def get_time_series(
        self, metric_name: str, last_n_seconds: int = 60
    ) -> List[Dict[str, float]]:
        """获取指标的时序数据"""
        self.sample()
        history = self._history.get(metric_name, [])
        cutoff = time.time() - last_n_seconds
        recent = [p for p in history if p.timestamp >= cutoff]

        return [
            {"timestamp": p.timestamp, "value": p.value}
            for p in recent
        ]

    def get_gpu_report(self) -> Dict[str, Any]:
        """获取 GPU 报告"""
        self.sample()
        return {
            "timestamp": time.time(),
            "gpu_utilization": self.metrics.get_metric(
                "gpu_utilization_ratio"
            ).get_value(),
            "gpu_memory_usage_gb": self.metrics.get_metric(
                "gpu_memory_usage_bytes"
            ).get_value() / (1024 ** 3),
            "kv_cache_utilization": self.metrics.get_metric(
                "kv_cache_utilization_ratio"
            ).get_value(),
            "active_requests": self.metrics.get_metric(
                "active_requests_count"
            ).get_value(),
        }

    def get_inference_report(self) -> Dict[str, Any]:
        """获取推理性能报告"""
        self.sample()
        summary = self.metrics.get_summary()
        return {
            "timestamp": time.time(),
            "total_requests": summary["total_requests"],
            "total_tokens": summary["total_tokens_generated"],
            "performance": {
                "avg_ttft_ms": summary["avg_ttft_ms"],
                "p95_ttft_ms": summary["p95_ttft_ms"],
                "p99_ttft_ms": summary["p99_ttft_ms"],
                "avg_e2e_ms": summary["avg_e2e_latency_ms"],
                "p95_e2e_ms": summary["p95_e2e_latency_ms"],
                "current_tps": summary["current_tps"],
            },
        }
