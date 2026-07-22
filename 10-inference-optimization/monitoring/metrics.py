"""指标收集器 - 收集和聚合性能指标"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Any


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricSample:
    """指标样本"""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class HistogramBucket:
    """直方图桶"""
    le: float  # upper bound (inclusive)
    count: int = 0


class Metric:
    """单个指标"""

    def __init__(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.metric_type = metric_type
        self.description = description
        self.label_names = labels or []

        # Counter/Gauge: 简单值
        self._value: float = 0.0

        # Histogram: 桶
        self._buckets: List[HistogramBucket] = []
        # 用于精确求和
        self._observations: List[float] = []
        self._count: int = 0

        # Summary: 分位数
        self._samples: deque = deque(maxlen=10000)

        # 时间序列 (用于趋势)
        self._history: deque = deque(maxlen=1000)

        if metric_type == MetricType.HISTOGRAM:
            # 默认桶: 10ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s
            default_bounds = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
            self._buckets = [HistogramBucket(le=le) for le in default_bounds]

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """增加计数"""
        self._value += value
        self._record(value, labels)

    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """减少计数"""
        self._value -= value
        self._record(value, labels)

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """设置值"""
        self._value = value
        self._record(value, labels)

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """观察值 (histogram/summary)"""
        self._observations.append(value)
        self._count += 1
        self._samples.append(value)

        # 更新桶
        for bucket in self._buckets:
            if value <= bucket.le + 1e-9:
                bucket.count += 1

        self._record(value, labels)

    def _record(self, value: float, labels: Optional[Dict[str, str]]) -> None:
        self._history.append(MetricSample(
            name=self.name,
            value=value,
            timestamp=time.time(),
            labels=labels or {},
        ))

    def get_value(self) -> float:
        return self._value

    def get_histogram(self) -> Dict[str, Any]:
        return {
            "count": self._count,
            "sum": round(sum(self._observations), 10) if self._observations else 0.0,
            "buckets": [
                {"le": b.le, "count": b.count}
                for b in self._buckets
            ],
        }

    def get_summary_percentile(self, percentile: float) -> float:
        """获取分位数"""
        if not self._samples:
            return 0.0
        sorted_samples = sorted(self._samples)
        idx = int(len(sorted_samples) * percentile)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._register_default_metrics()

    def _register_default_metrics(self) -> None:
        """注册默认指标"""
        # 推理指标
        self.create_counter(
            "inference_requests_total",
            "Total number of inference requests",
        )
        self.create_counter(
            "inference_tokens_generated_total",
            "Total number of tokens generated",
        )
        self.create_histogram(
            "inference_ttft_seconds",
            "Time to first token in seconds",
        )
        self.create_histogram(
            "inference_e2e_latency_seconds",
            "End-to-end inference latency in seconds",
        )
        self.create_gauge(
            "inference_tokens_per_second",
            "Current tokens per second generation rate",
        )

        # 系统指标
        self.create_gauge(
            "gpu_utilization_ratio",
            "Simulated GPU utilization ratio",
        )
        self.create_gauge(
            "gpu_memory_usage_bytes",
            "Simulated GPU memory usage in bytes",
        )
        self.create_gauge(
            "kv_cache_utilization_ratio",
            "KV Cache utilization ratio",
        )
        self.create_gauge(
            "active_requests_count",
            "Number of active inference requests",
        )

        # 批处理指标
        self.create_histogram(
            "batch_size",
            "Size of inference batches",
        )
        self.create_gauge(
            "batch_queue_length",
            "Current batch queue length",
        )

    def create_counter(self, name: str, description: str = "") -> Metric:
        metric = Metric(name, MetricType.COUNTER, description)
        self._metrics[name] = metric
        return metric

    def create_gauge(self, name: str, description: str = "") -> Metric:
        metric = Metric(name, MetricType.GAUGE, description)
        self._metrics[name] = metric
        return metric

    def create_histogram(
        self, name: str, description: str = ""
    ) -> Metric:
        metric = Metric(name, MetricType.HISTOGRAM, description)
        self._metrics[name] = metric
        return metric

    def get_metric(self, name: str) -> Optional[Metric]:
        return self._metrics.get(name)

    def record_inference(
        self,
        ttft: float,
        e2e_latency: float,
        tokens_generated: int,
        tokens_per_second: float,
    ) -> None:
        """记录一次推理"""
        self.get_metric("inference_requests_total").inc()
        self.get_metric("inference_tokens_generated_total").inc(tokens_generated)
        self.get_metric("inference_ttft_seconds").observe(ttft)
        self.get_metric("inference_e2e_latency_seconds").observe(e2e_latency)
        self.get_metric("inference_tokens_per_second").set(tokens_per_second)

    def set_gpu_stats(
        self,
        utilization: float,
        memory_bytes: int,
    ) -> None:
        """设置 GPU 统计"""
        self.get_metric("gpu_utilization_ratio").set(utilization)
        self.get_metric("gpu_memory_usage_bytes").set(memory_bytes)

    def set_cache_utilization(self, ratio: float) -> None:
        """设置缓存利用率"""
        self.get_metric("kv_cache_utilization_ratio").set(ratio)

    def get_all_metrics(self) -> Dict[str, Dict]:
        """获取所有指标"""
        result = {}
        for name, metric in self._metrics.items():
            data = {
                "type": metric.metric_type.value,
                "value": metric.get_value(),
            }
            if metric.metric_type == MetricType.HISTOGRAM:
                data["histogram"] = metric.get_histogram()
            result[name] = data
        return result

    def get_summary(self) -> Dict[str, Any]:
        """获取汇总统计"""
        total_requests = self.get_metric("inference_requests_total").get_value()
        total_tokens = self.get_metric("inference_tokens_generated_total").get_value()
        ttft_metric = self.get_metric("inference_ttft_seconds")
        e2e_metric = self.get_metric("inference_e2e_latency_seconds")

        return {
            "total_requests": total_requests,
            "total_tokens_generated": total_tokens,
            "avg_ttft_ms": ttft_metric.get_summary_percentile(0.5) * 1000,
            "p95_ttft_ms": ttft_metric.get_summary_percentile(0.95) * 1000,
            "p99_ttft_ms": ttft_metric.get_summary_percentile(0.99) * 1000,
            "avg_e2e_latency_ms": e2e_metric.get_summary_percentile(0.5) * 1000,
            "p95_e2e_latency_ms": e2e_metric.get_summary_percentile(0.95) * 1000,
            "current_tps": self.get_metric("inference_tokens_per_second").get_value(),
            "gpu_utilization": self.get_metric("gpu_utilization_ratio").get_value(),
            "gpu_memory_gb": self.get_metric("gpu_memory_usage_bytes").get_value() / (1024 ** 3),
            "kv_cache_utilization": self.get_metric("kv_cache_utilization_ratio").get_value(),
        }

    def reset(self) -> None:
        """重置所有指标"""
        self._metrics.clear()
        self._register_default_metrics()
