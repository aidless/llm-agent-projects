"""指标注册中心 - 管理 Counter/Histogram/Gauge 的注册和获取。"""

import threading
from typing import Dict, Optional

from metrics.collector import Counter, Gauge, Histogram


class MetricRegistry:
    """线程安全的指标注册中心。

    用法::

        registry = MetricRegistry()
        counter = registry.counter("requests_total", "Total requests")
        counter.increment()
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, Counter] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._gauges: Dict[str, Gauge] = {}

    def counter(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None) -> Counter:
        """获取或创建 Counter。"""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(
                    name=name, description=description, tags=tags or {}
                )
            return self._counters[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: Optional[list] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Histogram:
        """获取或创建 Histogram。"""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(
                    name=name,
                    description=description,
                    buckets=buckets or [],
                    tags=tags or {},
                )
            return self._histograms[name]

    def gauge(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None) -> Gauge:
        """获取或创建 Gauge。"""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(
                    name=name, description=description, tags=tags or {}
                )
            return self._gauges[name]

    def get_all_metrics(self) -> Dict[str, object]:
        """返回所有已注册的指标。"""
        with self._lock:
            all_metrics: Dict[str, object] = {}
            all_metrics.update(self._counters)
            all_metrics.update(self._histograms)
            all_metrics.update(self._gauges)
            return all_metrics

    def list_metrics(self) -> list:
        """列出所有指标名称和类型。"""
        with self._lock:
            result = []
            for c in self._counters.values():
                result.append({"name": c.name, "type": "counter", "description": c.description})
            for h in self._histograms.values():
                result.append({"name": h.name, "type": "histogram", "description": h.description})
            for g in self._gauges.values():
                result.append({"name": g.name, "type": "gauge", "description": g.description})
            return result

    def clear(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()


# 全局单例
_global_registry: Optional[MetricRegistry] = None


def get_registry() -> MetricRegistry:
    """获取全局 MetricRegistry 单例。"""
    global _global_registry
    if _global_registry is None:
        _global_registry = MetricRegistry()
    return _global_registry


def reset_registry() -> None:
    """重置全局注册中心（仅用于测试）。"""
    global _global_registry
    if _global_registry is not None:
        _global_registry.clear()
    _global_registry = None