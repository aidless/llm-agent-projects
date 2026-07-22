"""时间窗口聚合 - 支持按 1m/5m/15m/1h 窗口聚合指标数据。"""

import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# 支持的窗口配置
WINDOW_CONFIGS: Dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
}


class TimeWindowAggregator:
    """按时间窗口聚合指标数据点。

    支持的聚合方式:
    - sum: 求和
    - avg: 平均值
    - min: 最小值
    - max: 最大值
    - count: 计数
    - last: 最后一个值

    用法::

        aggregator = TimeWindowAggregator()
        aggregator.add_point("request_duration_ms", 42.5, timestamp=...)
        aggregator.add_point("request_duration_ms", 100.0, timestamp=...)
        result = aggregator.aggregate("request_duration_ms", window="5m")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._points: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add_point(self, name: str, value: float, timestamp: Optional[float] = None) -> None:
        """添加数据点。"""
        point = {
            "value": value,
            "timestamp": timestamp or time.time(),
        }
        with self._lock:
            self._points[name].append(point)

    def add_points(self, name: str, points: List[Dict[str, Any]]) -> None:
        """批量添加数据点。"""
        with self._lock:
            self._points[name].extend(points)

    def aggregate(
        self,
        name: str,
        window: str = "5m",
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """对指定指标进行时间窗口聚合。

        Args:
            name: 指标名
            window: 时间窗口 (1m/5m/15m/1h)
            methods: 聚合方法列表，默认 ["count", "sum", "avg", "min", "max", "p50", "p95", "p99"]

        Returns:
            聚合结果字典
        """
        if window not in WINDOW_CONFIGS:
            raise ValueError(f"不支持的时间窗口: {window}，可选: {list(WINDOW_CONFIGS.keys())}")

        window_seconds = WINDOW_CONFIGS[window]
        methods = methods or ["count", "sum", "avg", "min", "max", "p50", "p95", "p99"]
        cutoff = time.time() - window_seconds

        with self._lock:
            points = [
                p for p in self._points.get(name, [])
                if p["timestamp"] >= cutoff
            ]

        if not points:
            return {
                "metric_name": name,
                "window": window,
                "window_seconds": window_seconds,
                "point_count": 0,
                "aggregations": {m: None for m in methods},
            }

        values = [p["value"] for p in points]
        sorted_values = sorted(values)

        result: Dict[str, Any] = {}
        for method in methods:
            if method == "count":
                result[method] = len(values)
            elif method == "sum":
                result[method] = sum(values)
            elif method == "avg":
                result[method] = sum(values) / len(values)
            elif method == "min":
                result[method] = min(values)
            elif method == "max":
                result[method] = max(values)
            elif method in ("p50", "p95", "p99"):
                p = int(method[1:])
                result[method] = self._percentile(sorted_values, p)
            else:
                result[method] = None

        return {
            "metric_name": name,
            "window": window,
            "window_seconds": window_seconds,
            "point_count": len(values),
            "aggregations": result,
        }

    def aggregate_all(self, window: str = "5m") -> List[Dict[str, Any]]:
        """对所有已注册指标进行时间窗口聚合。"""
        with self._lock:
            names = list(self._points.keys())
        return [self.aggregate(name, window) for name in names]

    @staticmethod
    def _percentile(sorted_values: List[float], p: float) -> float:
        if not sorted_values:
            return 0.0
        import math
        idx = (p / 100.0) * (len(sorted_values) - 1)
        lower = int(math.floor(idx))
        upper = min(lower + 1, len(sorted_values) - 1)
        frac = idx - lower
        return sorted_values[lower] + frac * (sorted_values[upper] - sorted_values[lower])

    def clear(self) -> None:
        with self._lock:
            self._points.clear()


# 全局单例
_global_aggregator: Optional[TimeWindowAggregator] = None


def get_aggregator() -> TimeWindowAggregator:
    """获取全局 TimeWindowAggregator 单例。"""
    global _global_aggregator
    if _global_aggregator is None:
        _global_aggregator = TimeWindowAggregator()
    return _global_aggregator


def reset_aggregator() -> None:
    """重置全局聚合器（仅用于测试）。"""
    global _global_aggregator
    if _global_aggregator is not None:
        _global_aggregator.clear()
    _global_aggregator = None