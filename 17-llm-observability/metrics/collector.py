"""指标收集器 - Counter / Histogram / Gauge 三种指标类型的实现。"""

import math
import threading
import time
from typing import Any, Dict, List, Optional

from storage.memory_store import get_store


class Counter:
    """计数器 - 只增不减的累计指标。

    用法::

        counter = Counter("requests_total", "Total request count")
        counter.increment()
        counter.increment(5)
    """

    def __init__(
        self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None
    ) -> None:
        self.name = name
        self.description = description
        self.tags = tags or {}
        self._value: int = 0
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> "Counter":
        """增加计数。"""
        with self._lock:
            self._value += amount
        get_store().record_metric(self.name, self._value, self.tags)
        return self

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "counter",
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "value": self.value,
        }


class Gauge:
    """仪表盘 - 可增可减的瞬时值。

    用法::

        gauge = Gauge("active_connections", "Current active connections")
        gauge.set(10)
        gauge.increment(3)
        gauge.decrement(1)
    """

    def __init__(
        self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None
    ) -> None:
        self.name = name
        self.description = description
        self.tags = tags or {}
        self._value: float = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> "Gauge":
        """设置当前值。"""
        with self._lock:
            self._value = value
        get_store().record_metric(self.name, value, self.tags)
        return self

    def increment(self, amount: float = 1.0) -> "Gauge":
        """增加当前值。"""
        with self._lock:
            self._value += amount
        get_store().record_metric(self.name, self._value, self.tags)
        return self

    def decrement(self, amount: float = 1.0) -> "Gauge":
        """减少当前值。"""
        with self._lock:
            self._value -= amount
        get_store().record_metric(self.name, self._value, self.tags)
        return self

    @property
    def value(self) -> float:
        with self._lock:
            return self._value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "gauge",
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "value": self.value,
        }


class Histogram:
    """直方图 - 观测值分布，支持分位数计算。

    用法::

        histogram = Histogram("request_duration_ms", "Request latency")
        histogram.observe(42.5)
        histogram.observe(100.0)
        p50 = histogram.percentile(50)
    """

    # 默认延迟桶边界 (毫秒)
    DEFAULT_BUCKETS = [10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2500, 5000, 10000]

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.tags = tags or {}
        self._buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._bucket_counts: List[int] = [0] * (len(self._buckets) + 1)  # +inf
        self._sum: float = 0.0
        self._count: int = 0
        self._values: List[float] = []  # 保留所有值用于精确分位数
        self._lock = threading.Lock()

    def observe(self, value: float) -> "Histogram":
        """记录一个观测值。"""
        with self._lock:
            self._sum += value
            self._count += 1
            self._values.append(value)
            for i, boundary in enumerate(self._buckets):
                if value <= boundary:
                    self._bucket_counts[i] += 1
                    break
            else:
                self._bucket_counts[-1] += 1
        get_store().record_metric(self.name, value, self.tags)
        return self

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def sum(self) -> float:
        with self._lock:
            return self._sum

    @property
    def avg(self) -> float:
        with self._lock:
            return self._sum / self._count if self._count > 0 else 0.0

    def percentile(self, p: float) -> float:
        """计算分位数 (0-100)。"""
        with self._lock:
            if not self._values:
                return 0.0
            sorted_values = sorted(self._values)
            idx = (p / 100.0) * (len(sorted_values) - 1)
            lower = int(math.floor(idx))
            upper = min(lower + 1, len(sorted_values) - 1)
            frac = idx - lower
            return sorted_values[lower] + frac * (sorted_values[upper] - sorted_values[lower])

    def p50(self) -> float:
        return self.percentile(50)

    def p95(self) -> float:
        return self.percentile(95)

    def p99(self) -> float:
        return self.percentile(99)

    @property
    def min(self) -> float:
        with self._lock:
            return min(self._values) if self._values else 0.0

    @property
    def max(self) -> float:
        with self._lock:
            return max(self._values) if self._values else 0.0

    @property
    def bucket_counts(self) -> List[int]:
        with self._lock:
            return list(self._bucket_counts)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "type": "histogram",
                "name": self.name,
                "description": self.description,
                "tags": self.tags,
                "count": self._count,
                "sum": self._sum,
                "avg": self._sum / self._count if self._count > 0 else 0.0,
                "min": min(self._values) if self._values else 0.0,
                "max": max(self._values) if self._values else 0.0,
                "p50": self.percentile(50) if self._values else 0.0,
                "p95": self.percentile(95) if self._values else 0.0,
                "p99": self.percentile(99) if self._values else 0.0,
                "buckets": {
                    str(b): c for b, c in zip(self._buckets + ["+Inf"], self._bucket_counts)
                },
            }