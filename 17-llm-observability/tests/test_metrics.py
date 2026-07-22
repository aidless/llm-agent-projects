"""Metrics 模块测试 - Counter/Histogram/Gauge/Registry/Aggregation。"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from metrics.collector import Counter, Gauge, Histogram
from metrics.registry import MetricRegistry
from metrics.aggregation import TimeWindowAggregator


class TestCounter:
    def test_increment(self):
        c = Counter("test_counter")
        c.increment()
        c.increment(5)
        assert c.value == 6

    def test_to_dict(self):
        c = Counter("test_counter", description="A test counter")
        c.increment(3)
        d = c.to_dict()
        assert d["type"] == "counter"
        assert d["value"] == 3
        assert d["description"] == "A test counter"


class TestGauge:
    def test_set(self):
        g = Gauge("test_gauge")
        g.set(42.5)
        assert g.value == 42.5

    def test_increment_decrement(self):
        g = Gauge("test_gauge")
        g.set(10)
        g.increment(3)
        g.decrement(2)
        assert g.value == 11.0

    def test_to_dict(self):
        g = Gauge("test_gauge", tags={"service": "api"})
        g.set(5)
        d = g.to_dict()
        assert d["type"] == "gauge"
        assert d["value"] == 5
        assert d["tags"]["service"] == "api"


class TestHistogram:
    def test_observe(self):
        h = Histogram("test_hist")
        h.observe(10)
        h.observe(20)
        h.observe(30)
        assert h.count == 3
        assert h.sum == 60.0
        assert h.avg == 20.0

    def test_percentiles(self):
        h = Histogram("test_hist")
        values = list(range(1, 101))  # 1 to 100
        for v in values:
            h.observe(v)
        assert h.p50() == 50.5
        assert h.p95() == pytest.approx(95.05, abs=0.1)
        assert h.p99() == pytest.approx(99.05, abs=0.1)
        assert h.min == 1
        assert h.max == 100

    def test_empty_histogram(self):
        h = Histogram("empty")
        assert h.count == 0
        assert h.sum == 0
        assert h.avg == 0
        assert h.p50() == 0

    def test_bucket_counts(self):
        h = Histogram("test_hist", buckets=[10, 50, 100])
        h.observe(5)    # <=10
        h.observe(15)   # <=50
        h.observe(200)  # +inf
        bc = h.bucket_counts
        assert bc[0] == 1  # <=10
        assert bc[1] == 1  # <=50
        assert bc[2] == 0  # <=100
        assert bc[3] == 1  # +inf

    def test_to_dict(self):
        h = Histogram("test_hist")
        h.observe(100)
        d = h.to_dict()
        assert d["type"] == "histogram"
        assert d["count"] == 1
        assert "p50" in d
        assert "p95" in d
        assert "p99" in d


class TestRegistry:
    def test_register_counter(self):
        reg = MetricRegistry()
        c = reg.counter("my_counter", "Test counter")
        c.increment()
        assert c.value == 1
        # 同名返回同一实例
        c2 = reg.counter("my_counter")
        assert c2 is c

    def test_register_histogram(self):
        reg = MetricRegistry()
        h = reg.histogram("my_hist", "Test histogram")
        h.observe(42)
        assert h.count == 1

    def test_register_gauge(self):
        reg = MetricRegistry()
        g = reg.gauge("my_gauge", "Test gauge")
        g.set(100)
        assert g.value == 100

    def test_list_metrics(self):
        reg = MetricRegistry()
        reg.counter("c1")
        reg.histogram("h1")
        reg.gauge("g1")
        metrics = reg.list_metrics()
        assert len(metrics) == 3
        types = {m["type"] for m in metrics}
        assert types == {"counter", "histogram", "gauge"}

    def test_clear(self):
        reg = MetricRegistry()
        reg.counter("c1").increment()
        reg.clear()
        assert reg.list_metrics() == []


class TestTimeWindowAggregator:
    def test_basic_aggregation(self):
        agg = TimeWindowAggregator()
        for i in range(10):
            agg.add_point("latency", float(i * 10))
        result = agg.aggregate("latency", window="5m")
        assert result["point_count"] == 10
        assert result["aggregations"]["count"] == 10
        assert result["aggregations"]["sum"] == 450.0
        assert result["aggregations"]["avg"] == 45.0

    def test_invalid_window(self):
        agg = TimeWindowAggregator()
        with pytest.raises(ValueError):
            agg.aggregate("test", window="10m")

    def test_aggregate_all(self):
        agg = TimeWindowAggregator()
        agg.add_point("metric_a", 1.0)
        agg.add_point("metric_b", 2.0)
        results = agg.aggregate_all("1h")
        assert len(results) == 2
        names = {r["metric_name"] for r in results}
        assert names == {"metric_a", "metric_b"}