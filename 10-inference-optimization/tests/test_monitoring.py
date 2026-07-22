"""Monitoring 模块测试"""

import time
import pytest

from monitoring.metrics import (
    MetricsCollector, Metric, MetricType,
    MetricSample,
)
from monitoring.prometheus_exporter import PrometheusExporter
from monitoring.dashboard import DashboardAPI


class TestMetric:
    """测试单个指标"""

    def test_counter_inc(self):
        metric = Metric("test_counter", MetricType.COUNTER)
        metric.inc(5)
        assert metric.get_value() == 5
        metric.inc(3)
        assert metric.get_value() == 8

    def test_gauge_set(self):
        metric = Metric("test_gauge", MetricType.GAUGE)
        metric.set(42.5)
        assert metric.get_value() == 42.5
        metric.set(0)
        assert metric.get_value() == 0

    def test_gauge_inc_dec(self):
        metric = Metric("test_gauge", MetricType.GAUGE)
        metric.inc(10)
        assert metric.get_value() == 10
        metric.dec(3)
        assert metric.get_value() == 7

    def test_histogram_observe(self):
        metric = Metric("test_hist", MetricType.HISTOGRAM)
        for v in [0.01, 0.05, 0.1, 0.5, 1.0]:
            metric.observe(v)
        hist = metric.get_histogram()
        assert hist["count"] == 5
        assert abs(hist["sum"] - 1.66) < 0.001

    def test_histogram_buckets(self):
        metric = Metric("test_hist", MetricType.HISTOGRAM)
        metric.observe(0.02)  # <= 0.01? No. <= 0.05? Yes.
        hist = metric.get_histogram()
        bucket_counts = [b["count"] for b in hist["buckets"]]
        # 0.02 应该在 le=0.05 的桶中
        assert bucket_counts[1] == 1  # le=0.05

    def test_summary_percentile(self):
        metric = Metric("test_summary", MetricType.HISTOGRAM)
        for i in range(100):
            metric.observe(float(i))
        p50 = metric.get_summary_percentile(0.5)
        assert 45 <= p50 <= 55


class TestMetricsCollector:
    """测试指标收集器"""

    def test_create(self):
        collector = MetricsCollector()
        assert collector.get_metric("inference_requests_total") is not None

    def test_record_inference(self):
        collector = MetricsCollector()
        collector.record_inference(
            ttft=0.05,
            e2e_latency=1.0,
            tokens_generated=50,
            tokens_per_second=50.0,
        )
        assert collector.get_metric("inference_requests_total").get_value() == 1
        assert collector.get_metric("inference_tokens_generated_total").get_value() == 50

    def test_record_multiple(self):
        collector = MetricsCollector()
        for _ in range(10):
            collector.record_inference(0.05, 1.0, 50, 50.0)
        assert collector.get_metric("inference_requests_total").get_value() == 10

    def test_set_gpu_stats(self):
        collector = MetricsCollector()
        collector.set_gpu_stats(0.85, 16 * 1024**3)
        assert collector.get_metric("gpu_utilization_ratio").get_value() == 0.85
        assert collector.get_metric("gpu_memory_usage_bytes").get_value() == 16 * 1024**3

    def test_set_cache_utilization(self):
        collector = MetricsCollector()
        collector.set_cache_utilization(0.6)
        assert collector.get_metric("kv_cache_utilization_ratio").get_value() == 0.6

    def test_get_all_metrics(self):
        collector = MetricsCollector()
        collector.record_inference(0.05, 1.0, 50, 50.0)
        all_metrics = collector.get_all_metrics()
        assert "inference_requests_total" in all_metrics
        assert "inference_ttft_seconds" in all_metrics

    def test_get_summary(self):
        collector = MetricsCollector()
        collector.record_inference(0.05, 1.0, 50, 50.0)
        summary = collector.get_summary()
        assert "total_requests" in summary
        assert "current_tps" in summary
        assert "gpu_utilization" in summary

    def test_reset(self):
        collector = MetricsCollector()
        collector.record_inference(0.05, 1.0, 50, 50.0)
        collector.reset()
        assert collector.get_metric("inference_requests_total").get_value() == 0


class TestPrometheusExporter:
    """测试 Prometheus 导出器"""

    def test_generate_empty(self):
        collector = MetricsCollector()
        exporter = PrometheusExporter(collector)
        output = exporter.generate()
        assert "inference_requests_total" in output
        assert "# TYPE" in output

    def test_generate_with_data(self):
        collector = MetricsCollector()
        collector.record_inference(0.05, 1.0, 50, 50.0)
        exporter = PrometheusExporter(collector)
        output = exporter.generate()
        assert "inference_requests_total 1" in output
        assert "# TYPE inference_requests_total counter" in output

    def test_histogram_format(self):
        collector = MetricsCollector()
        collector.record_inference(0.05, 1.0, 50, 50.0)
        exporter = PrometheusExporter(collector)
        output = exporter.generate()
        assert "inference_ttft_seconds_bucket" in output
        assert "inference_ttft_seconds_count" in output
        assert "inference_ttft_seconds_sum" in output
        assert '+Inf' in output

    def test_scrape_output(self):
        collector = MetricsCollector()
        collector.record_inference(0.05, 1.0, 50, 50.0)
        exporter = PrometheusExporter(collector)
        output = exporter.generate_metrics_for_scrape()
        assert "llm_total_requests" in output
        assert "Prometheus Metrics" in output


class TestDashboardAPI:
    """测试 Dashboard API"""

    def test_get_realtime_stats(self):
        collector = MetricsCollector()
        collector.record_inference(0.05, 1.0, 50, 50.0)
        dashboard = DashboardAPI(collector)
        stats = dashboard.get_realtime_stats()
        assert "total_requests" in stats
        assert "uptime_seconds" in stats

    def test_get_metrics_overview(self):
        collector = MetricsCollector()
        collector.record_inference(0.05, 1.0, 50, 50.0)
        dashboard = DashboardAPI(collector)
        overview = dashboard.get_metrics_overview()
        assert "metrics" in overview
        assert "summary" in overview

    def test_get_gpu_report(self):
        collector = MetricsCollector()
        collector.set_gpu_stats(0.9, 20 * 1024**3)
        dashboard = DashboardAPI(collector)
        report = dashboard.get_gpu_report()
        assert report["gpu_utilization"] == 0.9
        assert report["gpu_memory_usage_gb"] > 0

    def test_get_inference_report(self):
        collector = MetricsCollector()
        collector.record_inference(0.05, 1.0, 50, 50.0)
        dashboard = DashboardAPI(collector)
        report = dashboard.get_inference_report()
        assert "performance" in report
        assert "avg_ttft_ms" in report["performance"]
