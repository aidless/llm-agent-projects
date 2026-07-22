"""API 集成测试 - FastAPI 路由端到端测试。"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthAPI:
    def test_root(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "LLM Observability Platform"
        assert data["status"] == "running"

    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestTraceAPI:
    def test_list_traces_empty(self):
        resp = client.get("/api/v1/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["traces"] == []

    def test_get_trace_not_found(self):
        resp = client.get("/api/v1/traces/nonexistent")
        assert resp.status_code == 404

    def test_list_traces_after_create(self):
        from tracing.tracer import Tracer
        from tracing.span import SpanKind
        tracer = Tracer("test-service")
        with tracer.start_span("api-test-op", kind=SpanKind.LLM):
            pass

        resp = client.get("/api/v1/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["traces"]) == 1
        assert data["traces"][0]["service_name"] == "test-service"

    def test_trace_filter_by_span_type(self):
        from tracing.tracer import Tracer
        from tracing.span import SpanKind
        tracer = Tracer("test-service")
        with tracer.start_span("llm-op", kind=SpanKind.LLM):
            pass
        with tracer.start_span("internal-op", kind=SpanKind.INTERNAL):
            pass

        resp = client.get("/api/v1/traces", params={"span_type": "llm"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1

    def test_trace_detail(self):
        from tracing.tracer import Tracer
        tracer = Tracer("test-service")
        with tracer.start_span("detail-test") as span:
            pass

        resp = client.get(f"/api/v1/traces/{span.trace_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trace"]["trace_id"] == span.trace_id

    def test_trace_spans(self):
        from tracing.tracer import Tracer
        tracer = Tracer("test-service")
        with tracer.start_span("parent") as parent:
            with tracer.start_span("child"):
                pass

        resp = client.get(f"/api/v1/traces/{parent.trace_id}/spans")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_latency_analysis(self):
        from tracing.tracer import Tracer
        from tracing.span import SpanKind
        tracer = Tracer("test-service")
        with tracer.start_span("llm-op", kind=SpanKind.LLM):
            pass

        resp = client.get("/api/v1/traces/analysis/latency", params={"span_type": "llm"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1

    def test_error_analysis(self):
        resp = client.get("/api/v1/traces/analysis/errors")
        assert resp.status_code == 200
        data = resp.json()
        assert "error_rate" in data
        assert "error_breakdown" in data

    def test_cost_analysis(self):
        resp = client.get("/api/v1/traces/analysis/cost")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_cost_usd" in data
        assert "by_model" in data

    def test_trends(self):
        resp = client.get("/api/v1/traces/analysis/trends", params={"window": "1h"})
        assert resp.status_code == 200
        data = resp.json()
        assert "window" in data


class TestMetricsAPI:
    def test_list_metrics_empty(self):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["metrics"] == []

    def test_list_metrics_after_record(self):
        from metrics.registry import get_registry
        reg = get_registry()
        reg.counter("test_counter", "Test").increment()

        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["metrics"]) >= 1

    def test_get_metric_not_found(self):
        resp = client.get("/api/v1/metrics/nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "nonexistent"
        assert data["points"] == []

    def test_aggregate_metrics(self):
        resp = client.get("/api/v1/metrics/aggregate/5m")
        assert resp.status_code == 200
        data = resp.json()
        assert data["window"] == "5m"

    def test_aggregate_invalid_window(self):
        resp = client.get("/api/v1/metrics/aggregate/10m")
        assert resp.status_code == 400


class TestLogsAPI:
    def test_query_logs_empty(self):
        resp = client.get("/api/v1/logs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_query_logs_after_log(self):
        from obs_logging.logger import LLMObservabilityLogger
        logger = LLMObservabilityLogger("api-test")
        logger.info("API test log message")

        resp = client.get("/api/v1/logs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    def test_query_logs_by_level(self):
        from obs_logging.logger import LLMObservabilityLogger
        logger = LLMObservabilityLogger("api-test-level")
        logger.error("Error log")

        resp = client.get("/api/v1/logs", params={"level": "ERROR"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert all(l["level"] == "ERROR" for l in data["logs"])

    def test_query_logs_by_message(self):
        from obs_logging.logger import LLMObservabilityLogger
        logger = LLMObservabilityLogger("api-test-msg")
        logger.info("UNIQUE_MARKER_12345")

        resp = client.get("/api/v1/logs", params={"message": "UNIQUE_MARKER_12345"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1


class TestFeedbackAPI:
    def test_create_thumbs_up(self):
        # 先创建一个 trace
        from tracing.tracer import Tracer
        tracer = Tracer("test-service")
        with tracer.start_span("fb-test") as span:
            pass

        resp = client.post("/api/v1/feedback", json={
            "trace_id": span.trace_id,
            "feedback_type": "thumbs_up",
            "comment": "Great!",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["feedback_type"] == "thumbs_up"
        assert data["trace_id"] == span.trace_id

    def test_create_rating(self):
        from tracing.tracer import Tracer
        tracer = Tracer("test-service")
        with tracer.start_span("rating-test") as span:
            pass

        resp = client.post("/api/v1/feedback", json={
            "trace_id": span.trace_id,
            "feedback_type": "rating",
            "value": 4.5,
            "comment": "Good",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["value"] == 4.5

    def test_list_feedbacks(self):
        resp = client.get("/api/v1/feedback")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "feedbacks" in data

    def test_feedback_stats(self):
        resp = client.get("/api/v1/feedback/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "thumbs_up" in data
        assert "error_rate" not in data  # stats 不包含 error_rate

    def test_export_finetuning(self):
        resp = client.get("/api/v1/feedback/export/finetuning")
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert "data" in data