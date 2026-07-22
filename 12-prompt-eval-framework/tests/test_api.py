"""API 测试。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthAPI:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Prompt 评估与优化框架"

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestEvaluationAPI:
    def test_list_metrics(self, client):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["metrics"]) >= 5

    def test_evaluate(self, client):
        resp = client.post("/api/v1/evaluate", json={
            "predictions": ["hello world", "foo bar"],
            "references": ["hello world", "foo bar"],
            "metric_names": ["exact_match", "f1"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_samples"] == 2
        assert "results" in data

    def test_judge_dimensions(self, client):
        resp = client.get("/api/v1/judge/dimensions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["dimensions"]) == 6

    def test_judge_single(self, client):
        resp = client.post("/api/v1/judge", json={
            "questions": ["What is AI?"],
            "predictions": ["AI is artificial intelligence."],
            "references": ["AI stands for artificial intelligence."],
            "dimensions": ["accuracy"],
            "num_judges": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_samples"] == 1

    def test_report_json(self, client):
        eval_result = {
            "num_samples": 2,
            "metrics_used": ["exact_match"],
            "results": {"exact_match": {"exact_match": 0.5, "exact_match_count": 1, "total_count": 2}},
            "elapsed_seconds": 0.1,
        }
        resp = client.post("/api/v1/reports/generate", json={
            "evaluation_result": eval_result,
            "format": "json",
        })
        assert resp.status_code == 200
        assert resp.json()["format"] == "json"

    def test_report_markdown(self, client):
        eval_result = {
            "num_samples": 2,
            "metrics_used": ["exact_match"],
            "results": {"exact_match": {"exact_match": 0.5, "exact_match_count": 1, "total_count": 2}},
            "elapsed_seconds": 0.1,
        }
        resp = client.post("/api/v1/reports/generate", json={
            "evaluation_result": eval_result,
            "format": "markdown",
        })
        assert resp.status_code == 200
        assert "# Prompt" in resp.json()["report"]

    def test_prompt_version_crud(self, client):
        # Create
        resp = client.post("/api/v1/prompts/versions", json={
            "version_id": "v1",
            "name": "Version 1",
            "prompt_template": "You are a helpful assistant. {input}",
        })
        assert resp.status_code == 200

        # List
        resp = client.get("/api/v1/prompts/versions")
        assert resp.status_code == 200
        assert len(resp.json()["versions"]) == 1

        # Delete
        resp = client.delete("/api/v1/prompts/versions/v1")
        assert resp.status_code == 200

    def test_prompt_analyze(self, client):
        resp = client.post("/api/v1/prompts/analyze", json={
            "prompt": "please can you help me?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "clarity_score" in data
        assert "suggestions" in data

    def test_template_parameterize(self, client):
        resp = client.post("/api/v1/prompts/template/parameterize", json={
            "template": "Translate {text} from {source_lang} to {target_lang}",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["parameterized"] is True
        assert len(data["variables"]) == 3