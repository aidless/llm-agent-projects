"""
tests/test_api.py - FastAPI 接口测试

测试所有 API 端点。
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestHealthEndpoints:
    """健康检查端点测试."""

    def test_health_root(self):
        """测试根路径健康检查."""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "mmlu" in data["benchmarks_available"]

    def test_health_endpoint(self):
        """测试 /health 端点."""
        resp = client.get("/health")
        assert resp.status_code == 200


class TestBenchmarkAPI:
    """评测 API 测试."""

    def test_list_benchmarks(self):
        """测试列出基准."""
        resp = client.get("/api/benchmark/list")
        assert resp.status_code == 200
        data = resp.json()
        assert "benchmarks" in data
        assert len(data["benchmarks"]) >= 4

    def test_benchmark_info(self):
        """测试基准信息."""
        resp = client.get("/api/benchmark/mmlu/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "mmlu"
        assert data["total_questions"] == 50

    def test_benchmark_info_not_found(self):
        """测试不存在的基准."""
        resp = client.get("/api/benchmark/nonexistent/info")
        assert resp.status_code == 404

    def test_evaluate_model(self):
        """测试执行评测."""
        resp = client.post("/api/benchmark/evaluate", json={
            "model_name": "test-model",
            "benchmark": "mmlu",
            "max_concurrent": 5,
            "question_limit": 5,
            "enable_cache": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "test-model"
        assert data["benchmark"] == "mmlu"
        assert data["total_questions"] == 5
        assert "scores" in data
        assert "results" in data

    def test_evaluate_unknown_benchmark(self):
        """测试评测未知基准."""
        resp = client.post("/api/benchmark/evaluate", json={
            "model_name": "test-model",
            "benchmark": "nonexistent",
        })
        assert resp.status_code == 404

    def test_evaluate_gsm8k(self):
        """测试 GSM8K 评测."""
        resp = client.post("/api/benchmark/evaluate", json={
            "model_name": "test-model-gsm8k",
            "benchmark": "gsm8k",
            "question_limit": 3,
            "enable_cache": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_questions"] == 3

    def test_get_progress(self):
        """测试获取进度."""
        resp = client.get("/api/benchmark/progress/test-model")
        assert resp.status_code == 200

    def test_get_results(self):
        """测试获取结果 (先执行评测)."""
        # 先执行评测
        client.post("/api/benchmark/evaluate", json={
            "model_name": "results-test-model",
            "benchmark": "mmlu",
            "question_limit": 3,
            "enable_cache": False,
        })
        # 获取结果
        resp = client.get("/api/benchmark/results/results-test-model/mmlu")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3


class TestModelAPI:
    """模型管理 API 测试."""

    def test_register_model(self):
        """测试注册模型."""
        resp = client.post("/api/models/register", json={
            "name": "test-gpt4",
            "api_endpoint": "https://api.openai.com/v1",
            "parameters": "1.8T",
            "provider": "openai",
            "description": "GPT-4",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-gpt4"
        assert data["latest_version"] == "v1"

    def test_register_duplicate(self):
        """测试重复注册."""
        client.post("/api/models/register", json={"name": "dup-model"})
        resp = client.post("/api/models/register", json={"name": "dup-model"})
        assert resp.status_code == 409

    def test_list_models(self):
        """测试列出模型."""
        resp = client.get("/api/models/list")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "models" in data

    def test_get_model(self):
        """测试获取模型信息."""
        client.post("/api/models/register", json={"name": "get-test-model"})
        resp = client.get("/api/models/get-test-model")
        assert resp.status_code == 200
        assert resp.json()["name"] == "get-test-model"

    def test_get_model_not_found(self):
        """测试获取不存在的模型."""
        resp = client.get("/api/models/nonexistent-model-xyz")
        assert resp.status_code == 404

    def test_update_model(self):
        """测试更新模型."""
        client.post("/api/models/register", json={"name": "update-model"})
        resp = client.put("/api/models/update-model", json={
            "description": "Updated description",
            "version": "v2",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["description"] == "Updated description"
        assert data["latest_version"] == "v2"

    def test_delete_model(self):
        """测试删除模型."""
        client.post("/api/models/register", json={"name": "delete-model"})
        resp = client.delete("/api/models/delete-model")
        assert resp.status_code == 200


class TestLeaderboardAPI:
    """排行榜 API 测试."""

    def _ensure_eval_data(self):
        """确保有评测数据."""
        client.post("/api/benchmark/evaluate", json={
            "model_name": "lb-model-a",
            "benchmark": "mmlu",
            "question_limit": 5,
            "enable_cache": False,
        })
        client.post("/api/benchmark/evaluate", json={
            "model_name": "lb-model-b",
            "benchmark": "mmlu",
            "question_limit": 5,
            "enable_cache": False,
        })

    def test_overall_leaderboard(self):
        """测试综合排行榜."""
        self._ensure_eval_data()
        resp = client.get("/api/leaderboard/overall")
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_ranking" in data

    def test_benchmark_leaderboard(self):
        """测试分基准排行榜."""
        self._ensure_eval_data()
        resp = client.get("/api/leaderboard/benchmark/mmlu")
        assert resp.status_code == 200
        data = resp.json()
        assert "ranking" in data

    def test_model_card(self):
        """测试模型卡片."""
        self._ensure_eval_data()
        resp = client.get("/api/leaderboard/model/lb-model-a")
        assert resp.status_code == 200

    def test_charts_endpoint(self):
        """测试图表生成."""
        self._ensure_eval_data()
        resp = client.get("/api/leaderboard/charts")
        assert resp.status_code == 200


class TestReportAPI:
    """报告 API 测试."""

    def _ensure_eval_data(self):
        """确保有评测数据."""
        client.post("/api/benchmark/evaluate", json={
            "model_name": "rpt-model-a",
            "benchmark": "mmlu",
            "question_limit": 3,
            "enable_cache": False,
        })

    def test_json_report(self):
        """测试 JSON 报告."""
        self._ensure_eval_data()
        resp = client.post("/api/report/generate", json={
            "format": "json",
        })
        assert resp.status_code == 200

    def test_markdown_report(self):
        """测试 Markdown 报告."""
        self._ensure_eval_data()
        resp = client.post("/api/report/generate", json={
            "format": "markdown",
        })
        assert resp.status_code == 200
        assert "text/markdown" in resp.headers.get("content-type", "")

    def test_csv_report(self):
        """测试 CSV 报告."""
        self._ensure_eval_data()
        resp = client.post("/api/report/generate", json={
            "format": "csv",
        })
        assert resp.status_code == 200
        assert "text/csv" in resp.headers.get("content-type", "")

    def test_compare_models(self):
        """测试模型对比."""
        client.post("/api/benchmark/evaluate", json={
            "model_name": "cmp-a",
            "benchmark": "mmlu",
            "question_limit": 5,
            "enable_cache": False,
        })
        client.post("/api/benchmark/evaluate", json={
            "model_name": "cmp-b",
            "benchmark": "mmlu",
            "question_limit": 5,
            "enable_cache": False,
        })
        resp = client.post("/api/report/compare", json={
            "model_a": "cmp-a",
            "model_b": "cmp-b",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_a"] == "cmp-a"
        assert data["model_b"] == "cmp-b"

    def test_compare_nonexistent(self):
        """测试对比不存在的模型."""
        resp = client.post("/api/report/compare", json={
            "model_a": "nonexistent",
            "model_b": "also-nonexistent",
        })
        assert resp.status_code == 404