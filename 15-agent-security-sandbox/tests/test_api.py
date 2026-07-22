"""API 集成测试"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthAPI:
    """健康检查 API 测试"""

    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "AI Agent Security Sandbox"

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"

    def test_ready_check(self):
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


class TestExecuteAPI:
    """代码执行 API 测试"""

    def test_execute_simple_code(self):
        response = client.post("/api/v1/execute", json={
            "code": "print('hello')",
            "policy_name": "medium",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "hello" in data["output"]

    def test_execute_with_variables(self):
        response = client.post("/api/v1/execute", json={
            "code": "print(name)",
            "policy_name": "medium",
            "variables": {"name": "test"},
        })
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_execute_dangerous_code(self):
        response = client.post("/api/v1/execute", json={
            "code": "import os\nos.system('ls')",
            "policy_name": "medium",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert len(data["security_violations"]) > 0

    def test_execute_nonexistent_policy(self):
        response = client.post("/api/v1/execute", json={
            "code": "print(1)",
            "policy_name": "nonexistent",
        })
        assert response.status_code == 400

    def test_execute_by_level(self):
        response = client.post("/api/v1/execute", json={
            "code": "print('strict')",
            "policy_name": "STRICT",
        })
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_execution_has_resource_usage(self):
        response = client.post("/api/v1/execute", json={
            "code": "x = sum(range(1000))",
            "policy_name": "medium",
        })
        data = response.json()
        assert "resource_usage" in data
        assert "execution_time" in data["resource_usage"]


class TestPolicyAPI:
    """策略管理 API 测试"""

    def test_list_policies(self):
        response = client.get("/api/v1/policies")
        assert response.status_code == 200
        data = response.json()
        assert len(data["policies"]) >= 4

    def test_get_policy(self):
        response = client.get("/api/v1/policies/medium")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "medium"
        assert data["level"] == "MEDIUM"

    def test_get_nonexistent_policy(self):
        response = client.get("/api/v1/policies/nonexistent")
        assert response.status_code == 404

    def test_create_custom_policy(self):
        response = client.post("/api/v1/policies", json={
            "name": "test_custom",
            "level": "HIGH",
            "description": "Test custom policy",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["policy"]["name"] == "test_custom"

    def test_delete_custom_policy(self):
        # 先创建
        client.post("/api/v1/policies", json={"name": "to_delete", "level": "LOW"})
        response = client.delete("/api/v1/policies/to_delete")
        assert response.status_code == 200

    def test_cannot_delete_preset(self):
        response = client.delete("/api/v1/policies/medium")
        assert response.status_code == 400

    def test_inherit_policy(self):
        response = client.post("/api/v1/policies/inherit", json={
            "base_name": "medium",
            "overrides": {"resource_limits": {"max_execution_time": 5.0}},
            "new_name": "inherited_api_test",
        })
        assert response.status_code == 200

    def test_combine_policies(self):
        response = client.post("/api/v1/policies/combine", json={
            "policy_names": ["low", "high"],
            "combined_name": "combined_api_test",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["policy"]["level"] == "HIGH"  # 取最高等级


class TestAuditAPI:
    """审计日志 API 测试"""

    def test_query_logs(self):
        response = client.get("/api/v1/audit/logs")
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data

    def test_query_events(self):
        response = client.get("/api/v1/audit/events")
        assert response.status_code == 200
        data = response.json()
        assert "events" in data

    def test_get_alerts(self):
        response = client.get("/api/v1/audit/alerts")
        assert response.status_code == 200

    def test_summary_report(self):
        response = client.get("/api/v1/audit/reports/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["report_type"] == "summary"

    def test_security_report(self):
        response = client.get("/api/v1/audit/reports/security")
        assert response.status_code == 200
        data = response.json()
        assert data["report_type"] == "security"

    def test_audit_stats(self):
        response = client.get("/api/v1/audit/stats")
        assert response.status_code == 200
        data = response.json()
        assert "log_stats" in data
        assert "event_stats" in data

    def test_get_snapshots(self):
        response = client.get("/api/v1/audit/snapshots")
        assert response.status_code == 200