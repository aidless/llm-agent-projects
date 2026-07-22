"""执行 API 测试"""

import pytest
from fastapi.testclient import TestClient


class TestExecutionAPI:
    """执行 API 端到端测试"""

    def setup_method(self):
        from app.main import app, workflow_store, execution_log
        self.app = app
        self.store = workflow_store
        self.exec_log = execution_log
        self.client = TestClient(app)
        # 清空
        self.store._workflows.clear()
        self.store._versions.clear()
        self.exec_log._logs.clear()

    def _create_simple_workflow(self, wf_id="exec-test"):
        """创建一个简单的工作流用于执行测试"""
        self.store.create({
            "id": wf_id,
            "name": "Exec Test",
            "nodes": {
                "step1": {
                    "type": "llm",
                    "inputs": {"prompt": "Summarize: hello world"},
                },
            },
            "edges": [],
        })
        return wf_id

    def test_execute_workflow_success(self):
        wf_id = self._create_simple_workflow()
        resp = self.client.post("/api/v1/executions", json={
            "workflow_id": wf_id,
            "inputs": {},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "step1" in data["node_results"]

    def test_execute_nonexistent_workflow(self):
        resp = self.client.post("/api/v1/executions", json={
            "workflow_id": "nope",
        })
        assert resp.status_code == 404

    def test_list_executions(self):
        wf_id = self._create_simple_workflow()
        self.client.post("/api/v1/executions", json={"workflow_id": wf_id})
        resp = self.client.get("/api/v1/executions")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    def test_get_execution_detail(self):
        wf_id = self._create_simple_workflow()
        exec_resp = self.client.post("/api/v1/executions", json={"workflow_id": wf_id})
        exec_id = exec_resp.json()["execution_id"]
        resp = self.client.get(f"/api/v1/executions/{exec_id}")
        assert resp.status_code == 200
        assert resp.json()["execution_id"] == exec_id

    def test_chain_workflow_execution(self):
        self.store.create({
            "id": "chain-wf",
            "name": "Chain",
            "nodes": {
                "code1": {
                    "type": "code",
                    "inputs": {"code": "result = 42"},
                },
                "code2": {
                    "type": "code",
                    "inputs": {"code": "result = input.get('value', 0) + 1", "value": "{{code1.output.result}}"},
                },
            },
            "edges": [
                {"source": "code1", "target": "code2"},
            ],
        })
        resp = self.client.post("/api/v1/executions", json={
            "workflow_id": "chain-wf",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["node_results"]["code1"]["output"]["result"] == 42
        assert data["node_results"]["code2"]["output"]["result"] == 43


class TestTemplateAPI:
    """模板 API 测试"""

    def setup_method(self):
        from app.main import app, workflow_store
        self.app = app
        self.store = workflow_store
        self.client = TestClient(app)
        self.store._workflows.clear()
        self.store._versions.clear()

    def test_list_templates(self):
        resp = self.client.get("/api/v1/templates")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 3

    def test_get_template(self):
        resp = self.client.get("/api/v1/templates/text-summary")
        assert resp.status_code == 200
        assert resp.json()["name"] == "文本摘要"

    def test_get_nonexistent_template(self):
        resp = self.client.get("/api/v1/templates/nonexistent")
        assert resp.status_code == 404

    def test_create_from_template(self):
        resp = self.client.post("/api/v1/templates/create", json={
            "template_id": "text-summary",
            "name": "My Summary",
        })
        assert resp.status_code == 201
        assert resp.json()["name"] == "My Summary"
        assert resp.json()["id"].startswith("user-")
