"""工作流存储和 API 测试"""

import pytest
from fastapi.testclient import TestClient

from storage.workflow_store import WorkflowStore
from engine.errors import WorkflowNotFoundError


class TestWorkflowStore:
    """工作流存储测试"""

    def setup_method(self):
        self.store = WorkflowStore()

    def test_create_workflow(self):
        wf = self.store.create({
            "id": "test-wf",
            "name": "Test Workflow",
            "description": "A test",
        })
        assert wf["id"] == "test-wf"
        assert wf["name"] == "Test Workflow"
        assert wf["version"] == 1

    def test_get_workflow(self):
        self.store.create({"id": "wf1", "name": "WF1"})
        wf = self.store.get("wf1")
        assert wf["id"] == "wf1"

    def test_get_nonexistent_raises(self):
        with pytest.raises(WorkflowNotFoundError):
            self.store.get("nonexistent")

    def test_update_workflow(self):
        self.store.create({"id": "wf1", "name": "Original"})
        updated = self.store.update("wf1", {"name": "Updated"})
        assert updated["name"] == "Updated"
        assert updated["version"] == 2

    def test_delete_workflow(self):
        self.store.create({"id": "wf1", "name": "To Delete"})
        result = self.store.delete("wf1")
        assert result is True
        with pytest.raises(WorkflowNotFoundError):
            self.store.get("wf1")

    def test_list_workflows(self):
        self.store.create({"id": "wf1", "name": "First"})
        self.store.create({"id": "wf2", "name": "Second", "tags": ["nlp"]})
        all_wfs = self.store.list_all()
        assert len(all_wfs) == 2

    def test_list_by_tag(self):
        self.store.create({"id": "wf1", "name": "NLP WF", "tags": ["nlp"]})
        self.store.create({"id": "wf2", "name": "Data WF", "tags": ["data"]})
        nlp_wfs = self.store.list_all(tag="nlp")
        assert len(nlp_wfs) == 1
        assert nlp_wfs[0]["id"] == "wf1"

    def test_version_management(self):
        self.store.create({"id": "wf1", "name": "V1"})
        self.store.update("wf1", {"name": "V2"})
        self.store.update("wf1", {"name": "V3"})
        versions = self.store.list_versions("wf1")
        assert len(versions) == 3

    def test_rollback(self):
        self.store.create({"id": "wf1", "name": "Original", "description": "old"})
        self.store.update("wf1", {"name": "Changed", "description": "new"})
        rolled = self.store.rollback("wf1", 1)
        assert rolled["description"] == "old"

    def test_export_import_json(self):
        self.store.create({"id": "wf1", "name": "Export Me"})
        json_str = self.store.export_json("wf1")
        assert '"id": "wf1"' in json_str or '"id": "wf1"' in json_str

        store2 = WorkflowStore()
        imported = store2.import_json(json_str)
        assert imported["name"] == "Export Me"

    def test_create_without_id_uses_name(self):
        wf = self.store.create({"name": "My Workflow"})
        assert wf["id"] == "my-workflow"

    def test_create_without_id_or_name_raises(self):
        with pytest.raises(ValueError):
            self.store.create({})

    def test_id_and_created_at_immutable_on_update(self):
        self.store.create({"id": "wf1", "name": "Original"})
        updated = self.store.update("wf1", {"id": "new-id", "name": "Changed"})
        assert updated["id"] == "wf1"  # id 不可变


class TestWorkflowAPI:
    """工作流 API 端到端测试"""

    def setup_method(self):
        from app.main import app, workflow_store
        self.app = app
        self.store = workflow_store
        self.client = TestClient(app)
        # 清空存储
        self.store._workflows.clear()
        self.store._versions.clear()

    def test_health_check(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_create_workflow_api(self):
        resp = self.client.post("/api/v1/workflows", json={
            "name": "Test API Workflow",
            "description": "Created via API",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test API Workflow"

    def test_list_workflows_api(self):
        self.client.post("/api/v1/workflows", json={"name": "WF1"})
        self.client.post("/api/v1/workflows", json={"name": "WF2"})
        resp = self.client.get("/api/v1/workflows")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_get_workflow_api(self):
        self.client.post("/api/v1/workflows", json={"name": "Get Me", "id": "get-me"})
        resp = self.client.get("/api/v1/workflows/get-me")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Get Me"

    def test_get_nonexistent_workflow_api(self):
        resp = self.client.get("/api/v1/workflows/nope")
        assert resp.status_code == 404

    def test_delete_workflow_api(self):
        self.client.post("/api/v1/workflows", json={"name": "Delete Me", "id": "del-me"})
        resp = self.client.delete("/api/v1/workflows/del-me")
        assert resp.status_code == 200

    def test_export_workflow_api(self):
        self.client.post("/api/v1/workflows", json={"name": "Export", "id": "exp-wf"})
        resp = self.client.get("/api/v1/workflows/exp-wf/export")
        assert resp.status_code == 200
