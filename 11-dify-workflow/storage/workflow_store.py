"""工作流存储 - 内存存储 + JSON 文件持久化"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from engine.errors import WorkflowNotFoundError


class WorkflowStore:
    """工作流存储层 (内存 + 可选 JSON 文件持久化)"""

    def __init__(self, persist_dir: Optional[str] = None):
        """
        Args:
            persist_dir: 若指定，工作流将持久化到该目录下的 JSON 文件
        """
        self._workflows: Dict[str, Dict[str, Any]] = {}
        self._versions: Dict[str, List[Dict[str, Any]]] = {}  # workflow_id -> [versions]
        self._persist_dir = persist_dir

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)

    # ── CRUD ───────────────────────────────────────────────

    def create(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建工作流"""
        wf_id = workflow_data.get("id") or workflow_data.get("name", "").lower().replace(" ", "-")
        if not wf_id:
            raise ValueError("工作流必须有 id 或 name")

        now = datetime.now(timezone.utc).isoformat()
        workflow = {
            "id": wf_id,
            "name": workflow_data.get("name", wf_id),
            "description": workflow_data.get("description", ""),
            "nodes": workflow_data.get("nodes", {}),
            "edges": workflow_data.get("edges", []),
            "global_inputs": workflow_data.get("global_inputs", {}),
            "version": 1,
            "created_at": now,
            "updated_at": now,
            "tags": workflow_data.get("tags", []),
            "metadata": workflow_data.get("metadata", {}),
        }

        self._workflows[wf_id] = workflow
        self._versions.setdefault(wf_id, []).append(workflow.copy())

        if self._persist_dir:
            self._save_to_file(wf_id, workflow)

        return workflow

    def get(self, workflow_id: str) -> Dict[str, Any]:
        """获取工作流"""
        if workflow_id not in self._workflows:
            raise WorkflowNotFoundError(f"工作流 '{workflow_id}' 不存在")
        return self._workflows[workflow_id].copy()

    def update(self, workflow_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """更新工作流"""
        if workflow_id not in self._workflows:
            raise WorkflowNotFoundError(f"工作流 '{workflow_id}' 不存在")

        workflow = self._workflows[workflow_id]
        for key, value in updates.items():
            if key in ("id", "created_at"):
                continue  # 不允许修改
            workflow[key] = value

        workflow["updated_at"] = datetime.now(timezone.utc).isoformat()
        workflow["version"] = workflow.get("version", 1) + 1

        self._versions.setdefault(workflow_id, []).append(workflow.copy())

        if self._persist_dir:
            self._save_to_file(workflow_id, workflow)

        return workflow.copy()

    def delete(self, workflow_id: str) -> bool:
        """删除工作流"""
        if workflow_id not in self._workflows:
            raise WorkflowNotFoundError(f"工作流 '{workflow_id}' 不存在")

        del self._workflows[workflow_id]

        if self._persist_dir:
            filepath = os.path.join(self._persist_dir, f"{workflow_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)

        return True

    def list_all(self, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有工作流"""
        workflows = list(self._workflows.values())
        if tag:
            workflows = [wf for wf in workflows if tag in wf.get("tags", [])]
        return [wf.copy() for wf in workflows]

    # ── 版本管理 ────────────────────────────────────────────

    def get_version(self, workflow_id: str, version: int) -> Dict[str, Any]:
        """获取特定版本"""
        versions = self._versions.get(workflow_id, [])
        for v in versions:
            if v.get("version") == version:
                return v.copy()
        raise WorkflowNotFoundError(
            f"工作流 '{workflow_id}' 的版本 {version} 不存在"
        )

    def list_versions(self, workflow_id: str) -> List[Dict[str, Any]]:
        """列出所有版本"""
        return [v.copy() for v in self._versions.get(workflow_id, [])]

    def rollback(self, workflow_id: str, version: int) -> Dict[str, Any]:
        """回滚到指定版本"""
        old_version = self.get_version(workflow_id, version)
        old_version["updated_at"] = datetime.now(timezone.utc).isoformat()
        old_version["version"] = old_version.get("version", 0) + 1
        self._workflows[workflow_id] = old_version
        self._versions.setdefault(workflow_id, []).append(old_version.copy())

        if self._persist_dir:
            self._save_to_file(workflow_id, old_version)

        return old_version.copy()

    # ── 导入/导出 ──────────────────────────────────────────

    def export_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """导出为字典 (可序列化为 JSON/YAML)"""
        return self.get(workflow_id)

    def import_workflow(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """从字典导入"""
        return self.create(data)

    def export_json(self, workflow_id: str) -> str:
        """导出为 JSON 字符串"""
        import json
        return json.dumps(self.export_workflow(workflow_id), ensure_ascii=False, indent=2)

    def import_json(self, json_str: str) -> Dict[str, Any]:
        """从 JSON 字符串导入"""
        import json
        data = json.loads(json_str)
        return self.import_workflow(data)

    # ── 文件持久化 ─────────────────────────────────────────

    def _save_to_file(self, workflow_id: str, data: Dict[str, Any]) -> None:
        filepath = os.path.join(self._persist_dir, f"{workflow_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_from_file(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_all_from_dir(self) -> int:
        """从持久化目录加载所有工作流"""
        if not self._persist_dir:
            return 0
        count = 0
        for filename in os.listdir(self._persist_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self._persist_dir, filename)
                try:
                    data = self._load_from_file(filepath)
                    self._workflows[data["id"]] = data
                    self._versions.setdefault(data["id"], []).append(data.copy())
                    count += 1
                except Exception:
                    continue
        return count
