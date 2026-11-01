"""执行日志 - 记录工作流执行历史"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ExecutionLog:
    """工作流执行日志存储"""

    def __init__(self, max_logs: int = 1000):
        self._logs: Dict[str, Dict[str, Any]] = {}
        self._max_logs = max_logs

    def create_log(
        self,
        workflow_id: str,
        execution_id: str = None,
        status: str = "pending",
        global_inputs: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """创建执行日志"""
        if execution_id is None:
            execution_id = str(uuid.uuid4())[:12]

        log = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": status,
            "global_inputs": global_inputs or {},
            "node_logs": {},
            "output": {},
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": 0.0,
        }

        self._logs[execution_id] = log
        self._trim_logs()

        return log

    def update_log(
        self,
        execution_id: str,
        status: Optional[str] = None,
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        node_logs: Optional[Dict[str, Any]] = None,
        elapsed_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """更新执行日志"""
        if execution_id not in self._logs:
            raise KeyError(f"执行日志 '{execution_id}' 不存在")

        log = self._logs[execution_id]
        if status:
            log["status"] = status
        if output:
            log["output"] = output
        if error:
            log["error"] = error
        if node_logs:
            log["node_logs"] = node_logs
        if elapsed_ms is not None:
            log["elapsed_ms"] = elapsed_ms
        log["updated_at"] = datetime.now(timezone.utc).isoformat()

        return log

    def add_node_log(
        self,
        execution_id: str,
        node_id: str,
        status: str,
        output: Dict[str, Any] = None,
        error: Optional[str] = None,
        elapsed_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """添加节点执行日志"""
        if execution_id not in self._logs:
            raise KeyError(f"执行日志 '{execution_id}' 不存在")

        log = self._logs[execution_id]
        log["node_logs"][node_id] = {
            "node_id": node_id,
            "status": status,
            "output": output or {},
            "error": error,
            "elapsed_ms": elapsed_ms,
        }
        return log["node_logs"][node_id]

    def get_log(self, execution_id: str) -> Dict[str, Any]:
        """获取执行日志"""
        if execution_id not in self._logs:
            raise KeyError(f"执行日志 '{execution_id}' 不存在")
        return self._logs[execution_id].copy()

    def list_logs(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """列出执行日志"""
        logs = list(self._logs.values())

        if workflow_id:
            logs = [l for l in logs if l["workflow_id"] == workflow_id]
        if status:
            logs = [l for l in logs if l["status"] == status]

        # 按时间倒序
        logs.sort(key=lambda x: x["created_at"], reverse=True)
        return logs[:limit]

    def delete_log(self, execution_id: str) -> bool:
        if execution_id in self._logs:
            del self._logs[execution_id]
            return True
        return False

    def _trim_logs(self) -> None:
        """修剪日志数量"""
        if len(self._logs) > self._max_logs:
            # 删除最旧的日志
            sorted_logs = sorted(
                self._logs.items(),
                key=lambda x: x[1]["created_at"],
            )
            for eid, _ in sorted_logs[: len(self._logs) - self._max_logs]:
                del self._logs[eid]
