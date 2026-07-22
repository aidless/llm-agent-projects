"""审计日志 - 记录每次执行的完整信息"""

import json
import time
import uuid
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from collections import OrderedDict
from security.syscall_interceptor import SecurityViolation


# 简易的 LRU 缓存（固定大小日志存储）
MAX_LOG_ENTRIES = 10000


@dataclass
class AuditLogEntry:
    """审计日志条目"""
    timestamp: str = ""
    event_type: str = ""
    execution_id: str = ""
    level: str = "INFO"
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "execution_id": self.execution_id,
            "level": self.level,
            "detail": self.detail,
        }


class AuditLogger:
    """审计日志记录器"""

    def __init__(self, max_entries: int = MAX_LOG_ENTRIES):
        self.max_entries = max_entries
        self._logs: List[AuditLogEntry] = []
        self._execution_snapshots: Dict[str, dict] = OrderedDict()

    def _add_log(self, event_type: str, execution_id: str, level: str, detail: dict):
        """添加日志条目"""
        entry = AuditLogEntry(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            event_type=event_type,
            execution_id=execution_id,
            level=level,
            detail=detail,
        )
        self._logs.append(entry)

        # LRU 限制
        while len(self._logs) > self.max_entries:
            self._logs.pop(0)

    def log_execution_start(self, execution_id: str, code: str, policy_name: str):
        """记录执行开始"""
        self._add_log("EXECUTION_START", execution_id, "INFO", {
            "code_length": len(code),
            "code_preview": code[:200],
            "policy": policy_name,
        })

    def log_execution_end(self, execution_id: str, result):
        """记录执行结束"""
        self._add_log("EXECUTION_END", execution_id, "INFO" if result.success else "WARNING", {
            "success": result.success,
            "error_type": result.error_type,
            "execution_time": result.execution_time,
            "output_length": len(result.output),
            "resource_usage": result.resource_usage,
        })

        # 保存执行快照
        snapshot = {
            "execution_id": execution_id,
            "success": result.success,
            "output": result.output[:2000],
            "error": result.error[:1000] if result.error else "",
            "error_type": result.error_type,
            "execution_time": result.execution_time,
            "resource_usage": result.resource_usage,
            "security_violations": result.security_violations,
        }
        self._execution_snapshots[execution_id] = snapshot

        # LRU 限制快照
        while len(self._execution_snapshots) > self.max_entries:
            self._execution_snapshots.popitem(last=False)

    def log_security_violation(self, execution_id: str, violations: List[SecurityViolation]):
        """记录安全违规"""
        self._add_log("SECURITY_VIOLATION", execution_id, "CRITICAL", {
            "violation_count": len(violations),
            "violations": [v.to_dict() for v in violations],
        })

    def log_security_event(self, execution_id: str, event_type: str, message: str):
        """记录安全事件"""
        self._add_log("SECURITY_EVENT", execution_id, "CRITICAL", {
            "event": event_type,
            "message": message,
        })

    def log_policy_change(self, policy_name: str, action: str, detail: dict = None):
        """记录策略变更"""
        self._add_log("POLICY_CHANGE", "", "INFO", {
            "policy": policy_name,
            "action": action,
            **(detail or {}),
        })

    def get_logs(
        self,
        execution_id: Optional[str] = None,
        event_type: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        """查询审计日志"""
        filtered = self._logs

        if execution_id:
            filtered = [l for l in filtered if l.execution_id == execution_id]
        if event_type:
            filtered = [l for l in filtered if l.event_type == event_type]
        if level:
            filtered = [l for l in filtered if l.level == level]

        paginated = filtered[offset:offset + limit]
        return [l.to_dict() for l in paginated]

    def get_snapshot(self, execution_id: str) -> Optional[dict]:
        """获取执行快照"""
        return self._execution_snapshots.get(execution_id)

    def get_all_snapshots(self) -> List[dict]:
        """获取所有执行快照"""
        return list(self._execution_snapshots.values())

    def clear(self):
        """清除所有日志"""
        self._logs.clear()
        self._execution_snapshots.clear()

    def get_stats(self) -> dict:
        """获取日志统计"""
        total = len(self._logs)
        by_type = {}
        by_level = {}
        for log in self._logs:
            by_type[log.event_type] = by_type.get(log.event_type, 0) + 1
            by_level[log.level] = by_level.get(log.level, 0) + 1

        return {
            "total_entries": total,
            "total_executions": len(self._execution_snapshots),
            "by_event_type": by_type,
            "by_level": by_level,
        }