"""事件追踪器 - 系统调用追踪和安全事件告警"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from security.syscall_interceptor import SecurityViolation


@dataclass
class TrackedEvent:
    """追踪事件"""
    execution_id: str
    event_type: str
    timestamp: float = 0.0
    detail: str = ""
    severity: str = "INFO"

    def to_dict(self) -> dict:
        return {
            "execution_id": self.execution_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "time_str": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)),
            "detail": self.detail,
            "severity": self.severity,
        }


class EventTracker:
    """事件追踪器"""

    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self._events: List[TrackedEvent] = []
        self._alerts: List[TrackedEvent] = []

    def track_event(self, execution_id: str, event_type: str, detail: str, severity: str = "INFO"):
        """追踪一个事件"""
        event = TrackedEvent(
            execution_id=execution_id,
            event_type=event_type,
            timestamp=time.time(),
            detail=detail,
            severity=severity,
        )
        self._events.append(event)

        # 安全事件产生告警
        if severity in ("CRITICAL", "HIGH"):
            self._alerts.append(event)

        # 限制大小
        while len(self._events) > self.max_events:
            self._events.pop(0)
        while len(self._alerts) > self.max_events:
            self._alerts.pop(0)

    def track_violations(self, execution_id: str, violations: List[SecurityViolation]):
        """追踪安全违规"""
        for v in violations:
            self.track_event(
                execution_id,
                f"VIOLATION_{v.violation_type}",
                v.detail,
                severity=v.severity,
            )

    def track_execution(self, execution_id: str, code: str, result):
        """追踪执行完成事件"""
        event_type = "EXECUTION_SUCCESS" if result.success else "EXECUTION_FAILURE"
        detail = f"code_len={len(code)}, exec_time={result.execution_time:.3f}s"
        if result.error:
            detail += f", error={result.error_type}: {result.error[:100]}"
        self.track_event(execution_id, event_type, detail)

    def get_events(
        self,
        execution_id: Optional[str] = None,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """查询追踪事件"""
        filtered = self._events
        if execution_id:
            filtered = [e for e in filtered if e.execution_id == execution_id]
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        if severity:
            filtered = [e for e in filtered if e.severity == severity]
        return [e.to_dict() for e in filtered[-limit:]]

    def get_alerts(self, limit: int = 100) -> List[dict]:
        """获取安全告警"""
        return [a.to_dict() for a in self._alerts[-limit:]]

    def get_execution_timeline(self, execution_id: str) -> List[dict]:
        """获取执行的完整时间线"""
        events = [e for e in self._events if e.execution_id == execution_id]
        events.sort(key=lambda e: e.timestamp)
        return [e.to_dict() for e in events]

    def clear(self):
        """清除所有事件"""
        self._events.clear()
        self._alerts.clear()

    def get_stats(self) -> dict:
        """获取事件统计"""
        by_type = {}
        by_severity = {}
        for e in self._events:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
            by_severity[e.severity] = by_severity.get(e.severity, 0) + 1

        return {
            "total_events": len(self._events),
            "total_alerts": len(self._alerts),
            "by_event_type": by_type,
            "by_severity": by_severity,
        }