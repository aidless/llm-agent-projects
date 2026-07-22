"""审计报告生成器"""

import json
import time
from typing import Optional, List
from audit.logger import AuditLogger
from audit.event_tracker import EventTracker


class AuditReporter:
    """审计报告生成器"""

    def __init__(self, audit_logger: AuditLogger, event_tracker: EventTracker):
        self.logger = audit_logger
        self.tracker = event_tracker

    def generate_summary_report(self) -> dict:
        """生成汇总报告"""
        log_stats = self.logger.get_stats()
        event_stats = self.tracker.get_stats()
        alerts = self.tracker.get_alerts(limit=50)

        # 最近执行
        snapshots = self.logger.get_all_snapshots()
        recent_executions = snapshots[-20:]

        # 成功率统计
        total_execs = len(snapshots)
        success_count = sum(1 for s in snapshots if s.get("success"))
        success_rate = (success_count / total_execs * 100) if total_execs > 0 else 0.0

        # 安全事件统计
        security_events = [
            e for e in self.tracker.get_events()
            if e["event_type"].startswith("VIOLATION_") or e["event_type"].startswith("SECURITY_")
        ]

        return {
            "report_type": "summary",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "statistics": {
                "total_executions": total_execs,
                "successful_executions": success_count,
                "failed_executions": total_execs - success_count,
                "success_rate": round(success_rate, 1),
                "total_security_events": len(security_events),
                "total_alerts": event_stats["total_alerts"],
            },
            "recent_executions": recent_executions,
            "active_alerts": alerts,
            "security_events": security_events[-20:],
            "log_stats": log_stats,
            "event_stats": event_stats,
        }

    def generate_execution_report(self, execution_id: str) -> Optional[dict]:
        """生成单次执行的详细报告"""
        snapshot = self.logger.get_snapshot(execution_id)
        if not snapshot:
            return None

        timeline = self.tracker.get_execution_timeline(execution_id)
        events = self.tracker.get_events(execution_id=execution_id)

        return {
            "report_type": "execution_detail",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "execution": snapshot,
            "timeline": timeline,
            "events": events,
            "violation_count": len([e for e in events if e["event_type"].startswith("VIOLATION_")]),
        }

    def generate_security_report(self) -> dict:
        """生成安全专项报告"""
        alerts = self.tracker.get_alerts(limit=200)
        security_events = [
            e for e in self.tracker.get_events()
            if e["event_type"].startswith("VIOLATION_") or e["event_type"].startswith("SECURITY_") or e["event_type"].startswith("RUNTIME_")
        ]

        # 按类型分组
        by_type = {}
        for event in security_events:
            et = event["event_type"]
            by_type[et] = by_type.get(et, 0) + 1

        # 按严重程度分组
        by_severity = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
        for event in security_events:
            sev = event.get("severity", "INFO")
            if sev in by_severity:
                by_severity[sev] += 1

        return {
            "report_type": "security",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "total_security_events": len(security_events),
            "total_active_alerts": len(alerts),
            "events_by_type": by_type,
            "events_by_severity": by_severity,
            "recent_alerts": alerts[-50:],
            "recent_security_events": security_events[-50:],
        }

    def generate_text_report(self, report_type: str = "summary") -> str:
        """生成文本格式的报告"""
        if report_type == "summary":
            data = self.generate_summary_report()
        elif report_type == "security":
            data = self.generate_security_report()
        else:
            data = self.generate_summary_report()

        lines = [
            "=" * 60,
            f"  AI Agent Security Sandbox - Audit Report",
            f"  Generated: {data['generated_at']}",
            "=" * 60,
            "",
        ]

        if report_type == "summary" or "statistics" in data:
            stats = data.get("statistics", {})
            lines.append(f"Total Executions: {stats.get('total_executions', 0)}")
            lines.append(f"Success Rate: {stats.get('success_rate', 0)}%")
            lines.append(f"Security Events: {stats.get('total_security_events', 0)}")
            lines.append(f"Active Alerts: {stats.get('total_alerts', 0)}")
            lines.append("")

        if "events_by_type" in data:
            lines.append("Events by Type:")
            for et, count in data["events_by_type"].items():
                lines.append(f"  {et}: {count}")
            lines.append("")

        if "events_by_severity" in data:
            lines.append("Events by Severity:")
            for sev, count in data["events_by_severity"].items():
                lines.append(f"  {sev}: {count}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)