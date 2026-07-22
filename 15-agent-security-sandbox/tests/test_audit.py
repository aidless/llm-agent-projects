"""审计模块测试"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit.logger import AuditLogger
from audit.event_tracker import EventTracker
from audit.reporter import AuditReporter
from security.syscall_interceptor import SecurityViolation


@pytest.fixture
def audit_logger():
    return AuditLogger()


@pytest.fixture
def event_tracker():
    return EventTracker()


@pytest.fixture
def reporter(audit_logger, event_tracker):
    return AuditReporter(audit_logger, event_tracker)


class TestAuditLogger:
    """审计日志测试"""

    def test_log_execution_start(self, audit_logger):
        """测试记录执行开始"""
        audit_logger.log_execution_start("exec_1", "print('hi')", "medium")
        logs = audit_logger.get_logs()
        assert len(logs) == 1
        assert logs[0]["event_type"] == "EXECUTION_START"
        assert logs[0]["execution_id"] == "exec_1"

    def test_log_execution_end(self, audit_logger):
        """测试记录执行结束"""
        from sandbox.executor import ExecutionResult
        result = ExecutionResult(
            execution_id="exec_2",
            success=True,
            output="hello",
            execution_time=0.1,
        )
        audit_logger.log_execution_end("exec_2", result)
        logs = audit_logger.get_logs()
        assert any(l["event_type"] == "EXECUTION_END" for l in logs)

    def test_log_security_violation(self, audit_logger):
        """测试记录安全违规"""
        violations = [
            SecurityViolation(
                violation_type="BLOCKED_IMPORT",
                node_type="Import",
                line=1, col=0,
                detail="import os",
            )
        ]
        audit_logger.log_security_violation("exec_3", violations)
        logs = audit_logger.get_logs(event_type="SECURITY_VIOLATION")
        assert len(logs) == 1
        assert logs[0]["level"] == "CRITICAL"

    def test_query_by_execution_id(self, audit_logger):
        """测试按执行ID查询"""
        audit_logger.log_execution_start("exec_a", "code", "medium")
        audit_logger.log_execution_start("exec_b", "code", "strict")
        logs_a = audit_logger.get_logs(execution_id="exec_a")
        assert len(logs_a) == 1
        assert logs_a[0]["execution_id"] == "exec_a"

    def test_query_by_level(self, audit_logger):
        """测试按级别查询"""
        audit_logger.log_execution_start("exec_4", "code", "medium")
        logs = audit_logger.get_logs(level="INFO")
        assert len(logs) >= 1

    def test_get_snapshot(self, audit_logger):
        """测试获取执行快照"""
        from sandbox.executor import ExecutionResult
        result = ExecutionResult(
            execution_id="snap_1",
            success=True,
            output="test output",
            execution_time=0.05,
        )
        audit_logger.log_execution_end("snap_1", result)
        snapshot = audit_logger.get_snapshot("snap_1")
        assert snapshot is not None
        assert snapshot["output"] == "test output"

    def test_get_stats(self, audit_logger):
        """测试获取统计"""
        audit_logger.log_execution_start("exec_5", "code", "medium")
        stats = audit_logger.get_stats()
        assert stats["total_entries"] >= 1

    def test_clear_logs(self, audit_logger):
        """测试清除日志"""
        audit_logger.log_execution_start("exec_6", "code", "medium")
        audit_logger.clear()
        assert len(audit_logger.get_logs()) == 0


class TestEventTracker:
    """事件追踪器测试"""

    def test_track_event(self, event_tracker):
        """测试追踪事件"""
        event_tracker.track_event("exec_1", "TEST_EVENT", "test detail")
        events = event_tracker.get_events()
        assert len(events) == 1
        assert events[0]["event_type"] == "TEST_EVENT"

    def test_track_alert(self, event_tracker):
        """测试安全事件产生告警"""
        event_tracker.track_event("exec_1", "VIOLATION", "danger", severity="CRITICAL")
        alerts = event_tracker.get_alerts()
        assert len(alerts) == 1

    def test_track_violations(self, event_tracker):
        """测试追踪安全违规"""
        violations = [
            SecurityViolation("BLOCKED_IMPORT", "Import", 1, 0, "import os"),
            SecurityViolation("BLOCKED_BUILTIN_CALL", "Call", 2, 0, "exec()"),
        ]
        event_tracker.track_violations("exec_2", violations)
        events = event_tracker.get_events(execution_id="exec_2")
        assert len(events) == 2

    def test_get_execution_timeline(self, event_tracker):
        """测试获取执行时间线"""
        event_tracker.track_event("exec_3", "EXECUTION_START", "start")
        event_tracker.track_event("exec_3", "EXECUTION_END", "end")
        timeline = event_tracker.get_execution_timeline("exec_3")
        assert len(timeline) == 2

    def test_get_stats(self, event_tracker):
        """测试获取事件统计"""
        event_tracker.track_event("exec_4", "TEST", "detail")
        stats = event_tracker.get_stats()
        assert stats["total_events"] >= 1


class TestAuditReporter:
    """审计报告测试"""

    def test_summary_report(self, reporter):
        """测试生成汇总报告"""
        report = reporter.generate_summary_report()
        assert report["report_type"] == "summary"
        assert "statistics" in report
        assert "generated_at" in report

    def test_security_report(self, reporter):
        """测试生成安全报告"""
        report = reporter.generate_security_report()
        assert report["report_type"] == "security"
        assert "events_by_type" in report
        assert "events_by_severity" in report

    def test_execution_report_not_found(self, reporter):
        """测试不存在的执行报告"""
        report = reporter.generate_execution_report("nonexistent")
        assert report is None

    def test_text_report(self, reporter):
        """测试文本报告生成"""
        text = reporter.generate_text_report("summary")
        assert "AI Agent Security Sandbox" in text
        assert "Audit Report" in text