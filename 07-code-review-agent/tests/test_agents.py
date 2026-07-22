"""Agent 测试。"""

from __future__ import annotations

import pytest

from agents.security_agent import SecurityAgent
from agents.performance_agent import PerformanceAgent
from agents.style_agent import StyleAgent
from agents.logic_agent import LogicAgent
from agents.summary_agent import SummaryAgent
from app.models import Category, Finding, ReviewResult, Severity


# ==================== 测试数据 ====================

SECURITY_CODE = '''
import pickle
import subprocess

password = "my_secret_password_123"

def query_user(user_id):
    sql = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(sql)
    return cursor.fetchall()

def process_data(data):
    result = eval(data)
    return result

def run_command(cmd):
    subprocess.call(cmd, shell=True)
    return os.system(cmd)
'''

PERFORMANCE_CODE = '''
def calculate(items):
    result = []
    for item in items:
        result.append(item * 2)
    return result

def complex_function(x, y, z, a, b, c, d, e, f):
    if x > 0:
        if y > 0:
            if z > 0:
                if a > 0:
                    if b > 0:
                        return x + y + z
    return 0
'''

STYLE_CODE = '''
class myclass:
    def GetData(self, x):
        return x
'''

LOGIC_CODE = '''
def process(data, items=[]):
    try:
        result = data / items
    except:
        pass
    if result == None:
        return result
'''

CLEAN_CODE = '''
"""Clean module."""

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''


# ==================== SecurityAgent 测试 ====================

class TestSecurityAgent:
    """SecurityAgent 测试。"""

    def setup_method(self) -> None:
        self.agent = SecurityAgent()

    def test_detect_hardcoded_password(self) -> None:
        """测试检测硬编码密码。"""
        findings = self.agent.analyze(SECURITY_CODE, "test.py")
        password_findings = [
            f for f in findings
            if "hardcoded" in f.message.lower() or "password" in f.message.lower() or "密钥" in f.message
        ]
        assert len(password_findings) > 0

    def test_detect_eval_usage(self) -> None:
        """测试检测 eval 使用。"""
        findings = self.agent.analyze(SECURITY_CODE, "test.py")
        eval_findings = [f for f in findings if "eval" in f.message]
        assert len(eval_findings) > 0
        assert eval_findings[0].severity in (Severity.CRITICAL, Severity.HIGH)

    def test_detect_subprocess_shell(self) -> None:
        """测试检测 subprocess shell=True。"""
        findings = self.agent.analyze(SECURITY_CODE, "test.py")
        subprocess_findings = [
            f for f in findings
            if "subprocess" in f.message.lower() or "shell" in f.message.lower()
        ]
        assert len(subprocess_findings) > 0

    def test_detect_pickle_import(self) -> None:
        """测试检测 pickle 导入。"""
        findings = self.agent.analyze(SECURITY_CODE, "test.py")
        pickle_findings = [f for f in findings if "pickle" in f.message]
        assert len(pickle_findings) > 0

    def test_clean_code_no_critical_findings(self) -> None:
        """测试干净代码无严重发现。"""
        findings = self.agent.analyze(CLEAN_CODE, "clean.py")
        critical = [f for f in findings if f.severity == Severity.CRITICAL]
        assert len(critical) == 0

    def test_findings_have_file_and_line(self) -> None:
        """测试发现包含文件和行号信息。"""
        findings = self.agent.analyze(SECURITY_CODE, "security_test.py")
        for f in findings:
            assert f.file == "security_test.py"
            assert f.line > 0


# ==================== PerformanceAgent 测试 ====================

class TestPerformanceAgent:
    """PerformanceAgent 测试。"""

    def setup_method(self) -> None:
        self.agent = PerformanceAgent()

    def test_detect_high_complexity(self) -> None:
        """测试检测高圈复杂度。"""
        findings = self.agent.analyze(PERFORMANCE_CODE, "test.py")
        complexity_findings = [f for f in findings if "复杂度" in f.message]
        assert len(complexity_findings) > 0

    def test_complexity_finding_severity(self) -> None:
        """测试复杂度发现严重等级。"""
        findings = self.agent.analyze(PERFORMANCE_CODE, "test.py")
        complexity_findings = [f for f in findings if "复杂度" in f.message]
        assert any(f.severity in (Severity.HIGH, Severity.MEDIUM) for f in complexity_findings)

    def test_clean_code_passes(self) -> None:
        """测试干净代码通过性能检查。"""
        findings = self.agent.analyze(CLEAN_CODE, "clean.py")
        high_severity = [f for f in findings if f.severity in (Severity.CRITICAL, Severity.HIGH)]
        assert len(high_severity) == 0


# ==================== StyleAgent 测试 ====================

class TestStyleAgent:
    """StyleAgent 测试。"""

    def setup_method(self) -> None:
        self.agent = StyleAgent()

    def test_detect_bad_class_name(self) -> None:
        """测试检测不合规类名。"""
        findings = self.agent.analyze(STYLE_CODE, "test.py")
        class_findings = [f for f in findings if "类名" in f.message]
        assert len(class_findings) > 0

    def test_detect_bad_function_name(self) -> None:
        """测试检测不合规函数名。"""
        findings = self.agent.analyze(STYLE_CODE, "test.py")
        func_findings = [f for f in findings if "函数名" in f.message]
        assert len(func_findings) > 0

    def test_detect_missing_docstring(self) -> None:
        """测试检测缺少 docstring。"""
        findings = self.agent.analyze(CLEAN_CODE, "test.py")
        # CLEAN_CODE has docstrings, so GetData in STYLE_CODE should be detected
        style_findings = StyleAgent().analyze(STYLE_CODE, "test.py")
        docstring_findings = [f for f in style_findings if "docstring" in f.message.lower()]
        assert len(docstring_findings) > 0

    def test_naming_findings_are_low_severity(self) -> None:
        """测试命名规范发现为低严重等级。"""
        findings = self.agent.analyze(STYLE_CODE, "test.py")
        for f in findings:
            if "命名" in f.message:
                assert f.severity == Severity.LOW


# ==================== LogicAgent 测试 ====================

class TestLogicAgent:
    """LogicAgent 测试。"""

    def setup_method(self) -> None:
        self.agent = LogicAgent()

    def test_detect_bare_except(self) -> None:
        """测试检测裸 except。"""
        findings = self.agent.analyze(LOGIC_CODE, "test.py")
        bare_except = [f for f in findings if "裸" in f.message or "except" in f.message.lower()]
        assert len(bare_except) > 0

    def test_detect_mutable_default(self) -> None:
        """测试检测可变默认参数。"""
        findings = self.agent.analyze(LOGIC_CODE, "test.py")
        mutable_findings = [f for f in findings if "可变" in f.message or "默认" in f.message]
        assert len(mutable_findings) > 0

    def test_detect_none_comparison(self) -> None:
        """测试检测 None 比较。"""
        findings = self.agent.analyze(LOGIC_CODE, "test.py")
        none_findings = [f for f in findings if "None" in f.message]
        assert len(none_findings) > 0

    def test_clean_code_no_logic_issues(self) -> None:
        """测试干净代码无逻辑问题。"""
        findings = self.agent.analyze(CLEAN_CODE, "clean.py")
        high_severity = [f for f in findings if f.severity in (Severity.CRITICAL, Severity.HIGH)]
        assert len(high_severity) == 0


# ==================== SummaryAgent 测试 ====================

class TestSummaryAgent:
    """SummaryAgent 测试。"""

    def setup_method(self) -> None:
        self.agent = SummaryAgent()

    def test_generate_summary(self) -> None:
        """测试生成汇总报告。"""
        results = [
            ReviewResult(
                agent_name="SecurityAgent",
                findings=[
                    Finding(
                        file="test.py", line=1,
                        severity=Severity.HIGH,
                        category=Category.SECURITY,
                        message="Test finding",
                    )
                ],
                summary="1 issue found",
            )
        ]

        summary = self.agent.generate_summary(code="", filename="test.py", agent_results=results, language="python")
        assert summary["total_findings"] == 1
        assert "report_markdown" in summary
        assert "report_json" in summary

    def test_generate_summary_empty(self) -> None:
        """测试空结果汇总。"""
        results = []
        summary = self.agent.generate_summary(code="", filename="test.py", agent_results=results, language="python")
        assert summary["total_findings"] == 0

    def test_get_overall_score_perfect(self) -> None:
        """测试完美评分。"""
        score = self.agent.get_overall_score([])
        assert score == 100.0

    def test_get_overall_score_with_findings(self) -> None:
        """测试有发现时的评分。"""
        results = [
            ReviewResult(
                agent_name="SecurityAgent",
                findings=[
                    Finding(
                        file="test.py", line=1,
                        severity=Severity.HIGH,
                        category=Category.SECURITY,
                        message="Test",
                    )
                ],
                summary="1 issue",
            )
        ]
        score = self.agent.get_overall_score(results)
        assert score < 100.0
        assert score > 0.0
