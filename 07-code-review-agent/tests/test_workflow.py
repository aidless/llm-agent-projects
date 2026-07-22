"""工作流测试。"""

from __future__ import annotations

import pytest

from workflows.review_graph import ReviewWorkflow, ReviewState, create_review_graph


# ==================== 测试数据 ====================

SAMPLE_CODE = '''
import os

password = "hardcoded_secret"

def process(data):
    try:
        result = eval(data)
    except:
        pass
    return result

if result == None:
    print("done")
'''


class TestReviewState:
    """ReviewState 测试。"""

    def test_default_state(self) -> None:
        """测试默认状态。"""
        state = ReviewState()
        assert state.code == ""
        assert state.language == "python"
        assert state.filename == "untitled.py"
        assert state.ast_result is None
        assert state.security_findings == []
        assert state.performance_findings == []
        assert state.style_findings == []
        assert state.logic_findings == []
        assert state.agent_results == []

    def test_custom_state(self) -> None:
        """测试自定义状态。"""
        state = ReviewState(
            code="x = 1",
            language="python",
            filename="test.py",
        )
        assert state.code == "x = 1"
        assert state.filename == "test.py"


class TestReviewWorkflow:
    """ReviewWorkflow 测试。"""

    def setup_method(self) -> None:
        self.workflow = ReviewWorkflow()

    def test_preprocess_valid_code(self) -> None:
        """测试预处理有效代码。"""
        state = ReviewState(code="def foo(): pass", language="python")
        result = self.workflow._preprocess(state)
        assert result["ast_result"] is not None
        assert result["error"] is None

    def test_preprocess_non_python(self) -> None:
        """测试预处理非 Python 代码。"""
        state = ReviewState(code="function foo() {}", language="javascript")
        result = self.workflow._preprocess(state)
        assert result["ast_result"] is None
        assert result["error"] is None

    def test_preprocess_invalid_syntax(self) -> None:
        """测试预处理无效语法。"""
        state = ReviewState(code="def broken(:", language="python")
        result = self.workflow._preprocess(state)
        assert result["ast_result"] is None
        assert result["error"] is not None

    def test_security_review(self) -> None:
        """测试安全审查节点。"""
        state = ReviewState(
            code='password = "secret"\neval("x")\n',
            filename="test.py",
            ast_result=None,
        )
        result = self.workflow._security_review(state)
        assert "security_findings" in result
        assert len(result["security_findings"]) > 0

    def test_performance_review(self) -> None:
        """测试性能审查节点。"""
        state = ReviewState(code="def f(x):\n    return x\n", filename="test.py")
        result = self.workflow._performance_review(state)
        assert "performance_findings" in result

    def test_style_review(self) -> None:
        """测试风格审查节点。"""
        state = ReviewState(code="x=1\n", filename="test.py")
        result = self.workflow._style_review(state)
        assert "style_findings" in result

    def test_logic_review(self) -> None:
        """测试逻辑审查节点。"""
        state = ReviewState(code="try:\n pass\nexcept:\n pass\n", filename="test.py")
        result = self.workflow._logic_review(state)
        assert "logic_findings" in result

    def test_summarize_empty_findings(self) -> None:
        """测试空结果汇总。"""
        state = ReviewState(
            code="x = 1\n",
            filename="test.py",
        )
        result = self.workflow._aggregate_and_summarize(state)
        assert "agent_results" in result
        assert "report_markdown" in result
        assert "report_json" in result

    def test_summarize_with_findings(self) -> None:
        """测试有发现的汇总。"""
        state = ReviewState(
            code='password = "secret"\nexcept:\n pass\n',
            filename="test.py",
            security_findings=[{
                "file": "test.py", "line": 1,
                "severity": "High", "category": "Security",
                "message": "Hardcoded password",
                "suggestion": None, "code_snippet": None,
                "column": None, "rule_id": "SEC-001",
            }],
            performance_findings=[],
            style_findings=[],
            logic_findings=[],
        )
        result = self.workflow._aggregate_and_summarize(state)
        assert len(result["agent_results"]) > 0
        assert len(result["report_markdown"]) > 0

    def test_build_graph(self) -> None:
        """测试构建图。"""
        graph = self.workflow.build_graph()
        assert graph is not None

    def test_create_review_graph(self) -> None:
        """测试创建编译后的图。"""
        compiled_graph = create_review_graph()
        assert compiled_graph is not None

    def test_full_workflow_execution(self) -> None:
        """测试完整工作流执行。"""
        graph = create_review_graph()
        state = ReviewState(
            code='password = "secret123"\neval("x")\ntry:\n pass\nexcept:\n pass\n',
            language="python",
            filename="test.py",
        )
        result = graph.invoke(state)
        assert "agent_results" in result
        assert "report_markdown" in result
        assert len(result["agent_results"]) > 0
