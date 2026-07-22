"""LangGraph StateGraph 审查工作流。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from langgraph.graph import StateGraph, START, END

from agents.security_agent import SecurityAgent
from agents.performance_agent import PerformanceAgent
from agents.style_agent import StyleAgent
from agents.logic_agent import LogicAgent
from agents.summary_agent import SummaryAgent
from analyzers.ast_analyzer import ASTAnalyzer
from app.models import Category, Finding, ReviewResult, Severity

logger = logging.getLogger(__name__)


@dataclass
class ReviewState:
    """审查工作流状态。

    作为 LangGraph StateGraph 的状态对象，在各节点间传递数据。
    """

    code: str = ""
    language: str = "python"
    filename: str = "untitled.py"
    context: str = ""
    ast_result: Optional[dict] = None
    security_findings: list[dict] = field(default_factory=list)
    performance_findings: list[dict] = field(default_factory=list)
    style_findings: list[dict] = field(default_factory=list)
    logic_findings: list[dict] = field(default_factory=list)
    agent_results: list[dict] = field(default_factory=list)
    report_markdown: str = ""
    report_json: dict = field(default_factory=dict)
    error: Optional[str] = None


class ReviewWorkflow:
    """代码审查工作流。

    使用 LangGraph StateGraph 编排多个审查 Agent:
    1. AST 分析（预处理）
    2. 并行执行安全/性能/风格/逻辑审查
    3. 汇总生成报告
    """

    def __init__(self) -> None:
        """初始化工作流。"""
        self._security_agent = SecurityAgent()
        self._performance_agent = PerformanceAgent()
        self._style_agent = StyleAgent()
        self._logic_agent = LogicAgent()
        self._summary_agent = SummaryAgent()
        self._ast_analyzer = ASTAnalyzer()

    def _preprocess(self, state: ReviewState) -> dict:
        """预处理节点：AST 分析。

        Args:
            state: 当前工作流状态。

        Returns:
            更新的状态部分。
        """
        logger.info(f"开始预处理: {state.filename}")

        if state.language == "python":
            try:
                ast_result = self._ast_analyzer.analyze(state.code)
                return {"ast_result": ast_result, "error": None}
            except SyntaxError as e:
                logger.warning(f"AST 解析失败: {e}")
                return {"ast_result": None, "error": str(e)}

        return {"ast_result": None, "error": None}

    def _security_review(self, state: ReviewState) -> dict:
        """安全审查节点。

        Args:
            state: 当前工作流状态。

        Returns:
            更新的状态部分。
        """
        logger.info("执行安全审查...")
        start = time.time()
        kwargs = {}
        if state.ast_result:
            kwargs["ast_result"] = state.ast_result

        findings = self._security_agent.analyze(state.code, state.filename, **kwargs)
        elapsed = time.time() - start
        result = self._security_agent.get_result(elapsed)

        logger.info(f"安全审查完成: 发现 {len(findings)} 个问题 ({elapsed:.2f}s)")
        return {"security_findings": [f.model_dump() for f in findings]}

    def _performance_review(self, state: ReviewState) -> dict:
        """性能审查节点。"""
        logger.info("执行性能审查...")
        start = time.time()
        kwargs = {}
        if state.ast_result:
            kwargs["ast_result"] = state.ast_result

        findings = self._performance_agent.analyze(state.code, state.filename, **kwargs)
        elapsed = time.time() - start

        logger.info(f"性能审查完成: 发现 {len(findings)} 个问题 ({elapsed:.2f}s)")
        return {"performance_findings": [f.model_dump() for f in findings]}

    def _style_review(self, state: ReviewState) -> dict:
        """风格审查节点。"""
        logger.info("执行风格审查...")
        start = time.time()
        kwargs = {}
        if state.ast_result:
            kwargs["ast_result"] = state.ast_result

        findings = self._style_agent.analyze(state.code, state.filename, **kwargs)
        elapsed = time.time() - start

        logger.info(f"风格审查完成: 发现 {len(findings)} 个问题 ({elapsed:.2f}s)")
        return {"style_findings": [f.model_dump() for f in findings]}

    def _logic_review(self, state: ReviewState) -> dict:
        """逻辑审查节点。"""
        logger.info("执行逻辑审查...")
        start = time.time()
        kwargs = {}
        if state.ast_result:
            kwargs["ast_result"] = state.ast_result

        findings = self._logic_agent.analyze(state.code, state.filename, **kwargs)
        elapsed = time.time() - start

        logger.info(f"逻辑审查完成: 发现 {len(findings)} 个问题 ({elapsed:.2f}s)")
        return {"logic_findings": [f.model_dump() for f in findings]}

    def _aggregate_and_summarize(self, state: ReviewState) -> dict:
        """汇总节点：聚合各 Agent 结果，生成报告。"""
        logger.info("汇总审查结果...")

        # 收集所有 Agent 结果
        all_findings_data = []
        agent_results = []

        for agent, findings_data, name in [
            (self._security_agent, state.security_findings, "SecurityAgent"),
            (self._performance_agent, state.performance_findings, "PerformanceAgent"),
            (self._style_agent, state.style_findings, "StyleAgent"),
            (self._logic_agent, state.logic_findings, "LogicAgent"),
        ]:
            # 重新运行以获取 ReviewResult（因为 findings_data 是 dict 列表）
            findings = [Finding(**f) for f in findings_data]
            agent._findings = findings
            result = agent.get_result()
            agent_results.append(result)
            all_findings_data.extend(findings_data)

        # 使用 SummaryAgent 生成报告
        summary = self._summary_agent.generate_summary(
            code=state.code,
            filename=state.filename,
            agent_results=agent_results,
            language=state.language,
        )

        logger.info(f"汇总完成: 总共 {summary['total_findings']} 个问题")
        return {
            "agent_results": [r.model_dump() for r in agent_results],
            "report_markdown": summary["report_markdown"],
            "report_json": summary["report_json"],
        }

    def build_graph(self) -> StateGraph:
        """构建 LangGraph StateGraph。

        Returns:
            编排完成的 StateGraph。
        """
        workflow = StateGraph(ReviewState)

        # 添加节点
        workflow.add_node("preprocess", self._preprocess)
        workflow.add_node("security_review", self._security_review)
        workflow.add_node("performance_review", self._performance_review)
        workflow.add_node("style_review", self._style_review)
        workflow.add_node("logic_review", self._logic_review)
        workflow.add_node("summarize", self._aggregate_and_summarize)

        # 定义边：预处理 -> 并行审查 -> 汇总
        workflow.add_edge(START, "preprocess")
        workflow.add_edge("preprocess", "security_review")
        workflow.add_edge("preprocess", "performance_review")
        workflow.add_edge("preprocess", "style_review")
        workflow.add_edge("preprocess", "logic_review")

        # 四个并行审查完成后进入汇总
        workflow.add_edge("security_review", "summarize")
        workflow.add_edge("performance_review", "summarize")
        workflow.add_edge("style_review", "summarize")
        workflow.add_edge("logic_review", "summarize")

        workflow.add_edge("summarize", END)

        return workflow


def create_review_graph() -> StateGraph:
    """创建审查工作流 StateGraph 实例。

    Returns:
        编译后的 StateGraph。
    """
    workflow = ReviewWorkflow()
    graph = workflow.build_graph()
    return graph.compile()
