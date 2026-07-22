"""汇总报告 Agent。"""

from __future__ import annotations

from typing import Any, Optional

from agents.base import BaseAgent
from app.models import Category, Finding, ReviewResult, Severity
from templates.report_template import ReportTemplate


class SummaryAgent(BaseAgent):
    """汇总报告 Agent。

    聚合所有 Agent 的审查结果，生成统一的审查报告。
    """

    def __init__(self) -> None:
        super().__init__(name="SummaryAgent", category=Category.SECURITY)

    def analyze(self, code: str, filename: str, **kwargs: Any) -> list[Finding]:
        """此 Agent 不直接分析代码，而是汇总其他 Agent 的结果。"""
        return []

    def generate_summary(
        self,
        code: str,
        filename: str,
        agent_results: list[ReviewResult],
        language: str = "python",
    ) -> dict[str, Any]:
        """汇总所有 Agent 的审查结果，生成报告。

        Args:
            code: 原始代码。
            filename: 文件名。
            agent_results: 各 Agent 的审查结果列表。
            language: 编程语言。

        Returns:
            包含 Markdown 报告和 JSON 报告的字典。
        """
        all_findings: list[Finding] = []
        for result in agent_results:
            all_findings.extend(result.findings)

        # 按严重等级排序
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }
        sorted_findings = sorted(
            all_findings,
            key=lambda f: severity_order.get(f.severity, 5),
        )

        # 统计各级别数量
        severity_counts: dict[str, int] = {}
        for f in sorted_findings:
            severity_counts[f.severity.value] = severity_counts.get(f.severity.value, 0) + 1

        # 按类别分组
        category_groups: dict[str, list[Finding]] = {}
        for f in sorted_findings:
            cat = f.category.value
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(f)

        # 生成 Markdown 报告
        report_markdown = ReportTemplate.generate_markdown_report(
            filename=filename,
            language=language,
            findings=sorted_findings,
            agent_results=agent_results,
            severity_counts=severity_counts,
            category_groups=category_groups,
        )

        # 生成 JSON 报告
        report_json = ReportTemplate.generate_json_report(
            filename=filename,
            language=language,
            findings=sorted_findings,
            agent_results=agent_results,
            severity_counts=severity_counts,
            category_groups=category_groups,
        )

        return {
            "report_markdown": report_markdown,
            "report_json": report_json,
            "total_findings": len(sorted_findings),
            "severity_counts": severity_counts,
            "category_counts": {cat: len(items) for cat, items in category_groups.items()},
        }

    def get_overall_score(self, agent_results: list[ReviewResult]) -> float:
        """计算总体审查评分（0-100）。"""
        if not agent_results:
            return 100.0

        total_findings = sum(len(r.findings) for r in agent_results)
        if total_findings == 0:
            return 100.0

        # 加权扣分
        deductions = 0.0
        for result in agent_results:
            for f in result.findings:
                weight = {
                    Severity.CRITICAL: 20,
                    Severity.HIGH: 10,
                    Severity.MEDIUM: 5,
                    Severity.LOW: 2,
                    Severity.INFO: 1,
                }.get(f.severity, 1)
                deductions += weight

        score = max(0.0, 100.0 - deductions)
        return round(score, 1)
