"""审查报告生成模板。"""

from __future__ import annotations

from datetime import datetime
from typing import Any


class ReportTemplate:
    """审查报告模板。

    生成 Markdown 格式和 JSON 格式的审查报告。
    """

    SEVERITY_ICONS = {
        "Critical": "X",
        "High": "!",
        "Medium": "~",
        "Low": "-",
        "Info": "i",
    }

    @staticmethod
    def generate_markdown_report(
        filename: str,
        language: str,
        findings: list[Any],
        agent_results: list[Any],
        severity_counts: dict[str, int],
        category_groups: dict[str, list[Any]],
    ) -> str:
        """生成 Markdown 格式报告。

        Args:
            filename: 文件名。
            language: 编程语言。
            findings: 所有问题列表。
            agent_results: 各 Agent 审查结果。
            severity_counts: 严重等级统计。
            category_groups: 按类别分组的问题。

        Returns:
            Markdown 格式报告字符串。
        """
        lines: list[str] = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 标题
        lines.append(f"# Code Review Report")
        lines.append("")
        lines.append(f"**File:** `{filename}`  ")
        lines.append(f"**Language:** {language}  ")
        lines.append(f"**Time:** {now}  ")
        lines.append(f"**Total Findings:** {len(findings)}")
        lines.append("")

        # 统计概览
        lines.append("## Summary")
        lines.append("")
        lines.append("| Severity | Count |")
        lines.append("|----------|-------|")
        for sev in ["Critical", "High", "Medium", "Low", "Info"]:
            count = severity_counts.get(sev, 0)
            if count > 0:
                icon = ReportTemplate.SEVERITY_ICONS.get(sev, "?")
                lines.append(f"| [{icon}] {sev} | {count} |")
        lines.append("")

        # 按 Agent 分组的摘要
        lines.append("## Agent Results")
        lines.append("")
        for result in agent_results:
            name = getattr(result, "agent_name", "unknown")
            findings_count = len(getattr(result, "findings", []))
            summary = getattr(result, "summary", "")
            elapsed = getattr(result, "elapsed_time", 0.0)
            lines.append(f"### {name}")
            lines.append(f"- **Findings:** {findings_count}")
            lines.append(f"- **Time:** {elapsed:.2f}s")
            lines.append(f"- **Summary:** {summary}")
            lines.append("")

        # 按类别分组的问题详情
        lines.append("## Findings by Category")
        lines.append("")

        for category, cat_findings in category_groups.items():
            lines.append(f"### {category}")
            lines.append("")

            for finding in cat_findings:
                sev = finding.severity.value if hasattr(finding, "severity") else "Unknown"
                icon = ReportTemplate.SEVERITY_ICONS.get(sev, "?")
                line_num = finding.line if hasattr(finding, "line") else 0
                message = finding.message if hasattr(finding, "message") else ""
                rule_id = finding.rule_id if hasattr(finding, "rule_id") else ""

                lines.append(f"**[{icon}] [{sev}]** L{line_num}: {message}")
                if rule_id:
                    lines.append(f"  Rule: `{rule_id}`")
                if hasattr(finding, "suggestion") and finding.suggestion:
                    lines.append(f"  Suggestion: {finding.suggestion}")
                if hasattr(finding, "code_snippet") and finding.code_snippet:
                    lines.append(f"  ```")
                    lines.append(f"  {finding.code_snippet}")
                    lines.append(f"  ```")
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def generate_json_report(
        filename: str,
        language: str,
        findings: list[Any],
        agent_results: list[Any],
        severity_counts: dict[str, int],
        category_groups: dict[str, list[Any]],
    ) -> dict[str, Any]:
        """生成 JSON 格式报告。

        Args:
            filename: 文件名。
            language: 编程语言。
            findings: 所有问题列表。
            agent_results: 各 Agent 审查结果。
            severity_counts: 严重等级统计。
            category_groups: 按类别分组的问题。

        Returns:
            JSON 可序列化的字典。
        """
        findings_data = []
        for f in findings:
            findings_data.append({
                "file": f.file,
                "line": f.line,
                "column": f.column,
                "severity": f.severity.value,
                "category": f.category.value,
                "message": f.message,
                "suggestion": f.suggestion,
                "code_snippet": f.code_snippet,
                "rule_id": f.rule_id,
            })

        agent_data = []
        for r in agent_results:
            agent_data.append({
                "agent_name": r.agent_name,
                "findings_count": len(r.findings),
                "summary": r.summary,
                "elapsed_time": r.elapsed_time,
            })

        return {
            "metadata": {
                "filename": filename,
                "language": language,
                "generated_at": datetime.now().isoformat(),
                "total_findings": len(findings),
            },
            "severity_summary": severity_counts,
            "category_summary": {
                cat: len(items) for cat, items in category_groups.items()
            },
            "agent_results": agent_data,
            "findings": findings_data,
        }
