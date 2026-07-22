"""Agent 基类定义。"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from app.models import Category, Finding, ReviewResult, Severity

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """代码审查 Agent 基类。

    所有专业审查 Agent 继承此基类，实现 analyze 方法。
    每个 Agent 负责特定维度的代码审查。
    """

    def __init__(self, name: str, category: Category) -> None:
        """初始化 Agent。

        Args:
            name: Agent 名称。
            category: 该 Agent 负责的审查类别。
        """
        self.name = name
        self.category = category
        self._findings: list[Finding] = []

    def reset(self) -> None:
        """重置 Agent 状态，清空之前的审查结果。"""
        self._findings = []

    def add_finding(
        self,
        file: str,
        line: int,
        severity: Severity,
        message: str,
        suggestion: Optional[str] = None,
        code_snippet: Optional[str] = None,
        column: Optional[int] = None,
        rule_id: Optional[str] = None,
    ) -> None:
        """添加一条审查发现。

        Args:
            file: 文件路径。
            line: 行号。
            severity: 严重等级。
            message: 问题描述。
            suggestion: 修复建议。
            code_snippet: 相关代码片段。
            column: 列号。
            rule_id: 规则标识。
        """
        finding = Finding(
            file=file,
            line=line,
            column=column,
            severity=severity,
            category=self.category,
            message=message,
            suggestion=suggestion,
            code_snippet=code_snippet,
            rule_id=rule_id,
        )
        self._findings.append(finding)

    @abstractmethod
    def analyze(self, code: str, filename: str, **kwargs: Any) -> list[Finding]:
        """分析代码，返回发现的问题列表。

        Args:
            code: 待分析的代码内容。
            filename: 文件名。
            **kwargs: 额外参数（如 AST 分析结果）。

        Returns:
            发现的问题列表。
        """
        ...

    async def analyze_async(self, code: str, filename: str, **kwargs: Any) -> list[Finding]:
        """异步分析代码。

        Args:
            code: 待分析的代码内容。
            filename: 文件名。
            **kwargs: 额外参数。

        Returns:
            发现的问题列表。
        """
        return self.analyze(code, filename, **kwargs)

    def get_result(self, elapsed_time: float = 0.0) -> ReviewResult:
        """获取审查结果。

        Args:
            elapsed_time: 审查耗时(秒)。

        Returns:
            结构化的审查结果。
        """
        summary = f"{self.name} 发现 {len(self._findings)} 个问题"
        if self._findings:
            critical = sum(1 for f in self._findings if f.severity == Severity.CRITICAL)
            high = sum(1 for f in self._findings if f.severity == Severity.HIGH)
            if critical > 0:
                summary += f"（其中 {critical} 个严重, {high} 个高危）"
            elif high > 0:
                summary += f"（其中 {high} 个高危）"

        return ReviewResult(
            agent_name=self.name,
            findings=self._findings.copy(),
            summary=summary,
            elapsed_time=elapsed_time,
        )
