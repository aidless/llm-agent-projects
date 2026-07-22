"""性能审查 Agent。"""

from __future__ import annotations

import ast
import re
from typing import Any

from agents.base import BaseAgent
from analyzers.ast_analyzer import ASTAnalyzer
from analyzers.complexity import ComplexityAnalyzer
from app.models import Category, Finding, Severity

# 性能反模式
PERFORMANCE_PATTERNS = {
    "list_append_in_loop": re.compile(
        r'(?:for\s+.+\s+in\s+.+:)\s*(?:(?:.+\n)*?)(?:\w+)\.append\s*\(',
        re.MULTILINE,
    ),
    "string_concat_in_loop": re.compile(
        r'for\s+.+\s+in\s+.+:\s*(?:(?:.+\n)*?)(\w+)\s*\+\s*=',
        re.MULTILINE,
    ),
    "global_import": re.compile(
        r'^\s*import\s+\w+\s*$',
        re.MULTILINE,
    ),
    "not_using_set": re.compile(
        r'(?:"\w+"\s+in\s+\w+|\'\w+\'\s+in\s+\w+)\s*',
    ),
    "re_compilation": re.compile(
        r'for\s+.+\s+in\s+.+:\s*(?:(?:.+\n)*?)re\.compile\s*\(',
        re.MULTILINE,
    ),
    "os_walk": re.compile(
        r'os\.walk\s*\(',
    ),
}


class PerformanceAgent(BaseAgent):
    """性能问题检测 Agent。

    检测 N+1 查询、内存泄漏、算法复杂度等性能问题。
    """

    def __init__(self) -> None:
        super().__init__(name="PerformanceAgent", category=Category.PERFORMANCE)
        self._ast_analyzer = ASTAnalyzer()
        self._complexity_analyzer = ComplexityAnalyzer()

    def analyze(self, code: str, filename: str, **kwargs: Any) -> list[Finding]:
        """分析代码中的性能问题。

        Args:
            code: 待分析的代码内容。
            filename: 文件名。
            **kwargs: 额外参数。

        Returns:
            发现的性能问题列表。
        """
        self.reset()
        lines = code.split("\n")

        # 1. 圈复杂度检测
        self._check_complexity(code, filename)

        # 2. 基于正则模式的性能反模式检测
        self._pattern_scan(code, lines, filename)

        # 3. AST 级别性能分析
        try:
            ast_result = kwargs.get("ast_result") or self._ast_analyzer.analyze(code)
            self._ast_performance_scan(ast_result, filename, lines)
        except SyntaxError:
            pass

        # 4. LLM 辅助分析
        llm_findings = kwargs.get("llm_findings")
        if llm_findings:
            for f in llm_findings:
                self.add_finding(
                    file=filename,
                    line=f.get("line", 1),
                    severity=Severity(f.get("severity", "Medium")),
                    message=f.get("message", ""),
                    suggestion=f.get("suggestion", ""),
                    rule_id=f.get("rule_id"),
                )

        return self._findings

    def _check_complexity(self, code: str, filename: str) -> None:
        """检查代码圈复杂度。"""
        try:
            functions = self._complexity_analyzer.analyze(code)
            for func in functions:
                if func.complexity > 10:
                    self.add_finding(
                        file=filename,
                        line=func.line_start,
                        severity=Severity.HIGH,
                        message=(
                            f"函数 {func.name} 圈复杂度过高 ({func.complexity})，"
                            f"建议拆分为更小的函数（推荐复杂度 < 10）"
                        ),
                        suggestion=(
                            f"将函数 {func.name} 拆分为多个职责单一的子函数，"
                            f"降低圈复杂度至 10 以下"
                        ),
                        rule_id="PERF-COMPLEXITY",
                    )
                elif func.complexity > 5:
                    self.add_finding(
                        file=filename,
                        line=func.line_start,
                        severity=Severity.MEDIUM,
                        message=(
                            f"函数 {func.name} 圈复杂度偏高 ({func.complexity})，"
                            f"建议适当简化（推荐复杂度 < 5）"
                        ),
                        suggestion=f"考虑简化 {func.name} 中的条件逻辑",
                        rule_id="PERF-COMPLEXITY",
                    )
        except SyntaxError:
            pass

    def _pattern_scan(self, code: str, lines: list[str], filename: str) -> None:
        """基于正则模式扫描性能反模式。"""
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # 检测字符串拼接
            if "+=" in stripped and any(
                kw in stripped for kw in ("'", '"', "str(", "format")
            ):
                # 检查是否在循环中
                context = "\n".join(lines[max(0, i - 5) : i])
                if "for " in context:
                    self.add_finding(
                        file=filename,
                        line=i,
                        severity=Severity.MEDIUM,
                        message="在循环中使用字符串拼接，性能较低",
                        suggestion="使用 ''.join() 或列表收集后 join 替代字符串拼接",
                        code_snippet=stripped[:100],
                        rule_id="PERF-STR-CONCAT",
                    )

            # 检测 re.compile 在循环中
            if "re.compile" in stripped:
                context = "\n".join(lines[max(0, i - 5) : i])
                if "for " in context:
                    self.add_finding(
                        file=filename,
                        line=i,
                        severity=Severity.MEDIUM,
                        message="在循环中重复编译正则表达式",
                        suggestion="将 re.compile() 移到循环外部，预编译正则表达式",
                        code_snippet=stripped[:100],
                        rule_id="PERF-RE-COMPILE",
                    )

            # 检测不必要的全局导入
            if stripped.startswith("import ") and "from" not in stripped:
                modules = stripped.replace("import ", "").strip().split(",")
                for mod in modules:
                    mod = mod.strip().split(".")[0]
                    if mod in ("os", "sys", "json", "re", "time", "datetime"):
                        pass  # 标准库模块通常没问题
                    else:
                        self.add_finding(
                            file=filename,
                            line=i,
                            severity=Severity.INFO,
                            message=f"检测到导入 {mod}，考虑使用局部导入以减少启动时间",
                            suggestion="如果是偶尔使用的模块，考虑在函数内部导入",
                            rule_id="PERF-IMPORT",
                        )

    def _ast_performance_scan(
        self,
        ast_result: dict,
        filename: str,
        lines: list[str],
    ) -> None:
        """基于 AST 分析结果进行性能扫描。"""
        # 检查函数参数数量（过多的参数可能表明函数职责过重）
        functions = ast_result.get("functions", [])
        for func in functions:
            arg_count = len(func.get("args", []))
            if arg_count > 7:
                self.add_finding(
                    file=filename,
                    line=func.get("line", 1),
                    severity=Severity.INFO,
                    message=f"函数 {func.get('name')} 参数过多 ({arg_count})，考虑使用数据类封装",
                    suggestion="使用 dataclass 或 NamedTuple 封装相关参数",
                    rule_id="PERF-MANY-ARGS",
                )

        # 检查嵌套深度
        functions_info = ast_result.get("functions", [])
        for func in functions_info:
            nesting = func.get("max_nesting", 0)
            if nesting > 4:
                self.add_finding(
                    file=filename,
                    line=func.get("line", 1),
                    severity=Severity.MEDIUM,
                    message=f"函数 {func.get('name')} 嵌套深度过深 ({nesting} 层)",
                    suggestion="使用 early return 减少嵌套，或提取子函数",
                    rule_id="PERF-DEEP-NESTING",
                )
