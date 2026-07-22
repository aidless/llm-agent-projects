"""逻辑缺陷审查 Agent。"""

from __future__ import annotations

import re
from typing import Any

from agents.base import BaseAgent
from analyzers.ast_analyzer import ASTAnalyzer
from app.models import Category, Finding, Severity

# 逻辑缺陷模式
LOGIC_PATTERNS = {
    "bare_except": re.compile(r"except\s*:"),
    "mutable_default": re.compile(
        r"def\s+\w+\s*\([^)]*=\s*(?:\[\]|\{\}|\[.*\]|\{.*\})[^)]*\)"
    ),
    "none_comparison": re.compile(r"(?:==|!=)\s+None\b"),
    "is_not_comparison": re.compile(r"\bis\s+(not\s+)?(True|False|None)\b"),
    "return_in_finally": re.compile(r"finally\s*:.*return"),
    "assign_in_condition": re.compile(
        r"(?:if|while|elif)\s*\([^)]*(?:=[^=])[^)]*\):",
    ),
    "unreachable_code": re.compile(r"return\s+\S.*\n\s+\S"),
    "useless_comparison": re.compile(
        r"(?:if|elif)\s+\w+\s*(?:==|!=)\s+(?:True|False)\s*:",
    ),
    "shadow_builtin": re.compile(
        r"^\s*(list|dict|set|tuple|str|int|float|bool|type|id|input|open|print|map|filter|zip|range|len|max|min|sum)\s*=",
        re.MULTILINE,
    ),
    "empty_body": re.compile(
        r"(?:if|else|for|while|def|class|try|except|finally)\s*.*:\s*$",
        re.MULTILINE,
    ),
}


class LogicAgent(BaseAgent):
    """逻辑缺陷检测 Agent。

    检测空指针引用、边界条件、异常处理问题等逻辑缺陷。
    """

    def __init__(self) -> None:
        super().__init__(name="LogicAgent", category=Category.LOGIC)
        self._ast_analyzer = ASTAnalyzer()

    def analyze(self, code: str, filename: str, **kwargs: Any) -> list[Finding]:
        """分析代码中的逻辑问题。

        Args:
            code: 待分析的代码内容。
            filename: 文件名。
            **kwargs: 额外参数。

        Returns:
            发现的逻辑问题列表。
        """
        self.reset()
        lines = code.split("\n")

        # 1. 基于正则模式的逻辑缺陷检测
        self._pattern_scan(lines, filename, code)

        # 2. AST 级别逻辑分析
        try:
            ast_result = kwargs.get("ast_result") or self._ast_analyzer.analyze(code)
            self._ast_logic_scan(ast_result, filename, lines)
        except SyntaxError:
            pass

        # 3. LLM 辅助分析
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

    def _pattern_scan(self, lines: list[str], filename: str, code: str) -> None:
        """基于正则模式扫描逻辑缺陷。"""
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # 裸 except
            if LOGIC_PATTERNS["bare_except"].search(line):
                self.add_finding(
                    file=filename,
                    line=i,
                    severity=Severity.HIGH,
                    message="使用了裸 except 子句，会捕获所有异常包括 KeyboardInterrupt",
                    suggestion="指定具体的异常类型，如 except Exception: 或 except (ValueError, TypeError):",
                    code_snippet=stripped[:100],
                    rule_id="LOGIC-BARE-EXCEPT",
                )

            # 可变默认参数
            if LOGIC_PATTERNS["mutable_default"].search(line):
                self.add_finding(
                    file=filename,
                    line=i,
                    severity=Severity.MEDIUM,
                    message="函数使用了可变默认参数，可能导致意外的共享状态",
                    suggestion="使用 None 作为默认值，在函数体内创建新实例",
                    code_snippet=stripped[:100],
                    rule_id="LOGIC-MUTABLE-DEFAULT",
                )

            # None 比较
            if LOGIC_PATTERNS["none_comparison"].search(line):
                self.add_finding(
                    file=filename,
                    line=i,
                    severity=Severity.LOW,
                    message="使用 ==/!= 比较 None，建议使用 is/is not",
                    suggestion="将 == None 改为 is None，将 != None 改为 is not None",
                    code_snippet=stripped[:100],
                    rule_id="LOGIC-NONE-COMPARISON",
                )

            # 覆盖内置函数
            if LOGIC_PATTERNS["shadow_builtin"].search(line):
                builtin_name = LOGIC_PATTERNS["shadow_builtin"].search(line).group(1)
                self.add_finding(
                    file=filename,
                    line=i,
                    severity=Severity.MEDIUM,
                    message=f"覆盖了内置函数 '{builtin_name}'",
                    suggestion=f"使用不同的变量名，避免覆盖 {builtin_name}",
                    code_snippet=stripped[:100],
                    rule_id="LOGIC-SHADOW-BUILTIN",
                )

            # 空函数体
            if LOGIC_PATTERNS["empty_body"].search(line):
                # 检查下一行是否有 pass、... 或 docstring
                if i < len(lines):
                    next_line = lines[i].strip()
                    is_valid_body = next_line in ("pass", "...", "pass  # ...", "")
                    # docstring 也是有效的函数体开始
                    if not is_valid_body and (next_line.startswith('"""') or next_line.startswith("'''") or next_line.startswith('r"""') or next_line.startswith("r'''")):
                        is_valid_body = True
                    if not is_valid_body:
                        self.add_finding(
                            file=filename,
                            line=i,
                            severity=Severity.HIGH,
                            message="代码块体为空，缺少 pass 或 ...",
                            suggestion="添加 pass 或 ... 作为占位符",
                            code_snippet=stripped[:100],
                            rule_id="LOGIC-EMPTY-BODY",
                        )

    def _ast_logic_scan(
        self,
        ast_result: dict,
        filename: str,
        lines: list[str],
    ) -> None:
        """基于 AST 分析结果进行逻辑缺陷扫描。"""
        # 检查异常处理
        exception_handlers = ast_result.get("exception_handlers", [])
        for handler in exception_handlers:
            handler_type = handler.get("type", "")
            if not handler_type:
                self.add_finding(
                    file=filename,
                    line=handler.get("line", 1),
                    severity=Severity.HIGH,
                    message="except 子句没有指定异常类型",
                    suggestion="指定具体的异常类型",
                    rule_id="LOGIC-BARE-EXCEPT-AST",
                )

        # 检查函数是否有返回值
        functions = ast_result.get("functions", [])
        for func in functions:
            name = func.get("name", "")
            has_return = func.get("has_return", False)
            is_generator = func.get("is_generator", False)

            # 跳过 __init__ 和特殊方法
            if name.startswith("__") and name.endswith("__"):
                continue

            # 跳过无返回的函数（如只做副作用的函数）
            has_return_annotation = func.get("has_return_annotation", False)
            if has_return_annotation and not has_return and not is_generator:
                self.add_finding(
                    file=filename,
                    line=func.get("line", 1),
                    severity=Severity.MEDIUM,
                    message=f"函数 '{name}' 声明了返回类型注解但没有 return 语句",
                    suggestion="添加 return 语句或修改返回类型注解为 -> None",
                    rule_id="LOGIC-MISSING-RETURN",
                )

            # 检查只有 return None 的函数
            returns_none = func.get("returns_none_only", False)
            if returns_none and has_return_annotation:
                self.add_finding(
                    file=filename,
                    line=func.get("line", 1),
                    severity=Severity.INFO,
                    message=f"函数 '{name}' 只返回 None，返回类型注解可能需要调整",
                    suggestion="确认返回类型注解与实际返回值一致",
                    rule_id="LOGIC-RETURN-NONE",
                )

        # 检查未处理的变量赋值
        assigned_vars = set()
        for func in functions:
            assigned_vars.update(func.get("assigned_vars", []))

        # 检查可能的除零错误
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if "/" in stripped and not stripped.startswith("#"):
                # 简单的除零检查提示
                if re.search(r"\w+\s*/\s*\w+", stripped):
                    # 检查周围是否有保护
                    context_start = max(0, i - 3)
                    context = "\n".join(lines[context_start:i])
                    if "if " not in context and "try:" not in context and "ZeroDivisionError" not in context:
                        match = re.search(r"(\w+)\s*/\s*(\w+)", stripped)
                        if match:
                            divisor = match.group(2)
                            if divisor not in ("2", "3", "4", "5", "6", "7", "8", "9", "10", "100", "1000"):
                                self.add_finding(
                                    file=filename,
                                    line=i,
                                    severity=Severity.LOW,
                                    message=f"除法操作可能存在除零错误（除数: {divisor}）",
                                    suggestion=f"在使用 {divisor} 做除数前检查其是否为 0",
                                    code_snippet=stripped[:100],
                                    rule_id="LOGIC-DIVISION-BY-ZERO",
                                )
