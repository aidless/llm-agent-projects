"""代码风格审查 Agent。"""

from __future__ import annotations

import re
from typing import Any

from agents.base import BaseAgent
from analyzers.ast_analyzer import ASTAnalyzer
from app.models import Category, Finding, Severity

# 命名规范模式
NAMING_PATTERNS = {
    "bad_class_name": re.compile(r"class\s+([a-z]\w*)\s*[:(]"),
    "bad_function_name": re.compile(r"def\s+([A-Z]\w*)\s*\("),
    "bad_variable_name": re.compile(r"\b([A-Z]{2,}[a-z]\w*)\s*="),
    "constant_not_upper": re.compile(r"^[A-Z_][A-Z_0-9]*\s*=\s*", re.MULTILINE),
}


class StyleAgent(BaseAgent):
    """代码规范检查 Agent。

    检查 PEP8 规范、命名规范、类型注解、docstring 等。
    """

    def __init__(self) -> None:
        super().__init__(name="StyleAgent", category=Category.STYLE)
        self._ast_analyzer = ASTAnalyzer()

    def analyze(self, code: str, filename: str, **kwargs: Any) -> list[Finding]:
        """分析代码中的风格问题。

        Args:
            code: 待分析的代码内容。
            filename: 文件名。
            **kwargs: 额外参数。

        Returns:
            发现的风格问题列表。
        """
        self.reset()
        lines = code.split("\n")

        # 1. 基本格式检查
        self._check_formatting(lines, filename)

        # 2. 命名规范检查
        self._check_naming(lines, filename)

        # 3. AST 级别风格分析
        try:
            ast_result = kwargs.get("ast_result") or self._ast_analyzer.analyze(code)
            self._ast_style_scan(ast_result, filename, lines)
        except SyntaxError:
            pass

        # 4. LLM 辅助分析
        llm_findings = kwargs.get("llm_findings")
        if llm_findings:
            for f in llm_findings:
                self.add_finding(
                    file=filename,
                    line=f.get("line", 1),
                    severity=Severity(f.get("severity", "Low")),
                    message=f.get("message", ""),
                    suggestion=f.get("suggestion", ""),
                    rule_id=f.get("rule_id"),
                )

        return self._findings

    def _check_formatting(self, lines: list[str], filename: str) -> None:
        """检查代码基本格式。"""
        for i, line in enumerate(lines, start=1):
            # 检查行长度
            if len(line) > 120:
                self.add_finding(
                    file=filename,
                    line=i,
                    severity=Severity.INFO,
                    message=f"行过长 ({len(line)} 字符)，建议不超过 120 字符",
                    suggestion="将长行拆分为多行，使用括号进行续行",
                    rule_id="STYLE-LINE-LENGTH",
                )
            elif len(line) > 88:
                self.add_finding(
                    file=filename,
                    line=i,
                    severity=Severity.INFO,
                    message=f"行较长 ({len(line)} 字符)，PEP8 推荐不超过 79 字符",
                    rule_id="STYLE-LINE-LENGTH",
                )

            # 检查行尾空格
            if line.rstrip() != line and line.strip():
                self.add_finding(
                    file=filename,
                    line=i,
                    severity=Severity.INFO,
                    message="行末有多余空格",
                    suggestion="删除行尾空格",
                    rule_id="STYLE-TRAILING-WHITESPACE",
                )

            # 检查 Tab 缩进
            if line.startswith("\t"):
                self.add_finding(
                    file=filename,
                    line=i,
                    severity=Severity.LOW,
                    message="使用了 Tab 缩进，PEP8 推荐使用 4 个空格",
                    suggestion="将 Tab 替换为 4 个空格",
                    rule_id="STYLE-TAB-INDENT",
                )

            # 检查多余空行（两个以上连续空行）
            if i > 1 and not line.strip() and not lines[i - 2].strip():
                if i - 3 >= 0 and not lines[i - 3].strip():
                    self.add_finding(
                        file=filename,
                        line=i,
                        severity=Severity.INFO,
                        message="连续多个空行，PEP8 推荐最多两个空行分隔顶层定义",
                        suggestion="减少多余空行",
                        rule_id="STYLE-MULTIPLE-BLANK-LINES",
                    )

    def _check_naming(self, lines: list[str], filename: str) -> None:
        """检查命名规范。"""
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()

            # 检查类名是否使用 PascalCase
            match = NAMING_PATTERNS["bad_class_name"].search(stripped)
            if match:
                class_name = match.group(1)
                self.add_finding(
                    file=filename,
                    line=i,
                    severity=Severity.LOW,
                    message=f"类名 '{class_name}' 应使用 PascalCase 命名",
                    suggestion=f"将类名改为 {class_name[0].upper()}{class_name[1:]}",
                    rule_id="STYLE-CLASS-NAMING",
                )

            # 检查函数名是否使用 snake_case
            match = NAMING_PATTERNS["bad_function_name"].search(stripped)
            if match:
                func_name = match.group(1)
                # 排除特殊方法（__init__, __str__ 等）
                if not func_name.startswith("__"):
                    snake = re.sub(r"([A-Z])", r"_\1", func_name).lower().lstrip("_")
                    self.add_finding(
                        file=filename,
                        line=i,
                        severity=Severity.LOW,
                        message=f"函数名 '{func_name}' 应使用 snake_case 命名",
                        suggestion=f"将函数名改为 '{snake}'",
                        rule_id="STYLE-FUNC-NAMING",
                    )

    def _ast_style_scan(
        self,
        ast_result: dict,
        filename: str,
        lines: list[str],
    ) -> None:
        """基于 AST 分析结果进行风格扫描。"""
        # 检查函数是否有 docstring
        functions = ast_result.get("functions", [])
        for func in functions:
            has_docstring = func.get("has_docstring", False)
            if not has_docstring:
                line = func.get("line", 1)
                name = func.get("name", "")
                # 跳过私有方法和特殊方法
                if not name.startswith("_") or name.startswith("__"):
                    if not name.startswith("__") or name.endswith("__"):
                        self.add_finding(
                            file=filename,
                            line=line,
                            severity=Severity.INFO,
                            message=f"函数 '{name}' 缺少 docstring",
                            suggestion="为函数添加 docstring，说明参数、返回值和功能",
                            rule_id="STYLE-MISSING-DOCSTRING",
                        )

        # 检查函数是否有类型注解
        for func in functions:
            has_return_annotation = func.get("has_return_annotation", False)
            annotated_args = func.get("annotated_args", [])
            total_args = func.get("args", [])

            if not has_return_annotation and total_args:
                name = func.get("name", "")
                self.add_finding(
                    file=filename,
                    line=func.get("line", 1),
                    severity=Severity.INFO,
                    message=f"函数 '{name}' 缺少返回值类型注解",
                    suggestion="为函数添加返回值类型注解，如 -> None 或 -> str",
                    rule_id="STYLE-MISSING-TYPE-ANNOTATION",
                )

            # 检查未注解的参数
            unannotated = [
                a for a in total_args
                if a not in annotated_args and a != "self" and a != "cls"
            ]
            if unannotated and total_args:
                name = func.get("name", "")
                self.add_finding(
                    file=filename,
                    line=func.get("line", 1),
                    severity=Severity.INFO,
                    message=f"函数 '{name}' 的参数 {unannotated} 缺少类型注解",
                    suggestion="为所有参数添加类型注解",
                    rule_id="STYLE-MISSING-ARG-TYPE",
                )

        # 检查模块 docstring
        has_module_docstring = ast_result.get("has_module_docstring", False)
        if not has_module_docstring:
            self.add_finding(
                file=filename,
                line=1,
                severity=Severity.INFO,
                message="模块缺少 docstring",
                suggestion="在文件开头添加模块 docstring，说明模块功能",
                rule_id="STYLE-MISSING-MODULE-DOCSTRING",
            )

        # 检查 imports 顺序
        imports = ast_result.get("imports", [])
        if imports:
            last_stdlib = -1
            last_third_party = -1
            last_local = -1
            for idx, imp in enumerate(imports):
                module = imp.get("module", "")
                if module.startswith(("os", "sys", "json", "re", "ast", "time", "typing", "datetime", "logging", "functools", "collections", "hashlib")):
                    if last_local >= 0 or last_third_party > idx:
                        pass  # 顺序问题
                    last_stdlib = idx
                elif module.startswith("."):
                    if last_stdlib > idx or last_third_party > idx:
                        pass
                    last_local = idx
                else:
                    if last_local >= 0:
                        pass
                    last_third_party = idx

        # 检查 TODO/FIXME 注释
        for i, line in enumerate(lines, start=1):
            if "# TODO" in line or "# FIXME" in line or "# XXX" in line:
                self.add_finding(
                    file=filename,
                    line=i,
                    severity=Severity.INFO,
                    message=f"发现 {line.strip().split('#')[1].strip()[:50]} 标记",
                    suggestion="处理或跟踪此标记项",
                    rule_id="STYLE-TODO-COMMENT",
                )
