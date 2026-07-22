"""安全审查 Agent。"""

from __future__ import annotations

import ast
import re
from typing import Any, Optional

from agents.base import BaseAgent
from analyzers.ast_analyzer import ASTAnalyzer
from app.models import Category, Finding, Severity

# 安全相关模式
PATTERNS = {
    "sql_injection": re.compile(
        r'(?:execute|cursor\.execute)\s*\(\s*(?:f["\']|".*%|".*\+|\'.*%|\'.*\+)',
        re.IGNORECASE,
    ),
    "hardcoded_password": re.compile(
        r'(?:password|passwd|pwd|secret|api_key|apikey|token|auth)\s*=\s*["\'][^"\']{3,}["\']',
        re.IGNORECASE,
    ),
    "hardcoded_url_with_creds": re.compile(
        r'(?:https?://)[^\s:]+:[^\s@]+@[^\s]+',
    ),
    "eval_usage": re.compile(
        r'\beval\s*\(',
    ),
    "exec_usage": re.compile(
        r'\bexec\s*\(',
    ),
    "pickle_usage": re.compile(
        r'\bpickle\.(?:loads|load|dumps|dump)\s*\(',
    ),
    "subprocess_shell": re.compile(
        r'subprocess\.(?:call|run|Popen)\s*\([^)]*shell\s*=\s*True',
    ),
    "assert_usage": re.compile(
        r'\bassert\s+',
    ),
    "md5_usage": re.compile(
        r'\bhashlib\.md5\b',
    ),
    "os_system": re.compile(
        r'\bos\.(?:system|popen)\s*\(',
    ),
    "dangerous_import": re.compile(
        r'\bimport\s+(?:pickle|subprocess|ctypes|shutil)\b',
    ),
}

# 不安全的依赖列表
UNSAFE_DEPENDENCIES = {
    "pickle": "存在反序列化漏洞风险，建议使用 json 或 msgpack 替代",
    "subprocess": "直接执行系统命令有命令注入风险，应使用参数列表形式",
    "ctypes": "直接调用 C 函数可能导致内存安全问题",
    "eval": "执行任意代码，存在代码注入风险",
    "exec": "执行任意代码，存在代码注入风险",
}


class SecurityAgent(BaseAgent):
    """安全漏洞检测 Agent。

    检测 SQL 注入、XSS、硬编码密钥、不安全依赖等安全问题。
    结合 AST 分析和正则模式匹配进行检测。
    """

    def __init__(self) -> None:
        super().__init__(name="SecurityAgent", category=Category.SECURITY)
        self._ast_analyzer = ASTAnalyzer()

    def analyze(self, code: str, filename: str, **kwargs: Any) -> list[Finding]:
        """分析代码中的安全问题。

        Args:
            code: 待分析的代码内容。
            filename: 文件名。
            **kwargs: 额外参数，支持 ast_result 直接传入 AST 分析结果。

        Returns:
            发现的安全问题列表。
        """
        self.reset()
        lines = code.split("\n")

        # 1. 基于正则模式的静态检测
        self._pattern_scan(lines, filename)

        # 2. 基于 AST 的安全检测
        try:
            ast_result = kwargs.get("ast_result") or self._ast_analyzer.analyze(code)
            self._ast_security_scan(ast_result, filename, lines)
        except SyntaxError:
            pass

        # 3. LLM 辅助分析（如有 LLM 配置）
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

    def _pattern_scan(self, lines: list[str], filename: str) -> None:
        """基于正则模式扫描安全问题。"""
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue

            for pattern_name, pattern in PATTERNS.items():
                if pattern.search(line):
                    severity = self._get_pattern_severity(pattern_name)
                    message = self._get_pattern_message(pattern_name)
                    suggestion = self._get_pattern_suggestion(pattern_name)
                    self.add_finding(
                        file=filename,
                        line=i,
                        severity=severity,
                        message=message,
                        suggestion=suggestion,
                        code_snippet=stripped[:100],
                        rule_id=f"SEC-{pattern_name.upper()}",
                    )
                    break  # 每行最多报告一个安全问题

    def _ast_security_scan(
        self,
        ast_result: dict,
        filename: str,
        lines: list[str],
    ) -> None:
        """基于 AST 分析结果进行安全扫描。"""
        imports = ast_result.get("imports", [])
        for imp in imports:
            module = imp.get("module", "")
            for unsafe_dep, reason in UNSAFE_DEPENDENCIES.items():
                if unsafe_dep in module:
                    self.add_finding(
                        file=filename,
                        line=imp.get("line", 1),
                        severity=Severity.HIGH,
                        message=f"不安全依赖: 导入了 {module}。{reason}",
                        suggestion=f"评估是否可以用更安全的替代方案替代 {module}",
                        rule_id="SEC-UNSAFE-DEP",
                    )

        # 检查函数调用中的危险模式
        function_calls = ast_result.get("function_calls", [])
        for call in function_calls:
            func_name = call.get("name", "")
            if func_name in ("eval", "exec"):
                self.add_finding(
                    file=filename,
                    line=call.get("line", 1),
                    severity=Severity.CRITICAL,
                    message=f"危险函数调用: 使用了 {func_name}()，可能导致代码注入",
                    suggestion=f"避免使用 {func_name}()，使用 ast.literal_eval() 或其他安全替代",
                    code_snippet=lines[call.get("line", 1) - 1].strip() if call.get("line", 1) <= len(lines) else None,
                    rule_id=f"SEC-{func_name.upper()}",
                )

    @staticmethod
    def _get_pattern_severity(pattern_name: str) -> Severity:
        """获取模式对应的严重等级。"""
        severity_map = {
            "sql_injection": Severity.CRITICAL,
            "hardcoded_password": Severity.HIGH,
            "hardcoded_url_with_creds": Severity.CRITICAL,
            "eval_usage": Severity.CRITICAL,
            "exec_usage": Severity.CRITICAL,
            "pickle_usage": Severity.HIGH,
            "subprocess_shell": Severity.HIGH,
            "assert_usage": Severity.LOW,
            "md5_usage": Severity.MEDIUM,
            "os_system": Severity.HIGH,
            "dangerous_import": Severity.MEDIUM,
        }
        return severity_map.get(pattern_name, Severity.MEDIUM)

    @staticmethod
    def _get_pattern_message(pattern_name: str) -> str:
        """获取模式对应的描述信息。"""
        messages = {
            "sql_injection": "疑似 SQL 注入: 使用了字符串拼接构造 SQL 查询",
            "hardcoded_password": "疑似硬编码密码/密钥: 检测到敏感凭证直接写入代码",
            "hardcoded_url_with_creds": "URL 中包含硬编码凭证",
            "eval_usage": "使用 eval() 可能导致任意代码执行",
            "exec_usage": "使用 exec() 可能导致任意代码执行",
            "pickle_usage": "使用 pickle 可能导致反序列化漏洞",
            "subprocess_shell": "subprocess 使用 shell=True 存在命令注入风险",
            "assert_usage": "assert 在生产环境中会被优化掉，不应用于安全检查",
            "md5_usage": "使用 MD5 哈希算法，已不安全，建议使用 SHA-256",
            "os_system": "使用 os.system() 存在命令注入风险",
            "dangerous_import": "导入了可能存在安全风险的模块",
        }
        return messages.get(pattern_name, "安全风险")

    @staticmethod
    def _get_pattern_suggestion(pattern_name: str) -> str:
        """获取模式对应的修复建议。"""
        suggestions = {
            "sql_injection": "使用参数化查询 (placeholder) 替代字符串拼接",
            "hardcoded_password": "使用环境变量或配置管理工具管理敏感凭证",
            "hardcoded_url_with_creds": "使用环境变量或密钥管理服务管理凭证",
            "eval_usage": "使用 ast.literal_eval() 或 json.loads() 替代",
            "exec_usage": "重构代码避免使用 exec()，使用函数或类替代",
            "pickle_usage": "使用 json 或 msgpack 等安全序列化格式",
            "subprocess_shell": "使用 shell=False 并传递参数列表",
            "assert_usage": "使用显式的条件检查替代 assert",
            "md5_usage": "使用 hashlib.sha256() 替代 MD5",
            "os_system": "使用 subprocess.run() 并传递参数列表",
            "dangerous_import": "评估该模块的使用场景，确保安全使用",
        }
        return suggestions.get(pattern_name, "请审查此代码的安全性")
