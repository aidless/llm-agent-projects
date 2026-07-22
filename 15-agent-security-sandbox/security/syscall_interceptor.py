"""系统调用拦截器 - AST 级别危险函数检测 + 运行时 hook"""

import ast
from typing import List, Set, Optional
from dataclasses import dataclass, field


@dataclass
class SecurityViolation:
    """安全违规事件"""
    violation_type: str
    node_type: str
    line: int
    col: int
    detail: str
    severity: str = "HIGH"

    def to_dict(self) -> dict:
        return {
            "violation_type": self.violation_type,
            "node_type": self.node_type,
            "line": self.line,
            "col": self.col,
            "detail": self.detail,
            "severity": self.severity,
        }


@dataclass
class InterceptorConfig:
    """拦截器配置"""
    blocked_builtins: Set[str] = field(default_factory=lambda: {
        "exec", "eval", "compile", "__import__",
        "open", "input", "breakpoint",
    })
    blocked_modules: Set[str] = field(default_factory=lambda: {
        "os", "subprocess", "shutil", "sys",
        "socket", "requests", "urllib",
        "ctypes", "multiprocessing", "threading",
        "signal", "resource", "pty", "fcntl",
    })
    blocked_attributes: Set[str] = field(default_factory=lambda: {
        "system", "popen", "spawn", "exec", "eval",
        "fork", "kill", "send", "connect", "bind",
        "listen", "accept", "getaddrinfo",
    })
    allowed_builtins: Set[str] = field(default_factory=lambda: {
        "print", "len", "range", "int", "str", "float",
        "list", "dict", "set", "tuple", "bool", "bytes",
        "type", "isinstance", "issubclass", "hasattr",
        "getattr", "setattr", "delattr", "repr", "format",
        "sorted", "reversed", "enumerate", "zip", "map",
        "filter", "abs", "max", "min", "sum", "round",
        "pow", "divmod", "hex", "oct", "bin", "ord", "chr",
        "id", "hash", "dir", "vars", "callable", "iter",
        "next", "all", "any", "slice", "super", "property",
        "staticmethod", "classmethod", "complex", "memoryview",
        "frozenset", "bytearray", "help", "object", "Exception",
        "ValueError", "TypeError", "KeyError", "IndexError",
        "RuntimeError", "AttributeError", "NameError",
        "StopIteration", "NotImplementedError", "IOError",
        "ZeroDivisionError", "AssertionError", "ImportError",
        "OSError", "FileNotFoundError", "PermissionError",
    })
    allowed_modules: Set[str] = field(default_factory=lambda: {
        "math", "random", "string", "re", "json",
        "datetime", "collections", "itertools", "functools",
        "decimal", "fractions", "statistics", "typing",
        "dataclasses", "enum", "copy", "operator",
        "abc", "numbers", "hashlib", "hmac", "base64",
        "uuid", "time", "struct", "array", "queue",
        "heapq", "bisect", "pprint", "textwrap",
    })


class SyscallInterceptor:
    """系统调用拦截器 - 基于 AST 分析检测危险代码"""

    def __init__(self, config: Optional[InterceptorConfig] = None):
        self.config = config or InterceptorConfig()
        self.violations: List[SecurityViolation] = []

    def clear_violations(self):
        self.violations.clear()

    def analyze(self, code: str) -> List[SecurityViolation]:
        """分析代码，检测危险调用"""
        self.clear_violations()
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            self.violations.append(SecurityViolation(
                violation_type="SYNTAX_ERROR",
                node_type="Module",
                line=e.lineno or 0,
                col=e.offset or 0,
                detail=str(e.msg),
                severity="MEDIUM",
            ))
            return self.violations

        self._visit(tree)
        return self.violations

    def _visit(self, node: ast.AST):
        """遍历 AST 节点"""
        for child in ast.iter_child_nodes(node):
            self._check_node(child)
            self._visit(child)

    def _check_node(self, node: ast.AST):
        """检查单个 AST 节点"""
        if isinstance(node, ast.Import):
            self._check_import(node)
        elif isinstance(node, ast.ImportFrom):
            self._check_import_from(node)
        elif isinstance(node, ast.Call):
            self._check_call(node)
        elif isinstance(node, ast.Attribute):
            self._check_attribute(node)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            self._check_function_def(node)
        elif isinstance(node, ast.ClassDef):
            self._check_class_def(node)

    def _check_import(self, node: ast.Import):
        """检查 import 语句"""
        for alias in node.names:
            module = alias.name.split(".")[0]
            if module in self.config.blocked_modules:
                if module not in self.config.allowed_modules:
                    self.violations.append(SecurityViolation(
                        violation_type="BLOCKED_IMPORT",
                        node_type="Import",
                        line=node.lineno,
                        col=node.col_offset,
                        detail=f"Import of blocked module: {alias.name}",
                        severity="HIGH",
                    ))

    def _check_import_from(self, node: ast.ImportFrom):
        """检查 from ... import ... 语句"""
        if node.module:
            module = node.module.split(".")[0]
            if module in self.config.blocked_modules:
                if module not in self.config.allowed_modules:
                    self.violations.append(SecurityViolation(
                        violation_type="BLOCKED_IMPORT_FROM",
                        node_type="ImportFrom",
                        line=node.lineno,
                        col=node.col_offset,
                        detail=f"Import from blocked module: {node.module}",
                        severity="HIGH",
                    ))

    def _check_call(self, node: ast.Call):
        """检查函数调用"""
        func_name = self._get_call_name(node)
        if func_name in self.config.blocked_builtins:
            self.violations.append(SecurityViolation(
                violation_type="BLOCKED_BUILTIN_CALL",
                node_type="Call",
                line=node.lineno,
                col=node.col_offset,
                detail=f"Call to blocked builtin: {func_name}",
                severity="HIGH",
            ))

        # 检查 getattr 用于绕过
        if func_name == "getattr":
            self._check_getattr_abuse(node)

    def _check_attribute(self, node: ast.Attribute):
        """检查属性访问"""
        attr = node.attr
        if attr in self.config.blocked_attributes:
            # 检查是否是允许的上下文
            full_name = self._get_full_attr_name(node)
            for blocked in self.config.blocked_attributes:
                if full_name.endswith(f".{blocked}") or attr == blocked:
                    # 排除字符串/bytes 方法
                    value_name = self._get_value_name(node.value)
                    safe_types = {"str", "bytes", "list", "dict", "set", "tuple"}
                    if value_name not in safe_types:
                        self.violations.append(SecurityViolation(
                            violation_type="BLOCKED_ATTRIBUTE_ACCESS",
                            node_type="Attribute",
                            line=node.lineno,
                            col=node.col_offset,
                            detail=f"Access to blocked attribute: {full_name}",
                            severity="HIGH",
                        ))
                    break

    def _check_function_def(self, node):
        """检查函数定义中的危险装饰器"""
        for decorator in node.decorator_list:
            dec_name = self._get_node_name(decorator)
            if dec_name and ("exec" in dec_name or "eval" in dec_name):
                self.violations.append(SecurityViolation(
                    violation_type="SUSPICIOUS_DECORATOR",
                    node_type="FunctionDef",
                    line=node.lineno,
                    col=node.col_offset,
                    detail=f"Suspicious decorator on function '{node.name}': {dec_name}",
                    severity="MEDIUM",
                ))

    def _check_class_def(self, node):
        """检查类定义"""
        for base in node.bases:
            base_name = self._get_node_name(base)
            if base_name and base_name.startswith("__"):
                self.violations.append(SecurityViolation(
                    violation_type="SUSPICIOUS_BASE_CLASS",
                    node_type="ClassDef",
                    line=node.lineno,
                    col=node.col_offset,
                    detail=f"Suspicious base class '{base_name}' in class '{node.name}'",
                    severity="MEDIUM",
                ))

    def _check_getattr_abuse(self, node: ast.Call):
        """检查 getattr 滥用尝试"""
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            attr_val = node.args[1].value
            if isinstance(attr_val, str) and attr_val in self.config.blocked_attributes:
                self.violations.append(SecurityViolation(
                    violation_type="GETATTR_BYPASS_ATTEMPT",
                    node_type="Call",
                    line=node.lineno,
                    col=node.col_offset,
                    detail=f"getattr used to access blocked attribute: {attr_val}",
                    severity="HIGH",
                ))

    def _get_call_name(self, node: ast.Call) -> str:
        """获取函数调用名称"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def _get_node_name(self, node: ast.AST) -> str:
        """获取节点名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _get_full_attr_name(self, node: ast.Attribute) -> str:
        """获取完整属性名"""
        parts = [node.attr]
        current = node.value
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    def _get_value_name(self, node: ast.AST) -> str:
        """获取值节点的类型名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            return self._get_call_name(node)
        return ""


def create_runtime_guard(config: Optional[InterceptorConfig] = None):
    """创建运行时安全守卫，用于 exec 环境中 hook 危险操作"""
    cfg = config or InterceptorConfig()

    class RuntimeGuard:
        def __init__(self):
            self.violations = []

        def safe_import(self, name, *args, **kwargs):
            module = name.split(".")[0]
            if module in cfg.blocked_modules and module not in cfg.allowed_modules:
                raise SecurityError(f"Runtime: blocked import of module '{name}'")
            return __builtins__["__import__"](name, *args, **kwargs) if isinstance(__builtins__, dict) else __import__(name, *args, **kwargs)

        def safe_open(self, *args, **kwargs):
            raise SecurityError("Runtime: open() is not allowed in sandbox")

        def safe_exec(self, *args, **kwargs):
            raise SecurityError("Runtime: exec() is not allowed in sandbox")

        def safe_eval(self, *args, **kwargs):
            raise SecurityError("Runtime: eval() is not allowed in sandbox")

    return RuntimeGuard()


class SecurityError(Exception):
    """安全错误"""
    pass