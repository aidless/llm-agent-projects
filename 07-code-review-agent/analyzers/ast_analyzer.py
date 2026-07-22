"""Python AST 分析器。"""

from __future__ import annotations

import ast
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ASTAnalyzer:
    """Python AST 代码分析器。

    使用 Python ast 模块解析代码，提取函数、类、导入、
    函数调用等结构信息，为审查 Agent 提供静态分析数据。
    """

    def analyze(self, code: str) -> dict[str, Any]:
        """分析 Python 代码的 AST 结构。

        Args:
            code: Python 源代码。

        Returns:
            包含分析结果的字典，包含 imports, functions, classes,
            function_calls, exception_handlers 等信息。
        """
        tree = ast.parse(code)
        result: dict[str, Any] = {
            "imports": [],
            "functions": [],
            "classes": [],
            "function_calls": [],
            "exception_handlers": [],
            "has_module_docstring": False,
        }

        # 检查模块 docstring
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            result["has_module_docstring"] = True

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result["imports"].append({
                        "module": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "type": "import",
                    })
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                for alias in node.names:
                    result["imports"].append({
                        "module": f"{module_name}.{alias.name}",
                        "alias": alias.asname,
                        "line": node.lineno,
                        "type": "from_import",
                    })
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                func_info = self._extract_function_info(node)
                result["functions"].append(func_info)
            elif isinstance(node, ast.ClassDef):
                class_info = self._extract_class_info(node)
                result["classes"].append(class_info)
            elif isinstance(node, ast.Call):
                call_info = self._extract_call_info(node)
                if call_info:
                    result["function_calls"].append(call_info)
            elif isinstance(node, ast.ExceptHandler):
                handler_info = self._extract_except_handler_info(node)
                result["exception_handlers"].append(handler_info)

        return result

    def _extract_function_info(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
        """提取函数信息。"""
        args = [arg.arg for arg in node.args.args]
        annotated_args = [
            arg.arg for arg in node.args.args
            if arg.annotation is not None
        ]

        # 检查 docstring
        has_docstring = False
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            has_docstring = True

        # 检查返回值注解
        has_return_annotation = node.returns is not None

        # 检查是否有 return 语句
        has_return = False
        returns_none_only = True
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Return):
                has_return = True
                if sub_node.value is not None:
                    returns_none_only = False

        # 检查是否是生成器
        is_generator = any(
            isinstance(n, ast.Yield | ast.YieldFrom)
            for n in ast.walk(node)
        )

        # 计算嵌套深度
        max_nesting = self._calculate_nesting_depth(node)

        # 获取赋值变量
        assigned_vars = set()
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Assign):
                for target in sub_node.targets:
                    if isinstance(target, ast.Name):
                        assigned_vars.add(target.id)

        return {
            "name": node.name,
            "line": node.lineno,
            "line_end": node.end_lineno or node.lineno,
            "args": args,
            "annotated_args": annotated_args,
            "has_docstring": has_docstring,
            "has_return_annotation": has_return_annotation,
            "has_return": has_return,
            "returns_none_only": returns_none_only if has_return else False,
            "is_generator": is_generator,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "max_nesting": max_nesting,
            "assigned_vars": list(assigned_vars),
            "decorators": [
                self._get_decorator_name(d) for d in node.decorator_list
            ],
        }

    def _extract_class_info(self, node: ast.ClassDef) -> dict[str, Any]:
        """提取类信息。"""
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)

        has_docstring = False
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            has_docstring = True

        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{self._get_attr_name(base)}")

        return {
            "name": node.name,
            "line": node.lineno,
            "line_end": node.end_lineno or node.lineno,
            "bases": bases,
            "methods": methods,
            "has_docstring": has_docstring,
        }

    def _extract_call_info(self, node: ast.Call) -> Optional[dict[str, Any]]:
        """提取函数调用信息。"""
        name = self._get_call_name(node)
        if not name:
            return None

        return {
            "name": name,
            "line": node.lineno,
            "args_count": len(node.args),
        }

    def _extract_except_handler_info(self, node: ast.ExceptHandler) -> dict[str, Any]:
        """提取异常处理信息。"""
        exception_type = None
        if node.type:
            if isinstance(node.type, ast.Name):
                exception_type = node.type.id
            elif isinstance(node.type, ast.Tuple):
                exception_type = ", ".join(
                    elt.id for elt in node.type.elts
                    if isinstance(elt, ast.Name)
                )

        return {
            "type": exception_type or "",
            "name": node.name or "",
            "line": node.lineno,
        }

    def _calculate_nesting_depth(self, node: ast.AST, depth: int = 0) -> int:
        """计算 AST 节点的最大嵌套深度。"""
        max_depth = depth
        nesting_nodes = (
            ast.If, ast.For, ast.While, ast.With,
            ast.Try, ast.ExceptHandler,
        )
        for child in ast.iter_child_nodes(node):
            if isinstance(child, nesting_nodes):
                child_depth = self._calculate_nesting_depth(child, depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._calculate_nesting_depth(child, depth)
                max_depth = max(max_depth, child_depth)
        return max_depth

    @staticmethod
    def _get_call_name(node: ast.Call) -> str:
        """获取函数调用的名称。"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return ASTAnalyzer._get_attr_name(node.func)
        return ""

    @staticmethod
    def _get_attr_name(node: ast.Attribute) -> str:
        """获取属性链名称（如 os.path.join）。"""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    @staticmethod
    def _get_decorator_name(node: ast.expr) -> str:
        """获取装饰器名称。"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return ASTAnalyzer._get_attr_name(node)
        elif isinstance(node, ast.Call):
            return ASTAnalyzer._get_decorator_name(node.func)
        return ""
