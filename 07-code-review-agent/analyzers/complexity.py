"""圈复杂度分析器。"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FunctionComplexity:
    """函数复杂度信息。"""

    name: str
    line_start: int
    line_end: int
    complexity: int
    is_method: bool = False
    class_name: Optional[str] = None
    args_count: int = 0


class ComplexityAnalyzer:
    """圈复杂度（Cyclomatic Complexity）分析器。

    通过分析 Python AST 计算函数/方法的圈复杂度。
    圈复杂度 = 1 + 分支判断数量
    """

    def analyze(self, code: str) -> list[FunctionComplexity]:
        """分析代码中所有函数的圈复杂度。

        Args:
            code: Python 源代码。

        Returns:
            函数复杂度信息列表。
        """
        tree = ast.parse(code)
        results: list[FunctionComplexity] = []

        # 先遍历类，标记方法
        class_names: dict[int, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_names[id(item)] = node.name

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._calculate_complexity(node)
                is_method = id(node) in class_names
                func_info = FunctionComplexity(
                    name=node.name,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    complexity=complexity,
                    is_method=is_method,
                    class_name=class_names.get(id(node)),
                    args_count=len(node.args.args),
                )
                results.append(func_info)

        return results

    def _calculate_complexity(self, node: ast.AST) -> int:
        """计算 AST 节点的圈复杂度。

        圈复杂度 = 1 + 分支判断数量（if, elif, for, while, except, and, or, etc.）
        使用广度优先遍历，只计算分支节点。

        Args:
            node: AST 节点。

        Returns:
            圈复杂度值。
        """
        complexity = 1  # 基础复杂度
        queue = list(ast.iter_child_nodes(node))

        while queue:
            child = queue.pop(0)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # 跳过嵌套的函数/类定义
                continue
            if isinstance(child, (ast.If, ast.IfExp)):
                complexity += 1
            elif isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                op_count = len(child.values) - 1
                complexity += op_count
            elif isinstance(child, (ast.If, ast.IfExp, ast.For, ast.AsyncFor,
                                     ast.While, ast.ExceptHandler, ast.Try)):
                pass  # Already handled above
            # 添加子节点继续遍历
            for grandchild in ast.iter_child_nodes(child):
                queue.append(grandchild)

        return complexity
