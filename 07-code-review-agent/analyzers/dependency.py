"""依赖关系分析器。"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ImportInfo:
    """导入信息。"""

    module: str
    alias: Optional[str]
    line: int
    import_type: str  # "import" or "from_import"


@dataclass
class FunctionDependency:
    """函数依赖关系。"""

    name: str
    line: int
    imports_used: list[str] = field(default_factory=list)
    functions_called: list[str] = field(default_factory=list)
    classes_used: list[str] = field(default_factory=list)


class DependencyAnalyzer:
    """代码依赖关系分析器。

    分析模块级别的导入关系和函数级别的依赖关系。
    """

    def analyze(self, code: str) -> dict:
        """分析代码的依赖关系。

        Args:
            code: Python 源代码。

        Returns:
            包含 imports, functions, dependency_graph 的字典。
        """
        tree = ast.parse(code)

        # 收集所有导入
        imports: list[ImportInfo] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        alias=alias.asname,
                        line=node.lineno,
                        import_type="import",
                    ))
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=f"{module_name}.{alias.name}",
                        alias=alias.asname,
                        line=node.lineno,
                        import_type="from_import",
                    ))

        # 分析函数依赖
        function_deps: list[FunctionDependency] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                dep = self._analyze_function_deps(node)
                function_deps.append(dep)

        # 构建依赖图
        dep_graph = self._build_dependency_graph(imports, function_deps)

        return {
            "imports": imports,
            "functions": function_deps,
            "dependency_graph": dep_graph,
            "import_count": len(imports),
            "external_modules": self._get_external_modules(imports),
        }

    def _analyze_function_deps(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionDependency:
        """分析单个函数的依赖关系。"""
        imports_used: list[str] = []
        functions_called: list[str] = []
        classes_used: list[str] = []

        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if isinstance(child.ctx, ast.Load):
                    functions_called.append(child.id)
            elif isinstance(child, ast.Attribute):
                if isinstance(child.ctx, ast.Load):
                    functions_called.append(child.attr)
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    functions_called.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    functions_called.append(child.func.attr)

        return FunctionDependency(
            name=node.name,
            line=node.lineno,
            imports_used=list(set(imports_used)),
            functions_called=list(set(functions_called)),
            classes_used=list(set(classes_used)),
        )

    def _build_dependency_graph(
        self,
        imports: list[ImportInfo],
        function_deps: list[FunctionDependency],
    ) -> dict[str, list[str]]:
        """构建模块依赖图。"""
        graph: dict[str, list[str]] = {}
        for imp in imports:
            module = imp.module.split(".")[0]
            if module not in graph:
                graph[module] = []

        for func in function_deps:
            graph[func.name] = func.functions_called

        return graph

    @staticmethod
    def _get_external_modules(imports: list[ImportInfo]) -> list[str]:
        """获取外部依赖模块列表（排除标准库和本地模块）。"""
        stdlib_modules = {
            "os", "sys", "json", "re", "ast", "time", "datetime", "logging",
            "functools", "collections", "hashlib", "typing", "dataclasses",
            "abc", "copy", "io", "pathlib", "threading", "multiprocessing",
            "subprocess", "pickle", "shutil", "tempfile", "enum", "math",
            "random", "string", "struct", "unittest", "pytest",
        }

        external = []
        for imp in imports:
            module = imp.module.split(".")[0]
            if module not in stdlib_modules and not module.startswith("."):
                external.append(imp.module)

        return list(set(external))
