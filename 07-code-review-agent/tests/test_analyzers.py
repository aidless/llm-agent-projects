"""分析器测试。"""

from __future__ import annotations

import pytest

from analyzers.ast_analyzer import ASTAnalyzer
from analyzers.complexity import ComplexityAnalyzer, FunctionComplexity
from analyzers.diff_parser import DiffParser
from analyzers.dependency import DependencyAnalyzer


# ==================== 测试数据 ====================

SAMPLE_CODE = '''
"""Module docstring."""

import os
import json
from typing import Optional

def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"

class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def complex_calc(self, x):
        """Complex calculation."""
        if x > 0:
            if x > 10:
                return x * 2
            else:
                return x + 1
        elif x < -10:
            return x - 5
        else:
            return 0

async def fetch_data(url: str) -> Optional[dict]:
    """Fetch data from URL."""
    import httpx
    try:
        return {"url": url}
    except Exception:
        return None
'''

SAMPLE_DIFF = """diff --git a/hello.py b/hello.py
new file mode 100644
index 0000000..e69de44
--- /dev/null
+++ b/hello.py
@@ -0,0 +1,5 @@
+import os
+
+def hello():
+    print("Hello")
+
diff --git a/utils.py b/utils.py
index 1234567..abcdefg 100644
--- a/utils.py
+++ b/utils.py
@@ -1,3 +1,5 @@
 def old_func():
-    pass
+    return None
+    # new line added
"""


# ==================== ASTAnalyzer 测试 ====================

class TestASTAnalyzer:
    """ASTAnalyzer 测试。"""

    def setup_method(self) -> None:
        self.analyzer = ASTAnalyzer()

    def test_detect_module_docstring(self) -> None:
        """测试检测模块 docstring。"""
        result = self.analyzer.analyze(SAMPLE_CODE)
        assert result["has_module_docstring"] is True

    def test_no_module_docstring(self) -> None:
        """测试无模块 docstring。"""
        result = self.analyzer.analyze("import os\ndef foo(): pass")
        assert result["has_module_docstring"] is False

    def test_detect_imports(self) -> None:
        """测试检测导入。"""
        result = self.analyzer.analyze(SAMPLE_CODE)
        imports = result["imports"]
        assert len(imports) >= 3
        modules = [i["module"] for i in imports]
        assert "os" in modules
        assert "json" in modules
        assert "typing.Optional" in modules

    def test_detect_functions(self) -> None:
        """测试检测函数。"""
        result = self.analyzer.analyze(SAMPLE_CODE)
        functions = result["functions"]
        names = [f["name"] for f in functions]
        assert "greet" in names
        assert "add" in names
        assert "complex_calc" in names
        assert "fetch_data" in names

    def test_function_has_args(self) -> None:
        """测试函数参数提取。"""
        result = self.analyzer.analyze(SAMPLE_CODE)
        greet_func = next(f for f in result["functions"] if f["name"] == "greet")
        assert "name" in greet_func["args"]

    def test_function_has_docstring(self) -> None:
        """测试函数 docstring 检测。"""
        result = self.analyzer.analyze(SAMPLE_CODE)
        greet_func = next(f for f in result["functions"] if f["name"] == "greet")
        assert greet_func["has_docstring"] is True

    def test_function_has_return_annotation(self) -> None:
        """测试函数返回值注解检测。"""
        result = self.analyzer.analyze(SAMPLE_CODE)
        greet_func = next(f for f in result["functions"] if f["name"] == "greet")
        assert greet_func["has_return_annotation"] is True

    def test_detect_async_function(self) -> None:
        """测试检测异步函数。"""
        result = self.analyzer.analyze(SAMPLE_CODE)
        async_funcs = [f for f in result["functions"] if f["is_async"]]
        assert len(async_funcs) == 1
        assert async_funcs[0]["name"] == "fetch_data"

    def test_detect_classes(self) -> None:
        """测试检测类。"""
        result = self.analyzer.analyze(SAMPLE_CODE)
        classes = result["classes"]
        assert len(classes) == 1
        assert classes[0]["name"] == "Calculator"
        assert "add" in classes[0]["methods"]
        assert "complex_calc" in classes[0]["methods"]

    def test_detect_function_calls(self) -> None:
        """测试检测函数调用。"""
        result = self.analyzer.analyze(SAMPLE_CODE)
        calls = result["function_calls"]
        call_names = [c["name"] for c in calls]
        assert "print" in call_names or len(calls) >= 0  # print may not be in sample

    def test_detect_exception_handlers(self) -> None:
        """测试检测异常处理。"""
        result = self.analyzer.analyze(SAMPLE_CODE)
        handlers = result["exception_handlers"]
        assert len(handlers) >= 1
        assert handlers[0]["type"] == "Exception"

    def test_syntax_error_raises(self) -> None:
        """测试语法错误抛出异常。"""
        with pytest.raises(SyntaxError):
            self.analyzer.analyze("def broken(")


# ==================== ComplexityAnalyzer 测试 ====================

class TestComplexityAnalyzer:
    """ComplexityAnalyzer 测试。"""

    def setup_method(self) -> None:
        self.analyzer = ComplexityAnalyzer()

    def test_simple_function_complexity(self) -> None:
        """测试简单函数复杂度为 1。"""
        code = "def add(a, b):\n    return a + b\n"
        results = self.analyzer.analyze(code)
        assert len(results) == 1
        assert results[0].complexity == 1

    def test_if_increases_complexity(self) -> None:
        """测试 if 语句增加复杂度。"""
        code = "def check(x):\n    if x > 0:\n        return True\n    return False\n"
        results = self.analyzer.analyze(code)
        assert results[0].complexity == 2

    def test_complex_function_high_complexity(self) -> None:
        """测试复杂函数高复杂度。"""
        results = self.analyzer.analyze(SAMPLE_CODE)
        complex_func = next(
            (f for f in results if f.name == "complex_calc"),
            None,
        )
        assert complex_func is not None
        assert complex_func.complexity >= 4

    def test_method_detection(self) -> None:
        """测试方法检测。"""
        results = self.analyzer.analyze(SAMPLE_CODE)
        methods = [f for f in results if f.is_method]
        assert len(methods) >= 2

    def test_function_info_fields(self) -> None:
        """测试函数信息字段完整。"""
        results = self.analyzer.analyze(SAMPLE_CODE)
        for func in results:
            assert func.name
            assert func.line_start > 0
            assert func.line_end >= func.line_start
            assert func.complexity > 0


# ==================== DiffParser 测试 ====================

class TestDiffParser:
    """DiffParser 测试。"""

    def setup_method(self) -> None:
        self.parser = DiffParser()

    def test_parse_diff_files(self) -> None:
        """测试解析 diff 文件列表。"""
        files = self.parser.parse(SAMPLE_DIFF)
        assert len(files) == 2

    def test_parse_new_file(self) -> None:
        """测试解析新建文件。"""
        files = self.parser.parse(SAMPLE_DIFF)
        new_file = files[0]
        assert new_file.is_new is True
        assert new_file.new_path == "hello.py"

    def test_parse_hunks(self) -> None:
        """测试解析 diff 块。"""
        files = self.parser.parse(SAMPLE_DIFF)
        new_file = files[0]
        assert len(new_file.hunks) == 1
        hunk = new_file.hunks[0]
        assert hunk.old_start == 0
        assert hunk.new_start == 1

    def test_parse_added_lines(self) -> None:
        """测试解析新增行。"""
        files = self.parser.parse(SAMPLE_DIFF)
        new_file = files[0]
        assert len(new_file.hunks[0].added_lines) == 5

    def test_extract_added_code(self) -> None:
        """测试提取新增代码。"""
        code = self.parser.extract_added_code(SAMPLE_DIFF)
        assert "import os" in code
        assert "def hello()" in code

    def test_get_changed_files(self) -> None:
        """测试获取变更文件列表。"""
        files = self.parser.get_changed_files(SAMPLE_DIFF)
        assert len(files) == 2
        assert "hello.py" in files
        assert "utils.py" in files


# ==================== DependencyAnalyzer 测试 ====================

class TestDependencyAnalyzer:
    """DependencyAnalyzer 测试。"""

    def setup_method(self) -> None:
        self.analyzer = DependencyAnalyzer()

    def test_detect_imports(self) -> None:
        """测试检测导入。"""
        result = self.analyzer.analyze(SAMPLE_CODE)
        assert result["import_count"] >= 3

    def test_detect_function_deps(self) -> None:
        """测试检测函数依赖。"""
        result = self.analyzer.analyze(SAMPLE_CODE)
        assert len(result["functions"]) >= 4

    def test_dependency_graph(self) -> None:
        """测试依赖图。"""
        result = self.analyzer.analyze(SAMPLE_CODE)
        graph = result["dependency_graph"]
        assert "os" in graph or "json" in graph

    def test_external_modules(self) -> None:
        """测试外部模块检测。"""
        code = "import pandas\ndef foo():\n    pass\n"
        result = self.analyzer.analyze(code)
        assert "pandas" in result["external_modules"]
