"""测试沙箱执行器"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox.executor import SandboxExecutor
from policy.presets import get_preset_policy
from policy.models import SecurityLevel
from audit.logger import AuditLogger


@pytest.fixture
def executor():
    """创建执行器实例"""
    audit_logger = AuditLogger()
    return SandboxExecutor(audit_logger=audit_logger)


@pytest.fixture
def medium_policy():
    return get_preset_policy(SecurityLevel.MEDIUM)


class TestBasicExecution:
    """基本执行测试"""

    def test_simple_print(self, executor, medium_policy):
        """测试简单打印"""
        result = executor.execute("print('hello world')", medium_policy)
        assert result.success is True
        assert "hello world" in result.output

    def test_arithmetic(self, executor, medium_policy):
        """测试算术运算"""
        result = executor.execute("print(2 + 3 * 4)", medium_policy)
        assert result.success is True
        assert "14" in result.output

    def test_variable_assignment(self, executor, medium_policy):
        """测试变量赋值"""
        result = executor.execute("x = 42\nprint(x)", medium_policy)
        assert result.success is True
        assert "42" in result.output

    def test_function_definition(self, executor, medium_policy):
        """测试函数定义和调用"""
        code = """
def add(a, b):
    return a + b
print(add(3, 5))
"""
        result = executor.execute(code, medium_policy)
        assert result.success is True
        assert "8" in result.output

    def test_loop(self, executor, medium_policy):
        """测试循环"""
        code = """
total = 0
for i in range(5):
    total += i
print(total)
"""
        result = executor.execute(code, medium_policy)
        assert result.success is True
        assert "10" in result.output

    def test_list_comprehension(self, executor, medium_policy):
        """测试列表推导式"""
        code = "print([x*x for x in range(5)])"
        result = executor.execute(code, medium_policy)
        assert result.success is True
        assert "[0, 1, 4, 9, 16]" in result.output

    def test_string_operations(self, executor, medium_policy):
        """测试字符串操作"""
        code = "s = 'hello'\nprint(s.upper())"
        result = executor.execute(code, medium_policy)
        assert result.success is True
        assert "HELLO" in result.output

    def test_import_allowed_module(self, executor, medium_policy):
        """测试导入允许的模块"""
        code = "import math\nprint(math.sqrt(16))"
        result = executor.execute(code, medium_policy)
        assert result.success is True
        assert "4.0" in result.output

    def test_execution_has_id(self, executor, medium_policy):
        """测试执行结果包含ID"""
        result = executor.execute("print(1)", medium_policy)
        assert result.execution_id
        assert len(result.execution_id) > 0

    def test_syntax_error(self, executor, medium_policy):
        """测试语法错误处理 - AST 解析阶段捕获"""
        result = executor.execute("print(", medium_policy)
        assert result.success is False
        # 语法错误可能被 AST 分析器先检测到
        assert result.error_type in ("SyntaxError", "SecurityViolation")

    def test_runtime_error(self, executor, medium_policy):
        """测试运行时错误处理"""
        result = executor.execute("x = 1/0", medium_policy)
        assert result.success is False
        assert result.error

    def test_with_variables(self, executor, medium_policy):
        """测试带预定义变量的执行"""
        result = executor.execute("print(data['key'])", medium_policy, variables={"data": {"key": "value"}})
        assert result.success is True
        assert "value" in result.output

    def test_dict_operations(self, executor, medium_policy):
        """测试字典操作"""
        code = """
d = {'a': 1, 'b': 2}
print(d['a'], len(d))
"""
        result = executor.execute(code, medium_policy)
        assert result.success is True
        assert "1 2" in result.output