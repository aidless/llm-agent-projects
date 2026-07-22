# ============================================
# 工具单元测试
# ============================================

import pytest
import os
import shutil
import tempfile
import asyncio
from tools.calculator import calculator_tool
from tools.code_executor import code_execute_tool
from tools.file_tool import file_read_tool, file_write_tool


class TestCalculatorTool:
    """计算器工具测试类"""

    @pytest.mark.asyncio
    async def test_basic_arithmetic(self):
        """测试基本算术运算"""
        result = await calculator_tool.ainvoke({"expression": "2 + 3"})
        assert "5" in result

    @pytest.mark.asyncio
    async def test_power(self):
        """测试幂运算"""
        result = await calculator_tool.ainvoke({"expression": "2 ** 10"})
        assert "1024" in result

    @pytest.mark.asyncio
    async def test_sqrt(self):
        """测试开方"""
        result = await calculator_tool.ainvoke({"expression": "sqrt(144)"})
        assert "12" in result

    @pytest.mark.asyncio
    async def test_complex_expression(self):
        """测试复杂表达式"""
        result = await calculator_tool.ainvoke({"expression": "(3 + 5) * 2 - 4 / 2"})
        assert "14" in result

    @pytest.mark.asyncio
    async def test_dangerous_keywords(self):
        """测试危险关键词拦截"""
        result = await calculator_tool.ainvoke({"expression": "import os"})
        assert "禁止" in result


class TestCodeExecutorTool:
    """代码执行工具测试类"""

    @pytest.mark.asyncio
    async def test_simple_print(self):
        """测试简单打印"""
        result = await code_execute_tool.ainvoke({"code": 'print("Hello, World!")'})
        assert "Hello, World!" in result

    @pytest.mark.asyncio
    async def test_math_calculation(self):
        """测试数学计算"""
        result = await code_execute_tool.ainvoke({"code": "print(sum(range(1, 101)))"})
        assert "5050" in result

    @pytest.mark.asyncio
    async def test_list_operations(self):
        """测试列表操作"""
        code = "data = [1, 2, 3, 4, 5]\ndata.sort(reverse=True)\nprint(data)"
        result = await code_execute_tool.ainvoke({"code": code})
        assert "[5, 4, 3, 2, 1]" in result

    @pytest.mark.asyncio
    async def test_banned_import(self):
        """测试禁止导入危险模块"""
        result = await code_execute_tool.ainvoke({"code": "import os"})
        assert "禁止" in result

    @pytest.mark.asyncio
    async def test_banned_function(self):
        """测试禁止调用危险函数"""
        result = await code_execute_tool.ainvoke({"code": "eval('1+1')"})
        assert "禁止" in result

    @pytest.mark.asyncio
    async def test_syntax_error(self):
        """测试语法错误处理"""
        result = await code_execute_tool.ainvoke({"code": "this is not valid python"})
        assert "语法错误" in result


class TestFileTools:
    """文件工具测试类"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        tmp = tempfile.mkdtemp(prefix="test_files_")
        original_allowed = os.environ.get("WORKSPACE_DIR")
        os.environ["WORKSPACE_DIR"] = tmp
        yield tmp
        # 恢复环境变量
        if original_allowed:
            os.environ["WORKSPACE_DIR"] = original_allowed
        else:
            os.environ.pop("WORKSPACE_DIR", None)
        shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_write_and_read(self, temp_dir):
        """测试写入和读取文件"""
        # 重新导入以获取新的 ALLOWED_ROOT
        import importlib
        import tools.file_tool
        importlib.reload(tools.file_tool)

        file_path = os.path.join(temp_dir, "test.txt")

        # 写入
        write_result = await file_write_tool.ainvoke({
            "file_path": file_path,
            "content": "测试文件内容",
        })
        assert "成功" in write_result

        # 读取
        read_result = await file_read_tool.ainvoke({"file_path": file_path})
        assert "测试文件内容" in read_result

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, temp_dir):
        """测试读取不存在的文件"""
        import importlib
        import tools.file_tool
        importlib.reload(tools.file_tool)

        result = await file_read_tool.ainvoke({"file_path": os.path.join(temp_dir, "nonexistent.txt")})
        assert "错误" in result
