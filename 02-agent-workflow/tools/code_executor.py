# ============================================
# 代码执行工具
# 在受限沙箱中执行 Python 代码，返回 stdout 和 stderr
# ============================================

import io
import contextlib
import traceback
from loguru import logger

try:
    from langchain_core.tools import tool
except ImportError:
    from langchain.tools import tool

# 代码执行最大时长（秒）
EXECUTION_TIMEOUT = 30

# 禁止使用的模块（安全限制）
BANNED_MODULES = {
    "os", "subprocess", "sys", "shutil", "pathlib",
    "socket", "http", "urllib", "requests",
    "ctypes", "multiprocessing", "threading",
    "signal", "importlib", "__import__",
}

# 禁止使用的内置函数
BANNED_BUILTINS = {
    "exec", "eval", "compile", "open",
    "__import__", "globals", "locals", "vars",
}


def _check_code_safety(code: str) -> tuple[bool, str]:
    """检查代码安全性，禁止危险操作"""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"语法错误: {e}"

    for node in ast.walk(tree):
        # 检查 import 语句
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    if module_name in BANNED_MODULES:
                        return False, f"禁止导入模块: {module_name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_name = node.module.split(".")[0]
                if module_name in BANNED_MODULES:
                    return False, f"禁止导入模块: {module_name}"

        # 检查函数调用
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BANNED_BUILTINS:
                    return False, f"禁止调用: {node.func.id}"

    return True, "安全检查通过"


@tool
async def code_execute_tool(code: str) -> str:
    """
    代码执行工具：在安全沙箱中执行 Python 代码。

    支持数学计算、数据处理、字符串操作等。
    禁止使用 os、subprocess、socket 等危险模块。

    Args:
        code: 要执行的 Python 代码字符串

    Returns:
        str: 执行输出（stdout）或错误信息（stderr）
    """
    logger.info(f"[代码工具] 执行代码，长度: {len(code)} 字符")

    # 安全检查
    safe, reason = _check_code_safety(code)
    if not safe:
        error_msg = f"代码安全检查未通过: {reason}"
        logger.warning(f"[代码工具] {error_msg}")
        return f"错误：{error_msg}"

    # 捕获 stdout 和 stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        # 构建受限的执行环境
        # 提供安全的内置函数
        safe_builtins = {
            "print": lambda *args, **kwargs: print(*args, file=stdout_capture, **kwargs),
            "range": range,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "sorted": sorted,
            "reversed": reversed,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "type": type,
            "isinstance": isinstance,
            "hasattr": hasattr,
            "getattr": getattr,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "ZeroDivisionError": ZeroDivisionError,
            "RuntimeError": RuntimeError,
            "Exception": Exception,
            "StopIteration": StopIteration,
            "AttributeError": AttributeError,
        }

        # 执行代码
        with contextlib.redirect_stdout(stdout_capture):
            with contextlib.redirect_stderr(stderr_capture):
                exec(code, {"__builtins__": safe_builtins}, {})

        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        if stdout_output:
            logger.debug(f"[代码工具] 执行成功，输出长度: {len(stdout_output)} 字符")
            result = f"--- 输出 ---\n{stdout_output}"
        else:
            result = "代码执行成功，无输出。"

        if stderr_output:
            result += f"\n--- 错误 ---\n{stderr_output}"

        return result

    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"代码执行异常: {type(e).__name__}: {str(e)}\n{error_trace}"
        logger.error(f"[代码工具] {error_msg}")
        return f"错误：{error_msg}"
