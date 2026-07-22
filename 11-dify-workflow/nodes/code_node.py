"""代码执行节点 - 安全的 Python 代码执行"""

from __future__ import annotations

import io
import contextlib
import logging
from typing import Any, Dict

from engine.errors import NodeExecutionError

from .base import BaseNode

logger = logging.getLogger(__name__)

# 白名单模块: 允许在代码中 import 的模块
_ALLOWED_MODULES = frozenset({
    "json", "re", "math", "datetime", "collections",
    "itertools", "functools", "operator", "string",
    "copy", "hashlib", "base64", "random",
})

# 危险内置函数 (禁止使用)
_BLOCKED_BUILTINS = frozenset({
    "exec", "eval", "compile", "open", "__import__",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr",
    "input", "breakpoint",
})


class CodeNode(BaseNode):
    """安全的 Python 代码执行节点 (白名单沙箱)"""

    node_type = "code"

    def execute(
        self,
        inputs: Dict[str, Any],
        context: "ExecutionContext",
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        code = inputs.get("code", "")
        if not code.strip():
            raise NodeExecutionError("代码执行节点缺少 code 参数")

        # 构建受限的执行环境
        safe_globals: Dict[str, Any] = {"__builtins__": {}}
        # 只暴露安全的内置函数
        safe_builtins = {
            "abs": abs, "all": all, "any": any, "bin": bin,
            "bool": bool, "chr": chr, "dict": dict, "divmod": divmod,
            "enumerate": enumerate, "filter": filter, "float": float,
            "format": format, "frozenset": frozenset, "hex": hex,
            "int": int, "isinstance": isinstance, "issubclass": issubclass,
            "len": len, "list": list, "map": map, "max": max,
            "min": min, "oct": oct, "ord": ord, "pow": pow,
            "print": print, "range": range, "repr": repr,
            "reversed": reversed, "round": round, "set": set,
            "slice": slice, "sorted": sorted, "str": str,
            "sum": sum, "tuple": tuple, "type": type, "zip": zip,
            "True": True, "False": False, "None": None,
            "enumerate": enumerate, "iter": iter, "next": next,
            "hasattr": hasattr, "callable": callable,
            "ValueError": ValueError, "TypeError": TypeError,
            "KeyError": KeyError, "IndexError": IndexError,
            "RuntimeError": RuntimeError, "StopIteration": StopIteration,
            "AttributeError": AttributeError,
        }
        safe_globals["__builtins__"] = safe_builtins

        # 注入输入变量
        local_vars: Dict[str, Any] = dict(inputs)
        local_vars["input"] = inputs  # 便捷别名
        local_vars["result"] = None

        # 安全检查: 禁止危险关键字
        code_str = code
        for blocked in _BLOCKED_BUILTINS:
            # 粗略检查，防止明显的危险调用
            if f"{blocked}(" in code_str:
                raise NodeExecutionError(
                    f"代码中禁止使用 '{blocked}' 函数"
                )

        # 捕获 stdout
        stdout_capture = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(compile(code, "<workflow_code>", "exec"), safe_globals, local_vars)
        except Exception as e:
            raise NodeExecutionError(
                f"代码执行错误: {type(e).__name__}: {e}"
            )

        output_result = local_vars.get("result", None)
        stdout_value = stdout_capture.getvalue()

        return {
            "result": output_result,
            "stdout": stdout_value,
            "variables": {
                k: v for k, v in local_vars.items()
                if k not in ("input", "__builtins__") and not k.startswith("_")
            },
        }


def create_code_handler():
    """工厂函数"""
    node = CodeNode()

    def handler(inputs, context, config):
        return node.execute(inputs, context, config)

    handler.node_type = CodeNode.node_type
    return handler
