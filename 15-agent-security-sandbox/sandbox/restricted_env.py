"""RestrictedPython 环境配置"""

from typing import Any, Dict, Optional, Set
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Eval import default_guarded_getiter
from RestrictedPython.Guards import (
    guarded_unpack_sequence,
    safer_getattr,
)
from RestrictedPython.PrintCollector import PrintCollector
from security.import_guard import ImportGuard, ImportGuardConfig
from security.syscall_interceptor import SecurityError


def _write_guard(obj):
    """RestrictedPython 写保护 - 阻止对安全对象的写入"""
    raise SecurityError("Write access is not permitted in the sandbox")


def _safe_builtins(import_guard: ImportGuard) -> dict:
    """构建安全的 builtins"""
    import builtins

    safe = {}
    dangerous = {
        "exec", "eval", "compile", "__import__",
        "open", "input", "breakpoint", "exit", "quit",
        "globals", "locals", "vars", "memoryview",
    }

    for name in dir(builtins):
        if name not in dangerous and not name.startswith("_"):
            try:
                safe[name] = getattr(builtins, name)
            except (AttributeError, TypeError):
                pass

    # 添加受控的 __import__
    safe["__import__"] = import_guard.create_import_function()

    # 添加安全异常类
    for exc_name in (
        "Exception", "ValueError", "TypeError", "KeyError",
        "IndexError", "RuntimeError", "AttributeError", "NameError",
        "StopIteration", "NotImplementedError", "ImportError",
        "OSError", "FileNotFoundError", "PermissionError",
        "ZeroDivisionError", "AssertionError", "IOError",
        "ArithmeticError", "LookupError", "OverflowError",
        "RecursionError", "UnicodeError", "UnicodeDecodeError",
        "UnicodeEncodeError",
    ):
        if hasattr(builtins, exc_name):
            safe[exc_name] = getattr(builtins, exc_name)

    return safe


def create_restricted_env(
    import_guard: Optional[ImportGuard] = None,
    extra_globals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """创建 RestrictedPython 执行环境

    Args:
        import_guard: 导入守卫
        extra_globals: 额外的全局变量

    Returns:
        安全的全局变量字典
    """
    if import_guard is None:
        import_guard = ImportGuard()

    # RestrictedPython 的安全全局变量
    env = safe_globals.copy()

    # 覆盖 builtins
    env["__builtins__"] = _safe_builtins(import_guard)

    # 设置安全守卫函数
    env["_getiter_"] = default_guarded_getiter
    env["_unpack_sequence_"] = guarded_unpack_sequence
    env["_getattr_"] = safer_getattr
    env["_write_"] = _write_guard
    # RestrictedPython 8.x: _inplacevar_(op_str, obj, value) -> new_value
    import operator as _op_module
    _inplace_ops = {
        '+=': _op_module.iadd,
        '-=': _op_module.isub,
        '*=': _op_module.imul,
        '/=': _op_module.itruediv,
        '//=': _op_module.ifloordiv,
        '%=': _op_module.imod,
        '**=': _op_module.ipow,
        '>>=': _op_module.irshift,
        '<<=': _op_module.ilshift,
        '&=': _op_module.iand,
        '|=': _op_module.ior,
        '^=': _op_module.ixor,
    }

    def _inplacevar(op, obj, value):
        func = _inplace_ops.get(op)
        if func:
            return func(obj, value)
        # Fallback for other operations
        if op == '+=':
            return obj + value
        elif op == '-=':
            return obj - value
        raise SecurityError(f"Inplace operation '{op}' is not allowed")

    env["_inplacevar_"] = _inplacevar
    # _getitem_ is needed for subscript access in RestrictedPython 8.x
    env["_getitem_"] = lambda obj, key: obj[key]

    # RestrictedPython 8.x 将 print() 重写为 _print_ / _write_
    # 提供 _print_ 函数使其正常工作
    env["_print_"] = PrintCollector

    # 添加额外全局变量
    if extra_globals:
        env.update(extra_globals)

    return env


def compile_restricted_code(code: str) -> Any:
    """编译受限代码

    Args:
        code: Python 源代码

    Returns:
        编译后的代码对象

    Raises:
        SyntaxError: 代码包含不允许的语法
        SecurityError: 代码违反安全策略
    """
    try:
        result = compile_restricted(code, filename="<sandbox>", mode="exec")
    except Exception as e:
        raise SyntaxError(f"RestrictedPython compilation error: {e}") from e

    # RestrictedPython 在编译出错时返回 None
    if result is None:
        raise SyntaxError("RestrictedPython rejected the code (unsafe constructs detected)")

    # 兼容不同版本: RestrictedPython 8.x 直接返回 code 对象
    # 旧版本返回包装对象 with .code 属性
    if hasattr(result, 'code') and result.code is None:
        raise SyntaxError("RestrictedPython compilation produced no code (unsafe constructs)")

    return result