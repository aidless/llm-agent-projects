"""导入守卫 - 控制模块导入"""

from typing import Set, Optional, List
from dataclasses import dataclass, field


@dataclass
class ImportGuardConfig:
    """导入守卫配置"""
    allowed_modules: Set[str] = field(default_factory=lambda: {
        "math", "random", "string", "re", "json",
        "datetime", "collections", "itertools", "functools",
        "decimal", "fractions", "statistics", "typing",
        "dataclasses", "enum", "copy", "operator",
        "abc", "numbers", "hashlib", "hmac", "base64",
        "uuid", "time", "struct", "array", "queue",
        "heapq", "bisect", "pprint", "textwrap",
        "contextlib", "warnings", "inspect",
    })
    blocked_modules: Set[str] = field(default_factory=lambda: {
        "os", "subprocess", "shutil", "sys",
        "socket", "requests", "urllib",
        "ctypes", "multiprocessing", "threading",
        "signal", "resource", "pty", "fcntl",
        "pathlib", "tempfile", "glob", "fnmatch",
        "importlib", "pkgutil", "code", "codeop",
        "compileall", "distutils", "setuptools",
        "pip", "numpy", "pandas",  # 大型模块可能泄露信息
    })
    allow_star_import: bool = False


class ImportGuard:
    """模块导入守卫"""

    def __init__(self, config: Optional[ImportGuardConfig] = None):
        self.config = config or ImportGuardConfig()

    def check_import(self, module_name: str) -> bool:
        """检查是否允许导入指定模块"""
        base_module = module_name.split(".")[0]

        # 优先检查黑名单
        if base_module in self.config.blocked_modules:
            return False

        # 如果白名单非空，只允许白名单中的模块
        if self.config.allowed_modules:
            return base_module in self.config.allowed_modules

        return True

    def check_from_import(self, module_name: str, names: List[str]) -> bool:
        """检查 from ... import ... 语句"""
        if not self.check_import(module_name):
            return False

        # 检查 star import
        if "*" in names and not self.config.allow_star_import:
            return False

        return True

    def get_safe_builtins(self) -> dict:
        """获取安全的 builtins 字典"""
        import builtins
        safe = {}
        dangerous = {
            "exec", "eval", "compile", "__import__",
            "open", "input", "breakpoint", "exit", "quit",
            "globals", "locals", "memoryview",
        }
        for name in dir(builtins):
            if name not in dangerous and not name.startswith("_"):
                safe[name] = getattr(builtins, name)
        return safe

    def create_import_function(self):
        """创建受控的 __import__ 函数"""
        guard = self
        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

        def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            if not guard.check_import(name):
                raise ImportError(f"Import of module '{name}' is not allowed by security policy")
            return original_import(name, globals, locals, fromlist, level)

        return restricted_import