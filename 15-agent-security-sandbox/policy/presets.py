"""预设安全策略 - 4级安全等级"""

from policy.models import (
    SecurityPolicy, SecurityLevel, ResourceLimits,
    ModulePolicy, FilesystemPolicy, NetworkPolicy,
)

SAFE_MODULES = {
    "math", "random", "string", "re", "json",
    "datetime", "collections", "itertools", "functools",
    "decimal", "fractions", "statistics", "typing",
    "dataclasses", "enum", "copy", "operator",
    "abc", "numbers", "hashlib", "hmac", "base64",
    "uuid", "time", "struct", "array", "queue",
    "heapq", "bisect", "pprint", "textwrap",
    "contextlib", "warnings", "inspect",
}

DANGEROUS_MODULES = {
    "os", "subprocess", "shutil", "sys",
    "socket", "requests", "urllib",
    "ctypes", "multiprocessing", "threading",
    "signal", "resource", "pty", "fcntl",
    "pathlib", "tempfile", "glob", "fnmatch",
    "importlib", "pkgutil", "code", "codeop",
    "compileall", "distutils", "setuptools",
    "pip", "numpy", "pandas",
}

DANGEROUS_BUILTINS = {
    "exec", "eval", "compile", "__import__",
    "open", "input", "breakpoint", "exit", "quit",
}


def _low_policy() -> SecurityPolicy:
    """LOW 级别 - 最宽松，仅拦截最危险的操作"""
    return SecurityPolicy(
        name="low",
        level=SecurityLevel.LOW,
        description="低安全级别 - 仅拦截最危险操作 (exec/eval/os.system/subprocess)",
        resource_limits=ResourceLimits(
            max_execution_time=60.0,
            max_memory_mb=1024,
            max_output_size=5 * 1024 * 1024,
            max_file_size=50 * 1024 * 1024,
            max_concurrent=20,
            cpu_time_limit=30.0,
        ),
        module_policy=ModulePolicy(
            allowed_modules=set(),
            blocked_modules={"os", "subprocess", "shutil", "signal", "pty", "fcntl"},
            allowed_builtins=set(),
            blocked_builtins={"exec", "eval", "compile", "__import__"},
            allow_star_import=True,
        ),
        filesystem_policy=FilesystemPolicy(
            allowed_read_dirs=set(),
            allowed_write_dirs=set(),
            block_all_writes=False,
            block_all_reads=False,
        ),
        network_policy=NetworkPolicy(
            allowed_domains=set(),
            blocked_domains={"169.254.169.254", "metadata.google.internal"},
            allowed_ports=set(),
            blocked_ports=set(),
            block_all_network=False,
        ),
    )


def _medium_policy() -> SecurityPolicy:
    """MEDIUM 级别 - 默认安全策略"""
    return SecurityPolicy(
        name="medium",
        level=SecurityLevel.MEDIUM,
        description="中等安全级别 - 拦截危险模块和系统调用，限制网络访问",
        resource_limits=ResourceLimits(
            max_execution_time=30.0,
            max_memory_mb=256,
            max_output_size=1024 * 1024,
            max_file_size=10 * 1024 * 1024,
            max_concurrent=10,
            cpu_time_limit=10.0,
        ),
        module_policy=ModulePolicy(
            allowed_modules=SAFE_MODULES,
            blocked_modules=DANGEROUS_MODULES,
            allowed_builtins=set(),
            blocked_builtins=DANGEROUS_BUILTINS,
            allow_star_import=False,
        ),
        filesystem_policy=FilesystemPolicy(
            allowed_read_dirs=set(),
            allowed_write_dirs=set(),
            block_all_writes=True,
            block_all_reads=False,
        ),
        network_policy=NetworkPolicy(
            allowed_domains=set(),
            blocked_domains={"localhost", "127.0.0.1", "0.0.0.0", "::1", "169.254.169.254"},
            allowed_ports=set(),
            blocked_ports={22, 23, 25, 3306, 5432, 6379, 27017},
            block_all_network=True,
        ),
    )


def _high_policy() -> SecurityPolicy:
    """HIGH 级别 - 严格安全策略"""
    return SecurityPolicy(
        name="high",
        level=SecurityLevel.HIGH,
        description="高安全级别 - 严格模块白名单，禁止文件写入和网络访问",
        resource_limits=ResourceLimits(
            max_execution_time=10.0,
            max_memory_mb=128,
            max_output_size=512 * 1024,
            max_file_size=1 * 1024 * 1024,
            max_concurrent=5,
            cpu_time_limit=5.0,
        ),
        module_policy=ModulePolicy(
            allowed_modules={
                "math", "random", "string", "re", "json",
                "datetime", "collections", "itertools", "functools",
                "decimal", "fractions", "statistics", "typing",
                "dataclasses", "enum", "copy", "operator",
                "abc", "numbers", "hashlib", "hmac", "base64",
                "uuid", "time", "struct", "array", "queue",
                "heapq", "bisect", "pprint", "textwrap",
            },
            blocked_modules=DANGEROUS_MODULES | {"numpy", "pandas", "scipy", "tkinter"},
            allowed_builtins=set(),
            blocked_builtins=DANGEROUS_BUILTINS | {"open", "input", "breakpoint", "globals", "locals"},
            allow_star_import=False,
        ),
        filesystem_policy=FilesystemPolicy(
            allowed_read_dirs=set(),
            allowed_write_dirs=set(),
            block_all_writes=True,
            block_all_reads=True,
        ),
        network_policy=NetworkPolicy(
            allowed_domains=set(),
            blocked_domains={"localhost", "127.0.0.1", "0.0.0.0", "::1", "169.254.169.254", "metadata.google.internal"},
            allowed_ports=set(),
            blocked_ports={22, 23, 25, 53, 80, 443, 3306, 5432, 6379, 8080, 8443, 27017},
            block_all_network=True,
        ),
    )


def _strict_policy() -> SecurityPolicy:
    """STRICT 级别 - 最严格安全策略"""
    return SecurityPolicy(
        name="strict",
        level=SecurityLevel.STRICT,
        description="最严格安全级别 - 最小模块集，严格资源限制，完全隔离",
        resource_limits=ResourceLimits(
            max_execution_time=5.0,
            max_memory_mb=512,
            max_output_size=256 * 1024,
            max_file_size=512 * 1024,
            max_concurrent=2,
            cpu_time_limit=3.0,
        ),
        module_policy=ModulePolicy(
            allowed_modules={
                "math", "random", "string", "re", "json",
                "datetime", "collections", "itertools", "functools",
                "decimal", "fractions", "statistics", "typing",
                "dataclasses", "enum", "copy", "operator",
                "abc", "numbers",
            },
            blocked_modules=DANGEROUS_MODULES | {
                "numpy", "pandas", "scipy", "tkinter", "hashlib",
                "hmac", "base64", "uuid", "time", "struct",
                "array", "queue", "heapq", "bisect", "pprint",
                "textwrap", "contextlib", "warnings", "inspect",
            },
            allowed_builtins=set(),
            blocked_builtins=DANGEROUS_BUILTINS | {
                "open", "input", "breakpoint", "globals", "locals",
                "exit", "quit", "vars", "dir", "type",
            },
            allow_star_import=False,
        ),
        filesystem_policy=FilesystemPolicy(
            allowed_read_dirs=set(),
            allowed_write_dirs=set(),
            block_all_writes=True,
            block_all_reads=True,
        ),
        network_policy=NetworkPolicy(
            allowed_domains=set(),
            blocked_domains={"*"},
            allowed_ports=set(),
            blocked_ports=set(range(0, 65536)),
            block_all_network=True,
        ),
    )


# 预设策略映射
PRESET_POLICIES = {
    SecurityLevel.LOW: _low_policy,
    SecurityLevel.MEDIUM: _medium_policy,
    SecurityLevel.HIGH: _high_policy,
    SecurityLevel.STRICT: _strict_policy,
}


def get_preset_policy(level: SecurityLevel) -> SecurityPolicy:
    """获取预设策略"""
    factory = PRESET_POLICIES.get(level)
    if factory:
        return factory()
    return _medium_policy()


def list_preset_policies() -> list:
    """列出所有预设策略"""
    return [factory().to_dict() for factory in PRESET_POLICIES.values()]