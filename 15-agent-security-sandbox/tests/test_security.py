"""安全测试 - 验证危险代码被正确拦截"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox.executor import SandboxExecutor
from policy.presets import get_preset_policy
from policy.models import SecurityLevel
from audit.logger import AuditLogger
from security.syscall_interceptor import SyscallInterceptor, InterceptorConfig
from security.import_guard import ImportGuard
from security.file_guard import FileGuard, FileGuardConfig
from security.network_guard import NetworkGuard, NetworkGuardConfig


@pytest.fixture
def executor():
    return SandboxExecutor(audit_logger=AuditLogger())


@pytest.fixture
def medium_policy():
    return get_preset_policy(SecurityLevel.MEDIUM)


@pytest.fixture
def strict_policy():
    return get_preset_policy(SecurityLevel.STRICT)


@pytest.fixture
def low_policy():
    return get_preset_policy(SecurityLevel.LOW)


class TestDangerousCodeBlocked:
    """危险代码拦截测试"""

    def test_os_import_blocked(self, executor, medium_policy):
        """测试 os 模块导入被拦截"""
        result = executor.execute("import os", medium_policy)
        assert result.success is False
        assert len(result.security_violations) > 0

    def test_subprocess_import_blocked(self, executor, medium_policy):
        """测试 subprocess 模块导入被拦截"""
        result = executor.execute("import subprocess", medium_policy)
        assert result.success is False
        assert len(result.security_violations) > 0

    def test_exec_blocked(self, executor, medium_policy):
        """测试 exec 调用被拦截"""
        result = executor.execute("exec('print(1)')", medium_policy)
        assert result.success is False
        assert len(result.security_violations) > 0

    def test_eval_blocked(self, executor, medium_policy):
        """测试 eval 调用被拦截"""
        result = executor.execute("eval('1+1')", medium_policy)
        assert result.success is False
        assert len(result.security_violations) > 0

    def test_open_blocked(self, executor, medium_policy):
        """测试 open 调用被拦截"""
        result = executor.execute("open('/etc/passwd')", medium_policy)
        assert result.success is False

    def test_os_system_blocked(self, executor, medium_policy):
        """测试 os.system 被拦截"""
        result = executor.execute("import os\nos.system('ls')", medium_policy)
        assert result.success is False
        assert len(result.security_violations) > 0

    def test_socket_import_blocked(self, executor, medium_policy):
        """测试 socket 导入被拦截"""
        result = executor.execute("import socket", medium_policy)
        assert result.success is False

    def test_from_os_import_blocked(self, executor, medium_policy):
        """测试 from os import 被拦截"""
        result = executor.execute("from os import system", medium_policy)
        assert result.success is False
        assert len(result.security_violations) > 0

    def test_compile_blocked(self, executor, medium_policy):
        """测试 compile 调用被拦截"""
        result = executor.execute("compile('1+1', '', 'eval')", medium_policy)
        assert result.success is False


class TestSyscallInterceptor:
    """系统调用拦截器测试"""

    def test_detect_exec_call(self):
        """测试检测 exec 调用"""
        interceptor = SyscallInterceptor()
        violations = interceptor.analyze("exec('code')")
        assert len(violations) > 0
        assert any(v.violation_type == "BLOCKED_BUILTIN_CALL" for v in violations)

    def test_detect_os_import(self):
        """测试检测 os 导入"""
        interceptor = SyscallInterceptor()
        violations = interceptor.analyze("import os")
        assert len(violations) > 0
        assert any(v.violation_type == "BLOCKED_IMPORT" for v in violations)

    def test_detect_getattr_bypass(self):
        """测试检测 getattr 绕过尝试"""
        interceptor = SyscallInterceptor()
        violations = interceptor.analyze("getattr(os, 'system')('ls')")
        assert len(violations) > 0

    def test_safe_code_no_violations(self):
        """测试安全代码无违规"""
        interceptor = SyscallInterceptor()
        violations = interceptor.analyze("x = 1 + 2\nprint(x)")
        assert len(violations) == 0

    def test_syntax_error_detection(self):
        """测试语法错误检测"""
        interceptor = SyscallInterceptor()
        violations = interceptor.analyze("def (")
        assert len(violations) > 0
        assert any(v.violation_type == "SYNTAX_ERROR" for v in violations)


class TestImportGuard:
    """导入守卫测试"""

    def test_blocked_module(self):
        guard = ImportGuard()
        assert guard.check_import("os") is False
        assert guard.check_import("subprocess") is False

    def test_allowed_module(self):
        guard = ImportGuard()
        assert guard.check_import("math") is True
        assert guard.check_import("json") is True

    def test_star_import_blocked(self):
        guard = ImportGuard()
        assert guard.check_from_import("math", ["*"]) is False

    def test_safe_builtins_no_dangerous(self):
        guard = ImportGuard()
        safe = guard.get_safe_builtins()
        assert "exec" not in safe
        assert "eval" not in safe
        assert "open" not in safe
        assert "__import__" not in safe

    def test_safe_builtins_has_safe(self):
        guard = ImportGuard()
        safe = guard.get_safe_builtins()
        assert "print" in safe
        assert "len" in safe
        assert "int" in safe


class TestFileGuard:
    """文件守卫测试"""

    def test_write_blocked_by_default(self):
        guard = FileGuard()
        assert guard.check_write("/tmp/test.txt") is False

    def test_read_allowed_by_default(self):
        guard = FileGuard()
        assert guard.check_read("/etc/passwd") is True

    def test_read_blocked_when_configured(self):
        guard = FileGuard(FileGuardConfig(block_all_reads=True))
        assert guard.check_read("/etc/passwd") is False

    def test_write_allowed_in_whitelist(self):
        guard = FileGuard(FileGuardConfig(
            block_all_writes=False,
            allowed_write_dirs={"/tmp/sandbox"},
        ))
        assert guard.check_write("/tmp/sandbox/test.txt") is True
        assert guard.check_write("/tmp/other/test.txt") is False

    def test_file_size_check(self):
        guard = FileGuard(FileGuardConfig(max_file_size=1024))
        assert guard.check_file_size(512) is True
        assert guard.check_file_size(2048) is False


class TestNetworkGuard:
    """网络守卫测试"""

    def test_all_blocked_by_default(self):
        guard = NetworkGuard()
        assert guard.check_domain("example.com") is False
        assert guard.check_port(80) is False
        assert guard.check_connection("example.com", 80)[0] is False

    def test_blocked_domains(self):
        guard = NetworkGuard(NetworkGuardConfig(block_all_network=False))
        assert guard.check_domain("localhost") is False
        assert guard.check_domain("127.0.0.1") is False
        assert guard.check_domain("169.254.169.254") is False

    def test_allowed_domain(self):
        guard = NetworkGuard(NetworkGuardConfig(
            block_all_network=False,
            allowed_domains={"api.example.com"},
        ))
        assert guard.check_domain("api.example.com") is True

    def test_blocked_ports(self):
        guard = NetworkGuard(NetworkGuardConfig(block_all_network=False))
        assert guard.check_port(22) is False
        assert guard.check_port(3306) is False