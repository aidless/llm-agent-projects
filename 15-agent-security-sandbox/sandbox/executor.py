"""代码执行器 - 沙箱的核心执行引擎"""

import time
import uuid
import sys
import io
import threading
from typing import Optional, Any, Dict
from dataclasses import dataclass, field

from sandbox.restricted_env import create_restricted_env, compile_restricted_code
from sandbox.resource_limiter import ResourceLimiter, ResourceUsage, TimeoutError as SandboxTimeout
from sandbox.output_captor import OutputCaptor, CapturedOutput
from security.syscall_interceptor import SyscallInterceptor, InterceptorConfig, SecurityError
from security.import_guard import ImportGuard, ImportGuardConfig
from security.file_guard import FileGuard, FileGuardConfig
from security.network_guard import NetworkGuard, NetworkGuardConfig
from policy.models import SecurityPolicy, SecurityLevel
from audit.logger import AuditLogger
from audit.event_tracker import EventTracker


@dataclass
class ExecutionResult:
    """执行结果"""
    execution_id: str = ""
    success: bool = False
    output: str = ""
    error: str = ""
    error_type: str = ""
    result_value: Any = None
    resource_usage: dict = field(default_factory=dict)
    security_violations: list = field(default_factory=list)
    execution_time: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "execution_id": self.execution_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "error_type": self.error_type,
            "resource_usage": self.resource_usage,
            "security_violations": self.security_violations,
            "execution_time": round(self.execution_time, 4),
        }
        try:
            d["result_value"] = repr(self.result_value) if self.result_value is not None else None
        except Exception:
            d["result_value"] = "<non-representable>"
        return d


class SandboxExecutor:
    """沙箱代码执行器"""

    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self.audit_logger = audit_logger or AuditLogger()
        self.event_tracker = EventTracker()

    def _build_security_config(self, policy: SecurityPolicy) -> tuple:
        """从安全策略构建各守卫配置"""
        interceptor_config = InterceptorConfig(
            blocked_builtins=policy.module_policy.blocked_builtins,
            blocked_modules=policy.module_policy.blocked_modules,
            allowed_builtins=policy.module_policy.allowed_builtins,
            allowed_modules=policy.module_policy.allowed_modules,
        )
        import_config = ImportGuardConfig(
            allowed_modules=policy.module_policy.allowed_modules,
            blocked_modules=policy.module_policy.blocked_modules,
            allow_star_import=policy.module_policy.allow_star_import,
        )
        file_config = FileGuardConfig(
            allowed_read_dirs=policy.filesystem_policy.allowed_read_dirs,
            allowed_write_dirs=policy.filesystem_policy.allowed_write_dirs,
            block_all_writes=policy.filesystem_policy.block_all_writes,
            block_all_reads=policy.filesystem_policy.block_all_reads,
            max_file_size=policy.resource_limits.max_file_size,
        )
        network_config = NetworkGuardConfig(
            allowed_domains=policy.network_policy.allowed_domains,
            blocked_domains=policy.network_policy.blocked_domains,
            allowed_ports=policy.network_policy.allowed_ports,
            blocked_ports=policy.network_policy.blocked_ports,
            block_all_network=policy.network_policy.block_all_network,
        )
        return interceptor_config, import_config, file_config, network_config

    def execute(
        self,
        code: str,
        policy: SecurityPolicy,
        variables: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
    ) -> ExecutionResult:
        """在沙箱中执行代码"""
        exec_id = execution_id or str(uuid.uuid4())[:8]
        start_time = time.monotonic()
        result = ExecutionResult(execution_id=exec_id)

        self.audit_logger.log_execution_start(exec_id, code, policy.name)

        # 1. AST 级别安全检查
        interceptor_config, import_config, file_config, network_config = self._build_security_config(policy)
        interceptor = SyscallInterceptor(interceptor_config)
        violations = interceptor.analyze(code)

        if violations:
            result.security_violations = [v.to_dict() for v in violations]
            result.success = False
            result.error = f"Security violation: {violations[0].detail}"
            result.error_type = "SecurityViolation"
            self.audit_logger.log_security_violation(exec_id, violations)
            self.event_tracker.track_violations(exec_id, violations)
            result.execution_time = time.monotonic() - start_time
            self.audit_logger.log_execution_end(exec_id, result)
            return result

        # 2. 编译受限代码
        try:
            compiled_code = compile_restricted_code(code)
        except SyntaxError as e:
            result.success = False
            result.error = str(e)
            result.error_type = "SyntaxError"
            self.audit_logger.log_execution_end(exec_id, result)
            result.execution_time = time.monotonic() - start_time
            return result

        # 3. 准备安全环境
        import_guard = ImportGuard(import_config)
        safe_env = create_restricted_env(import_guard, extra_globals=variables or {})

        # 4. 在单独的线程中执行，同时捕获输出和限制资源
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        error_holder = [None]
        max_output = policy.resource_limits.max_output_size

        def _run_sandbox():
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = stdout_buf
                sys.stderr = stderr_buf
                exec(compiled_code, safe_env)
            except Exception as e:
                error_holder[0] = e
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        # 资源限制执行
        limiter = ResourceLimiter(
            max_time=policy.resource_limits.max_execution_time,
            max_memory_mb=policy.resource_limits.max_memory_mb,
            cpu_time_limit=policy.resource_limits.cpu_time_limit,
        )

        _, resource_usage, timeout_error = limiter.execute_with_limits(_run_sandbox)

        # 收集输出
        stdout_text = stdout_buf.getvalue()
        stderr_text = stderr_buf.getvalue()

        # RestrictedPython PrintCollector: print() 输出存储在 _print 对象中
        for key in ("_print", "_printed_"):
            if key in safe_env and safe_env[key] is not None:
                rp_output = safe_env[key]
                if hasattr(rp_output, 'txt'):
                    collected = ''.join(rp_output.txt)
                    stdout_text = collected + stdout_text
                elif callable(rp_output):
                    collected = rp_output()
                    stdout_text = str(collected) + stdout_text
                break

        if len(stdout_text) > max_output:
            stdout_text = stdout_text[:max_output]
        if len(stderr_text) > max_output:
            stderr_text = stderr_text[:max_output]

        result.output = stdout_text

        # 处理错误
        error = timeout_error or error_holder[0]
        if error is not None:
            result.success = False
            result.error_type = type(error).__name__
            result.error = str(error)
            if stderr_text:
                result.error = stderr_text
            if isinstance(error, SecurityError):
                self.audit_logger.log_security_event(exec_id, "RUNTIME_SECURITY_ERROR", str(error))
                self.event_tracker.track_event(exec_id, "runtime_security_error", str(error), severity="CRITICAL")
        else:
            result.success = True
            if "__result__" in safe_env:
                result.result_value = safe_env["__result__"]

        result.resource_usage = resource_usage.to_dict()
        result.execution_time = time.monotonic() - start_time

        self.audit_logger.log_execution_end(exec_id, result)
        self.event_tracker.track_execution(exec_id, code, result)

        return result