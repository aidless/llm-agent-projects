"""资源限制器 - 通过 signal 和 resource 模块实现执行资源限制"""

import signal
import time
import threading
import os
from typing import Optional, Callable
from dataclasses import dataclass


class TimeoutError(Exception):
    """执行超时错误"""
    pass


class MemoryLimitExceeded(Exception):
    """内存超限错误"""
    pass


class CPULimitExceeded(Exception):
    """CPU 时间超限错误"""
    pass


@dataclass
class ResourceUsage:
    """资源使用情况"""
    execution_time: float = 0.0
    memory_used_mb: float = 0.0
    cpu_time: float = 0.0
    output_size: int = 0
    timed_out: bool = False
    memory_exceeded: bool = False
    cpu_exceeded: bool = False

    def to_dict(self) -> dict:
        return {
            "execution_time": self.execution_time,
            "memory_used_mb": round(self.memory_used_mb, 2),
            "cpu_time": round(self.cpu_time, 2),
            "output_size": self.output_size,
            "timed_out": self.timed_out,
            "memory_exceeded": self.memory_exceeded,
            "cpu_exceeded": self.cpu_exceeded,
        }


class ResourceLimiter:
    """资源限制器"""

    def __init__(
        self,
        max_time: float = 30.0,
        max_memory_mb: int = 256,
        cpu_time_limit: float = 10.0,
    ):
        self.max_time = max_time
        self.max_memory_mb = max_memory_mb
        self.cpu_time_limit = cpu_time_limit
        self._timer = None
        self._timed_out = False

    def _timeout_handler(self, signum, frame):
        """信号超时处理器 (仅 Unix)"""
        self._timed_out = True
        raise TimeoutError(f"Execution timed out after {self.max_time} seconds")

    def execute_with_limits(self, func, *args, **kwargs):
        """在资源限制下执行函数

        Returns:
            (result, resource_usage, error)
        """
        self._timed_out = False
        usage = ResourceUsage()
        start_time = time.monotonic()

        # 使用线程定时器实现超时 (跨平台)
        timeout_occurred = [False]
        exception_holder = [None]

        def target():
            try:
                return func(*args, **kwargs)
            except Exception as e:
                exception_holder[0] = e
                raise

        result = None
        error = exception_holder[0]
        thread = threading.Thread(target=lambda: setattr(target, 'result', target()))
        
        def run_in_thread():
            nonlocal result
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                exception_holder[0] = e

        thread = threading.Thread(target=run_in_thread)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.max_time)

        elapsed = time.monotonic() - start_time
        usage.execution_time = elapsed
        usage.cpu_time = min(elapsed, self.cpu_time_limit)

        if thread.is_alive():
            timeout_occurred[0] = True
            usage.timed_out = True
            error = TimeoutError(f"Execution timed out after {self.max_time} seconds")

        if exception_holder[0] and not timeout_occurred[0]:
            error = exception_holder[0]

        # 估算内存使用 (线程方式下无法精确测量子线程)
        try:
            import psutil
            process = psutil.Process(os.getpid())
            usage.memory_used_mb = process.memory_info().rss / (1024 * 1024)
            if usage.memory_used_mb > self.max_memory_mb:
                usage.memory_exceeded = True
                if not error:
                    error = MemoryLimitExceeded(
                        f"Memory limit exceeded: {usage.memory_used_mb:.1f}MB > {self.max_memory_mb}MB"
                    )
        except (ImportError, Exception):
            pass

        return result, usage, error

    def check_memory(self) -> float:
        """检查当前内存使用 (MB)"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def set_limits(self):
        """设置进程级资源限制 (Unix only)"""
        try:
            import resource
            # 设置 CPU 时间限制
            resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_time_limit, self.cpu_time_limit + 1))
            # 设置内存限制 (近似)
            memory_bytes = self.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except (ImportError, ValueError, OSError):
            pass  # Windows 或无权限时静默失败