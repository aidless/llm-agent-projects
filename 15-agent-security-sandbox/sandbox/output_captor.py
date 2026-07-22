"""输出捕获器 - 捕获 stdout/stderr 输出"""

import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class CapturedOutput:
    """捕获的输出结果"""
    stdout: str = ""
    stderr: str = ""
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "truncated": self.truncated,
        }


class OutputCaptor:
    """输出捕获器 - 安全地捕获代码执行的输出"""

    def __init__(self, max_output_size: int = 1024 * 1024):
        self.max_output_size = max_output_size
        self._stdout_buffer = io.StringIO()
        self._stderr_buffer = io.StringIO()

    def _truncate(self, text: str) -> tuple:
        """截断输出到最大大小"""
        if len(text) <= self.max_output_size:
            return text, False
        return text[:self.max_output_size], True

    def capture(self, func, *args, **kwargs) -> tuple:
        """捕获函数执行时的输出

        Returns:
            (result, captured_output) - 函数返回值和捕获的输出
        """
        self._stdout_buffer = io.StringIO()
        self._stderr_buffer = io.StringIO()

        result = None
        error = None

        try:
            with redirect_stdout(self._stdout_buffer), redirect_stderr(self._stderr_buffer):
                result = func(*args, **kwargs)
        except Exception as e:
            error = e

        stdout_text, stdout_truncated = self._truncate(self._stdout_buffer.getvalue())
        stderr_text, stderr_truncated = self._truncate(self._stderr_buffer.getvalue())

        output = CapturedOutput(
            stdout=stdout_text,
            stderr=stderr_text,
            truncated=stdout_truncated or stderr_truncated,
        )

        return result, output, error
