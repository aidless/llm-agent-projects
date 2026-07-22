"""结构化日志 Handler - 自动关联 Trace ID 并输出 JSON 格式日志。"""

import json
import logging
import sys
import time
import traceback
from typing import Any, Dict, Optional


class StructuredLogHandler(logging.Handler):
    """将日志以结构化 JSON 格式输出，并自动关联 Trace ID。

    用法::

        handler = StructuredLogHandler()
        logger = logging.getLogger("my-app")
        logger.addHandler(handler)
        logger.info("Hello", extra={"key": "value"})
    """

    def __init__(self, store=None, level: int = logging.DEBUG) -> None:
        super().__init__(level)
        self._store = store
        # 使用导入时获取 store 的方式避免循环引用
        if self._store is None:
            from storage.memory_store import get_store
            self._store = get_store()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = self._format_record(record)
            self._store.store_log(entry)
        except Exception:
            self.handleError(record)

    def _format_record(self, record: logging.LogRecord) -> Dict[str, Any]:
        """将 LogRecord 转换为结构化字典。"""
        from tracing.context import get_current_trace_id, get_current_span_id

        entry: Dict[str, Any] = {
            "timestamp": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "trace_id": get_current_trace_id(),
            "span_id": get_current_span_id(),
        }

        # 附加自定义字段
        if hasattr(record, "trace_id") and record.trace_id:
            entry["trace_id"] = record.trace_id
        if hasattr(record, "span_id") and record.span_id:
            entry["span_id"] = record.span_id

        # 异常信息
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "stacktrace": traceback.format_exception(*record.exc_info),
            }

        return entry


def get_structured_logger(
    name: str = "llm-observability",
    level: int = logging.INFO,
) -> logging.Logger:
    """获取配置了 StructuredLogHandler 的 Logger。

    用法::

        logger = get_structured_logger("my-service")
        logger.info("Service started")
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if not any(isinstance(h, StructuredLogHandler) for h in logger.handlers):
        handler = StructuredLogHandler(level=level)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger