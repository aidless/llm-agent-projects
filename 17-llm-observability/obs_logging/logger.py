"""结构化日志 - 提供便捷的日志 API，自动关联 Trace 上下文。"""

import logging
import time
from typing import Any, Dict, Optional

from obs_logging.handler import StructuredLogHandler, get_structured_logger
from storage.memory_store import get_store
from tracing.context import get_current_span_id, get_current_trace_id


class LLMObservabilityLogger:
    """LLM 可观测性专用 Logger，提供语义化的日志方法。

    用法::

        logger = LLMObservabilityLogger("my-service")
        logger.log_request("gpt-4", {"messages": [...]})
        logger.log_response("gpt-4", {"content": "..."}, prompt_tokens=100, completion_tokens=50)
    """

    def __init__(self, name: str = "llm-observability", level: int = logging.INFO) -> None:
        self._logger = get_structured_logger(name, level)

    def _enrich(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """注入 Trace 上下文到日志字段。"""
        base: Dict[str, Any] = {}
        trace_id = get_current_trace_id()
        span_id = get_current_span_id()
        if trace_id:
            base["trace_id"] = trace_id
        if span_id:
            base["span_id"] = span_id
        if extra:
            base.update(extra)
        return base

    def debug(self, message: str, **kwargs: Any) -> None:
        self._logger.debug(message, extra=self._enrich(kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        self._logger.info(message, extra=self._enrich(kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        self._logger.warning(message, extra=self._enrich(kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        self._logger.error(message, extra=self._enrich(kwargs))

    def log_request(self, model: str, input_data: Any, **kwargs: Any) -> None:
        """记录 LLM 请求日志。"""
        self.info(
            f"LLM request: model={model}",
            model=model,
            llm_direction="request",
            llm_input=input_data,
            **kwargs,
        )

    def log_response(
        self,
        model: str,
        output_data: Any,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        **kwargs: Any,
    ) -> None:
        """记录 LLM 响应日志。"""
        self.info(
            f"LLM response: model={model}, tokens={{p={prompt_tokens},c={completion_tokens}}}",
            model=model,
            llm_direction="response",
            llm_output=output_data,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            **kwargs,
        )

    def log_error(self, message: str, exception: Optional[Exception] = None, **kwargs: Any) -> None:
        """记录错误日志，支持异常对象。"""
        if exception:
            kwargs["exception_type"] = type(exception).__name__
            kwargs["exception_message"] = str(exception)
        self.error(message, **kwargs)