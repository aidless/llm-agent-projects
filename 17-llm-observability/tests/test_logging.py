"""Logging 模块测试 - 结构化日志 Handler 和 Logger。"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from obs_logging.handler import StructuredLogHandler, get_structured_logger
from obs_logging.logger import LLMObservabilityLogger
from storage.memory_store import get_store
from tracing.context import set_context, clear_context


class TestStructuredLogHandler:
    def test_handler_stores_log(self):
        handler = StructuredLogHandler()
        logger = logging.getLogger("test-handler")
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)

        logger.info("Hello World")
        store = get_store()
        logs = store.query_logs()
        assert len(logs) == 1
        assert logs[0]["message"] == "Hello World"
        assert logs[0]["level"] == "INFO"

    def test_handler_trace_context(self):
        set_context("trace-abc", "span-xyz")
        handler = StructuredLogHandler()
        logger = logging.getLogger("test-ctx")
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)

        logger.info("With trace")
        store = get_store()
        logs = store.query_logs()
        assert len(logs) == 1
        assert logs[0]["trace_id"] == "trace-abc"
        assert logs[0]["span_id"] == "span-xyz"

    def test_handler_log_levels(self):
        handler = StructuredLogHandler()
        logger = logging.getLogger("test-levels")
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)

        logger.debug("debug msg")
        logger.warning("warn msg")
        logger.error("error msg")

        store = get_store()
        logs = store.query_logs()
        assert len(logs) == 3
        levels = {l["level"] for l in logs}
        assert "DEBUG" in levels
        assert "WARNING" in levels
        assert "ERROR" in levels

    def test_handler_exception(self):
        handler = StructuredLogHandler()
        logger = logging.getLogger("test-exc")
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)

        try:
            raise ValueError("test exception")
        except ValueError:
            logger.exception("An error occurred")

        store = get_store()
        logs = store.query_logs()
        assert len(logs) == 1
        assert logs[0]["exception"] is not None
        assert logs[0]["exception"]["type"] == "ValueError"


class TestLLMObservabilityLogger:
    def test_basic_logging(self):
        obs_logger = LLMObservabilityLogger("test-obs")
        obs_logger.info("Test message")
        store = get_store()
        logs = store.query_logs()
        assert len(logs) >= 1
        assert any(l["message"] == "Test message" for l in logs)

    def test_llm_request_log(self):
        obs_logger = LLMObservabilityLogger("test-llm-log")
        obs_logger.log_request("gpt-4", {"messages": [{"role": "user", "content": "Hi"}]})
        store = get_store()
        logs = store.query_logs(message_contains="LLM request")
        assert len(logs) >= 1

    def test_llm_response_log(self):
        obs_logger = LLMObservabilityLogger("test-llm-log")
        obs_logger.log_response(
            model="gpt-4",
            output_data="Hello!",
            prompt_tokens=10,
            completion_tokens=5,
        )
        store = get_store()
        logs = store.query_logs(message_contains="LLM response")
        assert len(logs) >= 1

    def test_get_structured_logger_singleton(self):
        logger1 = get_structured_logger("singleton-test")
        logger2 = get_structured_logger("singleton-test")
        assert logger1 is logger2