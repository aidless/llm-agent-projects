"""全局测试 fixtures - 确保每个测试之间完全隔离。"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(autouse=True)
def _reset_global_state():
    """每个测试前后完全重置全局状态。"""
    from storage.memory_store import reset_store
    from tracing.tracer import reset_tracer
    from tracing.context import clear_context
    from metrics.registry import reset_registry
    from metrics.aggregation import reset_aggregator

    reset_store()
    reset_tracer()
    clear_context()
    reset_registry()
    reset_aggregator()

    yield

    reset_store()
    reset_tracer()
    clear_context()
    reset_registry()
    reset_aggregator()