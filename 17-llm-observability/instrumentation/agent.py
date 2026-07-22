"""Agent 埋点 - 追踪 Agent 的决策循环、工具调用和最终输出。"""

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

from metrics.registry import get_registry
from tracing.span import SpanKind
from tracing.tracer import get_tracer


@contextmanager
def trace_agent(
    name: str,
    agent_type: str = "react",
    input_text: Optional[str] = None,
):
    """上下文管理器方式追踪 Agent 执行。

    用法::

        with trace_agent(name="ResearchAgent", agent_type="react") as span:
            result = agent.run("Find information about X")
            span.set_attribute("agent.output", result)
    """
    tracer = get_tracer()
    registry = get_registry()

    with tracer.start_span(name=f"agent.{name}", kind=SpanKind.AGENT) as span:
        span.set_attribute("agent.type", agent_type)
        if input_text:
            span.set_attribute("agent.input", input_text[:500])

        registry.counter("agent.runs", "Total agent runs").increment()
        start = time.time()
        iteration_count = 0

        try:
            yield span
            registry.counter("agent.success", "Successful agent runs").increment()
        except Exception as exc:
            registry.counter("agent.errors", "Failed agent runs").increment()
            span.set_error(str(exc))
            raise
        finally:
            duration = (time.time() - start) * 1000
            span.set_attribute("agent.iterations", iteration_count)
            registry.histogram("agent.duration_ms", "Agent run latency").observe(duration)
            registry.histogram("agent.iterations", "Agent iterations per run").observe(iteration_count)


@contextmanager
def trace_agent_iteration(agent_name: str, iteration: int):
    """追踪 Agent 的一次迭代（思考 + 动作）。"""
    tracer = get_tracer()

    with tracer.start_span(
        name=f"agent.{agent_name}.iteration.{iteration}",
        kind=SpanKind.AGENT,
    ) as span:
        span.set_attribute("agent.iteration", iteration)
        yield span


@contextmanager
def trace_tool_call(
    tool_name: str,
    tool_args: Optional[Dict[str, Any]] = None,
):
    """追踪工具调用。

    用法::

        with trace_agent("MyAgent") as agent_span:
            with trace_tool_call("search", {"query": "hello"}) as tool_span:
                result = search("hello")
                tool_span.set_attribute("tool.output", str(result))
    """
    tracer = get_tracer()
    registry = get_registry()

    with tracer.start_span(name=f"tool.{tool_name}", kind=SpanKind.TOOL) as span:
        span.set_attribute("tool.name", tool_name)
        if tool_args:
            for k, v in tool_args.items():
                span.set_attribute(f"tool.arg.{k}", str(v)[:200])

        registry.counter("tool.calls", "Total tool calls").increment()
        start = time.time()

        try:
            yield span
            registry.counter("tool.success", "Successful tool calls").increment()
        except Exception as exc:
            registry.counter("tool.errors", "Failed tool calls").increment()
            span.set_error(str(exc))
            raise
        finally:
            duration = (time.time() - start) * 1000
            registry.histogram("tool.duration_ms", "Tool call latency").observe(duration)


def trace_agent_tool(tool_name: str):
    """装饰器方式追踪工具调用。

    用法::

        @trace_agent_tool("calculator")
        def calculator(expression: str) -> float:
            return eval(expression)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with trace_tool_call(tool_name=tool_name) as span:
                result = func(*args, **kwargs)
                span.set_attribute("tool.output", str(result)[:500])
                return result

        return wrapper

    return decorator