"""LangChain 埋点 (模拟) - 追踪 LangChain 风格的 Chain 执行。"""

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

from metrics.registry import get_registry
from tracing.span import SpanKind
from tracing.tracer import get_tracer


@contextmanager
def trace_chain(
    name: str,
    input_vars: Optional[Dict[str, Any]] = None,
    chain_type: str = "sequential",
):
    """上下文管理器方式追踪 Chain 执行。

    用法::

        with trace_chain(name="Q&A Chain", chain_type="retrieval_qa"):
            result = chain.invoke({"query": "..."})
    """
    tracer = get_tracer()
    registry = get_registry()

    with tracer.start_span(name=f"chain.{name}", kind=SpanKind.CHAIN) as span:
        span.set_attribute("chain.type", chain_type)
        if input_vars:
            for k, v in input_vars.items():
                span.set_attribute(f"chain.input.{k}", str(v)[:200])

        registry.counter("chain.invocations", "Total chain invocations").increment()
        start = time.time()
        try:
            yield span
            registry.counter("chain.success", "Successful chain runs").increment()
        except Exception as exc:
            registry.counter("chain.errors", "Failed chain runs").increment()
            span.set_error(str(exc))
            raise
        finally:
            duration = (time.time() - start) * 1000
            registry.histogram("chain.duration_ms", "Chain execution latency").observe(duration)


def trace_chain_step(step_name: str):
    """装饰器方式追踪 Chain 中的单步。

    用法::

        @trace_chain_step("retriever")
        def retrieve(query: str) -> list:
            return retriever.get_relevant_documents(query)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            registry = get_registry()

            with tracer.start_span(name=f"chain.step.{step_name}", kind=SpanKind.CHAIN) as span:
                span.set_attribute("chain.step.name", step_name)
                start = time.time()

                try:
                    result = func(*args, **kwargs)
                    span.set_ok()
                    return result
                except Exception as exc:
                    span.set_error(str(exc))
                    raise
                finally:
                    duration = (time.time() - start) * 1000
                    registry.histogram("chain.step.duration_ms", "Chain step latency").observe(duration)

        return wrapper

    return decorator


def trace_langchain_run(
    run_id: str,
    run_type: str = "chain",
    name: str = "langchain_run",
):
    """装饰器 - 模拟 LangChain 的回调式埋点。

    用法::

        @trace_langchain_run(run_id="run_123", run_type="llm_chain")
        def run_chain(inputs: dict) -> dict:
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with trace_chain(name=name, chain_type=run_type):
                result = func(*args, **kwargs)
                return result

        return wrapper

    return decorator