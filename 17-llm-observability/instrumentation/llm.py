"""LLM 调用埋点 - 提供装饰器和上下文管理器两种方式追踪 LLM 调用。"""

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

from metrics.registry import get_registry
from tracing.span import Span, SpanKind
from tracing.tracer import get_tracer


# 模拟 LLM 模型定价表 (每 1K tokens, USD)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
    "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """根据模型和 token 数计算成本。"""
    pricing = MODEL_PRICING.get(model, {"prompt": 0.01, "completion": 0.03})
    return (
        (prompt_tokens / 1000.0) * pricing["prompt"]
        + (completion_tokens / 1000.0) * pricing["completion"]
    )


@contextmanager
def trace_llm_call(
    model: str,
    input_text: Any = None,
    prompt_template: Optional[str] = None,
    prompt_template_version: Optional[str] = None,
):
    """上下文管理器方式追踪 LLM 调用。

    用法::

        with trace_llm_call(model="gpt-4", input_text="Hello") as span:
            response = call_openai(...)
            span.set_llm(input_text="Hello", output_text=response, model="gpt-4",
                        prompt_tokens=10, completion_tokens=50)
    """
    tracer = get_tracer()
    registry = get_registry()

    with tracer.start_span(name=f"llm.{model}", kind=SpanKind.LLM) as span:
        span.set_attribute("llm.model", model)
        if prompt_template:
            span.set_attribute("llm.prompt_template", prompt_template)
        if prompt_template_version:
            span.set_attribute("llm.prompt_template_version", prompt_template_version)

        # 记录请求指标
        registry.counter("llm.requests.total", "Total LLM requests").increment()

        start = time.time()
        try:
            yield span
            registry.counter("llm.requests.success", "Successful LLM requests").increment()
        except Exception:
            registry.counter("llm.requests.error", "Failed LLM requests").increment()
            raise
        finally:
            duration = (time.time() - start) * 1000
            registry.histogram("llm.request.duration_ms", "LLM request latency").observe(duration)

            # 记录 token 指标
            if span.total_tokens > 0:
                registry.counter("llm.tokens.total", "Total tokens used").increment(span.total_tokens)
                registry.counter("llm.tokens.prompt", "Prompt tokens used").increment(span.prompt_tokens)
                registry.counter("llm.tokens.completion", "Completion tokens used").increment(span.completion_tokens)

                cost = calculate_cost(model, span.prompt_tokens, span.completion_tokens)
                span.set_attribute("llm.cost_usd", cost)
                registry.counter("llm.cost_usd", "Total LLM cost in USD").increment(cost)


def trace_llm(
    model: str = "unknown",
    prompt_template: Optional[str] = None,
    prompt_template_version: Optional[str] = None,
):
    """装饰器方式追踪 LLM 调用。

    被装饰的函数应返回 (response_text, prompt_tokens, completion_tokens) 元组，
    或一个包含 llm_output, prompt_tokens, completion_tokens 的字典。

    用法::

        @trace_llm(model="gpt-4")
        def my_llm_call(prompt: str) -> dict:
            ...
            return {"llm_output": response, "prompt_tokens": 10, "completion_tokens": 50}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            registry = get_registry()

            with tracer.start_span(name=f"llm.{model}", kind=SpanKind.LLM) as span:
                span.set_attribute("llm.model", model)
                span.set_attribute("llm.input_args", str(args)[:500])
                if prompt_template:
                    span.set_attribute("llm.prompt_template", prompt_template)
                if prompt_template_version:
                    span.set_attribute("llm.prompt_template_version", prompt_template_version)

                registry.counter("llm.requests.total", "Total LLM requests").increment()
                start = time.time()

                try:
                    result = func(*args, **kwargs)
                    registry.counter("llm.requests.success", "Successful LLM requests").increment()
                except Exception:
                    registry.counter("llm.requests.error", "Failed LLM requests").increment()
                    raise
                finally:
                    duration = (time.time() - start) * 1000
                    registry.histogram("llm.request.duration_ms", "LLM request latency").observe(duration)

                # 解析返回值
                if isinstance(result, dict):
                    span.set_llm(
                        input_text=result.get("llm_input", kwargs.get("prompt", str(args))),
                        output_text=result.get("llm_output", ""),
                        model_name=model,
                        prompt_tokens=result.get("prompt_tokens", 0),
                        completion_tokens=result.get("completion_tokens", 0),
                        prompt_template=prompt_template,
                        prompt_template_version=prompt_template_version,
                    )
                elif isinstance(result, tuple) and len(result) >= 3:
                    span.set_llm(
                        input_text=kwargs.get("prompt", str(args)),
                        output_text=result[0],
                        model=model,
                        prompt_tokens=result[1],
                        completion_tokens=result[2],
                    )

                return result

        return wrapper

    return decorator


def record_stream_chunk(model: str, chunk_index: int, content: str) -> None:
    """记录流式响应 chunk（配合手动 Span 使用）。"""
    registry = get_registry()
    registry.counter("llm.stream.chunks", "Stream chunks count").increment()
    # 注意: 需要在 Span 上下文中调用才能获取当前 span
    from tracing.context import get_current_span_id
    span_id = get_current_span_id()
    if span_id:
        from storage.memory_store import get_store
        get_store().add_span_event(span_id, {
            "name": "stream_chunk",
            "attributes": {"chunk_index": chunk_index, "content": content},
        })