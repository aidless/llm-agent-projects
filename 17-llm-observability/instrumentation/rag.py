"""RAG 埋点 - 追踪检索增强生成的各个阶段。"""

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

from metrics.registry import get_registry
from tracing.span import SpanKind
from tracing.tracer import get_tracer


@contextmanager
def trace_rag(
    name: str = "RAG",
    query: Optional[str] = None,
    top_k: int = 4,
):
    """上下文管理器方式追踪 RAG pipeline。

    用法::

        with trace_rag(query="What is X?", top_k=5) as span:
            docs = retriever.get_relevant_documents("What is X?")
            span.set_attribute("rag.retrieved_docs", len(docs))
            response = llm.generate(context=docs, query="What is X?")
    """
    tracer = get_tracer()
    registry = get_registry()

    with tracer.start_span(name=f"rag.{name}", kind=SpanKind.CHAIN) as span:
        span.set_attribute("rag.top_k", top_k)
        if query:
            span.set_attribute("rag.query", query[:500])

        registry.counter("rag.queries", "Total RAG queries").increment()
        start = time.time()

        try:
            yield span
            registry.counter("rag.success", "Successful RAG queries").increment()
        except Exception as exc:
            registry.counter("rag.errors", "Failed RAG queries").increment()
            span.set_error(str(exc))
            raise
        finally:
            duration = (time.time() - start) * 1000
            registry.histogram("rag.duration_ms", "RAG pipeline latency").observe(duration)


@contextmanager
def trace_retrieval(
    retriever_name: str = "default",
    query: Optional[str] = None,
    top_k: int = 4,
):
    """追踪检索阶段。

    用法::

        with trace_retrieval(retriever_name="vector_store", query="X") as span:
            docs = retriever.similarity_search("X", k=4)
            span.set_attribute("retrieval.result_count", len(docs))
    """
    tracer = get_tracer()
    registry = get_registry()

    with tracer.start_span(name=f"retrieval.{retriever_name}", kind=SpanKind.RETRIEVAL) as span:
        span.set_attribute("retrieval.retriever", retriever_name)
        span.set_attribute("retrieval.top_k", top_k)
        if query:
            span.set_attribute("retrieval.query", query[:500])

        start = time.time()

        try:
            yield span
            registry.counter("retrieval.queries", "Total retrieval queries").increment()
        except Exception as exc:
            span.set_error(str(exc))
            raise
        finally:
            duration = (time.time() - start) * 1000
            registry.histogram("retrieval.duration_ms", "Retrieval latency").observe(duration)


@contextmanager
def trace_embedding(
    model: str = "text-embedding-ada-002",
    texts_count: int = 1,
):
    """追踪 Embedding 调用。"""
    tracer = get_tracer()
    registry = get_registry()

    with tracer.start_span(name=f"embedding.{model}", kind=SpanKind.LLM) as span:
        span.set_attribute("embedding.model", model)
        span.set_attribute("embedding.texts_count", texts_count)

        registry.counter("embedding.requests", "Total embedding requests").increment()
        start = time.time()

        try:
            yield span
        except Exception as exc:
            span.set_error(str(exc))
            raise
        finally:
            duration = (time.time() - start) * 1000
            registry.histogram("embedding.duration_ms", "Embedding latency").observe(duration)


def trace_retriever(retriever_name: str = "default"):
    """装饰器方式追踪检索器调用。

    用法::

        @trace_retriever(retriever_name="faiss")
        def search(query: str, k: int = 4) -> list:
            return vector_store.search(query, k=k)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with trace_retrieval(retriever_name=retriever_name) as span:
                result = func(*args, **kwargs)
                if isinstance(result, list):
                    span.set_attribute("retrieval.result_count", len(result))
                return result

        return wrapper

    return decorator