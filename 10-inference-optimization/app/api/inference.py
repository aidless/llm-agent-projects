"""推理 API - OpenAI 兼容的推理接口"""

from __future__ import annotations

import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException

from ..models import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionChoice, ChatMessage, UsageInfo,
    CompletionRequest, CompletionResponse,
    ErrorResponse,
)
from ...engine.simulator import InferenceSimulator
from ...engine.batcher import BatchResult
from ...monitoring.metrics import MetricsCollector

router = APIRouter(prefix="/v1", tags=["inference"])

# 全局实例 (在 main.py 中初始化)
simulator: InferenceSimulator = None  # type: ignore
metrics: MetricsCollector = None  # type: ignore


def init_inference(sim: InferenceSimulator, met: MetricsCollector) -> None:
    global simulator, metrics
    simulator = sim
    metrics = met


def _messages_to_prompt(messages: List[ChatMessage]) -> str:
    """将对话消息合并为单个 prompt"""
    parts = []
    for msg in messages:
        parts.append(f"{msg.role}: {msg.content}")
    return "\n".join(parts)


def _result_to_response(result: BatchResult, model: str) -> Dict:
    """将 BatchResult 转换为响应"""
    return {
        "generated_text": result.generated_text,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "ttft_ms": round(result.ttft_ms, 2),
        "total_latency_ms": round(result.total_latency_ms, 2),
        "tokens_per_second": round(result.tokens_per_second, 2),
    }


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    responses={429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI 兼容的 Chat Completions 接口"""
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")

    prompt = _messages_to_prompt(request.messages)

    # 熔断检查在中间件中处理

    result = await simulator.generate(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        priority=request.priority,
    )

    # 记录指标
    if metrics:
        metrics.record_inference(
            ttft=result.ttft_ms / 1000,
            e2e_latency=result.total_latency_ms / 1000,
            tokens_generated=result.completion_tokens,
            tokens_per_second=result.tokens_per_second,
        )

    total_tokens = result.prompt_tokens + result.completion_tokens

    return ChatCompletionResponse(
        id=result.request_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=result.generated_text,
                ),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=total_tokens,
        ),
    )


@router.post(
    "/completions",
    response_model=CompletionResponse,
)
async def completions(request: CompletionRequest):
    """文本补全接口"""
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")

    result = await simulator.generate(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        priority=request.priority,
    )

    if metrics:
        metrics.record_inference(
            ttft=result.ttft_ms / 1000,
            e2e_latency=result.total_latency_ms / 1000,
            tokens_generated=result.completion_tokens,
            tokens_per_second=result.tokens_per_second,
        )

    total_tokens = result.prompt_tokens + result.completion_tokens

    return CompletionResponse(
        id=result.request_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            {
                "text": result.generated_text,
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=total_tokens,
        ),
    )


@router.get("/models", response_model=Dict)
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "mock-llm-7b",
                "object": "model",
                "owned_by": "local",
                "quantization": "fp16",
            },
            {
                "id": "mock-llm-7b-int8",
                "object": "model",
                "owned_by": "local",
                "quantization": "int8",
            },
            {
                "id": "mock-llm-7b-int4",
                "object": "model",
                "owned_by": "local",
                "quantization": "int4",
            },
        ],
    }
