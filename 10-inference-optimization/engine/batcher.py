"""动态批处理 - Continuous Batching 模拟"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from .tokenizer_mock import MockTokenizer


class BatchState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"


@dataclass
class InferenceRequest:
    """推理请求"""
    request_id: str
    prompt: str
    max_tokens: int = 128
    priority: int = 0  # 数值越高优先级越高
    temperature: float = 1.0
    top_p: float = 1.0

    # 运行时状态
    input_tokens: List[int] = field(default_factory=list)
    generated_tokens: List[int] = field(default_factory=list)
    start_time: float = 0.0
    ttft: float = 0.0  # 首 token 延迟 (秒)
    state: BatchState = BatchState.PENDING

    def __lt__(self, other: InferenceRequest) -> bool:
        """优先级排序"""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.start_time < other.start_time


@dataclass
class BatchResult:
    """批处理结果"""
    request_id: str
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_latency_ms: float
    tokens_per_second: float


class DynamicBatcher:
    """动态批处理器 - 模拟 Continuous Batching"""

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_time_ms: float = 50.0,
        tokenizer: Optional[MockTokenizer] = None,
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.tokenizer = tokenizer or MockTokenizer()

        self._pending_queue: List[InferenceRequest] = []
        self._running_batches: Dict[str, List[InferenceRequest]] = {}
        self._completed: Dict[str, BatchResult] = {}
        self._lock = asyncio.Lock()

        # 统计
        self.total_batches_processed = 0
        self.total_requests_processed = 0
        self.total_tokens_generated = 0

    async def submit(self, request: InferenceRequest) -> None:
        """提交推理请求到队列"""
        async with self._lock:
            request.input_tokens = self.tokenizer.encode(request.prompt).token_ids
            request.start_time = time.perf_counter()
            request.state = BatchState.PENDING
            self._pending_queue.append(request)
            # 按优先级排序
            self._pending_queue.sort()

    async def get_next_batch(self) -> Optional[List[InferenceRequest]]:
        """获取下一个批次"""
        async with self._lock:
            if not self._pending_queue:
                return None

            batch = []
            batch_start = time.perf_counter()
            while self._pending_queue and len(batch) < self.max_batch_size:
                req = self._pending_queue.pop(0)
                batch.append(req)

                elapsed_ms = (time.perf_counter() - batch_start) * 1000
                if elapsed_ms >= self.max_wait_time_ms:
                    break

            return batch if batch else None

    async def process_batch(self, batch: List[InferenceRequest]) -> List[BatchResult]:
        """处理一个批次 - 模拟 token by token 生成"""
        batch_id = f"batch_{self.total_batches_processed}"
        results: List[BatchResult] = []

        for req in batch:
            req.state = BatchState.RUNNING

        # 模拟首 token 延迟 (prefill)
        prefill_latency = 0.02 + len(batch) * 0.002  # 模拟批大小对 prefill 的影响
        await asyncio.sleep(prefill_latency)

        for req in batch:
            req.ttft = time.perf_counter() - req.start_time

        # 模拟逐 token 生成 (decode)
        max_steps = max(r.max_tokens for r in batch)
        for step in range(max_steps):
            active = [r for r in batch if len(r.generated_tokens) < r.max_tokens]
            if not active:
                break

            # decode 阶段延迟与活跃请求数成正比
            decode_latency = 0.005 + len(active) * 0.001
            await asyncio.sleep(decode_latency)

            for req in active:
                # 模拟生成 token (使用简单确定性方法)
                import random
                token_id = hash((req.request_id, step)) % self.tokenizer.vocab_size
                req.generated_tokens.append(token_id)

                if token_id == self.tokenizer.eos_token_id:
                    break

        # 收集结果
        for req in batch:
            req.state = BatchState.COMPLETED
            total_latency = time.perf_counter() - req.start_time

            generated_text = self.tokenizer.decode(req.generated_tokens)
            n_completion = len(req.generated_tokens)
            n_prompt = len(req.input_tokens)

            tps = n_completion / total_latency if total_latency > 0 else 0

            result = BatchResult(
                request_id=req.request_id,
                generated_text=generated_text,
                prompt_tokens=n_prompt,
                completion_tokens=n_completion,
                ttft_ms=req.ttft * 1000,
                total_latency_ms=total_latency * 1000,
                tokens_per_second=tps,
            )
            results.append(result)
            self._completed[req.request_id] = result

        # 更新统计
        self.total_batches_processed += 1
        self.total_requests_processed += len(batch)
        self.total_tokens_generated += sum(len(r.generated_tokens) for r in batch)

        return results

    @property
    def pending_count(self) -> int:
        return len(self._pending_queue)

    @property
    def avg_tokens_per_second(self) -> float:
        if self.total_requests_processed == 0:
            return 0.0
        return self.total_tokens_generated / max(1, self.total_requests_processed)
