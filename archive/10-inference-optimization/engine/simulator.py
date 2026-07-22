"""推理模拟器 - 模拟 LLM 推理过程"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .tokenizer_mock import MockTokenizer, TokenizerConfig
from .batcher import DynamicBatcher, InferenceRequest, BatchResult
from .scheduler import RequestScheduler, SchedulingPolicy, SchedulerConfig


@dataclass
class SimulatorConfig:
    """模拟器配置"""
    model_name: str = "mock-llm-7b"
    num_layers: int = 32
    hidden_size: int = 4096
    num_heads: int = 32
    num_kv_heads: int = 32
    vocab_size: int = 32000
    max_model_length: int = 4096
    # 模拟延迟参数
    base_prefill_latency_ms: float = 20.0
    base_decode_latency_ms: float = 5.0
    batch_size_latency_factor: float = 0.2


@dataclass
class ModelStats:
    """模型统计信息"""
    num_parameters: int = 0
    memory_fp32_gb: float = 0.0
    memory_fp16_gb: float = 0.0
    memory_int8_gb: float = 0.0
    memory_int4_gb: float = 0.0


class InferenceSimulator:
    """LLM 推理模拟器 - 模拟推理引擎的核心行为"""

    def __init__(self, config: Optional[SimulatorConfig] = None):
        self.config = config or SimulatorConfig()

        self.tokenizer = MockTokenizer(
            TokenizerConfig(vocab_size=self.config.vocab_size)
        )
        self.batcher = DynamicBatcher(
            max_batch_size=32,
            tokenizer=self.tokenizer,
        )
        self.scheduler = RequestScheduler(
            batcher=self.batcher,
            config=SchedulerConfig(),
        )

        # 运行状态
        self._is_running = False
        self._request_counter = 0
        self._total_requests_served = 0
        self._total_tokens_generated = 0
        self._total_latency_s = 0.0

    def get_model_stats(self) -> ModelStats:
        """计算模型参数和内存"""
        # 参数量估算: layers * hidden_size^2 * 12 (Transformer MLP factor)
        params = self.config.num_layers * self.config.hidden_size ** 2 * 12
        bytes_per_param_fp32 = 4
        bytes_per_param_fp16 = 2
        bytes_per_param_int8 = 1
        bytes_per_param_int4 = 0.5

        return ModelStats(
            num_parameters=params,
            memory_fp32_gb=params * bytes_per_param_fp32 / (1024 ** 3),
            memory_fp16_gb=params * bytes_per_param_fp16 / (1024 ** 3),
            memory_int8_gb=params * bytes_per_param_int8 / (1024 ** 3),
            memory_int4_gb=params * bytes_per_param_int4 / (1024 ** 3),
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        priority: int = 0,
    ) -> BatchResult:
        """单请求生成"""
        self._request_counter += 1
        request = InferenceRequest(
            request_id=f"req_{self._request_counter}",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=priority,
            start_time=time.perf_counter(),
        )

        await self.scheduler.enqueue(request)
        batch = await self.scheduler.dequeue_batch()
        if batch is None:
            # 重新取回刚入队的请求
            batch = await self.scheduler.dequeue_batch()

        if batch:
            results = await self.batcher.process_batch(batch)
            for r in results:
                self._total_requests_served += 1
                self._total_tokens_generated += r.completion_tokens
                self._total_latency_s += r.total_latency_ms / 1000
            # 找到当前请求的结果
            for r in results:
                if r.request_id == request.request_id:
                    return r

        # Fallback: 空结果
        return BatchResult(
            request_id=request.request_id,
            generated_text="",
            prompt_tokens=0,
            completion_tokens=0,
            ttft_ms=0,
            total_latency_ms=0,
            tokens_per_second=0,
        )

    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 128,
    ) -> List[BatchResult]:
        """批量请求生成"""
        self._request_counter += 1
        requests = []
        for i, prompt in enumerate(prompts):
            req = InferenceRequest(
                request_id=f"batch_req_{self._request_counter}_{i}",
                prompt=prompt,
                max_tokens=max_tokens,
                priority=0,
                start_time=time.perf_counter(),
            )
            requests.append(req)
            await self.scheduler.enqueue(req)

        all_results: List[BatchResult] = []
        while True:
            batch = await self.scheduler.dequeue_batch()
            if batch is None:
                break
            results = await self.batcher.process_batch(batch)
            all_results.extend(results)
            for r in results:
                self._total_requests_served += 1
                self._total_tokens_generated += r.completion_tokens
                self._total_latency_s += r.total_latency_ms / 1000

        return all_results

    def get_summary(self) -> Dict:
        """获取运行摘要"""
        avg_latency = (
            self._total_latency_s / self._total_requests_served
            if self._total_requests_served > 0
            else 0
        )
        avg_tps = (
            self._total_tokens_generated / self._total_latency_s
            if self._total_latency_s > 0
            else 0
        )
        return {
            "model_name": self.config.model_name,
            "total_requests_served": self._total_requests_served,
            "total_tokens_generated": self._total_tokens_generated,
            "avg_latency_ms": avg_latency * 1000,
            "avg_tokens_per_second": avg_tps,
            "batcher_stats": {
                "total_batches": self.batcher.total_batches_processed,
                "pending_count": self.batcher.pending_count,
            },
            "scheduler_stats": {
                "queue_size": self.scheduler.get_stats().current_queue_size,
                "total_rejected": self.scheduler.get_stats().total_rejected,
            },
        }
