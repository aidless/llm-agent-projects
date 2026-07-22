"""请求调度器 - 优先级队列和请求管理"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .batcher import DynamicBatcher, InferenceRequest


class SchedulingPolicy(Enum):
    FIFO = "fifo"
    PRIORITY = "priority"
    SHORTEST_JOB_FIRST = "sjf"
    LEAST_MEMORY_FIRST = "lmf"


@dataclass
class SchedulerConfig:
    """调度器配置"""
    policy: SchedulingPolicy = SchedulingPolicy.PRIORITY
    max_queue_size: int = 1024
    max_concurrent_batches: int = 4
    timeout_seconds: float = 60.0


@dataclass
class QueueStats:
    """队列统计"""
    total_received: int = 0
    total_processed: int = 0
    total_rejected: int = 0
    total_timeout: int = 0
    current_queue_size: int = 0
    avg_wait_time_ms: float = 0.0


class RequestScheduler:
    """请求调度器 - 管理请求队列和调度策略"""

    def __init__(
        self,
        batcher: DynamicBatcher,
        config: Optional[SchedulerConfig] = None,
    ):
        self.batcher = batcher
        self.config = config or SchedulerConfig()
        self._queue: List[InferenceRequest] = []
        self._lock = asyncio.Lock()
        self._running = False
        self._stats = QueueStats()
        self._wait_times: List[float] = []

    async def enqueue(self, request: InferenceRequest) -> bool:
        """将请求加入调度队列"""
        async with self._lock:
            if len(self._queue) >= self.config.max_queue_size:
                self._stats.total_rejected += 1
                return False

            self._queue.append(request)
            self._stats.total_received += 1
            self._stats.current_queue_size = len(self._queue)
            return True

    async def dequeue_batch(self) -> Optional[List[InferenceRequest]]:
        """按调度策略取出下一批请求"""
        async with self._lock:
            if not self._queue:
                return None

            now = time.perf_counter()

            # 超时清理
            timed_out = []
            remaining = []
            for req in self._queue:
                if now - req.start_time > self.config.timeout_seconds:
                    timed_out.append(req)
                else:
                    remaining.append(req)
            self._queue = remaining
            self._stats.total_timeout += len(timed_out)

            if not self._queue:
                return None

            # 按策略排序
            if self.config.policy == SchedulingPolicy.PRIORITY:
                self._queue.sort(key=lambda r: (-r.priority, r.start_time))
            elif self.config.policy == SchedulingPolicy.SHORTEST_JOB_FIRST:
                self._queue.sort(key=lambda r: r.max_tokens)
            elif self.config.policy == SchedulingPolicy.LEAST_MEMORY_FIRST:
                self._queue.sort(key=lambda r: len(r.input_tokens))

            # 取出批
            batch_size = min(len(self._queue), self.batcher.max_batch_size)
            batch = self._queue[:batch_size]
            self._queue = self._queue[batch_size:]
            self._stats.current_queue_size = len(self._queue)

            # 统计等待时间
            for req in batch:
                wait = time.perf_counter() - req.start_time
                self._wait_times.append(wait)

            return batch

    def get_stats(self) -> QueueStats:
        """获取调度统计"""
        stats = QueueStats(
            total_received=self._stats.total_received,
            total_processed=self._stats.total_processed,
            total_rejected=self._stats.total_rejected,
            total_timeout=self._stats.total_timeout,
            current_queue_size=self._stats.current_queue_size,
        )
        if self._wait_times:
            stats.avg_wait_time_ms = (
                sum(self._wait_times) / len(self._wait_times) * 1000
            )
        return stats

    def set_policy(self, policy: SchedulingPolicy) -> None:
        """动态调整调度策略"""
        self.config.policy = policy
