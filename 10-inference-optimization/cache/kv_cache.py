"""KV Cache 管理 - 模拟 KV Cache 的分配与回收"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class CacheEventType(Enum):
    ALLOCATE = "allocate"
    ACCESS = "access"
    MISS = "miss"
    EVICT = "evict"
    FREE = "free"


@dataclass
class CacheEvent:
    """缓存事件"""
    event_type: CacheEventType
    request_id: str
    layer: int
    num_tokens: int
    timestamp: float
    size_bytes: int


@dataclass
class CacheStats:
    """缓存统计"""
    total_allocations: int = 0
    total_evictions: int = 0
    total_hits: int = 0
    total_misses: int = 0
    total_frees: int = 0
    current_usage_bytes: int = 0
    peak_usage_bytes: int = 0
    hit_rate: float = 0.0

    @property
    def total_accesses(self) -> int:
        return self.total_hits + self.total_misses


class KVCache:
    """单层 KV Cache"""

    def __init__(
        self,
        layer_id: int,
        max_capacity_bytes: int,
        num_heads: int = 32,
        head_dim: int = 128,
        dtype_bytes: int = 2,  # FP16
    ):
        self.layer_id = layer_id
        self.max_capacity_bytes = max_capacity_bytes
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype_bytes = dtype_bytes

        # 每个 token 的 KV 占用: 2 (K+V) * num_heads * head_dim * dtype_bytes
        self.bytes_per_token = 2 * num_heads * head_dim * dtype_bytes

        # 存储槽 (token_position -> (request_id, data))
        self._slots: Dict[Tuple[str, int], bytes] = {}
        self._size = 0
        self._events: List[CacheEvent] = []

    def allocate(self, request_id: str, num_tokens: int) -> int:
        """分配 KV Cache 空间，返回实际分配的 token 数"""
        needed = num_tokens * self.bytes_per_token
        available = self.max_capacity_bytes - self._size

        if available <= 0:
            self._log_event(CacheEventType.MISS, request_id, 0, 0)
            return 0

        actual_tokens = min(num_tokens, available // self.bytes_per_token)
        if actual_tokens <= 0:
            self._log_event(CacheEventType.MISS, request_id, 0, 0)
            return 0

        # 分配槽
        for i in range(actual_tokens):
            self._slots[(request_id, i)] = b'\x00' * self.bytes_per_token

        allocated_bytes = actual_tokens * self.bytes_per_token
        self._size += allocated_bytes

        self._log_event(CacheEventType.ALLOCATE, request_id, actual_tokens, allocated_bytes)
        self._log_event(CacheEventType.ACCESS, request_id, actual_tokens, allocated_bytes)

        return actual_tokens

    def access(self, request_id: str) -> bool:
        """访问缓存"""
        has_data = any(k[0] == request_id for k in self._slots)
        if has_data:
            self._log_event(CacheEventType.ACCESS, request_id, 0, 0)
            return True
        return False

    def free(self, request_id: str) -> int:
        """释放请求占用的缓存"""
        freed = 0
        keys_to_remove = [k for k in self._slots if k[0] == request_id]
        for k in keys_to_remove:
            del self._slots[k]
            freed += self.bytes_per_token

        self._size -= freed
        self._log_event(CacheEventType.FREE, request_id, len(keys_to_remove), freed)
        return freed

    @property
    def usage(self) -> int:
        return self._size

    @property
    def usage_ratio(self) -> float:
        if self.max_capacity_bytes == 0:
            return 0.0
        return self._size / self.max_capacity_bytes

    def _log_event(
        self, event_type: CacheEventType, request_id: str,
        num_tokens: int, size_bytes: int
    ) -> None:
        self._events.append(CacheEvent(
            event_type=event_type,
            request_id=request_id,
            layer=self.layer_id,
            num_tokens=num_tokens,
            timestamp=time.time(),
            size_bytes=size_bytes,
        ))


class KVCacheManager:
    """KV Cache 管理器 - 管理所有层的 KV Cache"""

    def __init__(
        self,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        total_memory_bytes: int = 4 * 1024 * 1024 * 1024,  # 4GB
        dtype_bytes: int = 2,
    ):
        self.num_layers = num_layers
        self.total_memory_bytes = total_memory_bytes

        # 均匀分配内存到各层
        per_layer = total_memory_bytes // num_layers

        self._caches: List[KVCache] = []
        for i in range(num_layers):
            self._caches.append(
                KVCache(
                    layer_id=i,
                    max_capacity_bytes=per_layer,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dtype_bytes=dtype_bytes,
                )
            )

        self._stats = CacheStats()

    def allocate(
        self, request_id: str, num_tokens: int, num_layers: Optional[int] = None
    ) -> Dict[str, int]:
        """在指定层分配 KV Cache"""
        num_layers = num_layers or self.num_layers
        result = {"allocated_tokens": 0, "allocated_layers": 0, "allocated_bytes": 0}

        for i in range(num_layers):
            allocated = self._caches[i].allocate(request_id, num_tokens)
            if allocated > 0:
                result["allocated_tokens"] = allocated
                result["allocated_layers"] += 1
                result["allocated_bytes"] += allocated * self._caches[i].bytes_per_token

        self._stats.total_allocations += 1
        self._update_peak()
        return result

    def free(self, request_id: str) -> Dict[str, int]:
        """释放请求在所有层的缓存"""
        result = {"freed_layers": 0, "freed_bytes": 0}

        for cache in self._caches:
            freed = cache.free(request_id)
            if freed > 0:
                result["freed_layers"] += 1
                result["freed_bytes"] += freed

        self._stats.total_frees += 1
        return result

    def get_stats(self) -> CacheStats:
        """获取缓存统计"""
        # 汇总各层统计
        total_hits = sum(
            1 for c in self._caches
            for e in c._events
            if e.event_type == CacheEventType.ACCESS
        )
        total_misses = sum(
            1 for c in self._caches
            for e in c._events
            if e.event_type == CacheEventType.MISS
        )

        total_allocs = sum(
            1 for c in self._caches
            for e in c._events
            if e.event_type == CacheEventType.ALLOCATE
        )

        total_evicts = sum(
            1 for c in self._caches
            for e in c._events
            if e.event_type == CacheEventType.EVICT
        )

        self._stats.total_hits = total_hits
        self._stats.total_misses = total_misses
        self._stats.total_allocations = total_allocs
        self._stats.total_evictions = total_evicts
        self._stats.current_usage_bytes = sum(c.usage for c in self._caches)
        self._stats.hit_rate = (
            total_hits / (total_hits + total_misses)
            if (total_hits + total_misses) > 0
            else 0.0
        )
        return self._stats

    def get_per_layer_usage(self) -> List[Dict]:
        """获取每层的缓存使用情况"""
        return [
            {
                "layer_id": c.layer_id,
                "usage_bytes": c.usage,
                "max_capacity_bytes": c.max_capacity_bytes,
                "usage_ratio": round(c.usage_ratio, 4),
                "num_slots": len(c._slots),
            }
            for c in self._caches
        ]

    def _update_peak(self) -> None:
        current = sum(c.usage for c in self._caches)
        self._stats.peak_usage_bytes = max(self._stats.peak_usage_bytes, current)
