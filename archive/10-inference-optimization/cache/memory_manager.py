"""内存管理器 - GPU 内存分配与回收"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class MemoryPool(Enum):
    KV_CACHE = "kv_cache"
    WEIGHTS = "weights"
    ACTIVATIONS = "activations"
    TEMP = "temp"


@dataclass
class MemoryBlock:
    """内存块"""
    block_id: int
    size_bytes: int
    pool: MemoryPool
    owner: str  # request_id 或 "model"
    allocated_at: float = 0.0

    def __lt__(self, other: MemoryBlock) -> bool:
        return self.allocated_at < other.allocated_at


@dataclass
class MemoryPoolStats:
    """内存池统计"""
    pool: str
    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    num_blocks: int = 0
    num_allocations: int = 0
    num_frees: int = 0
    utilization: float = 0.0
    fragmentation_ratio: float = 0.0


class MemoryManager:
    """GPU 内存管理器"""

    def __init__(
        self,
        total_memory_bytes: int = 24 * 1024 * 1024 * 1024,  # 24 GB (A100 80GB 的一部分)
        block_size_bytes: int = 2 * 1024 * 1024,  # 2MB blocks
    ):
        self.total_memory_bytes = total_memory_bytes
        self.block_size_bytes = block_size_bytes
        self.total_blocks = total_memory_bytes // block_size_bytes

        # 各内存池分配比例
        self._pool_ratios = {
            MemoryPool.WEIGHTS: 0.4,
            MemoryPool.KV_CACHE: 0.4,
            MemoryPool.ACTIVATIONS: 0.15,
            MemoryPool.TEMP: 0.05,
        }

        # 计算各池大小
        self._pool_sizes: Dict[MemoryPool, int] = {
            pool: int(self.total_memory_bytes * ratio)
            for pool, ratio in self._pool_ratios.items()
        }

        # 各池的空闲块 (用最大堆模拟)
        self._free_blocks: Dict[MemoryPool, List[int]] = {
            pool: list(range(self._pool_sizes[pool] // block_size_bytes))
            for pool in MemoryPool
        }

        # 各池的已分配块
        self._allocated_blocks: Dict[MemoryPool, Dict[str, List[MemoryBlock]]] = {
            pool: {} for pool in MemoryPool
        }

        self._block_counter = 0
        self._stats: Dict[MemoryPool, MemoryPoolStats] = {
            pool: MemoryPoolStats(pool=pool.value)
            for pool in MemoryPool
        }

    def allocate(
        self,
        pool: MemoryPool,
        owner: str,
        size_bytes: int,
    ) -> Optional[List[MemoryBlock]]:
        """分配内存块"""
        num_blocks_needed = (size_bytes + self.block_size_bytes - 1) // self.block_size_bytes
        free = self._free_blocks.get(pool, [])

        if len(free) < num_blocks_needed:
            return None

        blocks: List[MemoryBlock] = []
        for _ in range(num_blocks_needed):
            block_id = free.pop()
            block = MemoryBlock(
                block_id=self._block_counter,
                size_bytes=self.block_size_bytes,
                pool=pool,
                owner=owner,
                allocated_at=time.time(),
            )
            self._block_counter += 1
            blocks.append(block)

            if owner not in self._allocated_blocks[pool]:
                self._allocated_blocks[pool][owner] = []
            self._allocated_blocks[pool][owner].append(block)

        self._stats[pool].num_allocations += 1
        self._update_pool_stats(pool)
        return blocks

    def free(self, pool: MemoryPool, owner: str) -> int:
        """释放指定 owner 的所有块"""
        blocks = self._allocated_blocks[pool].pop(owner, [])
        freed_count = len(blocks)

        # 归还到空闲列表
        # 简化: 将物理 block_id 加回空闲列表
        for block in blocks:
            self._free_blocks[pool].append(block.block_id)

        self._stats[pool].num_frees += 1
        self._update_pool_stats(pool)
        return freed_count

    def free_all(self, owner: str) -> Dict[str, int]:
        """释放指定 owner 在所有池的内存"""
        result = {}
        for pool in MemoryPool:
            freed = self.free(pool, owner)
            if freed > 0:
                result[pool.value] = freed
        return result

    def get_pool_stats(self, pool: Optional[MemoryPool] = None) -> Dict:
        """获取内存统计"""
        if pool:
            return self._pool_stats_to_dict(pool)
        return {p.value: self._pool_stats_to_dict(p) for p in MemoryPool}

    def _pool_stats_to_dict(self, pool: MemoryPool) -> Dict:
        stats = self._stats[pool]
        total_blocks = self._pool_sizes[pool] // self.block_size_bytes
        used_blocks = total_blocks - len(self._free_blocks.get(pool, []))
        stats.total_bytes = self._pool_sizes[pool]
        stats.used_bytes = used_blocks * self.block_size_bytes
        stats.free_bytes = len(self._free_blocks.get(pool, [])) * self.block_size_bytes
        stats.num_blocks = used_blocks
        stats.utilization = stats.used_bytes / stats.total_bytes if stats.total_bytes > 0 else 0
        return {
            "total_bytes": stats.total_bytes,
            "used_bytes": stats.used_bytes,
            "free_bytes": stats.free_bytes,
            "num_blocks": stats.num_blocks,
            "utilization": round(stats.utilization, 4),
            "num_allocations": stats.num_allocations,
            "num_frees": stats.num_frees,
        }

    def _update_pool_stats(self, pool: MemoryPool) -> None:
        self._pool_stats_to_dict(pool)

    def get_overview(self) -> Dict:
        """获取整体内存概览"""
        total_used = sum(
            self._stats[p].used_bytes for p in MemoryPool
        )
        return {
            "total_memory_bytes": self.total_memory_bytes,
            "total_used_bytes": total_used,
            "total_free_bytes": self.total_memory_bytes - total_used,
            "overall_utilization": round(
                total_used / self.total_memory_bytes, 4
            ) if self.total_memory_bytes > 0 else 0,
            "block_size_bytes": self.block_size_bytes,
            "pools": self.get_pool_stats(),
        }
