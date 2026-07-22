"""PagedAttention 分页 KV Cache 模拟"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Page:
    """缓存页"""
    page_id: int
    num_tokens: int
    block_size: int
    # page_id -> 逻辑页号
    ref_count: int = 0

    @property
    def is_full(self) -> bool:
        return self.num_tokens >= self.block_size


@dataclass
class PageTableEntry:
    """页表条目"""
    request_id: str
    logical_page_idx: int
    physical_page_id: Optional[int] = None
    num_valid_tokens: int = 0


class PageTable:
    """页表 - 管理逻辑到物理页的映射"""

    def __init__(self):
        self._entries: Dict[Tuple[str, int], PageTableEntry] = {}
        self._request_pages: Dict[str, List[int]] = {}  # request_id -> [logical_idx]

    def add_page(self, request_id: str, logical_idx: int) -> PageTableEntry:
        """添加页表条目"""
        entry = PageTableEntry(
            request_id=request_id,
            logical_page_idx=logical_idx,
        )
        self._entries[(request_id, logical_idx)] = entry
        if request_id not in self._request_pages:
            self._request_pages[request_id] = []
        self._request_pages[request_id].append(logical_idx)
        return entry

    def map_physical(self, request_id: str, logical_idx: int, physical_page_id: int) -> None:
        """映射逻辑页到物理页"""
        key = (request_id, logical_idx)
        if key in self._entries:
            self._entries[key].physical_page_id = physical_page_id

    def get_entry(self, request_id: str, logical_idx: int) -> Optional[PageTableEntry]:
        """获取页表条目"""
        return self._entries.get((request_id, logical_idx))

    def get_all_entries(self, request_id: str) -> List[PageTableEntry]:
        """获取请求的所有页表条目"""
        logical_indices = self._request_pages.get(request_id, [])
        return [
            self._entries[(request_id, idx)]
            for idx in logical_indices
            if (request_id, idx) in self._entries
        ]

    def remove_request(self, request_id: str) -> List[PageTableEntry]:
        """移除请求的所有页表条目"""
        entries = self.get_all_entries(request_id)
        for idx in self._request_pages.get(request_id, []):
            self._entries.pop((request_id, idx), None)
        self._request_pages.pop(request_id, None)
        return entries

    @property
    def num_entries(self) -> int:
        return len(self._entries)

    @property
    def num_requests(self) -> int:
        return len(self._request_pages)


class PagedKVCache:
    """PagedAttention 分页 KV Cache"""

    def __init__(
        self,
        block_size: int = 16,  # 每页 16 个 token
        num_blocks: int = 1024,  # 总共 1024 个物理页
        num_layers: int = 32,
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_layers = num_layers

        # 物理页池 (每层独立)
        self._free_blocks: List[Set[int]] = [
            set(range(num_blocks)) for _ in range(num_layers)
        ]
        # 被占用的物理页
        self._used_blocks: List[Dict[int, str]] = [
            {} for _ in range(num_layers)  # physical_id -> request_id
        ]

        # 页表 (每层一个)
        self._page_tables: List[PageTable] = [PageTable() for _ in range(num_layers)]

        # 统计
        self.total_page_allocations = 0
        self.total_page_evictions = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def allocate(
        self, request_id: str, num_tokens: int
    ) -> Dict[str, any]:
        """为请求分配分页 KV Cache"""
        num_pages_needed = (num_tokens + self.block_size - 1) // self.block_size

        allocated_pages = 0
        failed_layers = []

        for layer_idx in range(self.num_layers):
            free = self._free_blocks[layer_idx]
            needed = num_pages_needed

            if len(free) < needed:
                failed_layers.append(layer_idx)
                needed = len(free)

            # 分配物理页
            for i in range(needed):
                physical_id = min(free)  # 取最小可用 id
                free.remove(physical_id)
                self._used_blocks[layer_idx][physical_id] = request_id

                # 更新页表
                entry = self._page_tables[layer_idx].add_page(request_id, i)
                self._page_tables[layer_idx].map_physical(request_id, i, physical_id)
                entry.num_valid_tokens = min(self.block_size, num_tokens - i * self.block_size)

            if len(free) >= needed:
                allocated_pages = needed

        self.total_page_allocations += allocated_pages * self.num_layers

        return {
            "request_id": request_id,
            "pages_per_layer": allocated_pages,
            "tokens_per_layer": min(allocated_pages * self.block_size, num_tokens),
            "failed_layers": failed_layers,
        }

    def append_tokens(
        self, request_id: str, num_new_tokens: int
    ) -> Dict[str, any]:
        """追加 token 到已有缓存"""
        # 找到最后一个逻辑页，看是否有空间
        results = {"allocated_new_pages": 0, "appended_tokens": 0}

        for layer_idx in range(self.num_layers):
            entries = self._page_tables[layer_idx].get_all_entries(request_id)
            if not entries:
                results["allocated_new_pages"] += 1
                continue

            last_entry = entries[-1]
            remaining = self.block_size - last_entry.num_valid_tokens
            can_append = min(remaining, num_new_tokens)
            last_entry.num_valid_tokens += can_append

            still_need = num_new_tokens - can_append
            new_pages = (still_need + self.block_size - 1) // self.block_size

            for j in range(new_pages):
                free = self._free_blocks[layer_idx]
                if free:
                    physical_id = min(free)
                    free.remove(physical_id)
                    self._used_blocks[layer_idx][physical_id] = request_id
                    logical_idx = len(entries) + j
                    entry = self._page_tables[layer_idx].add_page(request_id, logical_idx)
                    self._page_tables[layer_idx].map_physical(request_id, logical_idx, physical_id)
                    entry.num_valid_tokens = min(self.block_size, still_need - j * self.block_size)
                    results["allocated_new_pages"] += 1

        results["appended_tokens"] = num_new_tokens
        return results

    def free(self, request_id: str) -> Dict[str, int]:
        """释放请求占用的所有分页"""
        freed_pages = 0

        for layer_idx in range(self.num_layers):
            # 回收物理页
            to_free = [
                pid for pid, rid in self._used_blocks[layer_idx].items()
                if rid == request_id
            ]
            for pid in to_free:
                del self._used_blocks[layer_idx][pid]
                self._free_blocks[layer_idx].add(pid)
                freed_pages += 1

            # 移除页表条目
            self._page_tables[layer_idx].remove_request(request_id)

        return {"freed_pages_per_layer": freed_pages // self.num_layers if self.num_layers else 0}

    def get_usage(self) -> Dict[str, any]:
        """获取使用情况"""
        layer_usages = []
        for layer_idx in range(self.num_layers):
            used = len(self._used_blocks[layer_idx])
            layer_usages.append({
                "layer": layer_idx,
                "used_blocks": used,
                "free_blocks": len(self._free_blocks[layer_idx]),
                "utilization": used / self.num_blocks if self.num_blocks > 0 else 0,
            })

        total_used = sum(len(u) for u in self._used_blocks)
        total_available = self.num_blocks * self.num_layers

        return {
            "block_size": self.block_size,
            "total_blocks": self.num_blocks,
            "total_layers": self.num_layers,
            "used_blocks": total_used,
            "available_blocks": total_available - total_used,
            "overall_utilization": total_used / total_available if total_available > 0 else 0,
            "total_page_allocations": self.total_page_allocations,
            "total_page_evictions": self.total_page_evictions,
            "num_active_requests": self._page_tables[0].num_requests if self._page_tables else 0,
            "layer_details": layer_usages[:5],  # 前 5 层详情
        }
