"""Cache 模块测试"""

import pytest

from cache.kv_cache import KVCache, KVCacheManager, CacheEventType, CacheStats
from cache.paged_cache import PagedKVCache, PageTable, PageTableEntry
from cache.memory_manager import (
    MemoryManager, MemoryBlock, MemoryPool, MemoryPoolStats,
)


class TestKVCache:
    """测试 KV Cache"""

    def test_create(self):
        cache = KVCache(layer_id=0, max_capacity_bytes=1024)
        assert cache.layer_id == 0
        assert cache.usage == 0

    def test_allocate(self):
        cache = KVCache(layer_id=0, max_capacity_bytes=102400)
        tokens = cache.allocate("req_1", 10)
        assert tokens > 0
        assert cache.usage > 0

    def test_allocate_overflow(self):
        cache = KVCache(layer_id=0, max_capacity_bytes=100)  # 很小的容量
        tokens = cache.allocate("req_1", 10000)
        # 可能分配部分或全部
        assert isinstance(tokens, int)

    def test_free(self):
        cache = KVCache(layer_id=0, max_capacity_bytes=102400)
        cache.allocate("req_1", 10)
        freed = cache.free("req_1")
        assert freed > 0
        assert cache.usage == 0

    def test_free_nonexistent(self):
        cache = KVCache(layer_id=0, max_capacity_bytes=102400)
        freed = cache.free("nonexistent")
        assert freed == 0

    def test_usage_ratio(self):
        cache = KVCache(layer_id=0, max_capacity_bytes=1024)
        ratio = cache.usage_ratio
        assert 0.0 <= ratio <= 1.0

    def test_access(self):
        cache = KVCache(layer_id=0, max_capacity_bytes=102400)
        cache.allocate("req_1", 5)
        assert cache.access("req_1") is True
        assert cache.access("nonexistent") is False


class TestKVCacheManager:
    """测试 KV Cache 管理器"""

    def test_create(self):
        mgr = KVCacheManager(num_layers=8, total_memory_bytes=1024 * 1024)
        assert mgr.num_layers == 8

    def test_allocate(self):
        mgr = KVCacheManager(num_layers=4, total_memory_bytes=10 * 1024 * 1024)
        result = mgr.allocate("req_1", 10)
        assert result["allocated_tokens"] > 0
        assert result["allocated_layers"] > 0

    def test_free(self):
        mgr = KVCacheManager(num_layers=4, total_memory_bytes=10 * 1024 * 1024)
        mgr.allocate("req_1", 10)
        result = mgr.free("req_1")
        assert result["freed_layers"] > 0

    def test_stats(self):
        mgr = KVCacheManager(num_layers=4, total_memory_bytes=10 * 1024 * 1024)
        mgr.allocate("req_1", 10)
        mgr.allocate("req_2", 5)
        stats = mgr.get_stats()
        assert isinstance(stats, CacheStats)
        assert stats.total_allocations >= 0

    def test_per_layer_usage(self):
        mgr = KVCacheManager(num_layers=4, total_memory_bytes=10 * 1024 * 1024)
        usage = mgr.get_per_layer_usage()
        assert len(usage) == 4
        for u in usage:
            assert "layer_id" in u
            assert "usage_bytes" in u


class TestPagedKVCache:
    """测试 PagedAttention 分页缓存"""

    def test_create(self):
        cache = PagedKVCache(block_size=16, num_blocks=100, num_layers=4)
        assert cache.block_size == 16

    def test_allocate(self):
        cache = PagedKVCache(block_size=16, num_blocks=100, num_layers=4)
        result = cache.allocate("req_1", 32)
        assert result["pages_per_layer"] > 0
        assert result["tokens_per_layer"] > 0

    def test_append_tokens(self):
        cache = PagedKVCache(block_size=16, num_blocks=100, num_layers=4)
        cache.allocate("req_1", 16)
        result = cache.append_tokens("req_1", 10)
        assert result["appended_tokens"] == 10

    def test_free(self):
        cache = PagedKVCache(block_size=16, num_blocks=100, num_layers=4)
        cache.allocate("req_1", 32)
        result = cache.free("req_1")
        assert result["freed_pages_per_layer"] > 0

    def test_usage(self):
        cache = PagedKVCache(block_size=16, num_blocks=100, num_layers=4)
        cache.allocate("req_1", 32)
        usage = cache.get_usage()
        assert usage["used_blocks"] > 0
        assert usage["overall_utilization"] > 0
        assert usage["block_size"] == 16

    def test_multiple_requests(self):
        cache = PagedKVCache(block_size=16, num_blocks=100, num_layers=4)
        cache.allocate("req_1", 32)
        cache.allocate("req_2", 48)
        usage = cache.get_usage()
        assert usage["num_active_requests"] == 2


class TestPageTable:
    """测试页表"""

    def test_add_and_get(self):
        pt = PageTable()
        entry = pt.add_page("req_1", 0)
        assert pt.num_entries == 1

        found = pt.get_entry("req_1", 0)
        assert found is not None
        assert found.logical_page_idx == 0

    def test_map_physical(self):
        pt = PageTable()
        pt.add_page("req_1", 0)
        pt.map_physical("req_1", 0, 42)
        entry = pt.get_entry("req_1", 0)
        assert entry.physical_page_id == 42

    def test_remove_request(self):
        pt = PageTable()
        pt.add_page("req_1", 0)
        pt.add_page("req_1", 1)
        entries = pt.remove_request("req_1")
        assert len(entries) == 2
        assert pt.num_entries == 0


class TestMemoryManager:
    """测试内存管理器"""

    def test_create(self):
        mgr = MemoryManager(total_memory_bytes=1024 * 1024 * 1024)
        overview = mgr.get_overview()
        assert overview["total_memory_bytes"] == 1024 * 1024 * 1024

    def test_allocate(self):
        mgr = MemoryManager(
            total_memory_bytes=1024 * 1024 * 1024,
            block_size_bytes=1024,
        )
        blocks = mgr.allocate(MemoryPool.KV_CACHE, "req_1", 2048)
        assert blocks is not None
        assert len(blocks) > 0

    def test_allocate_insufficient(self):
        mgr = MemoryManager(
            total_memory_bytes=1024,
            block_size_bytes=1024,
        )
        blocks = mgr.allocate(MemoryPool.KV_CACHE, "req_1", 100000)
        assert blocks is None

    def test_free(self):
        mgr = MemoryManager(
            total_memory_bytes=1024 * 1024 * 1024,
            block_size_bytes=1024,
        )
        mgr.allocate(MemoryPool.WEIGHTS, "model", 2048)
        freed = mgr.free(MemoryPool.WEIGHTS, "model")
        assert freed > 0

    def test_free_all(self):
        mgr = MemoryManager(
            total_memory_bytes=1024 * 1024 * 1024,
            block_size_bytes=1024,
        )
        mgr.allocate(MemoryPool.KV_CACHE, "req_1", 1024)
        mgr.allocate(MemoryPool.ACTIVATIONS, "req_1", 1024)
        result = mgr.free_all("req_1")
        assert len(result) > 0

    def test_pool_stats(self):
        mgr = MemoryManager(total_memory_bytes=1024 * 1024 * 1024)
        stats = mgr.get_pool_stats(MemoryPool.KV_CACHE)
        assert "total_bytes" in stats
        assert "utilization" in stats

    def test_overview(self):
        mgr = MemoryManager(total_memory_bytes=1024 * 1024 * 1024)
        overview = mgr.get_overview()
        assert "pools" in overview
        assert "overall_utilization" in overview
