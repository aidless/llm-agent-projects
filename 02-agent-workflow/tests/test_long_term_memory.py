# ============================================
# 长期记忆单元测试
# ============================================

import pytest
import os
import tempfile
import shutil
from memory.long_term import LongTermMemory


@pytest.fixture
def temp_chroma_dir():
    """创建临时 ChromaDB 目录"""
    tmp_dir = tempfile.mkdtemp(prefix="test_chroma_")
    yield tmp_dir
    # 清理
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass


class TestLongTermMemory:
    """长期记忆测试类"""

    def test_init(self, temp_chroma_dir):
        """测试初始化"""
        mem = LongTermMemory(
            persist_dir=temp_chroma_dir,
            collection_name="test_collection",
        )
        assert mem.collection_name == "test_collection"
        assert mem.similarity_top_k == 5

    def test_add_and_count(self, temp_chroma_dir):
        """测试添加记忆和计数"""
        mem = LongTermMemory(
            persist_dir=temp_chroma_dir,
            collection_name="test_add",
        )

        # 初始应为空（新集合）
        mem._ensure_collection()

        mem_id = mem.add_memory("这是一条测试记忆")
        assert mem_id is not None
        assert isinstance(mem_id, str)

        # 等待索引更新
        count = mem.count()
        assert count >= 1

    def test_add_with_metadata(self, temp_chroma_dir):
        """测试带元数据的记忆添加"""
        mem = LongTermMemory(
            persist_dir=temp_chroma_dir,
            collection_name="test_meta",
        )

        mem_id = mem.add_memory(
            content="重要信息",
            metadata={"source": "test", "importance": "high"},
        )

        assert mem_id is not None

    def test_search(self, temp_chroma_dir):
        """测试语义搜索"""
        mem = LongTermMemory(
            persist_dir=temp_chroma_dir,
            collection_name="test_search",
        )

        # 添加几条记忆
        mem.add_memory("Python 是一种流行的编程语言", metadata={"topic": "programming"})
        mem.add_memory("机器学习是人工智能的一个分支", metadata={"topic": "ai"})
        mem.add_memory("FastAPI 是一个现代的 Python Web 框架", metadata={"topic": "web"})

        # 搜索相关记忆
        results = mem.search("编程语言")
        assert len(results) > 0

        # 验证返回结构
        for r in results:
            assert "id" in r
            assert "content" in r
            assert "metadata" in r
            assert "distance" in r

    def test_search_empty_collection(self, temp_chroma_dir):
        """测试空集合搜索"""
        mem = LongTermMemory(
            persist_dir=temp_chroma_dir,
            collection_name="test_empty",
        )

        results = mem.search("任何查询")
        assert results == []

    def test_delete_memory(self, temp_chroma_dir):
        """测试删除记忆"""
        mem = LongTermMemory(
            persist_dir=temp_chroma_dir,
            collection_name="test_delete",
        )

        mem_id = mem.add_memory("即将被删除的记忆")
        initial_count = mem.count()

        # 删除
        success = mem.delete_memory(mem_id)
        assert success is True

        # 验证数量减少
        after_count = mem.count()
        assert after_count < initial_count

    def test_update_memory(self, temp_chroma_dir):
        """测试更新记忆"""
        mem = LongTermMemory(
            persist_dir=temp_chroma_dir,
            collection_name="test_update",
        )

        mem_id = mem.add_memory("原始内容")
        success = mem.update_memory(mem_id, "更新后的内容")
        assert success is True

    def test_get_all_memories(self, temp_chroma_dir):
        """测试获取所有记忆"""
        mem = LongTermMemory(
            persist_dir=temp_chroma_dir,
            collection_name="test_get_all",
        )

        for i in range(5):
            mem.add_memory(f"记忆 {i}")

        all_memories = mem.get_all_memories(limit=10)
        assert len(all_memories) == 5

    def test_clear(self, temp_chroma_dir):
        """测试清空记忆"""
        mem = LongTermMemory(
            persist_dir=temp_chroma_dir,
            collection_name="test_clear",
        )

        for i in range(5):
            mem.add_memory(f"记忆 {i}")

        assert mem.count() >= 5

        mem.clear()
        assert mem.count() == 0

    def test_repr(self, temp_chroma_dir):
        """测试字符串表示"""
        mem = LongTermMemory(
            persist_dir=temp_chroma_dir,
            collection_name="test_repr",
        )
        repr_str = repr(mem)
        assert "LongTermMemory" in repr_str
        assert "test_repr" in repr_str
