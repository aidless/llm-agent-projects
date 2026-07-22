"""RAG 检索测试"""

import os
import pytest
from rag.knowledge_base import KnowledgeBase
from rag.retriever import Retriever


@pytest.fixture
def knowledge_base():
    data_path = os.path.join(os.path.dirname(__file__), "..", "rag", "data", "faq_data.json")
    return KnowledgeBase(data_path=data_path)


@pytest.fixture
def retriever(knowledge_base):
    return Retriever(knowledge_base=knowledge_base)


class TestKnowledgeBase:
    """知识库测试"""

    def test_load_faq_data(self, knowledge_base):
        """测试加载 FAQ 数据"""
        assert knowledge_base.count() >= 50

    def test_search_by_keyword(self, knowledge_base):
        """测试关键词搜索"""
        results = knowledge_base.search_by_keyword("退货")
        assert len(results) > 0
        # 第一条应该是退货相关的
        assert results[0][1] > 0

    def test_search_by_keyword_english(self, knowledge_base):
        """测试英文关键词搜索"""
        results = knowledge_base.search_by_keyword("refund")
        assert len(results) > 0

    def test_add_item(self, knowledge_base):
        """测试添加条目"""
        count_before = knowledge_base.count()
        item = knowledge_base.add_item(
            question="测试问题",
            answer="测试答案",
            category="测试",
            keywords=["测试"],
        )
        assert item.id is not None
        assert knowledge_base.count() == count_before + 1

    def test_remove_item(self, knowledge_base):
        """测试删除条目"""
        item = knowledge_base.add_item(
            question="临时测试",
            answer="临时答案",
            category="临时",
        )
        success = knowledge_base.remove_item(item.id)
        assert success is True
        assert knowledge_base.get_item(item.id) is None

    def test_get_by_category(self, knowledge_base):
        """测试按分类获取"""
        items = knowledge_base.get_by_category("物流")
        assert len(items) > 0

    def test_get_stats(self, knowledge_base):
        """测试统计信息"""
        stats = knowledge_base.get_stats()
        assert stats.total_count >= 50
        assert len(stats.categories) > 0


class TestRetriever:
    """检索器测试"""

    def test_retrieve_return(self, retriever):
        """测试退货检索"""
        results = retriever.retrieve("如何退货", top_k=3)
        assert len(results) > 0
        assert results[0].score > 0

    def test_retrieve_refund(self, retriever):
        """测试退款检索"""
        results = retriever.retrieve("退款", top_k=3)
        assert len(results) > 0

    def test_retrieve_with_category(self, retriever):
        """测试按分类检索"""
        results = retriever.retrieve("运费", top_k=3, category="物流")
        assert len(results) > 0

    def test_retrieve_result_fields(self, retriever):
        """测试检索结果字段完整"""
        results = retriever.retrieve("订单", top_k=1)
        if results:
            r = results[0]
            assert r.id is not None
            assert r.question is not None
            assert r.answer is not None
            assert r.category is not None
            assert r.score > 0
