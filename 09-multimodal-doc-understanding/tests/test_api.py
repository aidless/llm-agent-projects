"""API 和 RAG 测试。"""

import pytest
from PIL import Image

from app.models import DocumentFormat, ParsedDocument, ParsedPage
from rag.document_indexer import DocumentChunk, DocumentIndexer, SimpleVectorStore
from rag.qa_chain import QAChain, MockLLMClient
from rag.retriever import HybridRetriever
from utils.image_utils import create_test_image


def create_test_document(
    text: str = "测试文档内容。这是第一段。\n\n这是第二段，包含更多信息。",
    filename: str = "test.txt",
) -> ParsedDocument:
    """创建测试文档。"""
    return ParsedDocument(
        filename=filename,
        format=DocumentFormat.TEXT,
        pages=[
            ParsedPage(page_num=1, text=text),
        ],
        full_text=text,
        metadata={"test": True},
    )


class TestSimpleVectorStore:
    """简单向量存储测试。"""

    def test_add_and_count(self):
        """测试添加和计数。"""
        store = SimpleVectorStore()
        chunks = [
            DocumentChunk(text="这是第一段内容。", document_id="doc1"),
            DocumentChunk(text="这是第二段内容。", document_id="doc1"),
        ]
        store.add(chunks)
        assert store.count() == 2

    def test_search(self):
        """测试搜索。"""
        store = SimpleVectorStore()
        chunks = [
            DocumentChunk(text="Python 是一种编程语言。", document_id="doc1"),
            DocumentChunk(text="Java 也是一种编程语言。", document_id="doc1"),
            DocumentChunk(text="今天天气不错。", document_id="doc2"),
        ]
        store.add(chunks)

        results = store.search("编程语言", top_k=2)
        assert len(results) > 0
        # 第一个结果应该是关于编程语言的
        idx, score = results[0]
        assert "编程" in store.chunks[idx].text or "Python" in store.chunks[idx].text

    def test_search_empty(self):
        """测试空存储搜索。"""
        store = SimpleVectorStore()
        results = store.search("测试", top_k=5)
        assert results == []

    def test_delete_by_document(self):
        """测试按文档删除。"""
        store = SimpleVectorStore()
        chunks = [
            DocumentChunk(text="内容1", document_id="doc1"),
            DocumentChunk(text="内容2", document_id="doc2"),
            DocumentChunk(text="内容3", document_id="doc1"),
        ]
        store.add(chunks)
        assert store.count() == 3

        store.delete_by_document("doc1")
        assert store.count() == 1
        assert store.chunks[0].document_id == "doc2"


class TestDocumentIndexer:
    """文档索引器测试。"""

    def test_index_document(self):
        """测试文档索引。"""
        indexer = DocumentIndexer(chunk_size=100)
        doc = create_test_document(
            text="机器学习是人工智能的一个分支。"
                 "深度学习是机器学习的一个子集。"
                 "神经网络是深度学习的基础。",
            filename="ai.txt",
        )
        result = indexer.index_document(doc)
        assert result.status == "success"
        assert result.chunks_count > 0
        assert len(result.document_id) > 0

    def test_index_empty_document(self):
        """测试索引空文档。"""
        indexer = DocumentIndexer()
        doc = create_test_document(text="")
        result = indexer.index_document(doc)
        assert result.chunks_count == 0

    def test_delete_document(self):
        """测试删除文档索引。"""
        indexer = DocumentIndexer()
        doc = create_test_document(text="可删除的内容。")
        result = indexer.index_document(doc)
        doc_id = result.document_id

        success = indexer.delete_document(doc_id)
        assert success is True
        assert doc_id not in indexer.get_indexed_documents()

    def test_delete_nonexistent_document(self):
        """测试删除不存在的文档。"""
        indexer = DocumentIndexer()
        success = indexer.delete_document("nonexistent")
        assert success is False


class TestHybridRetriever:
    """混合检索器测试。"""

    def test_retrieve(self):
        """测试混合检索。"""
        store = SimpleVectorStore()
        chunks = [
            DocumentChunk(
                text="Python 是一种广泛使用的编程语言。",
                document_id="doc1",
                metadata={"filename": "python.txt"},
            ),
            DocumentChunk(
                text="机器学习使用 Python 进行数据分析。",
                document_id="doc2",
                metadata={"filename": "ml.txt"},
            ),
        ]
        store.add(chunks)

        retriever = HybridRetriever(store)
        results = retriever.retrieve("Python 编程", top_k=2)
        assert len(results) > 0
        chunk, score = results[0]
        assert isinstance(chunk, DocumentChunk)
        assert score > 0

    def test_retrieve_with_document_filter(self):
        """测试限定文档范围检索。"""
        store = SimpleVectorStore()
        chunks = [
            DocumentChunk(text="文档1的内容", document_id="doc1"),
            DocumentChunk(text="文档2的内容", document_id="doc2"),
        ]
        store.add(chunks)

        retriever = HybridRetriever(store)
        results = retriever.retrieve("内容", top_k=5, document_ids=["doc1"])
        for chunk, _ in results:
            assert chunk.document_id == "doc1"


class TestQAChain:
    """问答链测试。"""

    def test_ask_with_content(self):
        """测试有内容时的问答。"""
        store = SimpleVectorStore()
        indexer = DocumentIndexer(vector_store=store, chunk_size=100)

        doc = create_test_document(
            text="Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。"
                 "它广泛应用于 Web 开发、数据科学和人工智能领域。",
            filename="python_intro.txt",
        )
        indexer.index_document(doc)

        qa = QAChain(indexer=indexer)
        result = qa.ask("Python 是什么？", top_k=3)

        assert result.question == "Python 是什么？"
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_ask_no_content(self):
        """测试无内容时的问答。"""
        store = SimpleVectorStore()
        indexer = DocumentIndexer(vector_store=store)
        qa = QAChain(indexer=indexer)

        result = qa.ask("测试问题")
        assert "未找到" in result.answer
        assert result.score == 0.0

    def test_compare_documents(self):
        """测试多文档对比问答。"""
        store = SimpleVectorStore()
        indexer = DocumentIndexer(vector_store=store, chunk_size=200)

        doc1 = create_test_document(
            text="Python 是一种编程语言，以简洁著称。",
            filename="python.txt",
        )
        doc2 = create_test_document(
            text="Java 是一种编程语言，以跨平台著称。",
            filename="java.txt",
        )
        r1 = indexer.index_document(doc1)
        r2 = indexer.index_document(doc2)

        qa = QAChain(indexer=indexer)
        result = qa.compare_documents(
            "哪种语言更好？",
            document_ids=[r1.document_id, r2.document_id],
        )

        assert result["question"] == "哪种语言更好？"
        assert len(result["documents"]) == 2