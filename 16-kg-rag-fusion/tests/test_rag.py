"""Tests for RAG module."""

import pytest

from rag.vector_store import VectorStore
from rag.indexer import DocumentIndexer
from rag.retriever import HybridRetriever


@pytest.fixture
def vector_store():
    vs = VectorStore()
    vs.add_document(
        "doc1",
        "Transformer is a deep learning model. It uses self-attention to process data.",
        chunk_size=200,
        overlap=50,
    )
    vs.add_document(
        "doc2",
        "GPT is a language model by OpenAI. GPT uses Transformer for text generation.",
        chunk_size=200,
        overlap=50,
    )
    return vs


class TestVectorStore:
    def test_add_document(self, vector_store):
        assert vector_store.get_document_count() == 2
        assert vector_store.get_chunk_count() > 0

    def test_add_document_chunking(self, vector_store):
        chunks = vector_store.chunks.get("doc1", [])
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) > 0

    def test_search(self, vector_store):
        results = vector_store.search("Transformer architecture", top_k=3)
        assert len(results) > 0
        assert results[0]["score"] > 0

    def test_keyword_search(self, vector_store):
        results = vector_store.keyword_search("OpenAI GPT", top_k=3)
        assert len(results) > 0
        assert results[0]["score"] > 0

    def test_remove_document(self, vector_store):
        initial_count = vector_store.get_document_count()
        assert vector_store.remove_document("doc1") is True
        assert vector_store.get_document_count() == initial_count - 1

    def test_remove_document_not_found(self, vector_store):
        assert vector_store.remove_document("nonexistent") is False

    def test_get_all_documents(self, vector_store):
        docs = vector_store.get_all_documents()
        assert len(docs) == 2

    def test_empty_store_search(self):
        vs = VectorStore()
        results = vs.search("test query", top_k=5)
        assert results == []


class TestDocumentIndexer:
    def test_index_document(self):
        indexer = DocumentIndexer()
        result = indexer.index_document("new_doc", "This is a test document about machine learning.")
        assert result["status"] == "indexed"
        assert result["chunks_created"] > 0

    def test_list_documents(self, vector_store):
        indexer = DocumentIndexer(vector_store)
        docs = indexer.list_documents()
        assert len(docs) == 2

    def test_get_stats(self, vector_store):
        indexer = DocumentIndexer(vector_store)
        stats = indexer.get_stats()
        assert stats["document_count"] == 2
        assert stats["chunk_count"] > 0


class TestHybridRetriever:
    def test_retrieve(self, vector_store):
        retriever = HybridRetriever(vector_store)
        results = retriever.retrieve("Transformer model", top_k=3)
        assert len(results) > 0
        assert "fused_score" in results[0]

    def test_retrieve_semantic(self, vector_store):
        retriever = HybridRetriever(vector_store)
        results = retriever.retrieve_semantic("GPT language model", top_k=3)
        assert len(results) > 0

    def test_retrieve_keyword(self, vector_store):
        retriever = HybridRetriever(vector_store)
        results = retriever.retrieve_keyword("OpenAI", top_k=3)
        assert len(results) > 0

    def test_rrf_fusion(self):
        vs = VectorStore()
        vs.add_document("test", "Machine learning is a subset of artificial intelligence.")
        retriever = HybridRetriever(vs)
        results = retriever.retrieve("machine learning", top_k=5)
        for r in results:
            assert r["source"] == "hybrid"
