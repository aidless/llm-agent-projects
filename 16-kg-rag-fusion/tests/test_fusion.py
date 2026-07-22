"""Tests for Fusion module."""

import os
import pytest

from kg.store import KnowledgeGraphStore
from kg.query_engine import GraphQueryEngine
from kg.reasoning import Reasoner
from fusion.entity_linker import EntityLinker
from fusion.context_enricher import ContextEnricher
from fusion.result_fuser import ResultFuser
from fusion.answer_generator import AnswerGenerator
from rag.vector_store import VectorStore

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE_KG = os.path.join(DATA_DIR, "sample_kg.json")


@pytest.fixture
def sample_store():
    s = KnowledgeGraphStore()
    if os.path.exists(SAMPLE_KG):
        s.load_json(SAMPLE_KG)
    return s


@pytest.fixture
def vector_store():
    vs = VectorStore()
    vs.add_document(
        "doc_transformer",
        "Transformer is a neural network architecture based on self-attention. "
        "It was introduced in the paper Attention Is All You Need by Google researchers. "
        "Transformer has become the foundation for modern NLP models like GPT and BERT.",
        chunk_size=200,
        overlap=50,
    )
    vs.add_document(
        "doc_gpt",
        "GPT (Generative Pre-trained Transformer) is a series of language models by OpenAI. "
        "GPT models use autoregressive language modeling for text generation. "
        "GPT-4 introduced multimodal capabilities supporting both text and images.",
        chunk_size=200,
        overlap=50,
    )
    vs.add_document(
        "doc_kg",
        "Knowledge graphs represent information as entities and relations. "
        "They can enhance RAG systems by providing structured context. "
        "KG-RAG fusion combines graph reasoning with document retrieval.",
        chunk_size=200,
        overlap=50,
    )
    return vs


class TestEntityLinker:
    def test_link_entities(self, sample_store):
        if sample_store.entity_count() == 0:
            pytest.skip("No sample data")
        linker = EntityLinker(sample_store)
        results = linker.link_entities("Tell me about Transformer and GPT", top_k=5)
        assert len(results) > 0
        names = [r["name"] for r in results]
        # At least one should match
        assert any("Transformer" in n or "GPT" in n for n in names)

    def test_link_single_entity(self, sample_store):
        if sample_store.entity_count() == 0:
            pytest.skip("No sample data")
        linker = EntityLinker(sample_store)
        result = linker.link_single_entity("Transformer")
        assert result is not None
        assert "Transformer" in result["name"]

    def test_link_no_match(self, sample_store):
        linker = EntityLinker(sample_store)
        results = linker.link_entities("xyznonexistent12345", top_k=5)
        assert len(results) == 0


class TestContextEnricher:
    def test_enrich(self, sample_store):
        if sample_store.entity_count() == 0:
            pytest.skip("No sample data")
        enricher = ContextEnricher(sample_store)
        linked = [{"entity_id": "transformer", "name": "Transformer", "entity_type": "Technology", "link_score": 1.0}]
        result = enricher.enrich(linked, hop=2)
        assert "facts" in result
        assert "summary" in result
        assert len(result["facts"]) > 0
        assert "Transformer" in result["summary"]

    def test_enrich_empty(self, sample_store):
        enricher = ContextEnricher(sample_store)
        result = enricher.enrich([], hop=2)
        assert result["entities_context"] == []
        assert result["facts"] == []


class TestResultFuser:
    def test_fuse(self):
        fuser = ResultFuser(kg_weight=0.5, rag_weight=0.5)
        kg_results = [
            {"fact": "Transformer uses attention mechanism"},
            {"fact": "GPT is developed by OpenAI"},
        ]
        rag_results = [
            {"chunk_id": "c1", "doc_id": "d1", "text": "Transformer is a neural architecture", "score": 0.9},
            {"chunk_id": "c2", "doc_id": "d2", "text": "GPT generates text", "score": 0.8},
        ]
        fused = fuser.fuse(kg_results, rag_results, top_k=5)
        assert fused["total_results"] == 4
        assert fused["kg_result_count"] == 2
        assert fused["rag_result_count"] == 2
        assert len(fused["results"]) == 4

    def test_fuse_empty(self):
        fuser = ResultFuser()
        fused = fuser.fuse([], [], top_k=5)
        assert fused["total_results"] == 0

    def test_build_context(self):
        fuser = ResultFuser()
        fused_results = {
            "results": [
                {"type": "kg", "data": {"fact": "GPT uses Transformer"}},
                {"type": "rag", "data": {"text": "Transformer is widely used in NLP"}},
            ]
        }
        context = fuser.build_context(fused_results)
        assert "[知识图谱]" in context
        assert "[文档]" in context


class TestAnswerGenerator:
    def test_generate_with_context(self):
        gen = AnswerGenerator()
        context = "[知识图谱] GPT uses Transformer\n[文档] Transformer is a neural architecture"
        linked = [{"name": "GPT"}]
        result = gen.generate("What is GPT?", context, linked)
        assert result["answer"] != ""
        assert result["confidence"] > 0
        assert "GPT" in result["answer"]

    def test_generate_empty_context(self):
        gen = AnswerGenerator()
        result = gen.generate("What is X?", "", None)
        assert "抱歉" in result["answer"]
        assert result["confidence"] == 0.0

    def test_generate_no_entities(self):
        gen = AnswerGenerator()
        context = "[文档] Some information about AI."
        result = gen.generate("Tell me about AI", context, None)
        assert result["answer"] != ""
        assert result["confidence"] > 0


class TestFusionIntegration:
    def test_full_fusion_pipeline(self, sample_store, vector_store):
        if sample_store.entity_count() == 0:
            pytest.skip("No sample data")

        # Step 1: Entity linking
        linker = EntityLinker(sample_store)
        linked = linker.link_entities("What is Transformer and how does GPT use it?", top_k=3)
        assert len(linked) > 0

        # Step 2: Context enrichment
        enricher = ContextEnricher(sample_store)
        kg_context = enricher.enrich(linked, hop=2)
        kg_results = [{"fact": f} for f in kg_context["facts"]]

        # Step 3: RAG retrieval
        from rag.retriever import HybridRetriever
        retriever = HybridRetriever(vector_store)
        rag_results = retriever.retrieve("Transformer GPT architecture", top_k=3)

        # Step 4: Fusion
        fuser = ResultFuser()
        fused = fuser.fuse(kg_results, rag_results, top_k=5)
        context = fuser.build_context(fused)

        # Step 5: Answer generation
        generator = AnswerGenerator()
        answer = generator.generate("What is Transformer?", context, linked)

        assert answer["answer"] != ""
        assert answer["confidence"] > 0
        assert answer["kg_facts_count"] > 0 or answer["doc_chunks_count"] > 0
