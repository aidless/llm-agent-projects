"""FastAPI application entry point."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kg.builder import KnowledgeGraphBuilder
from kg.extractor import Extractor
from kg.store import KnowledgeGraphStore
from rag.vector_store import VectorStore
from rag.indexer import DocumentIndexer
from rag.retriever import HybridRetriever
from fusion.entity_linker import EntityLinker
from fusion.context_enricher import ContextEnricher
from fusion.result_fuser import ResultFuser
from fusion.answer_generator import AnswerGenerator

from app.api import graph, document, query

# ── Create app ────────────────────────────────────────────────────

app = FastAPI(
    title="KG-RAG Fusion System",
    description="Knowledge Graph + RAG Fusion System for intelligent knowledge QA",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize services ─────────────────────────────────────────────

# Knowledge Graph
kg_store = KnowledgeGraphStore()
kg_builder = KnowledgeGraphBuilder(store=kg_store, extractor=Extractor(use_llm_mock=True))

# Load sample data if available
_sample_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_kg.json")
if os.path.exists(_sample_data_path):
    kg_builder.load_from_json(_sample_data_path)

# RAG
vector_store = VectorStore()
doc_indexer = DocumentIndexer(vector_store=vector_store)
hybrid_retriever = HybridRetriever(vector_store=vector_store)

# Fusion
entity_linker = EntityLinker(kg_store)
context_enricher = ContextEnricher(kg_store)
result_fuser = ResultFuser(kg_weight=0.5, rag_weight=0.5)
answer_generator = AnswerGenerator()

# ── Register routers ────────────────────────────────────────────────

app.include_router(graph.router)
app.include_router(document.router)
app.include_router(query.router)

# ── Initialize API services ─────────────────────────────────────────

graph.init_graph_services(kg_builder)
document.init_document_service(doc_indexer)
query.init_query_services(
    entity_linker=entity_linker,
    context_enricher=context_enricher,
    result_fuser=result_fuser,
    answer_generator=answer_generator,
    hybrid_retriever=hybrid_retriever,
)


# ── Health check ────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "entities": kg_store.entity_count(),
        "relations": kg_store.relation_count(),
        "documents": vector_store.get_document_count(),
    }


@app.get("/")
def root():
    return {
        "name": "KG-RAG Fusion System",
        "version": "1.0.0",
        "docs": "/docs",
    }
