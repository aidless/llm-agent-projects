"""Query API endpoints - KG-RAG fusion queries."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.models import QueryRequest, FusionQueryResponse
from fusion.entity_linker import EntityLinker
from fusion.context_enricher import ContextEnricher
from fusion.result_fuser import ResultFuser
from fusion.answer_generator import AnswerGenerator
from rag.retriever import HybridRetriever

router = APIRouter(prefix="/query", tags=["Query"])

_entity_linker: EntityLinker = None  # type: ignore
_context_enricher: ContextEnricher = None  # type: ignore
_result_fuser: ResultFuser = None  # type: ignore
_answer_generator: AnswerGenerator = None  # type: ignore
_hybrid_retriever: HybridRetriever = None  # type: ignore


def init_query_services(
    entity_linker: EntityLinker,
    context_enricher: ContextEnricher,
    result_fuser: ResultFuser,
    answer_generator: AnswerGenerator,
    hybrid_retriever: HybridRetriever,
) -> None:
    global _entity_linker, _context_enricher, _result_fuser, _answer_generator, _hybrid_retriever
    _entity_linker = entity_linker
    _context_enricher = context_enricher
    _result_fuser = result_fuser
    _answer_generator = answer_generator
    _hybrid_retriever = hybrid_retriever


@router.post("/fusion", response_model=Dict[str, Any])
def fusion_query(request: QueryRequest):
    """Execute a KG-RAG fusion query."""
    if any(svc is None for svc in [_entity_linker, _context_enricher, _result_fuser, _answer_generator, _hybrid_retriever]):
        raise HTTPException(500, "Query services not initialized")

    # Step 1: Entity linking
    linked_entities = _entity_linker.link_entities(request.query, top_k=request.top_k)

    # Step 2: Context enrichment from KG
    kg_context = _context_enricher.enrich(linked_entities, hop=2)
    kg_results = kg_context.get("facts", [])

    # Step 3: RAG retrieval
    rag_results = _hybrid_retriever.retrieve(request.query, top_k=request.top_k)

    # Step 4: Result fusion
    fuser = ResultFuser(kg_weight=request.kg_weight, rag_weight=request.rag_weight)
    fused = fuser.fuse(
        kg_results=[{"fact": f} for f in kg_results],
        rag_results=rag_results,
        top_k=request.top_k,
    )
    context = fuser.build_context(fused)

    # Step 5: Answer generation
    answer = _answer_generator.generate(request.query, context, linked_entities)

    return answer


@router.post("/semantic", response_model=Dict[str, Any])
def semantic_search(request: QueryRequest):
    """Pure semantic (TF-IDF) search."""
    if _hybrid_retriever is None:
        raise HTTPException(500, "Retriever not initialized")
    results = _hybrid_retriever.retrieve_semantic(request.query, top_k=request.top_k)
    return {"query": request.query, "results": results, "count": len(results)}


@router.post("/keyword", response_model=Dict[str, Any])
def keyword_search(request: QueryRequest):
    """Pure keyword search."""
    if _hybrid_retriever is None:
        raise HTTPException(500, "Retriever not initialized")
    results = _hybrid_retriever.retrieve_keyword(request.query, top_k=request.top_k)
    return {"query": request.query, "results": results, "count": len(results)}
