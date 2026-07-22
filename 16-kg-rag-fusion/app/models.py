"""Pydantic models for request/response schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Entity models ──────────────────────────────────────────────────

class EntityCreate(BaseModel):
    entity_id: str
    name: str
    entity_type: str = "Unknown"
    properties: Optional[Dict[str, Any]] = None


class EntityUpdate(BaseModel):
    properties: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    entity_type: Optional[str] = None


class EntityResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., alias="id")
    name: str
    entity_type: str = "Unknown"
    properties: Dict[str, Any] = {}


# ── Relation models ─────────────────────────────────────────────────

class RelationCreate(BaseModel):
    source: str
    target: str
    relation: str
    properties: Optional[Dict[str, Any]] = None


class RelationResponse(BaseModel):
    source: str
    target: str
    relation: str
    properties: Dict[str, Any] = {}


# ── Document models ────────────────────────────────────────────────

class DocumentInput(BaseModel):
    doc_id: str
    text: str
    chunk_size: int = 200
    overlap: int = 50


class DocumentResponse(BaseModel):
    doc_id: str
    status: str
    chunks_created: int = 0
    total_documents: int = 0
    total_chunks: int = 0


# ── Query models ───────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    kg_weight: float = 0.5
    rag_weight: float = 0.5


class FusionQueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    linked_entities: List[str] = []
    sources: List[Dict[str, Any]] = []
    kg_facts_count: int = 0
    doc_chunks_count: int = 0


# ── Graph query models ─────────────────────────────────────────────

class MultiHopQuery(BaseModel):
    entity_id: str
    hop: int = 2
    direction: str = "both"


class PathQuery(BaseModel):
    source: str
    target: str


class TextExtractionRequest(BaseModel):
    text: str


# ── Graph visualization ─────────────────────────────────────────────

class GraphVisResponse(BaseModel):
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []


# ── Stats ──────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    entity_count: int = 0
    relation_count: int = 0
    document_count: int = 0
    chunk_count: int = 0
