"""Knowledge Graph API endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from app.models import (
    EntityCreate,
    EntityUpdate,
    EntityResponse,
    RelationCreate,
    RelationResponse,
    MultiHopQuery,
    PathQuery,
    TextExtractionRequest,
    GraphVisResponse,
    StatsResponse,
)
from kg.builder import KnowledgeGraphBuilder
from kg.query_engine import GraphQueryEngine
from kg.reasoning import Reasoner

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])

# ── These will be set via dependency injection in main.py ──────────
_builder: Optional[KnowledgeGraphBuilder] = None
_query_engine: Optional[GraphQueryEngine] = None
_reasoner: Optional[Reasoner] = None


def init_graph_services(builder: KnowledgeGraphBuilder) -> None:
    global _builder, _query_engine, _reasoner
    _builder = builder
    _query_engine = GraphQueryEngine(builder.store)
    _reasoner = Reasoner(builder.store)


def get_store():
    if _builder is None:
        raise HTTPException(500, "Graph services not initialized")
    return _builder.store


# ── Entity CRUD ────────────────────────────────────────────────────

@router.post("/entities", response_model=Dict[str, Any])
def create_entity(entity: EntityCreate):
    store = get_store()
    added = store.add_entity(entity.entity_id, entity.name, entity.entity_type, entity.properties)
    if not added:
        raise HTTPException(409, f"Entity '{entity.entity_id}' already exists")
    return {"status": "created", "entity_id": entity.entity_id}


@router.get("/entities", response_model=List[Dict[str, Any]])
def list_entities(entity_type: Optional[str] = None):
    store = get_store()
    if entity_type:
        return store.get_entities_by_type(entity_type)
    return store.get_all_entities()


@router.get("/entities/{entity_id}", response_model=Dict[str, Any])
def get_entity(entity_id: str):
    store = get_store()
    entity = store.get_entity(entity_id)
    if not entity:
        raise HTTPException(404, f"Entity '{entity_id}' not found")
    return entity


@router.put("/entities/{entity_id}", response_model=Dict[str, Any])
def update_entity(entity_id: str, update: EntityUpdate):
    store = get_store()
    props = update.properties or {}
    if update.name:
        props["name"] = update.name
    if update.entity_type:
        props["entity_type"] = update.entity_type
    success = store.update_entity(entity_id, props)
    if not success:
        raise HTTPException(404, f"Entity '{entity_id}' not found")
    return {"status": "updated", "entity_id": entity_id}


@router.delete("/entities/{entity_id}", response_model=Dict[str, Any])
def delete_entity(entity_id: str):
    store = get_store()
    success = store.remove_entity(entity_id)
    if not success:
        raise HTTPException(404, f"Entity '{entity_id}' not found")
    return {"status": "deleted", "entity_id": entity_id}


@router.get("/entities/search/{keyword}", response_model=List[Dict[str, Any]])
def search_entities(keyword: str):
    store = get_store()
    return store.search_entities_by_name(keyword)


# ── Relation CRUD ─────────────────────────────────────────────────

@router.post("/relations", response_model=Dict[str, Any])
def create_relation(relation: RelationCreate):
    store = get_store()
    added = store.add_relation(relation.source, relation.target, relation.relation, relation.properties)
    if not added:
        raise HTTPException(404, "Source or target entity not found")
    return {"status": "created", "source": relation.source, "target": relation.target, "relation": relation.relation}


@router.get("/relations", response_model=List[Dict[str, Any]])
def list_relations(entity_id: Optional[str] = None, direction: str = "both"):
    store = get_store()
    if entity_id:
        return store.get_relations(entity_id, direction)
    return store.get_all_relations()


@router.delete("/relations", response_model=Dict[str, Any])
def delete_relation(source: str, target: str, relation: str):
    store = get_store()
    success = store.remove_relation(source, target, relation)
    if not success:
        raise HTTPException(404, "Relation not found")
    return {"status": "deleted"}


# ── Graph queries ───────────────────────────────────────────────────

@router.post("/query/multi-hop", response_model=Dict[str, Any])
def multi_hop_query(query: MultiHopQuery):
    if _query_engine is None:
        raise HTTPException(500, "Query engine not initialized")
    return _query_engine.multi_hop_query(query.entity_id, query.hop, query.direction)


@router.post("/query/path", response_model=Dict[str, Any])
def find_path(query: PathQuery):
    if _query_engine is None:
        raise HTTPException(500, "Query engine not initialized")
    return _query_engine.find_path(query.source, query.target)


@router.get("/query/traverse/bfs/{entity_id}", response_model=Dict[str, Any])
def traverse_bfs(entity_id: str, max_depth: int = 3):
    if _query_engine is None:
        raise HTTPException(500, "Query engine not initialized")
    return _query_engine.traverse_bfs(entity_id, max_depth)


@router.get("/query/traverse/dfs/{entity_id}", response_model=Dict[str, Any])
def traverse_dfs(entity_id: str, max_depth: int = 3):
    if _query_engine is None:
        raise HTTPException(500, "Query engine not initialized")
    return _query_engine.traverse_dfs(entity_id, max_depth)


@router.post("/reason/explain/{entity_id}", response_model=Dict[str, Any])
def explain_entity(entity_id: str):
    if _reasoner is None:
        raise HTTPException(500, "Reasoner not initialized")
    return _reasoner.explain_entity(entity_id)


# ── Text extraction ─────────────────────────────────────────────────

@router.post("/extract", response_model=Dict[str, Any])
def extract_from_text(request: TextExtractionRequest):
    if _builder is None:
        raise HTTPException(500, "Builder not initialized")
    return _builder.build_from_text(request.text)


# ── Visualization ──────────────────────────────────────────────────

@router.get("/visualization", response_model=Dict[str, Any])
def get_visualization():
    store = get_store()
    return store.to_vis_data()


# ── Stats ─────────────────────────────────────────────────────────

@router.get("/stats", response_model=Dict[str, Any])
def graph_stats():
    if _query_engine is None:
        raise HTTPException(500, "Query engine not initialized")
    return _query_engine.get_graph_stats()
