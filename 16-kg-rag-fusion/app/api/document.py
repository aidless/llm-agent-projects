"""Document Management API endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from app.models import DocumentInput, DocumentResponse
from rag.indexer import DocumentIndexer

router = APIRouter(prefix="/documents", tags=["Documents"])

_indexer: Optional[DocumentIndexer] = None


def init_document_service(indexer: DocumentIndexer) -> None:
    global _indexer
    _indexer = indexer


def get_indexer():
    if _indexer is None:
        raise HTTPException(500, "Document service not initialized")
    return _indexer


@router.post("/", response_model=Dict[str, Any])
def index_document(doc: DocumentInput):
    indexer = get_indexer()
    result = indexer.index_document(doc.doc_id, doc.text, doc.chunk_size, doc.overlap)
    return result


@router.delete("/{doc_id}", response_model=Dict[str, Any])
def remove_document(doc_id: str):
    indexer = get_indexer()
    result = indexer.remove_document(doc_id)
    return result


@router.get("/", response_model=List[Dict[str, Any]])
def list_documents():
    indexer = get_indexer()
    return indexer.list_documents()


@router.get("/stats", response_model=Dict[str, Any])
def document_stats():
    indexer = get_indexer()
    return indexer.get_stats()
