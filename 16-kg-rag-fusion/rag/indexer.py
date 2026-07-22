"""Document Indexer - Indexes documents into the vector store."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag.vector_store import VectorStore


class DocumentIndexer:
    """Manages document indexing into the vector store."""

    def __init__(self, vector_store: Optional[VectorStore] = None) -> None:
        self.store = vector_store or VectorStore()

    def index_document(
        self, doc_id: str, text: str, chunk_size: int = 200, overlap: int = 50
    ) -> Dict[str, Any]:
        """Index a document into the vector store."""
        chunk_count = self.store.add_document(doc_id, text, chunk_size, overlap)
        return {
            "doc_id": doc_id,
            "status": "indexed",
            "chunks_created": chunk_count,
            "total_documents": self.store.get_document_count(),
            "total_chunks": self.store.get_chunk_count(),
        }

    def remove_document(self, doc_id: str) -> Dict[str, Any]:
        """Remove a document from the index."""
        success = self.store.remove_document(doc_id)
        return {
            "doc_id": doc_id,
            "status": "removed" if success else "not_found",
            "total_documents": self.store.get_document_count(),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "document_count": self.store.get_document_count(),
            "chunk_count": self.store.get_chunk_count(),
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        return self.store.get_all_documents()
