"""Hybrid Retriever - Combines semantic and keyword search with RRF fusion."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class HybridRetriever:
    """Hybrid retriever combining TF-IDF semantic search and keyword search
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, vector_store, rrf_k: int = 60) -> None:
        """
        Args:
            vector_store: A VectorStore instance.
            rrf_k: RRF constant (default 60).
        """
        self.store = vector_store
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """Retrieve using hybrid search with RRF fusion.

        Args:
            query: Search query.
            top_k: Number of results to return.
            semantic_weight: Weight for semantic (TF-IDF) results in RRF.
            keyword_weight: Weight for keyword results in RRF.

        Returns:
            Fused and ranked results.
        """
        # Get results from both retrievers
        semantic_results = self.store.search(query, top_k=top_k * 2)
        keyword_results = self.store.keyword_search(query, top_k=top_k * 2)

        # Apply RRF fusion
        fused = self._rrf_fusion(
            semantic_results,
            keyword_results,
            semantic_weight,
            keyword_weight,
        )

        return fused[:top_k]

    def retrieve_semantic(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Pure semantic (TF-IDF) retrieval."""
        return self.store.search(query, top_k=top_k)

    def retrieve_keyword(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Pure keyword retrieval."""
        return self.store.keyword_search(query, top_k=top_k)

    def _rrf_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        semantic_weight: float,
        keyword_weight: float,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion of two result lists."""
        scores: Dict[str, float] = {}
        chunk_data: Dict[str, Dict[str, Any]] = {}

        # Score semantic results
        for rank, result in enumerate(semantic_results):
            cid = result["chunk_id"]
            rrf_score = semantic_weight / (self.rrf_k + rank + 1)
            scores[cid] = scores.get(cid, 0.0) + rrf_score
            chunk_data[cid] = result

        # Score keyword results
        for rank, result in enumerate(keyword_results):
            cid = result["chunk_id"]
            rrf_score = keyword_weight / (self.rrf_k + rank + 1)
            scores[cid] = scores.get(cid, 0.0) + rrf_score
            if cid not in chunk_data:
                chunk_data[cid] = result

        # Sort by fused score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for cid, score in sorted_results:
            entry = dict(chunk_data[cid])
            entry["fused_score"] = score
            entry["source"] = "hybrid"
            results.append(entry)

        return results
