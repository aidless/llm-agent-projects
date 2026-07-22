"""Result Fuser - Fuses KG and RAG results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ResultFuser:
    """Fuses results from knowledge graph reasoning and RAG retrieval."""

    def __init__(self, kg_weight: float = 0.5, rag_weight: float = 0.5) -> None:
        self.kg_weight = kg_weight
        self.rag_weight = rag_weight

    def fuse(
        self,
        kg_results: List[Dict[str, Any]],
        rag_results: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Fuse KG and RAG results into a unified ranking.

        Args:
            kg_results: Results from knowledge graph (facts, entities, paths).
            rag_results: Results from RAG retrieval (document chunks).
            top_k: Number of final results.

        Returns:
            Fused result set with combined scores.
        """
        fused_items: List[Dict[str, Any]] = []

        # Process KG results
        for idx, item in enumerate(kg_results):
            fused_items.append(
                {
                    "type": "kg",
                    "data": item,
                    "score": self.kg_weight / (idx + 1),
                    "rank_kg": idx + 1,
                    "rank_rag": None,
                }
            )

        # Process RAG results
        for idx, item in enumerate(rag_results):
            rag_score = item.get("score", item.get("fused_score", 0.0))
            fused_items.append(
                {
                    "type": "rag",
                    "data": item,
                    "score": self.rag_weight * rag_score / (idx + 1),
                    "rank_kg": None,
                    "rank_rag": idx + 1,
                }
            )

        # Sort by combined score
        fused_items.sort(key=lambda x: x["score"], reverse=True)

        return {
            "total_results": len(fused_items),
            "results": fused_items[:top_k],
            "kg_result_count": len(kg_results),
            "rag_result_count": len(rag_results),
            "weights": {
                "kg": self.kg_weight,
                "rag": self.rag_weight,
            },
        }

    def build_context(
        self,
        fused_results: Dict[str, Any],
    ) -> str:
        """Build a combined context string from fused results."""
        parts = []

        for item in fused_results.get("results", []):
            if item["type"] == "kg":
                data = item["data"]
                if "fact" in data:
                    parts.append(f"[知识图谱] {data['fact']}")
                elif "summary" in data:
                    parts.append(f"[知识图谱] {data['summary']}")
                elif "name" in data:
                    parts.append(f"[知识图谱] {data.get('name', '')}: {data.get('entity_type', '')}")
            elif item["type"] == "rag":
                data = item["data"]
                text = data.get("text", "")
                if text:
                    parts.append(f"[文档] {text}")

        return "\n".join(parts)
