"""Entity Linker - Links query terms to entities in the knowledge graph."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from kg.store import KnowledgeGraphStore


class EntityLinker:
    """Links query text to entities in the knowledge graph."""

    def __init__(self, store: KnowledgeGraphStore) -> None:
        self.store = store

    def link_entities(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Link entities from the query to the knowledge graph.

        Uses name matching, alias matching, and fuzzy matching.

        Returns:
            List of linked entities with match scores.
        """
        query_lower = query.lower()
        query_tokens = set(re.findall(r"\b\w+\b", query_lower))

        candidates: List[Dict[str, Any]] = []

        for entity_id, data in self.store.graph.nodes(data=True):
            name = data.get("name", "")
            name_lower = name.lower()
            score = self._compute_link_score(query_lower, query_tokens, name_lower, data)

            if score > 0:
                candidates.append(
                    {
                        "entity_id": entity_id,
                        "name": name,
                        "entity_type": data.get("entity_type", "Unknown"),
                        "link_score": score,
                    }
                )

        # Sort by link score and return top_k
        candidates.sort(key=lambda x: x["link_score"], reverse=True)
        return candidates[:top_k]

    def link_single_entity(self, query: str) -> Optional[Dict[str, Any]]:
        """Try to link query to a single best matching entity."""
        results = self.link_entities(query, top_k=1)
        return results[0] if results else None

    def _compute_link_score(
        self,
        query_lower: str,
        query_tokens: Set[str],
        name_lower: str,
        entity_data: Dict[str, Any],
    ) -> float:
        """Compute link score between query and entity."""
        score = 0.0

        # Exact name match
        if name_lower in query_lower:
            score += 1.0
        # Entity name tokens in query
        name_tokens = set(re.findall(r"\b\w+\b", name_lower))
        overlap = query_tokens & name_tokens
        if overlap:
            score += len(overlap) / max(len(name_tokens), 1) * 0.5

        # Property matching (e.g., aliases)
        props = entity_data.get("properties", {})
        aliases = props.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        for alias in aliases:
            if alias.lower() in query_lower:
                score += 0.8

        # Type bonus if type mentioned in query
        etype = entity_data.get("entity_type", "").lower()
        if etype and etype in query_lower:
            score += 0.3

        return score
