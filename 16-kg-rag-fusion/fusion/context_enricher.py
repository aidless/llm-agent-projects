"""Context Enricher - Enriches context using knowledge graph subgraph extraction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from kg.store import KnowledgeGraphStore


class ContextEnricher:
    """Enriches retrieved context using knowledge graph subgraph data."""

    def __init__(self, store: KnowledgeGraphStore) -> None:
        self.store = store

    def enrich(
        self,
        linked_entities: List[Dict[str, Any]],
        hop: int = 2,
    ) -> Dict[str, Any]:
        """Extract subgraph context around linked entities.

        Args:
            linked_entities: Entities linked from the query.
            hop: Number of hops for subgraph extraction.

        Returns:
            Enrichment result with structured graph context.
        """
        if not linked_entities:
            return {"entities_context": [], "facts": [], "summary": ""}

        # Collect all relevant entity ids
        all_entity_ids: Set[str] = set()
        for ent in linked_entities:
            eid = ent["entity_id"]
            all_entity_ids.add(eid)
            neighbors = self.store.get_neighbors(eid, depth=hop)
            all_entity_ids.update(neighbors)

        # Extract subgraph
        subgraph = self.store.get_subgraph(all_entity_ids)

        # Build facts from subgraph edges
        facts: List[str] = []
        entity_names: Dict[str, str] = {}
        for nid, data in subgraph.nodes(data=True):
            entity_names[nid] = data.get("name", nid)

        for s, t, d in subgraph.edges(data=True):
            relation = d.get("relation", "related_to")
            src_name = entity_names.get(s, s)
            tgt_name = entity_names.get(t, t)
            facts.append(f"{src_name} --[{relation}]--> {tgt_name}")

        # Build entity descriptions
        entities_context = []
        for ent in linked_entities:
            eid = ent["entity_id"]
            entity_data = self.store.get_entity(eid)
            if entity_data:
                relations = self.store.get_relations(eid, "both")
                entities_context.append(
                    {
                        "entity": entity_data,
                        "relations": relations[:10],
                        "neighbor_count": len(self.store.get_neighbors(eid, depth=hop)),
                    }
                )

        summary = self._build_summary(linked_entities, facts)

        return {
            "linked_entities": linked_entities,
            "entities_context": entities_context,
            "facts": facts,
            "subgraph_size": len(all_entity_ids),
            "summary": summary,
        }

    def _build_summary(
        self,
        linked_entities: List[Dict[str, Any]],
        facts: List[str],
    ) -> str:
        """Build a natural language summary of the graph context."""
        if not linked_entities:
            return ""
        entity_names = [e["name"] for e in linked_entities]
        intro = f"知识图谱中找到相关实体: {', '.join(entity_names)}。"
        if facts:
            intro += f" 相关事实包括: {'; '.join(facts[:10])}。"
        return intro
