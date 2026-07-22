"""Multi-hop Reasoning over the knowledge graph."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from kg.store import KnowledgeGraphStore


class Reasoner:
    """Performs multi-hop reasoning and path-based inference on the KG."""

    def __init__(self, store: KnowledgeGraphStore) -> None:
        self.store = store

    def multi_hop_reason(
        self,
        entity_id: str,
        max_hops: int = 3,
        relation_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Reason over multiple hops from an entity.

        Args:
            entity_id: Starting entity.
            max_hops: Maximum number of hops (1-3).
            relation_filter: Optional list of relation types to follow.

        Returns:
            Dict with reasoning chain and discovered facts.
        """
        if not self.store.entity_exists(entity_id):
            return {"error": "Entity not found", "entity_id": entity_id}

        chain: List[Dict[str, Any]] = []
        visited: Set[str] = {entity_id}
        current_level: List[str] = [entity_id]

        for hop in range(1, max_hops + 1):
            next_level: List[str] = []
            for nid in current_level:
                relations = self.store.get_relations(nid, "outgoing")
                for rel in relations:
                    target = rel["target"]
                    rel_type = rel["relation"]

                    if relation_filter and rel_type not in relation_filter:
                        continue

                    if target not in visited:
                        visited.add(target)
                        next_level.append(target)

                        src_ent = self.store.get_entity(nid)
                        tgt_ent = self.store.get_entity(target)
                        chain.append(
                            {
                                "hop": hop,
                                "source": nid,
                                "source_name": src_ent.get("name", nid) if src_ent else nid,
                                "target": target,
                                "target_name": tgt_ent.get("name", target) if tgt_ent else target,
                                "relation": rel_type,
                                "fact": f"{src_ent.get('name', nid) if src_ent else nid} --[{rel_type}]--> {tgt_ent.get('name', target) if tgt_ent else target}",
                            }
                        )

            current_level = next_level
            if not current_level:
                break

        return {
            "entity_id": entity_id,
            "max_hops": max_hops,
            "total_discovered": len(chain),
            "reasoning_chain": chain,
            "discovered_entities": list(visited - {entity_id}),
        }

    def find_reasoning_path(
        self, source: str, target: str, max_depth: int = 5
    ) -> Dict[str, Any]:
        """Find a reasoning path between two entities.

        Returns path with annotated relations forming a reasoning chain.
        """
        path = self.store.shortest_path(source, target)
        if path is None:
            return {
                "source": source,
                "target": target,
                "found": False,
                "reasoning": None,
            }

        reasoning_steps = []
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            relation = "unknown"
            if self.store.graph.has_edge(src, tgt):
                relation = self.store.graph[src][tgt].get("relation", "unknown")
            elif self.store.graph.has_edge(tgt, src):
                relation = self.store.graph[tgt][src].get("relation", "unknown")

            src_ent = self.store.get_entity(src)
            tgt_ent = self.store.get_entity(tgt)
            reasoning_steps.append(
                {
                    "step": i + 1,
                    "from": src_ent.get("name", src) if src_ent else src,
                    "to": tgt_ent.get("name", tgt) if tgt_ent else tgt,
                    "relation": relation,
                }
            )

        return {
            "source": source,
            "target": target,
            "found": True,
            "path_length": len(path) - 1,
            "reasoning": reasoning_steps,
        }

    def explain_entity(
        self, entity_id: str, max_hops: int = 2
    ) -> Dict[str, Any]:
        """Generate an explanation of an entity based on its graph context."""
        if not self.store.entity_exists(entity_id):
            return {"error": "Entity not found"}

        entity = self.store.get_entity(entity_id)
        neighbors = self.store.get_neighbors(entity_id, depth=max_hops)
        relations = self.store.get_relations(entity_id, "both")

        # Collect types of connected entities
        connected_types: Dict[str, List[str]] = {}
        for nid in neighbors:
            ent = self.store.get_entity(nid)
            if ent:
                etype = ent.get("entity_type", "Unknown")
                connected_types.setdefault(etype, []).append(ent.get("name", nid))

        return {
            "entity": entity,
            "direct_relations": relations,
            "connected_entity_types": connected_types,
            "total_connections": len(neighbors),
            "summary": (
                f"{entity.get('name', entity_id)} 是一个 {entity.get('entity_type', 'Unknown')} 类型实体，"
                f"与 {len(neighbors)} 个实体有关系连接。"
                f"相关关系包括: {', '.join(r['relation'] for r in relations[:5])}。"
            ),
        }
