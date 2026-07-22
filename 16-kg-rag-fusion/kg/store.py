"""Knowledge Graph Store - Graph storage based on NetworkX DiGraph."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx


class KnowledgeGraphStore:
    """In-memory knowledge graph store using NetworkX DiGraph.

    Supports entity/relationship CRUD, persistence (JSON/GraphML),
    and graph analytics.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    # ── Entity operations ──────────────────────────────────────────

    def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str = "Unknown",
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add an entity (node) to the graph. Returns True if newly added."""
        if entity_id in self.graph:
            return False
        self.graph.add_node(
            entity_id,
            name=name,
            entity_type=entity_type,
            properties=properties or {},
        )
        return True

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Return entity attributes or None."""
        if entity_id not in self.graph:
            return None
        data = dict(self.graph.nodes[entity_id])
        data["id"] = entity_id
        return data

    def update_entity(
        self, entity_id: str, properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        if entity_id not in self.graph:
            return False
        if properties:
            self.graph.nodes[entity_id].update(properties)
        return True

    def remove_entity(self, entity_id: str) -> bool:
        """Remove entity and all its edges."""
        if entity_id not in self.graph:
            return False
        self.graph.remove_node(entity_id)
        return True

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Return list of all entities."""
        result = []
        for nid, data in self.graph.nodes(data=True):
            entry = dict(data)
            entry["id"] = nid
            result.append(entry)
        return result

    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        return [
            dict(d) | {"id": nid}
            for nid, d in self.graph.nodes(data=True)
            if d.get("entity_type") == entity_type
        ]

    def search_entities_by_name(self, keyword: str) -> List[Dict[str, Any]]:
        keyword = keyword.lower()
        return [
            dict(d) | {"id": nid}
            for nid, d in self.graph.nodes(data=True)
            if keyword in d.get("name", "").lower()
        ]

    def entity_exists(self, entity_id: str) -> bool:
        return entity_id in self.graph

    def entity_count(self) -> int:
        return self.graph.number_of_nodes()

    # ── Relationship operations ─────────────────────────────────────

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a directed relation (edge). Returns True if added."""
        if source_id not in self.graph or target_id not in self.graph:
            return False
        self.graph.add_edge(
            source_id, target_id, relation=relation, properties=properties or {}
        )
        return True

    def remove_relation(self, source_id: str, target_id: str, relation: str) -> bool:
        if self.graph.has_edge(source_id, target_id):
            edge_data = self.graph[source_id][target_id]
            if edge_data.get("relation") == relation:
                self.graph.remove_edge(source_id, target_id)
                return True
        return False

    def get_relations(
        self, entity_id: str, direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Get relations of an entity. direction: 'outgoing', 'incoming', 'both'."""
        result = []
        if direction in ("outgoing", "both"):
            for _, tid, data in self.graph.out_edges(entity_id, data=True):
                result.append(
                    {
                        "source": entity_id,
                        "target": tid,
                        "relation": data.get("relation", ""),
                        "properties": data.get("properties", {}),
                    }
                )
        if direction in ("incoming", "both"):
            for sid, _, data in self.graph.in_edges(entity_id, data=True):
                result.append(
                    {
                        "source": sid,
                        "target": entity_id,
                        "relation": data.get("relation", ""),
                        "properties": data.get("properties", {}),
                    }
                )
        return result

    def get_all_relations(self) -> List[Dict[str, Any]]:
        return [
            {
                "source": s,
                "target": t,
                "relation": d.get("relation", ""),
                "properties": d.get("properties", {}),
            }
            for s, t, d in self.graph.edges(data=True)
        ]

    def relation_count(self) -> int:
        return self.graph.number_of_edges()

    # ── Graph traversal / query ────────────────────────────────────

    def get_neighbors(
        self, entity_id: str, depth: int = 1, direction: str = "both"
    ) -> Set[str]:
        """Get neighbors up to *depth* hops. Returns set of entity ids."""
        expanded: Set[str] = set()
        discovered: Set[str] = {entity_id}
        frontier: Set[str] = {entity_id}
        for _ in range(depth):
            next_frontier: Set[str] = set()
            for nid in frontier:
                if nid in expanded:
                    continue
                expanded.add(nid)
                if direction in ("outgoing", "both"):
                    for _, nb in self.graph.out_edges(nid):
                        if nb not in discovered:
                            discovered.add(nb)
                            next_frontier.add(nb)
                if direction in ("incoming", "both"):
                    for nb, _ in self.graph.in_edges(nid):
                        if nb not in discovered:
                            discovered.add(nb)
                            next_frontier.add(nb)
            frontier = next_frontier
        discovered.discard(entity_id)
        return discovered

    def get_subgraph(self, entity_ids: Set[str]) -> nx.DiGraph:
        """Extract induced subgraph for given entity ids."""
        existing = entity_ids & set(self.graph.nodes)
        return self.graph.subgraph(existing).copy()

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Shortest path between two entities. Returns None if not connected."""
        if source not in self.graph or target not in self.graph:
            return None
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None

    def bfs_traverse(self, start: str, max_depth: int = 3) -> List[str]:
        """BFS traversal from start node, return visited node ids."""
        if start not in self.graph:
            return []
        return list(nx.bfs_tree(self.graph, start, depth_limit=max_depth).nodes)

    def dfs_traverse(self, start: str, max_depth: int = 3) -> List[str]:
        """DFS traversal from start node, return visited node ids."""
        if start not in self.graph:
            return []
        return list(nx.dfs_tree(self.graph, start, depth_limit=max_depth).nodes)

    # ── Persistence ────────────────────────────────────────────────

    def save_json(self, filepath: str) -> None:
        data = {
            "nodes": {
                nid: dict(data) for nid, data in self.graph.nodes(data=True)
            },
            "edges": [
                {"source": s, "target": t, "relation": d.get("relation", ""), "properties": d.get("properties", {})}
                for s, t, d in self.graph.edges(data=True)
            ],
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_json(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.graph.clear()
        for nid, attrs in data.get("nodes", {}).items():
            self.graph.add_node(nid, **attrs)
        for edge in data.get("edges", []):
            self.graph.add_edge(
                edge["source"],
                edge["target"],
                relation=edge.get("relation", ""),
                properties=edge.get("properties", {}),
            )

    def save_graphml(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        nx.write_graphml(self.graph, filepath)

    def load_graphml(self, filepath: str) -> None:
        self.graph = nx.read_graphml(filepath)

    # ── Visualization helpers ───────────────────────────────────────

    def to_vis_data(self) -> Dict[str, Any]:
        """Convert graph to vis.js / D3 compatible format."""
        nodes = [
            {
                "id": nid,
                "label": data.get("name", nid),
                "group": data.get("entity_type", "Unknown"),
                **data,
            }
            for nid, data in self.graph.nodes(data=True)
        ]
        edges = [
            {
                "from": s,
                "to": t,
                "label": d.get("relation", ""),
                **d.get("properties", {}),
            }
            for s, t, d in self.graph.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}
