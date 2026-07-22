"""Query Engine - High-level graph query interface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from kg.store import KnowledgeGraphStore


class GraphQueryEngine:
    """High-level query engine for the knowledge graph."""

    def __init__(self, store: KnowledgeGraphStore) -> None:
        self.store = store

    def query_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by id."""
        return self.store.get_entity(entity_id)

    def search_entities(self, keyword: str) -> List[Dict[str, Any]]:
        """Search entities by name keyword."""
        return self.store.search_entities_by_name(keyword)

    def get_entity_relations(
        self, entity_id: str, direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Get all relations of an entity."""
        if not self.store.entity_exists(entity_id):
            return []
        return self.store.get_relations(entity_id, direction)

    def multi_hop_query(
        self,
        entity_id: str,
        hop: int = 2,
        direction: str = "both",
    ) -> Dict[str, Any]:
        """Multi-hop neighbor query. Returns neighbors and subgraph info."""
        if not self.store.entity_exists(entity_id):
            return {"entity_id": entity_id, "error": "Entity not found", "neighbors": []}

        neighbors = self.store.get_neighbors(entity_id, depth=hop, direction=direction)
        neighbor_details = [
            self.store.get_entity(nid) for nid in neighbors if self.store.get_entity(nid)
        ]

        # Build subgraph
        sub_nodes = neighbors | {entity_id}
        subgraph = self.store.get_subgraph(sub_nodes)
        subgraph_edges = [
            {
                "source": s,
                "target": t,
                "relation": d.get("relation", ""),
            }
            for s, t, d in subgraph.edges(data=True)
        ]

        return {
            "entity_id": entity_id,
            "hop": hop,
            "direction": direction,
            "neighbor_count": len(neighbors),
            "neighbors": [n for n in neighbor_details if n],
            "subgraph_edges": subgraph_edges,
        }

    def find_path(self, source: str, target: str) -> Dict[str, Any]:
        """Find shortest path between two entities."""
        path = self.store.shortest_path(source, target)
        if path is None:
            return {
                "source": source,
                "target": target,
                "found": False,
                "path": None,
                "error": "No path found",
            }
        # Annotate path with entity names
        path_details = []
        for nid in path:
            ent = self.store.get_entity(nid)
            if ent:
                path_details.append(ent)
            else:
                path_details.append({"id": nid, "name": nid})

        # Annotate edges along the path
        edge_details = []
        for i in range(len(path) - 1):
            if self.store.graph.has_edge(path[i], path[i + 1]):
                edge_data = self.store.graph[path[i]][path[i + 1]]
                edge_details.append(
                    {
                        "source": path[i],
                        "target": path[i + 1],
                        "relation": edge_data.get("relation", ""),
                    }
                )
            elif self.store.graph.has_edge(path[i + 1], path[i]):
                edge_data = self.store.graph[path[i + 1]][path[i]]
                edge_details.append(
                    {
                        "source": path[i + 1],
                        "target": path[i],
                        "relation": edge_data.get("relation", ""),
                    }
                )

        return {
            "source": source,
            "target": target,
            "found": True,
            "path": path_details,
            "edges": edge_details,
            "length": len(path) - 1,
        }

    def traverse_bfs(self, start: str, max_depth: int = 3) -> Dict[str, Any]:
        """BFS traversal result."""
        if not self.store.entity_exists(start):
            return {"start": start, "error": "Entity not found", "visited": []}
        visited = self.store.bfs_traverse(start, max_depth)
        details = [self.store.get_entity(nid) for nid in visited]
        return {"start": start, "max_depth": max_depth, "visited": [d for d in details if d]}

    def traverse_dfs(self, start: str, max_depth: int = 3) -> Dict[str, Any]:
        """DFS traversal result."""
        if not self.store.entity_exists(start):
            return {"start": start, "error": "Entity not found", "visited": []}
        visited = self.store.dfs_traverse(start, max_depth)
        details = [self.store.get_entity(nid) for nid in visited]
        return {"start": start, "max_depth": max_depth, "visited": [d for d in details if d]}

    def get_subgraph_data(
        self, entity_ids: List[str]
    ) -> Dict[str, Any]:
        """Get subgraph for a list of entity ids."""
        id_set = set(entity_ids)
        subgraph = self.store.get_subgraph(id_set)
        nodes = [
            dict(d) | {"id": nid} for nid, d in subgraph.nodes(data=True)
        ]
        edges = [
            {
                "source": s,
                "target": t,
                "relation": d.get("relation", ""),
            }
            for s, t, d in subgraph.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}

    def get_graph_stats(self) -> Dict[str, Any]:
        return {
            "entity_count": self.store.entity_count(),
            "relation_count": self.store.relation_count(),
            "entity_types": list(
                set(
                    d.get("entity_type", "Unknown")
                    for _, d in self.store.graph.nodes(data=True)
                )
            ),
        }
