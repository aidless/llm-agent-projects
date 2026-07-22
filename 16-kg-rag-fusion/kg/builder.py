"""Knowledge Graph Builder - Orchestrates extraction and storage."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from kg.extractor import Extractor
from kg.store import KnowledgeGraphStore


class KnowledgeGraphBuilder:
    """Builds and manages a knowledge graph from text or pre-loaded data."""

    def __init__(
        self,
        store: Optional[KnowledgeGraphStore] = None,
        extractor: Optional[Extractor] = None,
    ) -> None:
        self.store = store or KnowledgeGraphStore()
        self.extractor = extractor or Extractor(use_llm_mock=True)

    def build_from_text(self, text: str) -> Dict[str, Any]:
        """Extract entities and relations from text and add to graph."""
        raw_entities = self.extractor.extract_entities(text)
        entities = self.extractor.disambiguate(raw_entities)
        relations = self.extractor.extract_relations(text, entities)

        added_entities = 0
        for ent in entities:
            if self.store.add_entity(
                ent["entity_id"], ent["name"], ent["type"]
            ):
                added_entities += 1

        added_relations = 0
        for rel in relations:
            if self.store.add_relation(rel["source"], rel["target"], rel["relation"]):
                added_relations += 1

        return {
            "entities_found": len(raw_entities),
            "entities_after_disambiguation": len(entities),
            "entities_added": added_entities,
            "relations_found": len(relations),
            "relations_added": added_relations,
        }

    def load_from_json(self, filepath: str) -> Dict[str, int]:
        """Load graph from a JSON file."""
        self.store.load_json(filepath)
        return {
            "entities": self.store.entity_count(),
            "relations": self.store.relation_count(),
        }

    def save_to_json(self, filepath: str) -> None:
        """Save graph to a JSON file."""
        self.store.save_json(filepath)

    def get_stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        return {
            "entity_count": self.store.entity_count(),
            "relation_count": self.store.relation_count(),
        }

    def reset(self) -> None:
        """Clear the graph."""
        self.store.graph.clear()
