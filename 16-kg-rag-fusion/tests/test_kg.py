"""Tests for Knowledge Graph module."""

import json
import os
import pytest

from kg.store import KnowledgeGraphStore
from kg.extractor import Extractor
from kg.builder import KnowledgeGraphBuilder
from kg.query_engine import GraphQueryEngine
from kg.reasoning import Reasoner

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE_KG = os.path.join(DATA_DIR, "sample_kg.json")


@pytest.fixture
def store():
    s = KnowledgeGraphStore()
    s.add_entity("e1", "Transformer", "Technology")
    s.add_entity("e2", "OpenAI", "Organization")
    s.add_entity("e3", "GPT", "Technology")
    s.add_entity("e4", "NLP", "Technology")
    s.add_entity("e5", "Google", "Organization")
    s.add_entity("e6", "BERT", "Technology")
    s.add_relation("e3", "e1", "uses")
    s.add_relation("e3", "e2", "developed_by")
    s.add_relation("e3", "e4", "applied_in")
    s.add_relation("e6", "e1", "uses")
    s.add_relation("e6", "e5", "developed_by")
    return s


@pytest.fixture
def sample_store():
    """Store loaded with sample_kg.json."""
    s = KnowledgeGraphStore()
    if os.path.exists(SAMPLE_KG):
        s.load_json(SAMPLE_KG)
    return s


class TestKnowledgeGraphStore:
    def test_add_entity(self, store):
        assert store.add_entity("e7", "NewEntity", "Concept") is True
        assert store.entity_count() == 7

    def test_add_duplicate_entity(self, store):
        assert store.add_entity("e1", "Transformer", "Technology") is False

    def test_get_entity(self, store):
        ent = store.get_entity("e1")
        assert ent is not None
        assert ent["name"] == "Transformer"
        assert ent["entity_type"] == "Technology"

    def test_get_entity_not_found(self, store):
        assert store.get_entity("nonexistent") is None

    def test_remove_entity(self, store):
        assert store.remove_entity("e1") is True
        assert store.entity_count() == 5
        # Removing an entity should also remove its edges
        assert store.relation_count() == 3  # e3->e1 and e6->e1 removed

    def test_remove_entity_not_found(self, store):
        assert store.remove_entity("nonexistent") is False

    def test_update_entity(self, store):
        assert store.update_entity("e1", {"name": "TransformerV2"}) is True
        ent = store.get_entity("e1")
        assert ent["name"] == "TransformerV2"

    def test_update_entity_not_found(self, store):
        assert store.update_entity("nonexistent", {"name": "X"}) is False

    def test_get_all_entities(self, store):
        entities = store.get_all_entities()
        assert len(entities) == 6

    def test_get_entities_by_type(self, store):
        tech = store.get_entities_by_type("Technology")
        assert len(tech) == 4  # e1, e3, e4, e6
        orgs = store.get_entities_by_type("Organization")
        assert len(orgs) == 2  # e2, e5

    def test_search_entities_by_name(self, store):
        results = store.search_entities_by_name("trans")
        assert len(results) == 1
        assert results[0]["name"] == "Transformer"

    def test_entity_exists(self, store):
        assert store.entity_exists("e1") is True
        assert store.entity_exists("nonexistent") is False

    def test_add_relation(self, store):
        assert store.add_relation("e1", "e4", "applied_in") is True
        assert store.relation_count() == 6

    def test_add_relation_missing_entity(self, store):
        assert store.add_relation("e1", "nonexistent", "uses") is False

    def test_remove_relation(self, store):
        assert store.remove_relation("e3", "e1", "uses") is True
        assert store.relation_count() == 4

    def test_remove_relation_wrong_type(self, store):
        assert store.remove_relation("e3", "e1", "wrong_relation") is False

    def test_get_relations(self, store):
        rels = store.get_relations("e3", "both")
        assert len(rels) == 3

    def test_get_relations_outgoing(self, store):
        rels = store.get_relations("e3", "outgoing")
        assert len(rels) == 3

    def test_get_relations_incoming(self, store):
        rels = store.get_relations("e2", "incoming")
        assert len(rels) == 1
        assert rels[0]["source"] == "e3"

    def test_get_neighbors(self, store):
        neighbors = store.get_neighbors("e3", depth=1)
        assert "e1" in neighbors
        assert "e2" in neighbors
        assert "e4" in neighbors
        assert "e3" not in neighbors

    def test_get_neighbors_multihop(self, store):
        neighbors = store.get_neighbors("e3", depth=2)
        # e3 depth=1: {e1,e2,e4}
        # e3 depth=2: also finds e6 via e1 (incoming edge e6->e1)
        assert len(neighbors) == 4

    def test_shortest_path(self, store):
        path = store.shortest_path("e3", "e5")
        # e3 -> e1 -> ? no, e1 has no edge to e5. e3 -> e2 -> ? no. So no path.
        # Let's check: e3->e1(uses), e3->e2(developed_by), e3->e4(applied_in)
        # e6->e1(uses), e6->e5(developed_by)
        # No directed path from e3 to e5
        assert path is None

    def test_shortest_path_connected(self, store):
        path = store.shortest_path("e3", "e1")
        assert path is not None
        assert path[0] == "e3"
        assert path[-1] == "e1"

    def test_bfs_traverse(self, store):
        visited = store.bfs_traverse("e3", max_depth=1)
        assert "e3" in visited
        assert len(visited) == 4  # e3 + e1, e2, e4

    def test_dfs_traverse(self, store):
        visited = store.dfs_traverse("e3", max_depth=1)
        assert "e3" in visited

    def test_get_subgraph(self, store):
        sub = store.get_subgraph({"e1", "e2", "e3"})
        assert sub.number_of_nodes() == 3

    def test_persistence_json(self, store, tmp_path):
        filepath = str(tmp_path / "test_kg.json")
        store.save_json(filepath)
        assert os.path.exists(filepath)

        new_store = KnowledgeGraphStore()
        new_store.load_json(filepath)
        assert new_store.entity_count() == store.entity_count()
        assert new_store.relation_count() == store.relation_count()

    def test_to_vis_data(self, store):
        vis = store.to_vis_data()
        assert "nodes" in vis
        assert "edges" in vis
        assert len(vis["nodes"]) == 6
        assert len(vis["edges"]) == 5


class TestExtractor:
    def test_extract_entities_tech(self):
        ext = Extractor(use_llm_mock=False)
        text = "Transformer是一种深度学习模型，用于自然语言处理。"
        entities = ext.extract_entities(text)
        names = [e["name"] for e in entities]
        assert "Transformer" in names
        assert "深度学习" in names
        assert "自然语言处理" in names

    def test_extract_entities_org(self):
        ext = Extractor(use_llm_mock=False)
        text = "OpenAI开发了GPT模型。"
        entities = ext.extract_entities(text)
        names = [e["name"] for e in entities]
        assert "OpenAI" in names

    def test_extract_entities_person(self):
        ext = Extractor(use_llm_mock=False)
        text = "Geoffrey Hinton是深度学习的先驱。"
        entities = ext.extract_entities(text)
        names = [e["name"] for e in entities]
        assert "Geoffrey Hinton" in names

    def test_extract_relations(self):
        ext = Extractor(use_llm_mock=True)
        text = "OpenAI开发了GPT模型，GPT基于Transformer架构。"
        entities = ext.extract_entities(text)
        relations = ext.extract_relations(text, entities)
        assert len(relations) > 0

    def test_disambiguate(self):
        ext = Extractor(use_llm_mock=False)
        entities = [
            {"name": "Transformer", "type": "Technology", "entity_id": "transformer"},
            {"name": "transformer", "type": "Unknown", "entity_id": "transformer_2"},
        ]
        merged = ext.disambiguate(entities)
        assert len(merged) == 1


class TestKnowledgeGraphBuilder:
    def test_build_from_text(self):
        builder = KnowledgeGraphBuilder()
        text = "OpenAI开发了GPT模型，GPT基于Transformer架构。"
        result = builder.build_from_text(text)
        assert result["entities_found"] > 0
        assert result["relations_found"] > 0

    def test_load_from_json(self):
        builder = KnowledgeGraphBuilder()
        if os.path.exists(SAMPLE_KG):
            stats = builder.load_from_json(SAMPLE_KG)
            assert stats["entities"] >= 30
            assert stats["relations"] >= 50

    def test_save_and_load_json(self, tmp_path):
        builder = KnowledgeGraphBuilder()
        builder.store.add_entity("test", "Test", "Concept")
        filepath = str(tmp_path / "saved.json")
        builder.save_to_json(filepath)

        builder2 = KnowledgeGraphBuilder()
        builder2.load_from_json(filepath)
        assert builder2.store.entity_count() == 1

    def test_reset(self):
        builder = KnowledgeGraphBuilder()
        builder.store.add_entity("a", "A", "Type")
        builder.reset()
        assert builder.store.entity_count() == 0


class TestGraphQueryEngine:
    def test_query_entity(self, sample_store):
        if sample_store.entity_count() == 0:
            pytest.skip("No sample data")
        engine = GraphQueryEngine(sample_store)
        ent = engine.query_entity("transformer")
        assert ent is not None
        assert ent["name"] == "Transformer"

    def test_search_entities(self, sample_store):
        if sample_store.entity_count() == 0:
            pytest.skip("No sample data")
        engine = GraphQueryEngine(sample_store)
        results = engine.search_entities("GPT")
        assert len(results) > 0

    def test_multi_hop_query(self, sample_store):
        if sample_store.entity_count() == 0:
            pytest.skip("No sample data")
        engine = GraphQueryEngine(sample_store)
        result = engine.multi_hop_query("transformer", hop=2)
        assert "neighbors" in result
        assert result["neighbor_count"] > 0

    def test_find_path(self, sample_store):
        if sample_store.entity_count() == 0:
            pytest.skip("No sample data")
        engine = GraphQueryEngine(sample_store)
        result = engine.find_path("gpt", "openai")
        assert result["found"] is True
        assert result["length"] >= 1

    def test_bfs_traverse(self, sample_store):
        if sample_store.entity_count() == 0:
            pytest.skip("No sample data")
        engine = GraphQueryEngine(sample_store)
        result = engine.traverse_bfs("transformer", max_depth=2)
        assert len(result["visited"]) > 1

    def test_get_graph_stats(self, sample_store):
        if sample_store.entity_count() == 0:
            pytest.skip("No sample data")
        engine = GraphQueryEngine(sample_store)
        stats = engine.get_graph_stats()
        assert stats["entity_count"] >= 30
        assert stats["relation_count"] >= 50


class TestReasoner:
    def test_multi_hop_reason(self, sample_store):
        if sample_store.entity_count() == 0:
            pytest.skip("No sample data")
        reasoner = Reasoner(sample_store)
        result = reasoner.multi_hop_reason("transformer", max_hops=2)
        assert "reasoning_chain" in result
        assert result["total_discovered"] > 0

    def test_find_reasoning_path(self, sample_store):
        if sample_store.entity_count() == 0:
            pytest.skip("No sample data")
        reasoner = Reasoner(sample_store)
        result = reasoner.find_reasoning_path("gpt", "openai")
        assert result["found"] is True
        assert len(result["reasoning"]) > 0

    def test_explain_entity(self, sample_store):
        if sample_store.entity_count() == 0:
            pytest.skip("No sample data")
        reasoner = Reasoner(sample_store)
        result = reasoner.explain_entity("transformer")
        assert "summary" in result
        assert "total_connections" in result

    def test_multi_hop_reason_not_found(self, sample_store):
        reasoner = Reasoner(sample_store)
        result = reasoner.multi_hop_reason("nonexistent")
        assert "error" in result
