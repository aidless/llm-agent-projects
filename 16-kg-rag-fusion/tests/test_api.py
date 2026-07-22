"""Tests for API endpoints using FastAPI TestClient."""

import os
import pytest
from fastapi.testclient import TestClient

# We need to import AFTER setting up sys.path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app

client = TestClient(app)


class TestHealthAndRoot:
    def test_root(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "KG-RAG Fusion System"
        assert data["version"] == "1.0.0"

    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "entities" in data
        assert "relations" in data


class TestGraphAPI:
    def test_list_entities(self):
        resp = client.get("/graph/entities")
        assert resp.status_code == 200
        entities = resp.json()
        assert len(entities) >= 30

    def test_get_entity(self):
        resp = client.get("/graph/entities/transformer")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Transformer"

    def test_get_entity_not_found(self):
        resp = client.get("/graph/entities/nonexistent_xyz")
        assert resp.status_code == 404

    def test_search_entities(self):
        resp = client.get("/graph/entities/search/GPT")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) > 0

    def test_list_relations(self):
        resp = client.get("/graph/relations")
        assert resp.status_code == 200
        relations = resp.json()
        assert len(relations) >= 50

    def test_get_relations_by_entity(self):
        resp = client.get("/graph/relations?entity_id=transformer")
        assert resp.status_code == 200
        relations = resp.json()
        assert len(relations) > 0

    def test_create_and_delete_entity(self):
        # Create
        resp = client.post(
            "/graph/entities",
            json={"entity_id": "test_entity", "name": "TestEntity", "entity_type": "Test"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "created"

        # Verify
        resp = client.get("/graph/entities/test_entity")
        assert resp.status_code == 200

        # Delete
        resp = client.delete("/graph/entities/test_entity")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_create_and_delete_relation(self):
        # Create
        resp = client.post(
            "/graph/relations",
            json={"source": "transformer", "target": "nlp", "relation": "test_rel"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "created"

        # Delete
        resp = client.delete(
            "/graph/relations?source=transformer&target=nlp&relation=test_rel"
        )
        assert resp.status_code == 200

    def test_multi_hop_query(self):
        resp = client.post(
            "/graph/query/multi-hop",
            json={"entity_id": "transformer", "hop": 2, "direction": "both"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "neighbors" in data
        assert data["neighbor_count"] > 0

    def test_find_path(self):
        resp = client.post(
            "/graph/query/path",
            json={"source": "gpt", "target": "openai"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True

    def test_bfs_traverse(self):
        resp = client.get("/graph/query/traverse/bfs/transformer?max_depth=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["visited"]) > 1

    def test_dfs_traverse(self):
        resp = client.get("/graph/query/traverse/dfs/transformer?max_depth=2")
        assert resp.status_code == 200

    def test_extract_from_text(self):
        resp = client.post(
            "/graph/extract",
            json={"text": "OpenAI开发了GPT模型，GPT基于Transformer架构。"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["entities_found"] > 0

    def test_visualization(self):
        resp = client.get("/graph/visualization")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) >= 30

    def test_graph_stats(self):
        resp = client.get("/graph/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_count"] >= 30
        assert data["relation_count"] >= 50


class TestDocumentAPI:
    def test_index_document(self):
        resp = client.post(
            "/documents/",
            json={
                "doc_id": "test_doc",
                "text": "Machine learning is a field of AI. Deep learning is a subset of ML.",
                "chunk_size": 100,
                "overlap": 20,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "indexed"

    def test_list_documents(self):
        resp = client.get("/documents/")
        assert resp.status_code == 200

    def test_document_stats(self):
        resp = client.get("/documents/stats")
        assert resp.status_code == 200

    def test_remove_document(self):
        # First add
        client.post(
            "/documents/",
            json={"doc_id": "to_delete", "text": "Temporary document for testing."},
        )
        resp = client.delete("/documents/to_delete")
        assert resp.status_code == 200


class TestQueryAPI:
    def test_fusion_query(self):
        resp = client.post(
            "/query/fusion",
            json={"query": "What is Transformer and how is it used?", "top_k": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "confidence" in data

    def test_semantic_search(self):
        resp = client.post(
            "/query/semantic",
            json={"query": "GPT language model", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data

    def test_keyword_search(self):
        resp = client.post(
            "/query/keyword",
            json={"query": "OpenAI", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
