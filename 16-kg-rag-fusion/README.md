# KG-RAG Fusion System

A complete Knowledge Graph + RAG Fusion system that combines the structured reasoning capability of knowledge graphs with the semantic retrieval capability of RAG (Retrieval-Augmented Generation).

## Architecture

```
                    ┌─────────────────┐
                    │   User Query    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Entity Linker │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
     ┌────────▼────────┐          ┌─────────▼───────┐
     │  KG Subgraph    │          │  RAG Retriever  │
     │  Extraction     │          │  (TF-IDF + KW)  │
     └────────┬────────┘          └─────────┬───────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │ Result Fusion   │
                    │   (RRF)         │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Answer Generator│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    Response     │
                    └─────────────────┘
```

## Features

### Knowledge Graph
- Entity extraction (NER) using rules + LLM mock
- Relation extraction using LLM mock
- Entity disambiguation and merging
- Graph storage using NetworkX DiGraph
- Multi-hop reasoning (1-3 hops)
- Shortest path finding
- BFS/DFS graph traversal
- Subgraph extraction
- Persistence (JSON / GraphML)

### RAG
- Document chunking with configurable size and overlap
- TF-IDF vectorization (scikit-learn)
- Semantic search (cosine similarity)
- Keyword search (BM25-like scoring)
- Hybrid retrieval with Reciprocal Rank Fusion (RRF)

### KG-RAG Fusion
- Entity linking (query -> graph entities)
- Graph-enhanced context (subgraph extraction)
- Result fusion (KG facts + RAG chunks)
- Answer generation with source attribution

### API
- RESTful API built with FastAPI
- Interactive docs at `/docs`
- Knowledge graph CRUD operations
- Document management
- Fusion query endpoint
- Graph visualization data

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

### Run tests

```bash
pytest tests/ -v
```

## Pre-loaded Data

The system comes with a pre-loaded knowledge graph of the AI/technology domain containing:
- 38 entities (technologies, organizations, people, concepts)
- 55 relations (developed_by, uses, applied_in, belongs_to, etc.)

## API Endpoints

### Health
- `GET /` - System info
- `GET /health` - Health check

### Knowledge Graph
- `POST /graph/entities` - Create entity
- `GET /graph/entities` - List entities
- `GET /graph/entities/{id}` - Get entity
- `PUT /graph/entities/{id}` - Update entity
- `DELETE /graph/entities/{id}` - Delete entity
- `GET /graph/entities/search/{keyword}` - Search by name
- `POST /graph/relations` - Create relation
- `GET /graph/relations` - List relations
- `DELETE /graph/relations` - Delete relation
- `POST /graph/query/multi-hop` - Multi-hop query
- `POST /graph/query/path` - Find path
- `GET /graph/query/traverse/bfs/{id}` - BFS traversal
- `GET /graph/query/traverse/dfs/{id}` - DFS traversal
- `POST /graph/extract` - Extract from text
- `GET /graph/visualization` - Graph visualization data
- `GET /graph/stats` - Graph statistics

### Documents
- `POST /documents/` - Index document
- `GET /documents/` - List documents
- `DELETE /documents/{id}` - Remove document
- `GET /documents/stats` - Index stats

### Query
- `POST /query/fusion` - KG-RAG fusion query
- `POST /query/semantic` - Semantic search
- `POST /query/keyword` - Keyword search

## Docker

```bash
docker-compose up --build
```

## Tech Stack

- Python 3.11+
- FastAPI - Web framework
- NetworkX - Graph database (in-memory)
- scikit-learn - TF-IDF vectorization
- Pydantic - Data validation
