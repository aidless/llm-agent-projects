# Multi-Agent Paper Review System

A multi-agent system for academic paper review, built with FastAPI and LangGraph.
5 specialised agents collaborate to review papers across 4 dimensions with cross-model validation.

## Architecture

```
Paper Text --> Parser --> [RelevanceAgent, MethodologyAgent, NoveltyAgent, ClarityAgent]
                                (parallel, each with 2 models for cross-validation)
                                     |
                               ConsensusEngine (agreement check + arbitration)
                                     |
                               MetaReviewerAgent --> Structured Report
```

## Agents

| Agent | Dimension | Description |
|-------|-----------|-------------|
| RelevanceAgent | Relevance | Topic match with target venue |
| MethodologyAgent | Methodology | Experiment design, baselines, ablation |
| NoveltyAgent | Novelty | Innovation and contribution |
| ClarityAgent | Clarity | Structure, figures, writing quality |
| MetaReviewerAgent | Meta | Synthesises all reviews into final recommendation |

## Quick Start

```bash
pip install -r requirements.txt
pytest tests/ -v
uvicorn app.main:app --reload
```

## API Endpoints

- `POST /api/v1/review` -- Submit paper for review
- `GET /api/v1/report/sample` -- Get sample report
- `POST /api/v1/report/render` -- Render a report to markdown/JSON
- `GET /health` -- Health check

## Cross-Model Validation

Each dimension is reviewed by two models independently (e.g. GPT-4o and DeepSeek).
The ConsensusEngine computes agreement scores and arbitrates disagreements using
confidence-weighted averaging.

## Recommendation Scale

| Score Range | Recommendation |
|-------------|----------------|
| 8.5 - 10.0 | Strong Accept |
| 7.0 - 8.5 | Accept |
| 5.5 - 7.0 | Weak Accept |
| 4.5 - 5.5 | Borderline |
| 3.0 - 4.5 | Weak Reject |
| 1.5 - 3.0 | Reject |
| 1.0 - 1.5 | Strong Reject |

## Docker

```bash
docker-compose up --build
```

## Project Structure

```
app/            -- FastAPI app, models, API routes
agents/         -- 5 specialised review agents
cross_validation/ -- Cross-model validator & consensus engine
parser/         -- Paper text parser & structure extractor
workflows/      -- LangGraph StateGraph orchestration
templates/      -- Report templates & review criteria
tests/          -- Comprehensive test suite (30+ tests)
```
