# LLM Agent Projects — System Map

> How the 17 active projects interconnect — a narrative map, not a runtime that `make check` executes.

## Architecture Layers

```
┌────────────────────────────────────────────────────────┐
│                    EVALUATION LAYER                     │
│  12-prompt-eval-framework    20-llm-eval-benchmark     │
│  (prompt quality, A/B test)  (MMLU/GSM8K/HumanEval)   │
└──────────────────────────┬─────────────────────────────┘
                           │ measures
┌──────────────────────────▼─────────────────────────────┐
│                   APPLICATION LAYER                     │
│  01-rag-knowledge-base    16-kg-rag-fusion             │
│  09-multimodal-doc         07-code-review-agent        │
│  19-voice-ai-assistant     11-dify-workflow            │
└──────────────────────────┬─────────────────────────────┘
                           │ uses
┌──────────────────────────▼─────────────────────────────┐
│                    AGENT LAYER                          │
│  02-agent-workflow       18-agent-long-term-memory     │
│  14-mcp-tool-integration                               │
└──────────────────────────┬─────────────────────────────┘
                           │ routes through
┌──────────────────────────▼─────────────────────────────┐
│                 INFRASTRUCTURE LAYER                    │
│  03-multi-model-gateway  17-llm-observability          │
│  05-vector-db-manager                                  │
└──────────────────────────┬─────────────────────────────┘
                           │ secured by
┌──────────────────────────▼─────────────────────────────┐
│                   SAFETY LAYER                          │
│  04-guardrails-chat      15-agent-security-sandbox     │
└────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│                  TRAINING LAYER                         │
│  06-lora-finetune (custom model improvement)           │
└────────────────────────────────────────────────────────┘
```

## End-to-End Pipeline (recommended demo)

```
User Query
    │
    ▼
[03 Gateway] ──► Route to best model, apply rate limiting
    │
    ▼
[04 Guardrails] ──► Input safety: prompt injection check, sensitive content filter
    │
    ▼
[02 Agent Workflow] ──► Planner → Executor → Reviewer → Summarizer
    │                       │
    │              ┌────────┴────────┐
    │              ▼                 ▼
    │    [01 RAG Knowledge]  [16 KG Fusion]
    │    (vector retrieval)  (entity enrichment)
    │              │                 │
    │              └────────┬────────┘
    │                       ▼
    │              [18 Memory] ──► long-term context
    │
    ▼
[04 Guardrails] ──► Output safety: hallucination detection
    │
    ▼
[17 Observability] ──► Trace spans, latency, token cost
    │
    ▼
[12 Prompt Eval] or [20 Benchmark] ──► Measure quality
```

## Quick Pipelines by Use Case

| Use Case | Pipeline |
|---|---|
| **Enterprise RAG Chat** | 03 → 04 → 01 + 16 → 04 → 17 |
| **Code Review Bot** | 03 → 04 → 07 → 17 |
| **Safe Agent Sandbox** | 03 → 04 → 15 → 17 |
| **Multimodal Doc QA** | 03 → 09 → 01 → 04 → 17 |
| **Voice Assistant** | 19 ASR → 03 LLM → 19 TTS, with 04 guardrails |
| **Custom Model Eval** | 06 train → 20 benchmark → 12 prompt optimize |
| **Dify-style Workflow** | 11 nodes powered by 03 gateway, monitored by 17 |

## Dependency Graph

```
06 (training)     ──► generates adapter for ──► 03 (gateway)
03 (gateway)      ──► routes to ──► 01, 02, 07, 09, 11, 16, 19
04 (guardrails)   ──► wraps ──► 01, 02, 07, 09, 11, 19
17 (observability)──► instruments ──► all application-layer projects
14 (MCP)          ──► provides tools for ──► 02, 07, 11
18 (memory)       ──► backs ──► 02, 07
05 (vector-db)    ──► manages storage for ──► 01, 16
12, 20 (eval)     ──► benchmarks ──► 01, 02, 03, 06, 07
```

## Project quality tiers

Tiers are the offline matrix in [`results/status-matrix.json`](./results/status-matrix.json), copied into the root README. This file does not keep a second tier list. A diagram above is a narrative of how the scaffolds could be talked about together. `make check` does not boot that diagram.

## Interview talking points

1. **Breadth**: 17 active directories plus 3 archived ones, covering RAG, agents, gateways, safety, eval, memory, fine-tuning, multimodal, and voice.
2. **What was measured**: `make check` runs each active project's `tests/` with mock providers and no API key. The tier names the outcome of that run.
3. **Honesty**: archived projects state why they were moved; default-path mocks are a column in the matrix.
4. **Security notes in project READMEs** (SSRF, auth, mock guards) are code comments and tests from 2026-07-22. They are not a fresh penetration test.
5. **Composition**: `demo_pipeline.py` is a script in this repo. The matrix does not score it.
