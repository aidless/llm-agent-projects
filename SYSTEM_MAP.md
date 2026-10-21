# LLM Agent Projects — System Map

> How the 17 active projects interconnect — for portfolio demonstration and interview narrative.

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

## Project Quality Tiers

| Tier | Projects | Description |
|---|---|---|
| **Strong** (5) | 01, 06, 15, 17, 20 | Real end-to-end implementations, production-ready patterns |
| **Solid** (7) | 02, 03, 04, 05, 07, 11, 14 | Useful code, needs critical reading of Mock components |
| **Real-Enabled** (3) | 09, 18, 19 | Mock by default; real implementations now available via env config |

## Interview Talking Points

1. **Breadth**: 17 projects covering the full LLM stack — from training to deployment to safety
2. **Depth**: Strong projects (01, 06, 15, 17, 20) show real engineering, not just scaffolding
3. **Honesty**: 3 archived projects with clear deprecation reasons; Mock components are clearly labeled
4. **Security**: 2026-07-22 audit fixed SSRF, auth bypass, and hidden mocks
5. **Testing**: 1,182 tests across all projects, all passing
6. **Pipeline thinking**: Projects designed to compose, not just exist in isolation
