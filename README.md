# LLM Agent Projects — Portfolio Collection

> **⚠️ Important context for hiring managers and reviewers (2026-07-22)**
>
> This is a **portfolio collection**, not a unified production codebase. The
> 17 numbered projects (`01`–`20`, with `08`/`10`/`13` archived — see below)
> were generated as independent scaffolds to demonstrate breadth across the
> LLM/Agent engineering space. They were **not** all built to the same
> engineering standard. Specifically:
>
> - **Strong (recommended review priority)**: `01`, `06`, `15`, `20`, `17` —
>   real implementations end-to-end.
> - **Solid scaffolding**: `02`, `03`, `04`, `05`, `07`, `11`, `14` — useful
>   reference code, but read with a critical eye.
> - **Archived (moved to `archive/`)**: `08` (Jaccard-only FAQ),
>   `10` (pure simulator), `13` (mocked LLM calls). These have prominent
>   ⚠ DEPRECATED banners explaining why.
> - **Default-Mock components**: `09`, `18`, `19` ship with mock vision/ASR/TTS/
>   memory-consolidator by default; READMEs call these out at the top.
>
> Every project has its own README with project-specific notes. **Read each
> README's top section before evaluating the code.**

---

A curated collection of 20 production-grade LLM / AI Agent projects,
spanning RAG, multi-agent orchestration, inference infrastructure, safety
governance, evaluation, memory, fine-tuning, multimodal and voice.

Each project is an independent, self-contained FastAPI service with a
Docker setup, test suite, and a dedicated README. Pick one that matches
your interest and follow its README to run.

## Quick Index

| # | Project | Domain | Highlights | Status |
|---|---------|--------|------------|--------|
| 01 | [rag-knowledge-base](./01-rag-knowledge-base) | RAG | ChromaDB · BGE · BM25+vector RRF · cross-encoder rerank · citation | ✅ Strong |
| 02 | [agent-workflow](./02-agent-workflow) | Agent | LangGraph · Planner/Executor/Reviewer/Summarizer · dual-layer memory | ⚠️ SSRF fix in 2026-07-22 |
| 03 | [multi-model-gateway](./03-multi-model-gateway) | Infra | 4 routing strategies · auto-fallback · multi-key rotation | ✅ Solid |
| 04 | [guardrails-chat](./04-guardrails-chat) | Safety | prompt-injection detection · 7-class sensitive filter · hallucination detection | ✅ Solid |
| 05 | [vector-db-manager](./05-vector-db-manager) | RAG | web UI · model swap · chunking comparison | ✅ Solid |
| 06 | [lora-finetune](./06-lora-finetune) | Training | LoRA / QLoRA · full SFT pipeline · eval · export | ✅ Strong |
| 07 | [code-review-agent](./07-code-review-agent) | Agent | 5 specialised agents · AST analysis · GitHub PR integration | ✅ Solid |
| — | [08-smart-customer-service](./archive/08-smart-customer-service) | Agent | (originally: RAG + human transfer) | ⚠️ **DEPRECATED** — Jaccard-only |
| 09 | [multimodal-doc-understanding](./09-multimodal-doc-understanding) | Multimodal | OCR · table/layout · VQA · RAG | ⚠️ Default Mock vision client |
| — | [10-inference-optimization](./archive/10-inference-optimization) | Infra | (originally: continuous batching · PagedAttention) | ⚠️ **DEPRECATED** — pure simulator |
| 11 | [dify-workflow](./11-dify-workflow) | Agent | DAG engine · 8 node types · templates | ✅ Solid (LLMNode defaults to Mock) |
| 12 | [prompt-eval-framework](./12-prompt-eval-framework) | Eval | BLEU/ROUGE/semantic-sim · LLM-as-Judge · A/B (t + bootstrap) | ✅ Fixed (2026-07-22 BERTScore → sentence-transformers) |
| — | [13-paper-review-agent](./archive/13-paper-review-agent) | Agent | (originally: 5-dim review · cross-model validation) | ⚠️ **DEPRECATED** — all-LLM-mock |
| 14 | [mcp-tool-integration](./14-mcp-tool-integration) | Protocol | MCP server/client · JSON-RPC 2.0 · registry · chain call | ✅ Solid |
| 15 | [agent-security-sandbox](./15-agent-security-sandbox) | Safety | RestrictedPython · 4 policy levels · syscall interception · audit · **Bearer auth (2026-07-22)** | ✅ Strong |
| 16 | [kg-rag-fusion](./16-kg-rag-fusion) | RAG | NetworkX KG · multi-hop · RRF fusion | ⚠️ NER is regex |
| 17 | [llm-observability](./17-llm-observability) | Infra | OpenTelemetry-style · trace/span/metric/log · cost analysis | ✅ Strong |
| 18 | [agent-long-term-memory](./18-agent-long-term-memory) | Memory | semantic/episodic/procedural/working · time-decay · importance | ⚠️ TF-IDF + Mock consolidator |
| 19 | [voice-ai-assistant](./19-voice-ai-assistant) | Voice | ASR/LLM/TTS pipeline · WebSocket · VAD · barge-in | ⚠️ Default Mock ASR/TTS |
| 20 | [llm-eval-benchmark](./20-llm-eval-benchmark) | Eval | MMLU/GSM8K/HumanEval/MT-Bench · Wilson CI · McNemar | ✅ Strong |

### Archive

Three projects have been moved to [`archive/`](./archive/) with prominent
deprecation notices:

| Project | Why archived |
|---|---|
| `08-smart-customer-service` | Retrieval is Jaccard similarity + substring scoring, not a real LLM agent |
| `10-inference-optimization` | Pure mathematical simulator with hardcoded latency formulas; no real GPU code |
| `13-paper-review-agent` | All LLM calls return hardcoded dictionaries; the README's "GPT-4o + DeepSeek cross-validation" was fictional |

See [`archive/README.md`](./archive/README.md) for details.

## Domains at a Glance

- **RAG & Retrieval (4)** — 01, 05, 09, 16
- **Agent & Multi-agent (5)** — 02, 07, ~~08~~, 11, ~~13~~
- **Inference Infrastructure (3)** — 03, ~~10~~, 17
- **Safety & Governance (3)** — 04, 14, 15
- **Evaluation (2)** — 12, 20
- **Training / Memory / Voice (3)** — 06, 18, 19
- **Safety & Governance (3)** — 04, 14, 15
- **Evaluation & Benchmarking (2)** — 12, 20
- **Training / Memory / Voice (3)** — 06, 18, 19

## Running

Each project is independent. See `XX-name/README.md` for setup. Most follow
this pattern:

```bash
cd XX-name
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
# or
docker-compose up --build
```

## Tech Stack (common)

Python 3.11+ · FastAPI · LangChain / LangGraph · ChromaDB · SentenceTransformers · OpenAI / DeepSeek / Qwen / Anthropic · Docker · pytest

## License

Each project is released under the MIT License — see the project-level
`LICENSE` file. The collection as a whole is also MIT.

## Author

[@aidless](https://github.com/aidless) — Liu Zewen