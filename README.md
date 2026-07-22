# LLM Agent Projects — Portfolio Collection

A curated collection of 20 production-grade LLM / AI Agent projects,
spanning RAG, multi-agent orchestration, inference infrastructure, safety
governance, evaluation, memory, fine-tuning, multimodal and voice.

Each project is an independent, self-contained FastAPI service with a
Docker setup, test suite, and a dedicated README. Pick one that matches
your interest and follow its README to run.

## Quick Index

| # | Project | Domain | Highlights |
|---|---------|--------|------------|
| 01 | [rag-knowledge-base](./01-rag-knowledge-base) | RAG | ChromaDB · BGE · BM25+vector RRF · cross-encoder rerank · citation |
| 02 | [agent-workflow](./02-agent-workflow) | Agent | LangGraph · Planner/Executor/Reviewer/Summarizer · dual-layer memory |
| 03 | [multi-model-gateway](./03-multi-model-gateway) | Infra | 4 routing strategies · auto-fallback · multi-key rotation |
| 04 | [guardrails-chat](./04-guardrails-chat) | Safety | prompt-injection detection · 7-class sensitive filter · hallucination detection |
| 05 | [vector-db-manager](./05-vector-db-manager) | RAG | web UI · model swap · chunking comparison |
| 06 | [lora-finetune](./06-lora-finetune) | Training | LoRA / QLoRA · full SFT pipeline · eval · export |
| 07 | [code-review-agent](./07-code-review-agent) | Agent | 5 specialised agents · AST analysis · GitHub PR integration |
| 08 | [smart-customer-service](./08-smart-customer-service) | Agent | RAG + human transfer |
| 09 | [multimodal-doc-understanding](./09-multimodal-doc-understanding) | Multimodal | OCR · table/layout · VQA · RAG |
| 10 | [inference-optimization](./10-inference-optimization) | Infra | continuous batching · PagedAttention · KV cache · FP/INT quant · rate-limit / circuit breaker |
| 11 | [dify-workflow](./11-dify-workflow) | Agent | DAG engine · 8 node types · templates |
| 12 | [prompt-eval-framework](./12-prompt-eval-framework) | Eval | BLEU/ROUGE/BERTScore · LLM-as-Judge · A/B (t + bootstrap) |
| 13 | [paper-review-agent](./13-paper-review-agent) | Agent | 5-dim review · cross-model validation · MetaReviewer |
| 14 | [mcp-tool-integration](./14-mcp-tool-integration) | Protocol | MCP server/client · JSON-RPC 2.0 · registry · chain call |
| 15 | [agent-security-sandbox](./15-agent-security-sandbox) | Safety | RestrictedPython · 4 policy levels · syscall interception · audit |
| 16 | [kg-rag-fusion](./16-kg-rag-fusion) | RAG | NetworkX KG · multi-hop · RRF fusion |
| 17 | [llm-observability](./17-llm-observability) | Infra | OpenTelemetry-style · trace/span/metric/log · cost analysis |
| 18 | [agent-long-term-memory](./18-agent-long-term-memory) | Memory | semantic/episodic/procedural/working · time-decay · importance |
| 19 | [voice-ai-assistant](./19-voice-ai-assistant) | Voice | ASR/LLM/TTS pipeline · WebSocket · VAD · barge-in |
| 20 | [llm-eval-benchmark](./20-llm-eval-benchmark) | Eval | MMLU/GSM8K/HumanEval/MT-Bench · Wilson CI · McNemar |

## Domains at a Glance

- **RAG & Retrieval (4)** — 01, 05, 09, 16
- **Agent & Multi-agent (5)** — 02, 07, 08, 11, 13
- **Inference Infrastructure (3)** — 03, 10, 17
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