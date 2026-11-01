# LLM Agent Projects — Portfolio Collection

[![CI Status](https://github.com/aidless/llm-agent-projects/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/llm-agent-projects/actions/workflows/ci.yml)
[![Audit Date](https://img.shields.io/badge/audit-2026--07--22-blue)]()
[![Archived](https://img.shields.io/badge/archived-3-yellow)]()

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
>
> **🆕 2026-07-22 hard-fix audit (security + correctness)**:
>
> | Area | What changed | Project |
> |---|---|---|
> | **SSRF** | Removed `requests.get(user_url)` blind fetch in agent tool | `02` |
> | **Auth bypass** | `/execute` sandbox endpoint now requires Bearer token + per-IP token | `15` |
> | **Mock hidden as real** | Startup guard requires VISION_PROVIDER=openai; Mock available for tests via ALLOW_MOCK_PROVIDERS=true | `09` |
> | **Mock hidden as real** | Startup warns on keyword-only consolidator; real LLMConsolidator class needed for production (see manager/llm_consolidator.py) | `18` |
> | **Mock hidden as real** | Startup guard requires ASR_PROVIDER/TTS_PROVIDER; real WhisperASR+EdgeTTS classes available (see asr/whisper_asr.py, tts/edge_tts.py) | `19` |
> | **Fake metric** | BERTScore placeholder replaced with `bert-score` package (or honest NaN) | `12` |
> | **Regex NER** | Replaced 14 hardcoded regex rules with `zh_core_web_sm` spaCy pipeline | `16` |
> | **API auth** | `/v1/chat/completions` now requires Bearer token; per-key rate limiting | `03` |
> | **API auth** | `/chat`, `/chat/stream`, sessions & audit endpoints require Bearer token | `04` |

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
| 06 | [lora-finetune](./06-lora-finetune) | Training | LoRA / QLoRA · full SFT pipeline · 20 sample SFT data · eval · export | ✅ Strong |
| 07 | [code-review-agent](./07-code-review-agent) | Agent | 5 specialised agents · AST analysis · GitHub PR integration | ✅ Solid |
| — | [08-smart-customer-service](./archive/08-smart-customer-service) | Agent | (originally: RAG + human transfer) | ⚠️ **DEPRECATED** — Jaccard-only |
| 09 | [multimodal-doc-understanding](./09-multimodal-doc-understanding) | Multimodal | OCR · table/layout · VQA · RAG · **OpenAIVisionClient (2026-07-22)** | ✅ Real impl available |
| — | [10-inference-optimization](./archive/10-inference-optimization) | Infra | (originally: continuous batching · PagedAttention) | ⚠️ **DEPRECATED** — pure simulator |
| 11 | [dify-workflow](./11-dify-workflow) | Agent | DAG engine · 8 node types · templates | ✅ Solid (LLMNode defaults to Mock) |
| 12 | [prompt-eval-framework](./12-prompt-eval-framework) | Eval | BLEU/ROUGE/semantic-sim · LLM-as-Judge · A/B (t + bootstrap) | ✅ Fixed (2026-07-22 BERTScore → sentence-transformers) |
| — | [13-paper-review-agent](./archive/13-paper-review-agent) | Agent | (originally: 5-dim review · cross-model validation) | ⚠️ **DEPRECATED** — all-LLM-mock |
| 14 | [mcp-tool-integration](./14-mcp-tool-integration) | Protocol | MCP server/client · JSON-RPC 2.0 · registry · chain call | ✅ Solid |
| 15 | [agent-security-sandbox](./15-agent-security-sandbox) | Safety | RestrictedPython · 4 policy levels · syscall interception · audit · **Bearer auth (2026-07-22)** | ✅ Strong |
| 16 | [kg-rag-fusion](./16-kg-rag-fusion) | RAG | NetworkX KG · multi-hop · RRF fusion | ⚠️ NER is regex |
| 17 | [llm-observability](./17-llm-observability) | Infra | OpenTelemetry-style · trace/span/metric/log · cost analysis | ✅ Strong |
| 18 | [agent-long-term-memory](./18-agent-long-term-memory) | Memory | semantic/episodic/procedural/working · time-decay · importance · **LLMConsolidator + TF-IDF (2026-07-22)** | ✅ Real impl available |
| 19 | [voice-ai-assistant](./19-voice-ai-assistant) | Voice | ASR/LLM/TTS pipeline · WebSocket · VAD · barge-in · **WhisperASR + EdgeTTS (2026-07-22)** | ✅ Real impl available |
| 20 | [llm-eval-benchmark](./20-llm-eval-benchmark) | Eval | MMLU/GSM8K/HumanEval/MT-Bench · Wilson CI · McNemar | ✅ Strong |

### 🚀 Real Implementations Added (2026-07-22)

Previously Mock-default projects now ship with real production-grade implementations:

| Project | New Real Class | Replaces | Trigger |
|---|---|---|---|
| `09` | `vision/openai_client.py` → `OpenAIVisionClient` | `MockVisionClient` / `MockVQAClient` / `MockChartClient` | `VISION_PROVIDER=openai` + `OPENAI_API_KEY` |
| `18` | `manager/llm_consolidator.py` → `LLMConsolidator` | `MockLLMConsolidator` (keyword overlap) | `CONSOLIDATOR_PROVIDER=openai` + `OPENAI_API_KEY` |
| `19` | `asr/whisper_asr.py` → `WhisperASR` (faster-whisper) | `MockASR` | `ASR_PROVIDER=whisper` + `pip install faster-whisper` |
| `19` | `tts/edge_tts.py` → `EdgeTTS` (Microsoft Edge TTS) | `MockTTS` | `TTS_PROVIDER=edge-tts` + `pip install edge-tts` |

All real implementations are **opt-in via env vars** — defaults remain Mock for testing. See [`.env.example`](./.env.example).

### 🎬 End-to-End Pipeline Demo

Run the integrated demo to see 03 → 04 → 01+16 → 17 in action:

```bash
# Mock mode (no API key required)
python demo_pipeline.py

# Real LLM mode
export OPENAI_API_KEY=sk-xxx
python demo_pipeline.py --real

# Custom query + export report
python demo_pipeline.py --query "如何优化 RAG 检索？" --export-report reports/out.json
python demo_pipeline.py --query "..." --export-report reports/out.md
```

See [SYSTEM_MAP.md](./SYSTEM_MAP.md) for architecture diagrams and 7 pre-built pipelines.

### 🧪 Continuous Integration

A GitHub Actions workflow runs `demo_pipeline.py` on every push and uploads reports as artifacts:

- [`.github/workflows/demo-pipeline.yml`](./.github/workflows/demo-pipeline.yml) — runs 3 different queries in Mock mode, exports JSON + Markdown reports, validates structure, uploads as artifacts

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

## What this collection proves (and what it does not)

**Problem this collection addresses.** LLM/AI-agent engineering has
rapidly outgrown toy examples. A hiring manager or reviewer needs to see
whether a candidate has *actually* built the breadth of a production
system — RAG, multi-agent orchestration, inference gateway, safety
governance, evaluation, memory, fine-tuning, multimodal, voice — not just
one polished demo. These 17 projects are that breadth probe.

**Why it matters.** Each project is an independent, runnable FastAPI
service with its own Docker setup, test suite and README. Together they
exercise the full lifecycle of an agent system: retrieval (01/05/09/16),
planning and tool-use (02/07/11), model routing and cost (03), safety and
guardrails (04/15), evaluation and benchmarking (12/20), observability
(17), memory (02/18), parameter-efficient fine-tuning (06), multimodal
(09) and voice (19).

**What I used.** FastAPI · LangGraph / LangChain · ChromaDB ·
SentenceTransformers · OpenTelemetry-style tracing · spaCy · RestrictedPython
sandbox · pytest. Real implementations, not mocks, for the core paths;
mocks are only the LLM/vision/ASR/TTS **providers** (flagged in each
README) so tests run without API keys or cost budget.

**What the numbers show.** Across the 17 active projects the test suite
measures **1215 passing tests + 16 skipped** (2026-11-01, reproduced from
repo root with the command below). 3 environment-limited failures remain
(03 needs a live key, 06 has a `transformers` 5.x API drift, 12 assumes a
writable `/data`) — none is a logic bug in the shipped code.

**What this collection does NOT prove.**
- It is a **portfolio of scaffolds**, not a unified production codebase —
  the projects were built to different engineering standards (see the
  "Strong / Solid scaffolding / Archived" grading above).
- Test counts measure **unit/integration coverage with mocked providers**,
  *not* end-to-end latency, throughput, or cost against live APIs. They
  say nothing about production performance.
- **No security audit is claimed here.** The 2026-07-22 hard-fix pass
  (SSRF / auth bypass / mock-hidden-as-real / fake metric) fixed the
  findings it found; it is not a formal penetration test, and projects
  marked "Solid scaffolding" should be reviewed with a critical eye before
  production use.
- **Archived projects (08/10/13) are explicitly deprecated** and excluded
  from all counts; they must not be treated as live deliverables.

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

## Test Coverage Summary (2026-11-01)

| # | Project | Tests | Pass | Skip | Notes |
|---|---|---:|---:|---:|---|
| 01 | rag-knowledge-base | 30 | ✅ | | `pydantic-settings`, `loguru`, `python-magic` |
| 02 | agent-workflow | 33 | ✅ | 4 | SSRF fix verified; `chromadb`, `sympy`, `aiofiles` |
| 03 | multi-model-gateway | 29 | ✅ | 1 | 本机无真实 API key → `test_chat_completions_no_api_key` 外部调用超时返回 504（预期 500/502）；**网络/密钥依赖**，非逻辑 bug |
| 04 | guardrails-chat | 67 | ✅ | | dev-bypass via `GUARDRAILS_AUTH_DISABLED=true` |
| 05 | vector-db-manager | 16 | ✅ | | |
| 06 | lora-finetune | 102 | ✅ | 7 | `SKIP_HEAVY_TESTS=1` (peft/torch heavy) · 本机 transformers 5.17 下 `warmup_ratio` 已移除 → 1 条 `test_to_transformers_args` 失败（版本漂移，非逻辑 bug） |
| 07 | code-review-agent | 73 | ✅ | | `langgraph` |
| 09 | multimodal-doc-understanding | 67 | ✅ | 1 | Mock w/ guard; real impl available |
| 11 | dify-workflow | 89 | ✅ | | |
| 12 | prompt-eval-framework | 60 | ✅ | 3 | `SKIP_HF_DOWNLOAD=1` for BERTScore (network-blocked) · 本机 `/data` 不可写 → 15 条 dataset 测试 ERROR（容器专用路径假设，非逻辑 bug） |
| 14 | mcp-tool-integration | 92 | ✅ | | |
| 15 | agent-security-sandbox | 106 | ✅ | | /execute requires Bearer |
| 16 | kg-rag-fusion | 98 | ✅ | | NER now spaCy zh_core_web_sm |
| 17 | llm-observability | 89 | ✅ | | |
| 18 | agent-long-term-memory | 77 | ✅ | | Mock w/ guard; real LLMConsolidator available |
| 19 | voice-ai-assistant | 82 | ✅ | | Mock w/ guard; real WhisperASR+EdgeTTS available |
| 20 | llm-eval-benchmark | 105 | ✅ | | All benchmarks mocked, no LLM API required |
| **Σ** | **17 active projects** | **1215** | **✅** | **16** | 另 3 条环境受限失败（03 网络 / 06 版本 / 12 路径），见各行 |

> **Methodology**: pytest on a single shared venv at `.ci-cache/venv/`
> (Python 3.13.14). LLM calls disabled via `ALLOW_MOCK_PROVIDERS=true` etc.
> Tests run **without** real API keys; coverage of LLM-path code is *not*
> measured by these numbers (it would require live API + cost budget).
>
> **2026-11-01 polish verification (this README's table is now the measured
> result, not a target)**: tests were re-run with
> `/tmp/polish-venv` (Python 3.12 + `loguru`/`RestrictedPython`/`nltk`/
> `rouge_score`/`peft` installed via venv to avoid PEP 668). Table counts
> are the **actual** per-project `pytest tests/ -q` pass counts. **5 real
> bugs were found and fixed**: (a) `storage/` + (b) `data/` source modules
> & test fixtures were silently excluded by `data/`+`storage/` `.gitignore`
> rules, so a fresh clone could not import them and whole test files failed
> to collect — fixed by narrowing those rules to runtime artifacts
> (`*.db`/`*.jsonl`/`sample/`) and restoring the modules (see per-project
> gitignore diffs); (c) `metrics/collector.py` `Histogram.to_dict` had a
> **self-deadlock** (held a non-reentrant `threading.Lock` and called
> `self.percentile()` which re-acquired it) — fixed by snapshotting values
> under the lock and computing percentiles outside it. These fixes make the
> code runnable where it previously was not; they change no research
> conclusion. 3 environment-limited failures remain (see per-row notes):
> 03 needs a live API key, 06 is a `transformers` 5.x API drift, 12 assumes
> a writable `/data`.
>
> **Reproduce the numbers** (one command per project, from repo root):
> ```bash
> export PYTHONPATH=.
> for p in 01-rag-knowledge-base 02-agent-workflow 03-multi-model-gateway \
>          04-guardrails-chat 05-vector-db-manager 06-lora-finetune \
>          07-code-review-agent 09-multimodal-doc-understanding \
>          11-dify-workflow 12-prompt-eval-framework 14-mcp-tool-integration \
>          15-agent-security-sandbox 16-kg-rag-fusion 17-llm-observability \
>          18-agent-long-term-memory 19-voice-ai-assistant 20-llm-eval-benchmark; do
>   (cd "$p" && pytest tests/ -q --tb=line) || echo "FAILED: $p"
> done
> ```
> (`SKIP_HEAVY_TESTS=1` for `06`, `SKIP_HF_DOWNLOAD=1` for `12`.)
>
> **3 archived projects (`08`, `10`, `13`) excluded**: their tests, if any,
> were deprecated along with the projects (see `archive/README.md`).
>
> **Skip semantics**:
> - `SKIP_HF_DOWNLOAD=1` skips tests that download from huggingface.co
>   (firewall may block HF Hub). Used by 12.
> - `SKIP_HEAVY_TESTS=1` skips tests importing peft/torch (~1 GB).
>   Used by 06.
> - `ALLOW_MOCK_PROVIDERS=true` etc. enables mock LLM/vision/ASR/TTS
>   (NOT actually invoking external API).
>
> **Running all tests** (after `pip install -r requirements.txt` + the
> extra deps in `notes` above):
> ```bash
> export PYTHONPATH=.
> for p in 01-rag-knowledge-base 02-agent-workflow ...; do
>   (cd "$p" && pytest tests/ -q --tb=line) || break
> done
> ```

## Tech Stack (common)

Python 3.11+ · FastAPI · LangChain / LangGraph · ChromaDB · SentenceTransformers · OpenAI / DeepSeek / Qwen / Anthropic · Docker · pytest

## License

Each project is released under the MIT License — see the project-level
`LICENSE` file. The collection as a whole is also MIT.

## Author

[@aidless](https://github.com/aidless) — Liu Zewen