# LLM Agent Projects

[![CI](https://github.com/aidless/llm-agent-projects/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/llm-agent-projects/actions/workflows/ci.yml)
[![offline matrix](https://img.shields.io/badge/offline%20matrix-results%2Fstatus--matrix.json-blue)](./results/status-matrix.json)
[![archived 3](https://img.shields.io/badge/archived-3-yellow)](./archive/README.md)

Independent scaffolds for LLM and agent engineering: 17 numbered projects stay on disk at the repository root (`01`–`07`, `09`, `11`, `12`, `14`–`20`), and `08` / `10` / `13` sit in [`archive/`](./archive/) with deprecation banners. Each directory is its own FastAPI-style tree. They were not built to one engineering standard.
Machine-checked claims are the rows [`make check`](./Makefile) writes into [`results/status-matrix.json`](./results/status-matrix.json): per active project, offline pytest with mock providers and no API key, plus the tier copied into the table below. [`scripts/drift_check.py`](./scripts/drift_check.py) fails when that table and the JSON disagree, and when a `Strong` row is not backed by an exit-0 run with zero skips.
[`SYSTEM_MAP.md`](./SYSTEM_MAP.md), per-project READMEs, and [`demo_pipeline.py`](./demo_pipeline.py) are narrative. This check does not execute a live model, and it does not certify them.

**STATUS: WORK-IN-PROGRESS — not A yet**

- GitHub About is a repository setting this change cannot write. The observed text and a replacement are in [Author and About](#author-and-about).
- The old "Strong" list is not what this matrix measured. `06-lora-finetune` is `Unverified` because tests import `datasets`, `peft`, `transformers`, and `torch`, and `requirements-check.txt` does not install those. `20-llm-eval-benchmark` is `Default-mock` because `app/api/benchmark.py` builds `MockLLMClient` on the evaluate route. `05-vector-db-manager` is `Unverified` because `embeddings/manager.py` imports `sentence_transformers`.
- `12-prompt-eval-framework` and `16-kg-rag-fusion` are in the table. `16` can be `Strong` under the suite rule while its default NER provider stays the regex dictionary; that limit is in the row notes.
- `make check` does not run `demo_pipeline.py`. Archived import results are in `archived_import` (`08` and `13` imported `app.main` on the refresh that wrote the JSON; `10` raised `ImportError`).
- Wall clock on the machine that wrote the JSON: venv plus `pip install -r requirements-check.txt` was 46 seconds with a warm wheel cache (2s + 44s), then the pytest refresh recorded as `duration_seconds`. A cold download of the same wheels was not timed separately. NLTK `punkt` was downloaded once in this session before that install timing. The pytest run itself is under a minute, inside the 15-minute fresh-clone budget.

## Status matrix

Tier rules applied by [`scripts/status_matrix.py`](./scripts/status_matrix.py) (the same function checks the JSON):

| Tier | Rule |
|---|---|
| Strong | `pytest` exit 0, `passed > 0`, `failed = 0`, `errors = 0`, `skipped = 0`, `mock-default = no` |
| Solid | exit 0 with the same failure counts, and `skipped > 0` |
| Default-mock | exit 0, and non-test code still constructs a mock or keyword stand-in on the default path |
| Unverified | the suite did not exit 0, timed out, or collected nothing |

`needs-key = yes` means a non-test `.py` file mentions `API_KEY` or `api_key`. The check deletes key, token, secret, and password variables from the child environment before pytest starts. `run = passed + skipped + failed + errors`.

<!-- STATUS-MATRIX:BEGIN -->
| project | tier | run | passed | skipped | failed | errors | needs-key | mock-default | notes |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| 01-rag-knowledge-base | Strong | 30 | 30 | 0 | 0 | 0 | yes | no |  |
| 02-agent-workflow | Solid | 37 | 33 | 4 | 0 | 0 | yes | no |  |
| 03-multi-model-gateway | Strong | 30 | 30 | 0 | 0 | 0 | yes | no |  |
| 04-guardrails-chat | Strong | 67 | 67 | 0 | 0 | 0 | yes | no |  |
| 05-vector-db-manager | Unverified | 16 | 13 | 0 | 2 | 1 | yes | no | failed_node=tests/test_services.py::TestEmbeddingManager::test_list_empty_models; missing=sentence_transformers |
| 06-lora-finetune | Unverified | 110 | 91 | 7 | 12 | 0 | yes | no | failed_node=tests/test_data.py::TestDatasetBuilder::test_build_hf_dataset; missing=datasets,peft,torch,transformers |
| 07-code-review-agent | Strong | 73 | 73 | 0 | 0 | 0 | yes | no |  |
| 09-multimodal-doc-understanding | Default-mock | 68 | 67 | 1 | 0 | 0 | yes | yes | mock:app/main.py:36 (+7) |
| 11-dify-workflow | Default-mock | 89 | 89 | 0 | 0 | 0 | no | yes | mock:nodes/llm_node.py:48 |
| 12-prompt-eval-framework | Solid | 78 | 75 | 3 | 0 | 0 | no | no | semantic_similarity is not Zhang BERTScore; SKIP_HF_DOWNLOAD=1 |
| 14-mcp-tool-integration | Strong | 92 | 92 | 0 | 0 | 0 | no | no |  |
| 15-agent-security-sandbox | Strong | 106 | 106 | 0 | 0 | 0 | no | no |  |
| 16-kg-rag-fusion | Strong | 98 | 98 | 0 | 0 | 0 | no | no | default NER=regex; spaCy excluded by requirements-check.txt |
| 17-llm-observability | Strong | 89 | 89 | 0 | 0 | 0 | no | no |  |
| 18-agent-long-term-memory | Default-mock | 77 | 77 | 0 | 0 | 0 | yes | yes | mock:app/main.py:32 (+3) |
| 19-voice-ai-assistant | Default-mock | 82 | 82 | 0 | 0 | 0 | yes | yes | mock:app/main.py:30 (+4) |
| 20-llm-eval-benchmark | Default-mock | 105 | 105 | 0 | 0 | 0 | no | yes | mock:app/api/benchmark.py:90 |
<!-- STATUS-MATRIX:END -->

## How to run the check

Fresh clone, from the repository root. `make check-install` is the networked step. `make check` is offline and must finish in the same environment that produced the committed matrix (a clean venv from `requirements-check.txt`).

```bash
python3 -m venv .venv
. .venv/bin/activate
make check-install
make check
```

`make check-refresh` overwrites `results/status-matrix.json`, `results/status-matrix.md`, and the table above. CI runs `make check` (compare only) in [`.github/workflows/ci.yml`](./.github/workflows/ci.yml) job `offline-check`.

The older `verify` job in that workflow still compares raw pytest exit codes to [`results/baseline.txt`](./results/baseline.txt). That file was recorded on a machine without per-project dependencies. It is a legacy ratchet, not this tier table.

`make eval` is a separate 20-case scorer fixture for `20-llm-eval-benchmark` (`eval/cases.jsonl`). It does not grade the other projects. Project 20's sample benchmarks, as named in its own code, are MMLU/GSM8K/HumanEval/MT-Bench · Wilson CI · McNemar.

## FAILURE_MODES

### Mock masking real failures

`make check` exports `ALLOW_MOCK_PROVIDERS=true` and does not set an API key. A `Default-mock` row passed while the default path still returns a canned string (vision, ASR/TTS, keyword memory consolidation, an LLM node, or `MockLLMClient` on an evaluate route). That run will not show a bad prompt, a downed API, or a wrong tool call. The `notes` cell points at the first matching source line. Passing tests under the mock flag is the measurement; it is not evidence that the opt-in provider path works.

### Dependency rot across 17 projects

Each active project ships its own `requirements.txt`, and the pins disagree (for example `18-agent-long-term-memory` pins `pytest==7.4.3` while `20-llm-eval-benchmark` pins `fastapi==0.115.0` and `numpy==2.1.0`). `make check` installs the single file `requirements-check.txt`. A green row means the suite ran on that shared set. It does not mean `pip install -r <project>/requirements.txt` still resolves, and it does not mean the excluded heavy stacks import.

### Archived code still importable

`08-smart-customer-service`, `10-inference-optimization`, and `13-paper-review-agent` are under `archive/` and are excluded from the tier table. Hyphenated directory names are not Python packages, so a root `PYTHONPATH` does not import them. Putting an archived directory on `sys.path` still reaches `app.main` when that tree's imports resolve. `archived_import` in `results/status-matrix.json` is the probe from the last `make check-refresh` (`ok app.main`, or the exception type). Archived code is kept on purpose and is unmaintained; it can still be executed.

## Running one project

```bash
cd 20-llm-eval-benchmark
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Dependencies and commands differ per directory. Read that directory's README before treating a scaffold as something you would deploy. Where a README names a real provider, the switch is an environment variable and the default path is the one in the matrix `notes`.

## Author and About

Repository owner path: [github.com/aidless/llm-agent-projects](https://github.com/aidless/llm-agent-projects). The display name "Liu Zewen" is the string committed in earlier revisions of this file. This check does not verify a legal identity.

GitHub About (observed, not edited here): `Portfolio of LLM/agent project scaffolds — RAG, multi-agent, safety, eval — tiered by quality, three archived after an honest audit`

Replacement text, so About matches this README:

`Independent LLM/agent scaffolds: 17 numbered projects, 3 archived. Tiers come from make check (offline pytest, mock providers, no API keys). Not a single production system.`

## License

Each project directory carries its own MIT `LICENSE`. The collection is MIT as well. See [`LICENSE`](./LICENSE).
