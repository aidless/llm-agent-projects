# Archive — Deprecated Projects

This directory contains three projects that were **moved out of the main
portfolio** on 2026-07-22 after an honest audit revealed they would mislead
reviewers or hiring managers. The original code is preserved for reference,
but each project has a prominent ⚠ DEPRECATED banner at the top of its
README explaining what is real and what is not.

## Projects

| Project | Why archived | Estimated effort to revive |
|---|---|---|
| [`08-smart-customer-service`](./08-smart-customer-service/) | Retrieval is Jaccard token-overlap, not real semantic search; the "agent" framing is misleading | 2–3 h (swap in real embeddings) |
| [`10-inference-optimization`](./10-inference-optimization/) | The "simulator" is hardcoded math formulas; no `torch`, no `vllm`, no `sglang`. Useful teaching material; not a real inference system | Not recommended — point to upstream vLLM/SGLang |
| [`13-paper-review-agent`](./13-paper-review-agent/) | `BaseReviewAgent._call_llm` always returns hardcoded dicts; the README's "GPT-4o + DeepSeek cross-validation" is fictional | 4–8 h (replace mock with real LLM calls + add real evaluation harness) |

## Audit methodology

The deprecation decision was based on a 2026-07-22 code review (Explore-1,
sampling each project's `main.py`, README, core services, and tests):

- If a project's README claimed functionality that the code did not actually
  implement, it was either fixed in place (e.g., `12` BERTScore) or archived
  (e.g., `13` paper review).
- If a project's "agent" or "service" framing was misleading about the
  underlying technical depth (e.g., `08` Jaccard), it was archived.
- If a project was a faithful educational simulator without claiming to be
  production infrastructure (e.g., `10` vLLM teaching simulator), it was
  archived but the README's "this is a simulator" disclaimer was promoted to
  the top.

## Why keep them at all?

Three reasons:

1. **Historical honesty** — pretending these projects never existed would
   misrepresent the work that went into the collection. Future readers can
   see the original code in git history if curious.
2. **Reusable scaffolding** — the FastAPI structure, Pydantic schemas, and
   test patterns in each archived project are still worth studying.
3. **Recovery path documented** — each archived README ends with an
   "Estimated effort to revive" section so future work knows the cost.