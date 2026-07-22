# ⚠️ DEPRECATED — Paper Review Agent (2026-07-22)

This project has been **moved to `archive/`** because its core "LLM" calls are
**not real**. The README originally claimed *cross-model validation with
GPT-4o + DeepSeek*, but the actual implementation in `agents/base.py`
(`_call_llm`) always returns a hardcoded dictionary from `_default_response()` —
no `openai`, `anthropic`, or `httpx` call is ever made.

This project is **kept for reference only** (LangGraph orchestration scaffolding
is real and worth studying). For a production paper-review pipeline see
[`aidless/reviewer-sim`](https://github.com/aidless/reviewer-sim) instead.

---

# Original README (preserved for reference)

# Multi-Agent Paper Review System

A multi-agent system for academic paper review, built with FastAPI and LangGraph.
5 specialised agents collaborate to review papers across 4 dimensions with cross-model validation.

> ⚠️ **Reality check (2026-07-22 audit)**: The "cross-model validation with GPT-4o + DeepSeek"
> described above is **not implemented**. All 5 agents return hardcoded default
> dictionaries from `BaseReviewAgent._call_llm`. The LangGraph orchestration
> is real; the LLM content is not.

## Architecture

```
Paper Text --> Parser --> [RelevanceAgent, MethodologyAgent, NoveltyAgent, ClarityAgent]
                                (parallel, each with 2 models for cross-validation)
                                     |
                               ConsensusEngine (agreement check + arbitration)
                                     |
                               MetaReviewerAgent --> Structured Report
```

## Why it's archived

- **Mock LLM responses** — every agent returns canned dicts; the README's
  "GPT-4o + DeepSeek cross-validation" is fictional.
- **No real evaluation** — no `results.md`, no benchmark numbers, no comparison
  against human reviews.
- **Better alternative exists** — the user's private `aidless/reviewer-sim`
  repo has a working 6-agent pipeline.

## Recommended fix (if you want to revive)

1. Replace `agents/base.py:BaseReviewAgent._call_llm` with a real call to
   `langchain_openai.ChatOpenAI` (or Anthropic).
2. Add `cross_validation/validator.py` that genuinely samples two different
   models per dimension.
3. Wire up a small eval harness (50 papers + ground-truth scores from
   OpenReview) and report Pearson / Cohen's κ against human reviewers.
4. Update README to remove all "GPT-4o / DeepSeek" claims until they are real.

Estimated effort: 4–8 hours.