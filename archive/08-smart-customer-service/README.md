# ⚠️ DEPRECATED — Smart Customer Service (2026-07-22)

This project has been **moved to `archive/`** because the retrieval mechanism
is **Jaccard similarity** (token set overlap) rather than true semantic
retrieval. It is a FAQ chatbot with substring scoring, not an "Agent" in the
LLM-era sense.

This project is **kept for reference only**. For a real customer-service bot
that uses embeddings, see [`01-rag-knowledge-base`](../../01-rag-knowledge-base/).

---

## What it actually is

- FAQ matching via Jaccard similarity + substring boost
- Fallback to hardcoded scripted flows
- Dialog state machine (real, but stateless)
- No embedding model, no LLM call, no real "agent" reasoning

## Why it's archived

- **Misleading naming** — "Smart Customer Service" suggests an LLM agent;
  in fact it's a string-similarity FAQ matcher.
- **Below portfolio bar** — doesn't demonstrate any modern AI skill.

## Recommended fix (if you want to revive)

Replace the `_compute_similarity` function in `core/` with a real
embedding-based retrieval (e.g., from `01-rag-knowledge-base`'s
`core/embeddings/embeddings.py`). Then this could legitimately be a
small customer-service RAG demo.

Estimated effort: 2–3 hours.

---

# Original (no README existed)

This project was committed without a top-level `README.md`. The original code
is intact in git history; this file documents the deprecation reason and
provides enough context for any future revival.