from typing import Any, Dict, List

from .base import BaseMetric


class BERTScoreMetric(BaseMetric):
    """Semantic similarity metric using sentence-transformers.

    ⚠️ **Honesty note (2026-07-22 audit fix)**:
    The previous implementation computed TF-IDF cosine similarity + token
    overlap, and was misleadingly named "BERTScore". That was incorrect:
    the original BERTScore (Zhang et al., ICLR 2020) uses contextual
    BERT embeddings for token-level greedy matching.

    This class now uses `sentence-transformers` to compute *semantic*
    similarity (a related but distinct metric). It is honest about what it
    computes. The class name `BERTScoreMetric` is preserved for backward
    compatibility but the returned metric `name` is `semantic_similarity`.

    For the true BERTScore metric, install `bert-score` and call
    `bert_score.score(cands, refs, lang="en", model_type=...)` directly.
    This class does NOT depend on the original BERTScore paper formulas.

    Dependencies: `pip install sentence-transformers` (~80MB model on first
    `compute()` call, runs on CPU).
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        super().__init__(
            name="semantic_similarity",
            description=(
                "Sentence-transformer cosine similarity (NOT the original "
                "BERTScore from Zhang et al., ICLR 2020). Honest semantic "
                "similarity using all-MiniLM-L6-v2 by default."
            ),
        )
        self.model_name = model_name
        self._model = None  # lazy load

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is not installed. Run:\n"
                "  pip install sentence-transformers\n"
                "Then retry. The first call will download the model "
                f"({self.model_name}, ~80MB)."
            ) from e
        self._model = SentenceTransformer(self.model_name)

    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, Any]:
        if len(predictions) != len(references):
            raise ValueError("predictions 和 references 长度必须相同")
        if not predictions:
            return {"semantic_similarity": 0.0, "n": 0}

        self._ensure_model()
        import numpy as np

        pred_emb = self._model.encode(predictions, convert_to_numpy=True, normalize_embeddings=True)
        ref_emb = self._model.encode(references, convert_to_numpy=True, normalize_embeddings=True)

        # Cosine similarity (embeddings are L2-normalized, so dot = cosine)
        similarities = (pred_emb * ref_emb).sum(axis=1)

        return {
            # Backward-compatible keys (labelled honestly in description)
            "bertscore_precision": float(np.mean(similarities)),  # legacy key
            "bertscore_recall": float(np.mean(similarities)),     # legacy key
            "bertscore_f1": float(np.mean(similarities)),         # legacy key
            # Honest key — what this actually is
            "semantic_similarity": float(np.mean(similarities)),
            "model": self.model_name,
            "n": len(predictions),
        }