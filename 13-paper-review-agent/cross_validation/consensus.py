"""Consensus computation and disagreement arbitration for cross-validation."""

from __future__ import annotations

import logging
from typing import Optional

from app.models import (
    CrossValidationResult,
    DimensionReview,
    ReviewDimension,
)

logger = logging.getLogger(__name__)

# Threshold: if score difference > this, reviews are considered inconsistent
_SCORE_DIFF_THRESHOLD = 2.0
# Threshold: agreement score above this is considered consistent
_AGREEMENT_THRESHOLD = 0.7


class ConsensusEngine:
    """Compute agreement between two reviews and arbitrate disagreements."""

    def __init__(
        self,
        score_diff_threshold: float = _SCORE_DIFF_THRESHOLD,
        agreement_threshold: float = _AGREEMENT_THRESHOLD,
    ) -> None:
        self.score_diff_threshold = score_diff_threshold
        self.agreement_threshold = agreement_threshold

    def compute_agreement(self, review_a: DimensionReview, review_b: DimensionReview) -> float:
        """Compute a 0-1 agreement score between two reviews.

        Uses:
        - Score similarity (weight 0.5)
        - Strengths overlap (weight 0.25)
        - Weaknesses overlap (weight 0.25)
        """
        # Score similarity: 1 - normalised difference
        score_diff = abs(review_a.score - review_b.score)
        score_sim = max(0.0, 1.0 - score_diff / 9.0)

        # Text overlap via set intersection
        strengths_a = set(review_a.strengths)
        strengths_b = set(review_b.strengths)
        weaknesses_a = set(review_a.weaknesses)
        weaknesses_b = set(review_b.weaknesses)

        def _jaccard(sa: set[str], sb: set[str]) -> float:
            if not sa and not sb:
                return 1.0
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / len(sa | sb)

        strength_sim = _jaccard(strengths_a, strengths_b)
        weakness_sim = _jaccard(weaknesses_a, weaknesses_b)

        agreement = 0.5 * score_sim + 0.25 * strength_sim + 0.25 * weakness_sim
        return round(agreement, 3)

    def is_consistent(self, review_a: DimensionReview, review_b: DimensionReview) -> bool:
        """Determine if two reviews are consistent enough."""
        score_diff = abs(review_a.score - review_b.score)
        if score_diff > self.score_diff_threshold:
            return False
        agreement = self.compute_agreement(review_a, review_b)
        return agreement >= self.agreement_threshold

    def arbitrate(
        self,
        review_a: DimensionReview,
        review_b: DimensionReview,
    ) -> DimensionReview:
        """Arbitrate between two disagreeing reviews.

        Strategy: weighted average by confidence, merge unique strengths/weaknesses.
        """
        total_conf = review_a.confidence + review_b.confidence
        if total_conf == 0:
            weight_a = weight_b = 0.5
        else:
            weight_a = review_a.confidence / total_conf
            weight_b = review_b.confidence / total_conf

        avg_score = round(review_a.score * weight_a + review_b.score * weight_b, 1)
        avg_confidence = round((review_a.confidence + review_b.confidence) / 2, 2)

        # Merge strengths and weaknesses (dedup by set)
        all_strengths = list(dict.fromkeys(review_a.strengths + review_b.strengths))
        all_weaknesses = list(dict.fromkeys(review_a.weaknesses + review_b.weaknesses))
        all_suggestions = list(dict.fromkeys(review_a.suggestions + review_b.suggestions))

        # Pick the model with higher confidence
        primary_model = review_a.model_name if weight_a >= weight_b else review_b.model_name

        return DimensionReview(
            dimension=review_a.dimension,
            score=max(1.0, min(10.0, avg_score)),
            confidence=avg_confidence,
            strengths=all_strengths,
            weaknesses=all_weaknesses,
            suggestions=all_suggestions,
            summary=f"[Arbitrated] Weighted avg score {avg_score} (models: {review_a.model_name}, {review_b.model_name})",
            model_name=primary_model,
        )

    def build_cross_validation_result(
        self,
        review_a: DimensionReview,
        review_b: DimensionReview,
    ) -> CrossValidationResult:
        """Build a full CrossValidationResult, arbitrating if needed."""
        agreement_score = self.compute_agreement(review_a, review_b)
        consistent = self.is_consistent(review_a, review_b)

        arbitrated: Optional[DimensionReview] = None
        disagreement_reason = ""

        if not consistent:
            arbitrated = self.arbitrate(review_a, review_b)
            disagreement_reason = (
                f"Score diff: {abs(review_a.score - review_b.score):.1f}, "
                f"Agreement: {agreement_score:.2f}"
            )
            logger.info(
                "Disagreement on %s: %s", review_a.dimension.value, disagreement_reason
            )

        return CrossValidationResult(
            dimension=review_a.dimension,
            reviews=[review_a, review_b],
            agreement_score=agreement_score,
            is_consistent=consistent,
            arbitrated_review=arbitrated,
            disagreement_reason=disagreement_reason,
        )