"""Cross-model validator -- runs the same dimension review with two different models."""

from __future__ import annotations

import logging
from typing import Optional

from agents.base import BaseReviewAgent
from agents.clarity_agent import ClarityAgent
from agents.methodology_agent import MethodologyAgent
from agents.novelty_agent import NoveltyAgent
from agents.relevance_agent import RelevanceAgent
from app.models import DimensionReview, ParsedPaper, ReviewDimension

logger = logging.getLogger(__name__)

_AGENT_MAP: dict[ReviewDimension, type[BaseReviewAgent]] = {
    ReviewDimension.RELEVANCE: RelevanceAgent,
    ReviewDimension.METHODOLOGY: MethodologyAgent,
    ReviewDimension.NOVELTY: NoveltyAgent,
    ReviewDimension.CLARITY: ClarityAgent,
}


class CrossValidator:
    """Run a dimension review with two different models for cross-validation."""

    def __init__(
        self,
        model_a: str = "gpt-4o",
        model_b: str = "deepseek-chat",
    ) -> None:
        self.model_a = model_a
        self.model_b = model_b

    def validate(
        self,
        dimension: ReviewDimension,
        paper: ParsedPaper,
        venue: str = "NeurIPS",
    ) -> tuple[DimensionReview, DimensionReview]:
        """Run two independent reviews for the same dimension using different models."""
        agent_cls = _AGENT_MAP[dimension]

        agent_a = agent_cls(model_name=self.model_a)
        agent_b = agent_cls(model_name=self.model_b)

        review_a = agent_a.review(paper, venue)
        review_b = agent_b.review(paper, venue)

        return review_a, review_b