"""Base class for all review agents."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from app.models import DimensionReview, ParsedPaper, ReviewDimension

logger = logging.getLogger(__name__)


# Global mock registry: tests can register mock responses keyed by
# (dimension, model_name).
_MOCK_REGISTRY: dict[tuple[str, str], dict[str, Any]] = {}


def register_mock(dimension: str, model_name: str, response: dict[str, Any]) -> None:
    """Register a mock LLM response for testing purposes."""
    _MOCK_REGISTRY[(dimension, model_name)] = response


def clear_mocks() -> None:
    """Clear all registered mocks."""
    _MOCK_REGISTRY.clear()


def get_mock(dimension: str, model_name: str) -> Optional[dict[str, Any]]:
    """Retrieve a registered mock response."""
    return _MOCK_REGISTRY.get((dimension, model_name))


class BaseReviewAgent(ABC):
    """Abstract base for a specialised review agent."""

    dimension: ReviewDimension
    agent_name: str = "base"

    def __init__(self, model_name: str = "gpt-4o") -> None:
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Template method
    # ------------------------------------------------------------------

    def review(self, paper: ParsedPaper, venue: str = "NeurIPS") -> DimensionReview:
        """Run the full review pipeline for this agent's dimension."""
        prompt = self._build_prompt(paper, venue)
        raw_response = self._call_llm(prompt)
        result = self._parse_response(raw_response)
        return result

    # ------------------------------------------------------------------
    # LLM call (mocked by default)
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call the LLM (or return a mock response in test mode)."""
        mock = get_mock(self.dimension.value, self.model_name)
        if mock is not None:
            logger.debug("Using mock response for %s/%s", self.dimension.value, self.model_name)
            return mock
        # Default deterministic response when no mock is registered
        return self._default_response()

    @abstractmethod
    def _default_response(self) -> dict[str, Any]:
        """Return a deterministic default response for this agent."""
        ...

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_prompt(self, paper: ParsedPaper, venue: str) -> str:
        """Build the review prompt for this dimension."""
        ...

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: dict[str, Any]) -> DimensionReview:
        """Parse the raw LLM response into a DimensionReview."""
        score = float(raw.get("score", 5.0))
        score = max(1.0, min(10.0, score))
        confidence = float(raw.get("confidence", 0.8))
        confidence = max(0.0, min(1.0, confidence))
        return DimensionReview(
            dimension=self.dimension,
            score=round(score, 1),
            confidence=round(confidence, 2),
            strengths=raw.get("strengths", []),
            weaknesses=raw.get("weaknesses", []),
            suggestions=raw.get("suggestions", []),
            summary=raw.get("summary", ""),
            model_name=self.model_name,
        )
