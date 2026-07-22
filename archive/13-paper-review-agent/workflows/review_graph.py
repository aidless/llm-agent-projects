"""LangGraph StateGraph workflow for multi-agent paper review."""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.meta_reviewer import MetaReviewerAgent
from app.models import (
    CrossValidationResult,
    ParsedPaper,
    ReviewDimension,
    ReviewReport,
)
from cross_validation.consensus import ConsensusEngine
from cross_validation.validator import CrossValidator

logger = logging.getLogger(__name__)

# Dimensions to review in parallel
DIMENSIONS = list(ReviewDimension)


# ---------------------------------------------------------------------------
# State definition using TypedDict with Annotated reducers
# ---------------------------------------------------------------------------

class ReviewState(TypedDict, total=False):
    """State passed through the LangGraph nodes.

    list fields use operator.add so parallel branches accumulate results.
    """
    paper: ParsedPaper
    venue: str
    model_names: list[str]
    cross_validation_results: Annotated[list[CrossValidationResult], operator.add]
    report: Optional[ReviewReport]
    errors: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def _review_dimension(dimension: ReviewDimension) -> callable:
    """Factory that creates a node function for a specific dimension."""

    def _node(state: ReviewState) -> dict:
        paper: ParsedPaper = state["paper"]
        venue: str = state.get("venue", "NeurIPS")
        model_names: list[str] = state.get("model_names", ["gpt-4o", "deepseek-chat"])

        if len(model_names) < 2:
            model_names = ["gpt-4o", "deepseek-chat"]

        validator = CrossValidator(model_a=model_names[0], model_b=model_names[1])
        try:
            review_a, review_b = validator.validate(dimension, paper, venue)
            engine = ConsensusEngine()
            cv_result = engine.build_cross_validation_result(review_a, review_b)
            return {"cross_validation_results": [cv_result]}
        except Exception as e:
            logger.error("Error reviewing dimension %s: %s", dimension.value, e)
            return {"errors": [f"{dimension.value}: {str(e)}"]}

    _node.__name__ = f"review_{dimension.value}"
    return _node


def _meta_review(state: ReviewState) -> dict:
    """Meta-review node: synthesise all cross-validation results."""
    paper: ParsedPaper = state["paper"]
    venue: str = state.get("venue", "NeurIPS")
    cv_results: list[CrossValidationResult] = state.get("cross_validation_results", [])

    meta = MetaReviewerAgent()
    report = meta.review(paper, cv_results, venue)
    return {"report": report}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_review_graph() -> StateGraph:
    """Build and compile the review LangGraph.

    Flow:
        START -> [relevance, methodology, novelty, clarity] (parallel)
              -> meta_review
              -> END
    """
    graph = StateGraph(ReviewState)

    # Add parallel review nodes
    node_names: list[str] = []
    for dim in DIMENSIONS:
        name = f"review_{dim.value}"
        graph.add_node(name, _review_dimension(dim))
        node_names.append(name)

    # Add meta-review node
    graph.add_node("meta_review", _meta_review)

    # START -> all four dimension nodes (parallel fan-out)
    for name in node_names:
        graph.add_edge(START, name)

    # All dimension nodes -> meta_review (fan-in)
    for name in node_names:
        graph.add_edge(name, "meta_review")

    # meta_review -> END
    graph.add_edge("meta_review", END)

    return graph.compile()


# Compiled singleton
_compiled_graph: Optional[Any] = None


def get_review_graph() -> Any:
    """Return the compiled review graph (cached)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_review_graph()
    return _compiled_graph