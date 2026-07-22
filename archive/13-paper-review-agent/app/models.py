"""Pydantic models for the paper review system."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Recommendation(str, Enum):
    STRONG_ACCEPT = "Strong Accept"
    ACCEPT = "Accept"
    WEAK_ACCEPT = "Weak Accept"
    BORDERLINE = "Borderline"
    WEAK_REJECT = "Weak Reject"
    REJECT = "Reject"
    STRONG_REJECT = "Strong Reject"


class ModelName(str, Enum):
    GPT4O = "gpt-4o"
    DEEPSEEK = "deepseek-chat"
    CLAUDE = "claude-3-opus"


class ReviewDimension(str, Enum):
    RELEVANCE = "relevance"
    METHODOLOGY = "methodology"
    NOVELTY = "novelty"
    CLARITY = "clarity"


# ---------------------------------------------------------------------------
# Paper models
# ---------------------------------------------------------------------------

class AuthorInfo(BaseModel):
    name: str = ""
    affiliation: str = ""
    email: str = ""


class PaperStructure(BaseModel):
    title: str = ""
    abstract: str = ""
    introduction: str = ""
    method: str = ""
    experiments: str = ""
    conclusion: str = ""
    references: str = ""


class ParsedPaper(BaseModel):
    raw_text: str
    structure: PaperStructure
    authors: list[AuthorInfo] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    page_count: int = 0


# ---------------------------------------------------------------------------
# Single review dimension result
# ---------------------------------------------------------------------------

class DimensionReview(BaseModel):
    """Result from a single reviewer agent on a single dimension."""

    dimension: ReviewDimension
    score: float = Field(ge=1.0, le=10.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    summary: str = ""
    model_name: str = ""


# ---------------------------------------------------------------------------
# Cross-validation models
# ---------------------------------------------------------------------------

class CrossValidationResult(BaseModel):
    dimension: ReviewDimension
    reviews: list[DimensionReview] = Field(min_length=2, max_length=2)
    agreement_score: float = Field(ge=0.0, le=1.0)
    is_consistent: bool
    arbitrated_review: Optional[DimensionReview] = None
    disagreement_reason: str = ""


# ---------------------------------------------------------------------------
# Final review report
# ---------------------------------------------------------------------------

class ReviewReport(BaseModel):
    paper_title: str = ""
    paper_abstract: str = ""
    dimension_reviews: list[DimensionReview] = Field(default_factory=list)
    cross_validation_results: list[CrossValidationResult] = Field(default_factory=list)
    overall_score: float = Field(ge=1.0, le=10.0, default=5.0)
    recommendation: Recommendation = Recommendation.BORDERLINE
    meta_summary: str = ""
    meta_strengths: list[str] = Field(default_factory=list)
    meta_weaknesses: list[str] = Field(default_factory=list)
    meta_suggestions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# API request / response
# ---------------------------------------------------------------------------

class ReviewRequest(BaseModel):
    """Incoming review request."""

    text: str = Field(..., min_length=50, description="Full paper text or extracted content")
    venue: str = Field(default="NeurIPS", description="Target conference or journal")
    model_names: list[str] = Field(
        default=["gpt-4o", "deepseek-chat"],
        description="Models used for cross-validation (2 per dimension)",
    )
    output_format: str = Field(default="markdown", description="markdown or json")


class ReviewResponse(BaseModel):
    success: bool = True
    report: Optional[ReviewReport] = None
    markdown: Optional[str] = None
    json_report: Optional[dict[str, Any]] = None
    error: Optional[str] = None