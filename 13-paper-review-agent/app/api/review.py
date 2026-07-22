"""Review API endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from agents.base import clear_mocks, register_mock
from agents.meta_reviewer import MetaReviewerAgent
from app.models import ReviewRequest, ReviewResponse, ReviewDimension
from cross_validation.consensus import ConsensusEngine
from cross_validation.validator import CrossValidator
from parser.paper_parser import PaperParser
from templates.report_template import render_report

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/review", response_model=ReviewResponse)
async def submit_review(request: ReviewRequest):
    """Submit a paper for multi-agent review."""
    try:
        # 1. Parse paper
        parser = PaperParser()
        paper = parser.parse(request.text)

        # 2. Cross-validate each dimension
        model_names = request.model_names
        if len(model_names) < 2:
            model_names = ["gpt-4o", "deepseek-chat"]

        cv_results = []
        for dim in ReviewDimension:
            validator = CrossValidator(model_a=model_names[0], model_b=model_names[1])
            review_a, review_b = validator.validate(dim, paper, request.venue)
            engine = ConsensusEngine()
            cv_result = engine.build_cross_validation_result(review_a, review_b)
            cv_results.append(cv_result)

        # 3. Meta review
        meta = MetaReviewerAgent()
        report = meta.review(paper, cv_results, request.venue)

        # 4. Render
        if request.output_format == "json":
            return ReviewResponse(
                success=True,
                report=report,
                json_report=report.model_dump(),
            )
        else:
            md = render_report(report, fmt="markdown")
            return ReviewResponse(
                success=True,
                report=report,
                markdown=md,
            )

    except Exception as e:
        logger.exception("Review failed")
        return ReviewResponse(success=False, error=str(e))