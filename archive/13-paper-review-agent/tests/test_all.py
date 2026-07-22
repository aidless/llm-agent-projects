"""Comprehensive test suite for the multi-agent paper review system.

Contains 30+ tests covering:
- Pydantic model validation
- Paper parser & structure extractor
- Individual agent behaviour (with mocks)
- Cross-validation & consensus
- LangGraph workflow
- Report template rendering
- FastAPI endpoints
"""

from __future__ import annotations

import json
import sys
import os
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.base import BaseReviewAgent, clear_mocks, register_mock, get_mock
from agents.clarity_agent import ClarityAgent
from agents.methodology_agent import MethodologyAgent
from agents.novelty_agent import NoveltyAgent
from agents.relevance_agent import RelevanceAgent
from agents.meta_reviewer import MetaReviewerAgent
from app.models import (
    AuthorInfo,
    CrossValidationResult,
    DimensionReview,
    ParsedPaper,
    PaperStructure,
    Recommendation,
    ReviewDimension,
    ReviewReport,
    ReviewRequest,
    ReviewResponse,
    ModelName,
)
from cross_validation.consensus import ConsensusEngine
from cross_validation.validator import CrossValidator
from parser.paper_parser import PaperParser
from parser.structure_extractor import StructureExtractor
from templates.report_template import render_markdown, render_json, render_report
from templates.review_criteria import (
    get_criteria,
    get_recommendation,
    get_all_criteria,
    REVIEW_CRITERIA,
)


# ===========================================================================
# Sample data
# ===========================================================================

SAMPLE_PAPER_TEXT = """Title: A Novel Attention Mechanism for Graph Neural Networks

Abstract: This paper proposes a novel attention mechanism for graph neural networks
that improves performance on node classification tasks. Our method, GraphAttention++,
introduces a multi-head attention scheme with adaptive weighting.

Keywords: graph neural networks, attention mechanism, node classification

1. Introduction
Graph neural networks have shown remarkable success in various domains.
However, existing attention mechanisms for GNNs have limitations.
In this paper, we propose GraphAttention++, a novel attention mechanism.

2. Methodology
Our proposed method consists of three components: an adaptive attention layer,
a multi-head attention scheme, and a residual connection module.
The adaptive attention layer computes attention weights using both node features
and graph topology information.

3. Experiments
We evaluate our method on three datasets: Cora, Citeseer, and Pubmed.
We compare with GCN, GAT, GraphSAGE, and APPNP.
Our method achieves state-of-the-art results on all three datasets.
Ablation study: each component contributes to the final performance.

4. Conclusion
We proposed GraphAttention++, a novel attention mechanism for GNNs.
Experiments show consistent improvements over existing methods.

References
[1] Kipf & Welling, Semi-Supervised Classification with GCN, ICLR 2017.
[2] Velickovic et al., GAT, ICLR 2018.
"""

SAMPLE_SHORT_TEXT = "A" * 100  # 100 chars, below min_length for ReviewRequest


def _make_parsed_paper(text: str = SAMPLE_PAPER_TEXT) -> ParsedPaper:
    parser = PaperParser()
    return parser.parse(text)


def _make_dimension_review(
    dimension: ReviewDimension = ReviewDimension.RELEVANCE,
    score: float = 7.0,
    model_name: str = "gpt-4o",
    strengths: list[str] | None = None,
    weaknesses: list[str] | None = None,
) -> DimensionReview:
    return DimensionReview(
        dimension=dimension,
        score=score,
        confidence=0.85,
        strengths=strengths or ["good point"],
        weaknesses=weaknesses or ["minor issue"],
        suggestions=["improve X"],
        summary="A review summary.",
        model_name=model_name,
    )


# ===========================================================================
# 1. Model tests
# ===========================================================================

class TestModels:
    """Tests for Pydantic models."""

    def test_recommendation_enum_values(self):
        assert Recommendation.STRONG_ACCEPT.value == "Strong Accept"
        assert Recommendation.REJECT.value == "Reject"
        assert len(Recommendation) == 7

    def test_model_name_enum(self):
        assert ModelName.GPT4O.value == "gpt-4o"
        assert ModelName.DEEPSEEK.value == "deepseek-chat"
        assert ModelName.CLAUDE.value == "claude-3-opus"

    def test_dimension_review_score_bounds(self):
        # Score must be 1-10
        dr = DimensionReview(dimension=ReviewDimension.RELEVANCE, score=1.0)
        assert dr.score == 1.0
        dr2 = DimensionReview(dimension=ReviewDimension.RELEVANCE, score=10.0)
        assert dr2.score == 10.0

    def test_dimension_review_score_clamped_in_agent(self):
        agent = RelevanceAgent()
        raw = {"score": 15.0, "confidence": 0.5}
        result = agent._parse_response(raw)
        assert result.score == 10.0

    def test_dimension_review_score_floor_in_agent(self):
        agent = RelevanceAgent()
        raw = {"score": 0.0, "confidence": 0.5}
        result = agent._parse_response(raw)
        assert result.score == 1.0

    def test_review_request_min_length(self):
        with pytest.raises(Exception):
            ReviewRequest(text="too short")

    def test_review_request_valid(self):
        req = ReviewRequest(text=SAMPLE_SHORT_TEXT + " more text to meet the threshold")
        assert req.venue == "NeurIPS"
        assert req.output_format == "markdown"

    def test_review_response_success(self):
        resp = ReviewResponse(success=True)
        assert resp.error is None
        resp2 = ReviewResponse(success=False, error="test error")
        assert resp2.error == "test error"

    def test_cross_validation_min_reviews(self):
        dr = _make_dimension_review()
        # Must have at least 2 reviews
        with pytest.raises(Exception):
            CrossValidationResult(
                dimension=ReviewDimension.RELEVANCE,
                reviews=[dr],
                agreement_score=1.0,
                is_consistent=True,
            )

    def test_parsed_paper_creation(self):
        paper = _make_parsed_paper()
        assert paper.raw_text == SAMPLE_PAPER_TEXT
        assert paper.page_count >= 1


# ===========================================================================
# 2. Parser tests
# ===========================================================================

class TestPaperParser:
    """Tests for PaperParser."""

    def setup_method(self):
        self.parser = PaperParser()

    def test_parse_returns_parsed_paper(self):
        result = self.parser.parse(SAMPLE_PAPER_TEXT)
        assert isinstance(result, ParsedPaper)

    def test_parse_extracts_title(self):
        result = self.parser.parse(SAMPLE_PAPER_TEXT)
        assert "Attention" in result.structure.title

    def test_parse_extracts_abstract(self):
        result = self.parser.parse(SAMPLE_PAPER_TEXT)
        assert len(result.structure.abstract) > 50

    def test_parse_extracts_keywords(self):
        result = self.parser.parse(SAMPLE_PAPER_TEXT)
        assert "attention mechanism" in result.keywords

    def test_parse_extracts_datasets(self):
        text = "We evaluate on dataset Cora and dataset Citeseer."
        result = self.parser.parse(text)
        assert "Cora" in result.datasets
        assert "Citeseer" in result.datasets

    def test_parse_fallback_title(self):
        text = "First line is the title\n\nSome content here." + "x" * 200
        result = self.parser.parse(text)
        assert result.structure.title == "First line is the title"

    def test_parse_unknown_author_fallback(self):
        result = self.parser.parse("No authors mentioned." + "x" * 200)
        assert len(result.authors) == 1
        assert result.authors[0].name == "Unknown Author"

    def test_page_count_estimate(self):
        short = "x" * 100
        result = self.parser.parse(short)
        assert result.page_count == 1

    def test_page_count_long(self):
        long_text = "x" * 10000
        result = self.parser.parse(long_text)
        assert result.page_count >= 3


# ===========================================================================
# 3. Structure extractor tests
# ===========================================================================

class TestStructureExtractor:

    def setup_method(self):
        self.extractor = StructureExtractor()
        self.paper = _make_parsed_paper()

    def test_section_completion_returns_dict(self):
        comp = self.extractor.get_section_completion(self.paper)
        assert isinstance(comp, dict)
        assert "title" in comp
        assert "abstract" in comp

    def test_section_completion_range(self):
        comp = self.extractor.get_section_completion(self.paper)
        for key, val in comp.items():
            assert 0.0 <= val <= 1.0

    def test_missing_sections(self):
        missing = self.extractor.get_missing_sections(self.paper)
        assert isinstance(missing, list)

    def test_structure_summary(self):
        summary = self.extractor.get_structure_summary(self.paper)
        assert "Title" in summary
        assert "Authors" in summary

    def test_extract_key_sentences(self):
        sents = self.extractor.extract_key_sentences(self.paper, "abstract")
        assert isinstance(sents, list)

    def test_extract_key_sentences_empty_section(self):
        empty_paper = ParsedPaper(raw_text="short", structure=PaperStructure())
        sents = self.extractor.extract_key_sentences(empty_paper, "abstract")
        assert sents == []


# ===========================================================================
# 4. Agent tests
# ===========================================================================

class TestAgents:
    """Tests for individual review agents."""

    def setup_method(self):
        clear_mocks()

    def teardown_method(self):
        clear_mocks()

    def test_relevance_agent_default(self):
        agent = RelevanceAgent()
        paper = _make_parsed_paper()
        result = agent.review(paper)
        assert result.dimension == ReviewDimension.RELEVANCE
        assert 1.0 <= result.score <= 10.0
        assert result.model_name == "gpt-4o"

    def test_relevance_agent_with_mock(self):
        register_mock("relevance", "gpt-4o", {
            "score": 8.5, "confidence": 0.9,
            "strengths": ["highly relevant"],
            "weaknesses": [],
            "suggestions": [],
            "summary": "Great match.",
        })
        agent = RelevanceAgent()
        result = agent.review(_make_parsed_paper())
        assert result.score == 8.5
        assert result.strengths == ["highly relevant"]

    def test_methodology_agent_default(self):
        agent = MethodologyAgent()
        result = agent.review(_make_parsed_paper())
        assert result.dimension == ReviewDimension.METHODOLOGY

    def test_novelty_agent_default(self):
        agent = NoveltyAgent()
        result = agent.review(_make_parsed_paper())
        assert result.dimension == ReviewDimension.NOVELTY

    def test_clarity_agent_default(self):
        agent = ClarityAgent()
        result = agent.review(_make_parsed_paper())
        assert result.dimension == ReviewDimension.CLARITY

    def test_agent_different_models(self):
        register_mock("relevance", "claude-3-opus", {
            "score": 9.0, "confidence": 0.95,
            "strengths": ["perfect"], "weaknesses": [], "suggestions": [], "summary": "ok",
        })
        agent = RelevanceAgent(model_name="claude-3-opus")
        result = agent.review(_make_parsed_paper())
        assert result.model_name == "claude-3-opus"
        assert result.score == 9.0

    def test_mock_registry(self):
        register_mock("test_dim", "test_model", {"score": 7.0})
        assert get_mock("test_dim", "test_model") == {"score": 7.0}
        assert get_mock("test_dim", "other") is None
        clear_mocks()
        assert get_mock("test_dim", "test_model") is None

    def test_meta_reviewer(self):
        dr1 = _make_dimension_review(ReviewDimension.RELEVANCE, 8.0)
        dr2 = _make_dimension_review(ReviewDimension.METHODOLOGY, 7.0)
        dr3 = _make_dimension_review(ReviewDimension.NOVELTY, 6.0)
        dr4 = _make_dimension_review(ReviewDimension.CLARITY, 7.5)

        for dim, dr in [(ReviewDimension.RELEVANCE, dr1), (ReviewDimension.METHODOLOGY, dr2),
                        (ReviewDimension.NOVELTY, dr3), (ReviewDimension.CLARITY, dr4)]:
            cv = CrossValidationResult(
                dimension=dim, reviews=[dr, dr],
                agreement_score=1.0, is_consistent=True,
            )

        paper = _make_parsed_paper()
        meta = MetaReviewerAgent()
        report = meta.review(paper, [
            CrossValidationResult(dimension=ReviewDimension.RELEVANCE, reviews=[dr1, dr1], agreement_score=1.0, is_consistent=True),
            CrossValidationResult(dimension=ReviewDimension.METHODOLOGY, reviews=[dr2, dr2], agreement_score=1.0, is_consistent=True),
            CrossValidationResult(dimension=ReviewDimension.NOVELTY, reviews=[dr3, dr3], agreement_score=1.0, is_consistent=True),
            CrossValidationResult(dimension=ReviewDimension.CLARITY, reviews=[dr4, dr4], agreement_score=1.0, is_consistent=True),
        ])
        assert report.overall_score > 0
        assert report.recommendation in list(Recommendation)
        assert len(report.dimension_reviews) == 4

    def test_prompt_contains_venue(self):
        agent = RelevanceAgent()
        paper = _make_parsed_paper()
        prompt = agent._build_prompt(paper, "ICML")
        assert "ICML" in prompt


# ===========================================================================
# 5. Cross-validation tests
# ===========================================================================

class TestCrossValidation:

    def setup_method(self):
        clear_mocks()

    def teardown_method(self):
        clear_mocks()

    def test_validator_returns_two_reviews(self):
        register_mock("relevance", "gpt-4o", {"score": 7.0, "confidence": 0.8, "strengths": ["s1"], "weaknesses": ["w1"], "suggestions": [], "summary": ""})
        register_mock("relevance", "deepseek-chat", {"score": 7.5, "confidence": 0.85, "strengths": ["s1"], "weaknesses": ["w2"], "suggestions": [], "summary": ""})
        validator = CrossValidator()
        a, b = validator.validate(ReviewDimension.RELEVANCE, _make_parsed_paper())
        assert a.model_name == "gpt-4o"
        assert b.model_name == "deepseek-chat"
        assert a.score == 7.0
        assert b.score == 7.5

    def test_consensus_agreement_identical(self):
        dr1 = _make_dimension_review(ReviewDimension.RELEVANCE, 7.0, strengths=["good"], weaknesses=["bad"])
        dr2 = _make_dimension_review(ReviewDimension.RELEVANCE, 7.0, strengths=["good"], weaknesses=["bad"])
        engine = ConsensusEngine()
        agreement = engine.compute_agreement(dr1, dr2)
        assert agreement == 1.0
        assert engine.is_consistent(dr1, dr2) is True

    def test_consensus_disagreement(self):
        dr1 = _make_dimension_review(ReviewDimension.RELEVANCE, 2.0, strengths=["s1"], weaknesses=["w1"])
        dr2 = _make_dimension_review(ReviewDimension.RELEVANCE, 9.0, strengths=["s2"], weaknesses=["w2"])
        engine = ConsensusEngine()
        agreement = engine.compute_agreement(dr1, dr2)
        assert agreement < 0.5
        assert engine.is_consistent(dr1, dr2) is False

    def test_arbitration_weighted_average(self):
        dr1 = DimensionReview(
            dimension=ReviewDimension.RELEVANCE, score=4.0, confidence=0.9,
            strengths=["a"], weaknesses=["b"], suggestions=[], summary="", model_name="m1",
        )
        dr2 = DimensionReview(
            dimension=ReviewDimension.RELEVANCE, score=8.0, confidence=0.1,
            strengths=["c"], weaknesses=["d"], suggestions=[], summary="", model_name="m2",
        )
        engine = ConsensusEngine()
        arbitrated = engine.arbitrate(dr1, dr2)
        # Higher confidence (0.9) on dr1 (score 4.0) -> weighted toward 4.0
        assert 3.5 <= arbitrated.score <= 5.0
        assert arbitrated.model_name == "m1"

    def test_build_cv_result_consistent(self):
        dr1 = _make_dimension_review(ReviewDimension.CLARITY, 6.0)
        dr2 = _make_dimension_review(ReviewDimension.CLARITY, 6.5)
        engine = ConsensusEngine()
        cv = engine.build_cross_validation_result(dr1, dr2)
        assert cv.is_consistent is True
        assert cv.arbitrated_review is None
        assert cv.disagreement_reason == ""

    def test_build_cv_result_inconsistent(self):
        dr1 = _make_dimension_review(ReviewDimension.CLARITY, 2.0, strengths=["a"], weaknesses=["b"])
        dr2 = _make_dimension_review(ReviewDimension.CLARITY, 9.0, strengths=["c"], weaknesses=["d"])
        engine = ConsensusEngine()
        cv = engine.build_cross_validation_result(dr1, dr2)
        assert cv.is_consistent is False
        assert cv.arbitrated_review is not None
        assert len(cv.disagreement_reason) > 0


# ===========================================================================
# 6. Workflow tests
# ===========================================================================

class TestWorkflow:

    def setup_method(self):
        clear_mocks()
        # Register mocks for all 4 dimensions x 2 models
        for dim in ReviewDimension:
            for model in ["gpt-4o", "deepseek-chat"]:
                register_mock(dim.value, model, {
                    "score": 7.0, "confidence": 0.8,
                    "strengths": ["good"], "weaknesses": ["minor"],
                    "suggestions": ["improve"], "summary": "ok",
                })

    def teardown_method(self):
        clear_mocks()

    def test_build_graph(self):
        from workflows.review_graph import build_review_graph
        graph = build_review_graph()
        assert graph is not None

    def test_run_workflow(self):
        from workflows.review_graph import get_review_graph
        graph = get_review_graph()
        paper = _make_parsed_paper()
        state = {
            "paper": paper,
            "venue": "NeurIPS",
            "model_names": ["gpt-4o", "deepseek-chat"],
            "cross_validation_results": [],
            "report": None,
            "errors": [],
        }
        result = graph.invoke(state)
        assert "report" in result
        assert result["report"] is not None
        assert result["report"].overall_score > 0

    def test_workflow_has_four_cv_results(self):
        from workflows.review_graph import get_review_graph
        graph = get_review_graph()
        paper = _make_parsed_paper()
        result = graph.invoke({
            "paper": paper,
            "venue": "ICML",
            "model_names": ["gpt-4o", "deepseek-chat"],
            "cross_validation_results": [],
            "report": None,
            "errors": [],
        })
        report = result["report"]
        assert len(report.cross_validation_results) == 4


# ===========================================================================
# 7. Report template tests
# ===========================================================================

class TestReportTemplate:

    def _make_report(self) -> ReviewReport:
        dr = _make_dimension_review(ReviewDimension.RELEVANCE, 7.5)
        dr2 = _make_dimension_review(ReviewDimension.RELEVANCE, 7.0, model_name="deepseek-chat")
        cv = CrossValidationResult(
            dimension=ReviewDimension.RELEVANCE,
            reviews=[dr, dr2],
            agreement_score=0.85,
            is_consistent=True,
        )
        return ReviewReport(
            paper_title="Test Paper",
            paper_abstract="Test abstract",
            cross_validation_results=[cv],
            overall_score=7.2,
            recommendation=Recommendation.ACCEPT,
            meta_summary="Good paper.",
            meta_strengths=["strength 1"],
            meta_weaknesses=["weakness 1"],
            meta_suggestions=["suggestion 1"],
            metadata={"venue": "NeurIPS", "models_used": ["gpt-4o", "deepseek-chat"]},
        )

    def test_render_markdown(self):
        report = self._make_report()
        md = render_markdown(report)
        assert "# 论文评审报告" in md
        assert "Test Paper" in md
        assert "Accept" in md

    def test_render_json(self):
        report = self._make_report()
        j = render_json(report)
        assert j["paper_title"] == "Test Paper"
        assert j["overall_score"] == 7.2

    def test_render_report_markdown_format(self):
        report = self._make_report()
        output = render_report(report, fmt="markdown")
        assert isinstance(output, str)
        assert "# " in output

    def test_render_report_json_format(self):
        report = self._make_report()
        output = render_report(report, fmt="json")
        parsed = json.loads(output)
        assert parsed["paper_title"] == "Test Paper"

    def test_render_markdown_contains_cv_table(self):
        report = self._make_report()
        md = render_markdown(report)
        assert "| 评审维度 |" in md
        assert "一致性评分" in md

    def test_render_markdown_disagreement_note(self):
        dr1 = _make_dimension_review(ReviewDimension.METHODOLOGY, 2.0, strengths=["a"], weaknesses=["b"])
        dr2 = _make_dimension_review(ReviewDimension.METHODOLOGY, 9.0, strengths=["c"], weaknesses=["d"])
        cv = CrossValidationResult(
            dimension=ReviewDimension.METHODOLOGY,
            reviews=[dr1, dr2],
            agreement_score=0.1,
            is_consistent=False,
            disagreement_reason="Score diff: 7.0",
        )
        report = ReviewReport(
            paper_title="T", cross_validation_results=[cv],
            overall_score=5.0, recommendation=Recommendation.BORDERLINE,
        )
        md = render_markdown(report)
        assert "分歧" in md


# ===========================================================================
# 8. Review criteria tests
# ===========================================================================

class TestReviewCriteria:

    def test_get_criteria_returns_dict(self):
        c = get_criteria(ReviewDimension.RELEVANCE)
        assert "name" in c
        assert "weight" in c
        assert "scoring_guide" in c
        assert "check_points" in c

    def test_get_all_criteria(self):
        all_c = get_all_criteria()
        assert len(all_c) == 4
        for dim in ReviewDimension:
            assert dim in all_c

    def test_get_recommendation_strong_accept(self):
        assert get_recommendation(9.0) == "Strong Accept"

    def test_get_recommendation_accept(self):
        assert get_recommendation(7.5) == "Accept"

    def test_get_recommendation_weak_accept(self):
        assert get_recommendation(6.0) == "Weak Accept"

    def test_get_recommendation_borderline(self):
        assert get_recommendation(5.0) == "Borderline"

    def test_get_recommendation_weak_reject(self):
        assert get_recommendation(4.0) == "Weak Reject"

    def test_get_recommendation_reject(self):
        assert get_recommendation(2.0) == "Reject"

    def test_get_recommendation_strong_reject(self):
        assert get_recommendation(1.2) == "Strong Reject"


# ===========================================================================
# 9. API tests
# ===========================================================================

class TestAPI:

    def setup_method(self):
        clear_mocks()
        for dim in ReviewDimension:
            for model in ["gpt-4o", "deepseek-chat"]:
                register_mock(dim.value, model, {
                    "score": 7.0, "confidence": 0.8,
                    "strengths": ["good"], "weaknesses": ["minor"],
                    "suggestions": ["improve"], "summary": "ok review",
                })

    def teardown_method(self):
        clear_mocks()

    def test_health_check(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_review_endpoint(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.post("/api/v1/review", json={
            "text": SAMPLE_PAPER_TEXT,
            "venue": "NeurIPS",
            "output_format": "markdown",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["report"] is not None
        assert data["report"]["overall_score"] > 0

    def test_review_endpoint_json_format(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.post("/api/v1/review", json={
            "text": SAMPLE_PAPER_TEXT,
            "output_format": "json",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["json_report"] is not None

    def test_sample_report_endpoint(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.get("/api/v1/report/sample")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "report" in data
        assert data["report"]["paper_title"] == "Sample Paper Title"