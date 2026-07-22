"""LLM-as-Judge 测试。"""

import pytest

from judge.llm_judge import LLMJudge
from judge.consensus import ConsensusJudge
from judge.templates import JUDGE_TEMPLATES, get_template


class TestLLMJudge:
    def test_judge_single_returns_all_dimensions(self):
        judge = LLMJudge(dimensions=["accuracy", "relevance"], seed=42)
        result = judge.judge_single(
            question="What is AI?",
            prediction="AI is artificial intelligence.",
            reference="AI stands for artificial intelligence, which simulates human thinking.",
        )
        assert "scores" in result
        assert "accuracy" in result["scores"]
        assert "relevance" in result["scores"]

    def test_judge_scores_in_range(self):
        judge = LLMJudge(dimensions=["accuracy"], seed=42)
        result = judge.judge_single(
            question="What is 1+1?",
            prediction="1+1 equals 2.",
            reference="1+1 equals 2.",
        )
        score = result["scores"]["accuracy"]["score"]
        assert 1 <= score <= 5

    def test_judge_batch(self):
        judge = LLMJudge(dimensions=["accuracy"], seed=42)
        results = judge.judge_batch(
            questions=["Q1", "Q2"],
            predictions=["A1", "A2"],
            references=["R1", "R2"],
        )
        assert len(results) == 2

    def test_judge_batch_length_mismatch_raises(self):
        judge = LLMJudge()
        with pytest.raises(ValueError, match="长度"):
            judge.judge_batch(["Q1"], ["A1", "A2"], ["R1"])

    def test_judge_aggregate(self):
        judge = LLMJudge(dimensions=["accuracy"], seed=42)
        results = judge.judge_batch(
            questions=["Q1", "Q2", "Q3"],
            predictions=["A1", "A2", "A3"],
            references=["R1", "R2", "R3"],
        )
        agg = judge.aggregate(results)
        assert "accuracy" in agg
        assert "mean" in agg["accuracy"]
        assert agg["accuracy"]["count"] == 3

    def test_judge_invalid_dimension_raises(self):
        with pytest.raises(ValueError, match="无效维度"):
            LLMJudge(dimensions=["invalid_dim"])


class TestConsensusJudge:
    def test_consensus_returns_multiple_judges(self):
        judge = ConsensusJudge(num_judges=3, dimensions=["accuracy"], seed=42)
        result = judge.judge_single(
            question="Q",
            prediction="P",
            reference="R",
        )
        assert result["num_judges"] == 3
        assert "accuracy" in result["dimensions"]

    def test_consensus_agreement_field(self):
        judge = ConsensusJudge(num_judges=3, dimensions=["accuracy"], seed=42)
        result = judge.judge_single(
            question="Q",
            prediction="P",
            reference="R",
        )
        assert "overall_agreement" in result
        assert 0 <= result["overall_agreement"] <= 1

    def test_consensus_batch(self):
        judge = ConsensusJudge(num_judges=2, dimensions=["accuracy"], seed=42)
        results = judge.judge_batch(
            questions=["Q1", "Q2"],
            predictions=["P1", "P2"],
            references=["R1", "R2"],
        )
        assert len(results) == 2


class TestTemplates:
    def test_get_all_templates(self):
        templates = get_template()
        assert len(templates) == 6

    def test_get_specific_dimensions(self):
        templates = get_template(["accuracy", "safety"])
        assert len(templates) == 2
        assert "accuracy" in templates
        assert "safety" in templates

    def test_template_has_required_fields(self):
        for key, t in JUDGE_TEMPLATES.items():
            assert "name" in t
            assert "description" in t
            assert "template" in t
