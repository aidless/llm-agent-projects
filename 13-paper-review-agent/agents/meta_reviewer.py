"""Meta-reviewer agent that aggregates dimension reviews into a final recommendation."""

from __future__ import annotations

from typing import Any, Optional

from app.models import (
    CrossValidationResult,
    DimensionReview,
    ParsedPaper,
    Recommendation,
    ReviewDimension,
    ReviewReport,
)
from templates.review_criteria import get_recommendation


META_REVIEW_PROMPT_TEMPLATE = """你是一位高级学术论文评审主席（Meta-Reviewer），负责汇总多个评审维度的意见并给出最终推荐。

## 论文标题：{title}

## 各维度评审结果（经交叉验证）

{dimension_summaries}

## 你的任务
请综合以上评审结果，给出：
1. 总体评分（1-10）
2. 最终推荐意见
3. 综合评述

### 输出要求
请以JSON格式返回：
{{
    "overall_score": <1-10的浮点数>,
    "recommendation": "<Strong Accept|Accept|Weak Accept|Borderline|Weak Reject|Reject|Strong Reject>",
    "summary": "<2-3句综合评述>",
    "strengths": ["综合优点1", "综合优点2", ...],
    "weaknesses": ["综合缺点1", "综合缺点2", ...],
    "suggestions": ["综合建议1", "综合建议2", ...]
}}
"""


class MetaReviewerAgent:
    """Meta-reviewer that synthesises dimension reviews into a final report.

    Unlike the four dimension agents, this agent does NOT extend BaseReviewAgent
    because its input/output interface is different.
    """

    agent_name = "meta_reviewer"

    def __init__(self, model_name: str = "gpt-4o") -> None:
        self.model_name = model_name

    def review(
        self,
        paper: ParsedPaper,
        cv_results: list[CrossValidationResult],
        venue: str = "NeurIPS",
    ) -> ReviewReport:
        """Produce a final ReviewReport from cross-validated dimension reviews."""
        # Build dimension review list (use arbitrated review if available,
        # otherwise average)
        dimension_reviews: list[DimensionReview] = []
        for cv in cv_results:
            if cv.arbitrated_review is not None:
                dimension_reviews.append(cv.arbitrated_review)
            else:
                # Use the first review as representative
                dimension_reviews.append(cv.reviews[0])

        # Compute weighted overall score
        from templates.review_criteria import get_all_criteria, REVIEW_CRITERIA
        criteria = get_all_criteria()
        total_score = 0.0
        total_weight = 0.0
        for dr in dimension_reviews:
            w = criteria[dr.dimension]["weight"]
            total_score += dr.score * w
            total_weight += w

        overall_score = round(total_score / total_weight, 1) if total_weight > 0 else 5.0
        overall_score = max(1.0, min(10.0, overall_score))
        recommendation = self._map_recommendation(overall_score)

        # Build meta summary
        meta_summary, meta_strengths, meta_weaknesses, meta_suggestions = (
            self._synthesize(paper, cv_results, venue)
        )

        return ReviewReport(
            paper_title=paper.structure.title,
            paper_abstract=paper.structure.abstract,
            dimension_reviews=dimension_reviews,
            cross_validation_results=cv_results,
            overall_score=overall_score,
            recommendation=recommendation,
            meta_summary=meta_summary,
            meta_strengths=meta_strengths,
            meta_weaknesses=meta_weaknesses,
            meta_suggestions=meta_suggestions,
            metadata={"venue": venue, "models_used": list(set(dr.model_name for dr in dimension_reviews))},
        )

    def _map_recommendation(self, score: float) -> Recommendation:
        label = get_recommendation(score)
        for member in Recommendation:
            if member.value == label:
                return member
        return Recommendation.BORDERLINE

    def _synthesize(
        self,
        paper: ParsedPaper,
        cv_results: list[CrossValidationResult],
        venue: str,
    ) -> tuple[str, list[str], list[str], list[str]]:
        """Synthesize cross-validation results into meta-level observations."""
        all_strengths: list[str] = []
        all_weaknesses: list[str] = []
        all_suggestions: list[str] = []
        dimension_scores: dict[str, float] = {}

        for cv in cv_results:
            # Use the first review or arbitrated one
            review = cv.arbitrated_review if cv.arbitrated_review else cv.reviews[0]
            all_strengths.extend(review.strengths)
            all_weaknesses.extend(review.weaknesses)
            all_suggestions.extend(review.suggestions)
            dimension_scores[cv.dimension.value] = review.score

        # Build summary
        high_dims = [k for k, v in dimension_scores.items() if v >= 7.0]
        low_dims = [k for k, v in dimension_scores.items() if v < 4.0]

        summary_parts = [f"论文《{paper.structure.title}》的综合评审结果："]
        if high_dims:
            summary_parts.append(f"在{', '.join(high_dims)}方面表现优秀。")
        if low_dims:
            summary_parts.append(f"在{', '.join(low_dims)}方面存在明显不足。")
        summary_parts.append(f"目标会议/期刊：{venue}。")
        meta_summary = " ".join(summary_parts)

        return (
            meta_summary,
            all_strengths[:10],
            all_weaknesses[:10],
            all_suggestions[:10],
        )