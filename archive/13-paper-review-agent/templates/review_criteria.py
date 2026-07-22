"""Review criteria and scoring rubrics for each dimension."""

from __future__ import annotations

from typing import Any

from app.models import ReviewDimension


# ---------------------------------------------------------------------------
# Criteria definitions per dimension
# ---------------------------------------------------------------------------

REVIEW_CRITERIA: dict[ReviewDimension, dict[str, Any]] = {
    ReviewDimension.RELEVANCE: {
        "name": "相关性评审",
        "description": "评估论文是否匹配目标会议/期刊的主题范围和研究方向",
        "weight": 0.20,
        "scoring_guide": {
            9: "论文核心主题与目标会议高度契合，直接解决该领域核心问题",
            7: "论文主题与会议相关，属于会议广泛关注的领域",
            5: "论文与会议主题有一定关联，但不是核心匹配",
            3: "论文主题与会议方向偏离较大",
            1: "论文完全不匹配目标会议的研究范围",
        },
        "check_points": [
            "论文主题是否在会议/期刊的征稿范围内",
            "研究问题是否为该领域当前关注的热点",
            "方法论是否适合该会议的读者群体",
            "论文贡献是否对该领域有意义",
        ],
    },
    ReviewDimension.METHODOLOGY: {
        "name": "方法论评审",
        "description": "评估论文的实验设计、数据集选择、基线对比和消融实验的严谨性",
        "weight": 0.30,
        "scoring_guide": {
            9: "实验设计严谨完整，包含充分的基线对比和消融实验",
            7: "实验设计合理，基线对比较充分，有少量改进空间",
            5: "实验基本合理，但基线或消融实验不够充分",
            3: "实验设计存在明显缺陷，缺少关键对比实验",
            1: "实验设计严重不足，缺乏说服力",
        },
        "check_points": [
            "实验设计是否合理且可复现",
            "数据集选择是否恰当且有代表性",
            "是否与足够的基线方法进行了对比",
            "消融实验是否充分验证了各组件的贡献",
            "统计显著性是否正确报告",
            "超参数选择是否有合理解释",
        ],
    },
    ReviewDimension.NOVELTY: {
        "name": "创新性评审",
        "description": "评估论文的创新程度，包括与现有工作的对比和实际贡献度",
        "weight": 0.30,
        "scoring_guide": {
            9: "提出了全新的方法或理论框架，具有重大突破性贡献",
            7: "在现有方法上有显著创新，贡献明确且有价值",
            5: "有一定创新但增量性质，贡献适中",
            3: "创新性有限，与现有工作差异不大",
            1: "缺乏实质性创新，属于已有工作的简单重复",
        },
        "check_points": [
            "是否清楚阐述了与已有工作的区别",
            "技术创新点是否真实有效而非表面改动",
            "理论贡献或实际应用价值如何",
            "相关工作是否被充分讨论和引用",
            "创新点是否经过充分验证",
        ],
    },
    ReviewDimension.CLARITY: {
        "name": "可读性评审",
        "description": "评估论文的结构清晰度、图表质量和写作规范性",
        "weight": 0.20,
        "scoring_guide": {
            9: "论文结构清晰，写作精炼，图表质量极高，易于理解",
            7: "论文结构合理，写作流畅，图表清晰可读",
            5: "论文结构基本合理，但部分章节可读性有待提高",
            3: "论文组织混乱，写作质量差，图表不清晰",
            1: "论文几乎无法理解，缺少基本的写作规范",
        },
        "check_points": [
            "论文整体结构是否逻辑清晰",
            "各章节之间的过渡是否自然",
            "图表是否有清晰的标题和说明",
            "数学符号和公式是否定义清楚",
            "语言表达是否准确、简洁、无歧义",
            "参考文献格式是否规范",
        ],
    },
}


# ---------------------------------------------------------------------------
# Recommendation thresholds
# ---------------------------------------------------------------------------

SCORE_TO_RECOMMENDATION = {
    (8.5, 10.0): "Strong Accept",
    (7.0, 8.5): "Accept",
    (5.5, 7.0): "Weak Accept",
    (4.5, 5.5): "Borderline",
    (3.0, 4.5): "Weak Reject",
    (1.5, 3.0): "Reject",
    (1.0, 1.5): "Strong Reject",
}


def get_recommendation(overall_score: float) -> str:
    """Map an overall score to a recommendation label."""
    for (lo, hi), label in SCORE_TO_RECOMMENDATION.items():
        if lo <= overall_score < hi:
            return label
    if overall_score >= 8.5:
        return "Strong Accept"
    return "Strong Reject"


def get_criteria(dimension: ReviewDimension) -> dict[str, Any]:
    """Return the criteria dict for a given dimension."""
    return REVIEW_CRITERIA[dimension]


def get_all_criteria() -> dict[ReviewDimension, dict[str, Any]]:
    """Return all criteria."""
    return REVIEW_CRITERIA