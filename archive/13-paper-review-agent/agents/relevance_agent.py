"""Relevance review agent."""

from __future__ import annotations

from typing import Any

from agents.base import BaseReviewAgent
from app.models import ParsedPaper, ReviewDimension


RELEVANCE_PROMPT_TEMPLATE = """你是一位经验丰富的学术论文评审专家，专注于评估论文与目标会议/期刊的相关性。

## 评审维度：相关性 (Relevance)
## 目标会议/期刊：{venue}

### 评审标准
- 论文主题是否在会议/期刊的征稿范围内
- 研究问题是否为该领域当前关注的热点
- 方法论是否适合该会议的读者群体
- 论文贡献是否对该领域有意义

### 论文信息
**标题**: {title}
**摘要**: {abstract}
**关键词**: {keywords}

### 全文
{full_text}

### 输出要求
请以JSON格式返回评审结果：
{{
    "score": <1-10的浮点数>,
    "confidence": <0-1的置信度>,
    "strengths": ["优点1", "优点2", ...],
    "weaknesses": ["缺点1", "缺点2", ...],
    "suggestions": ["建议1", "建议2", ...],
    "summary": "<一句话总结>"
}}
"""


class RelevanceAgent(BaseReviewAgent):
    dimension = ReviewDimension.RELEVANCE
    agent_name = "relevance"

    def _build_prompt(self, paper: ParsedPaper, venue: str) -> str:
        return RELEVANCE_PROMPT_TEMPLATE.format(
            venue=venue,
            title=paper.structure.title,
            abstract=paper.structure.abstract or "(no abstract)",
            keywords=", ".join(paper.keywords) or "(none)",
            full_text=paper.raw_text[:8000],
        )

    def _default_response(self) -> dict[str, Any]:
        return {
            "score": 5.0,
            "confidence": 0.7,
            "strengths": ["论文主题与会议有一定相关性"],
            "weaknesses": ["相关性不够突出，需要更明确地阐述与会议主题的匹配"],
            "suggestions": ["在引言中更明确地说明研究问题与目标会议/期刊的契合点"],
            "summary": "论文与目标会议主题有一定关联，但相关性的论述不够充分。",
        }