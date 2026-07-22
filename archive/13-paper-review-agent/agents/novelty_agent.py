"""Novelty review agent."""

from __future__ import annotations

from typing import Any

from agents.base import BaseReviewAgent
from app.models import ParsedPaper, ReviewDimension


NOVELTY_PROMPT_TEMPLATE = """你是一位经验丰富的学术论文评审专家，专注于评估论文的创新性和贡献度。

## 评审维度：创新性 (Novelty)
## 目标会议/期刊：{venue}

### 评审标准
- 是否清楚阐述了与已有工作的区别
- 技术创新点是否真实有效而非表面改动
- 理论贡献或实际应用价值如何
- 相关工作是否被充分讨论和引用
- 创新点是否经过充分验证

### 论文信息
**标题**: {title}
**摘要**: {abstract}
**引言**: {introduction}
**方法**: {method}
**参考文献**: {references}

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


class NoveltyAgent(BaseReviewAgent):
    dimension = ReviewDimension.NOVELTY
    agent_name = "novelty"

    def _build_prompt(self, paper: ParsedPaper, venue: str) -> str:
        return NOVELTY_PROMPT_TEMPLATE.format(
            venue=venue,
            title=paper.structure.title,
            abstract=paper.structure.abstract or "(no abstract)",
            introduction=paper.structure.introduction or "(no introduction)",
            method=paper.structure.method or "(no method section)",
            references=paper.structure.references or "(no references)",
            full_text=paper.raw_text[:8000],
        )

    def _default_response(self) -> dict[str, Any]:
        return {
            "score": 5.0,
            "confidence": 0.7,
            "strengths": ["提出了一个新的视角/方法"],
            "weaknesses": ["与现有工作的区别阐述不够清晰", "创新增量有限"],
            "suggestions": ["更详细地对比与现有方法的差异", "增加理论分析支持创新性"],
            "summary": "论文有一定创新性，但与现有工作的差异化论述需要加强。",
        }