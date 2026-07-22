"""Clarity review agent."""

from __future__ import annotations

from typing import Any

from agents.base import BaseReviewAgent
from app.models import ParsedPaper, ReviewDimension


CLARITY_PROMPT_TEMPLATE = """你是一位经验丰富的学术论文评审专家，专注于评估论文的可读性和写作质量。

## 评审维度：可读性 (Clarity)
## 目标会议/期刊：{venue}

### 评审标准
- 论文整体结构是否逻辑清晰
- 各章节之间的过渡是否自然
- 图表是否有清晰的标题和说明
- 数学符号和公式是否定义清楚
- 语言表达是否准确、简洁、无歧义
- 参考文献格式是否规范

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


class ClarityAgent(BaseReviewAgent):
    dimension = ReviewDimension.CLARITY
    agent_name = "clarity"

    def _build_prompt(self, paper: ParsedPaper, venue: str) -> str:
        return CLARITY_PROMPT_TEMPLATE.format(
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
            "strengths": ["论文结构基本合理"],
            "weaknesses": ["部分段落表述不够清晰", "图表缺少充分说明"],
            "suggestions": ["改进段落间的过渡", "为所有图表添加详细说明"],
            "summary": "论文可读性一般，结构和表述有改进空间。",
        }