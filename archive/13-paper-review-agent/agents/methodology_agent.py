"""Methodology review agent."""

from __future__ import annotations

from typing import Any

from agents.base import BaseReviewAgent
from app.models import ParsedPaper, ReviewDimension


METHODOLOGY_PROMPT_TEMPLATE = """你是一位经验丰富的学术论文评审专家，专注于评估论文的方法论严谨性。

## 评审维度：方法论 (Methodology)
## 目标会议/期刊：{venue}

### 评审标准
- 实验设计是否合理且可复现
- 数据集选择是否恰当且有代表性
- 是否与足够的基线方法进行了对比
- 消融实验是否充分验证了各组件的贡献
- 统计显著性是否正确报告
- 超参数选择是否有合理解释

### 论文信息
**标题**: {title}
**方法部分**: {method}
**实验部分**: {experiments}
**数据集**: {datasets}

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


class MethodologyAgent(BaseReviewAgent):
    dimension = ReviewDimension.METHODOLOGY
    agent_name = "methodology"

    def _build_prompt(self, paper: ParsedPaper, venue: str) -> str:
        return METHODOLOGY_PROMPT_TEMPLATE.format(
            venue=venue,
            title=paper.structure.title,
            method=paper.structure.method or "(no method section found)",
            experiments=paper.structure.experiments or "(no experiments section found)",
            datasets=", ".join(paper.datasets) or "(none identified)",
            full_text=paper.raw_text[:8000],
        )

    def _default_response(self) -> dict[str, Any]:
        return {
            "score": 5.0,
            "confidence": 0.7,
            "strengths": ["实验设计基本合理"],
            "weaknesses": ["缺少充分的消融实验", "基线对比方法不够全面"],
            "suggestions": ["增加消融实验验证各模块贡献", "补充更多SOTA基线对比"],
            "summary": "方法论基本合理，但实验严谨性有待提高，需要补充消融实验和基线对比。",
        }