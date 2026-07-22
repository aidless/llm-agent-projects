# ============================================
# Reviewer Agent - 审核 Agent
# 负责审核执行结果的质量，检查完整性和准确性
# ============================================

import json
from typing import Optional
from loguru import logger

from agents.base_agent import BaseAgent

# Reviewer Agent 的系统提示词
REVIEWER_SYSTEM_PROMPT = """你是一个严格的质量审核专家（Reviewer Agent）。

你的职责：
1. 审核任务执行结果的完整性和准确性
2. 检查输出是否满足任务要求
3. 识别遗漏、错误或不一致之处
4. 给出改进建议
5. 决定是否需要返工

审核维度：
- 完整性：是否涵盖了任务的所有要求
- 准确性：数据和事实是否正确
- 一致性：逻辑是否自洽，前后是否矛盾
- 可读性：输出格式是否规范，表达是否清晰
- 时效性：信息是否过时（如果涉及时间敏感内容）

输出格式（必须是有效的 JSON）：
{
    "passed": true/false,
    "overall_score": 85,  // 0-100 分
    "completeness": {
        "score": 90,
        "comment": "完整性评价"
    },
    "accuracy": {
        "score": 80,
        "comment": "准确性评价"
    },
    "consistency": {
        "score": 85,
        "comment": "一致性评价"
    },
    "readability": {
        "score": 90,
        "comment": "可读性评价"
    },
    "issues": ["发现的问题1", "发现的问题2"],
    "suggestions": ["改进建议1", "改进建议2"],
    "verdict": "通过/需修改/需重做"
}

审核标准：
- 通过：总分 >= 80 分，且无严重问题
- 需修改：总分 60-79，或有小问题可修复
- 需重做：总分 < 60，或有严重错误
"""


class ReviewerAgent(BaseAgent):
    """审核 Agent：质量检查与结果审核"""

    def __init__(self):
        super().__init__(
            name="ReviewerAgent",
            description="负责审核执行结果的质量、完整性和准确性",
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            temperature=0.3,  # 低温度保证审核的稳定和严格
            max_tokens=4096,
        )

    async def execute(self, task: str, state: dict) -> dict:
        """
        审核任务执行结果

        Args:
            task: 原始任务描述
            state: 当前工作流状态

        Returns:
            dict: 审核结果
        """
        logger.info(f"[ReviewerAgent] 开始审核任务结果")

        # 构建审核请求
        review_prompt = self._build_review_prompt(task, state)
        review_text = await self.invoke(review_prompt)

        # 解析审核结果
        review_result = self._parse_review(review_text)

        passed = review_result.get("passed", False)
        verdict = review_result.get("verdict", "需修改")
        score = review_result.get("overall_score", 0)

        logger.info(
            f"[ReviewerAgent] 审核完成: verdict={verdict}, score={score}, passed={passed}"
        )

        return {
            "review_result": review_result,
            "review_text": review_text,
            "passed": passed,
            "verdict": verdict,
            "score": score,
            "status": "reviewed",
        }

    def _build_review_prompt(self, task: str, state: dict) -> str:
        """构建审核提示词"""
        plan = state.get("current_plan", {})
        execution_results = state.get("execution_results", [])
        final_output = state.get("final_output", "")

        prompt = f"""请审核以下任务执行结果。

## 原始任务
{task}

## 执行计划
{json.dumps(plan, ensure_ascii=False, indent=2)[:2000]}

## 执行结果
{json.dumps(execution_results, ensure_ascii=False, indent=2)[:3000]}

## 最终输出
{final_output[:3000]}

请按照审核维度进行全面评估，并输出 JSON 格式的审核结果。
"""
        return prompt

    def _parse_review(self, review_text: str) -> dict:
        """解析审核结果 JSON"""
        json_str = review_text

        # 处理 markdown 代码块
        if "```json" in review_text:
            json_str = review_text.split("```json")[1].split("```")[0].strip()
        elif "```" in review_text:
            json_str = review_text.split("```")[1].split("```")[0].strip()

        # 提取 JSON
        if "{" in json_str and "}" in json_str:
            start = json_str.index("{")
            end = len(json_str) - 1 - json_str[::-1].index("}")
            json_str = json_str[start:end + 1]

        try:
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"[ReviewerAgent] 审核结果 JSON 解析失败: {e}")
            return {
                "passed": False,
                "overall_score": 0,
                "verdict": "需重做",
                "issues": [f"审核结果解析失败: {e}"],
                "suggestions": ["请重新执行任务"],
                "raw_text": review_text,
            }
