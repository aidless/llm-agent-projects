# ============================================
# Summarizer Agent - 总结 Agent
# 负责将执行结果整理为结构化的最终输出
# ============================================

import json
from typing import Optional
from datetime import datetime
from loguru import logger

from agents.base_agent import BaseAgent

# Summarizer Agent 的系统提示词
SUMMARIZER_SYSTEM_PROMPT = """你是一个专业的文档撰写和总结专家（Summarizer Agent）。

你的职责：
1. 将各步骤的执行结果整理为结构化的最终报告
2. 确保报告逻辑清晰、层次分明
3. 对关键信息进行提炼和归纳
4. 生成可读性强的最终输出

报告格式要求：
- 使用 Markdown 格式
- 包含标题、摘要、正文、结论
- 数据部分使用表格或列表展示
- 适当使用加粗和引用来突出重点

输出结构：
# {报告标题}

## 摘要
{一段话总结整个报告的核心内容}

## 背景与目标
{描述任务背景和目标}

## 详细内容
{按主题分节的详细内容}

## 关键发现
- 发现1
- 发现2
- 发现3

## 结论与建议
{总结性结论和行动建议}

---
*报告生成时间: {timestamp}*
*由 AI Agent 工作流自动生成*
"""

SUMMARIZER_PROMPT_TEMPLATE = """请根据以下信息生成一份结构化的研究报告。

## 任务主题
{task}

## 执行计划概要
{plan_summary}

## 搜索与分析数据
{search_data}

## 执行步骤结果
{execution_data}

## 审核信息
审核评分: {review_score}
审核结论: {review_verdict}

请生成一份完整、专业的研究报告。确保内容详实、逻辑清晰、格式规范。
"""


class SummarizerAgent(BaseAgent):
    """总结 Agent：结果整理与报告生成"""

    def __init__(self):
        super().__init__(
            name="SummarizerAgent",
            description="负责整理执行结果、生成结构化报告和总结",
            system_prompt=SUMMARIZER_SYSTEM_PROMPT,
            temperature=0.7,  # 较高温度以获得更自然的写作风格
            max_tokens=8192,  # 更大的 token 限制用于生成完整报告
        )

    async def execute(self, task: str, state: dict) -> dict:
        """
        生成最终报告

        Args:
            task: 原始任务描述
            state: 当前工作流状态

        Returns:
            dict: 包含最终报告的状态更新
        """
        logger.info(f"[SummarizerAgent] 开始生成报告: {task[:100]}...")

        # 构建总结提示词
        summarize_prompt = self._build_summarize_prompt(task, state)
        report = await self.invoke(summarize_prompt)

        # 构建报告元数据
        report_meta = {
            "title": task[:100],
            "generated_at": datetime.now().isoformat(),
            "task": task,
            "review_score": state.get("score", "N/A"),
            "total_steps": len(state.get("execution_results", [])),
        }

        # 保存报告到长期记忆
        memory_id = None
        if "long_term_memory" in state:
            try:
                memory_id = state["long_term_memory"].add_memory(
                    content=f"报告标题: {task}\n报告内容摘要: {report[:500]}",
                    metadata={"type": "report", "task": task[:200]},
                )
                logger.debug(f"[SummarizerAgent] 报告已保存到长期记忆, id={memory_id}")
            except Exception as e:
                logger.warning(f"[SummarizerAgent] 保存长期记忆失败: {e}")

        return {
            "final_report": report,
            "report_metadata": report_meta,
            "memory_id": memory_id,
            "status": "completed",
        }

    def _build_summarize_prompt(self, task: str, state: dict) -> str:
        """构建总结提示词"""
        plan = state.get("current_plan", {})
        search_results = state.get("search_results", "")
        execution_results = state.get("execution_results", [])
        review_result = state.get("review_result", {})
        collected_data = state.get("collected_data", {})

        # 计划摘要
        plan_summary = ""
        if plan:
            steps = plan.get("steps", [])
            plan_summary = f"共 {len(steps)} 个步骤"
            for step in steps:
                plan_summary += f"\n  {step.get('step_id', '?')}. {step.get('action', '未知')} - {step.get('description', '')[:100]}"

        # 搜索数据
        search_data = search_results if search_results else "未进行搜索"

        # 执行数据
        execution_data = ""
        if execution_results:
            for result in execution_results:
                execution_data += (
                    f"\n步骤 {result.get('step_id', '?')}: {result.get('action', '')}\n"
                    f"  状态: {result.get('status', 'unknown')}\n"
                    f"  结果: {str(result.get('result', ''))[:500]}\n"
                )

        # 审核信息
        review_score = review_result.get("overall_score", "N/A")
        review_verdict = review_result.get("verdict", "未审核")

        prompt = SUMMARIZER_PROMPT_TEMPLATE.format(
            task=task,
            plan_summary=plan_summary,
            search_data=str(search_data)[:3000],
            execution_data=execution_data[:3000],
            review_score=review_score,
            review_verdict=review_verdict,
        )

        return prompt
