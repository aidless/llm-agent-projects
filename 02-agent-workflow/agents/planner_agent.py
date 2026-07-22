# ============================================
# Planner Agent - 规划 Agent
# 负责分析任务需求，制定执行计划，将复杂任务分解为子任务
# ============================================

import json
from typing import Optional
from loguru import logger

from agents.base_agent import BaseAgent

# Planner Agent 的系统提示词
PLANNER_SYSTEM_PROMPT = """你是一个专业的任务规划专家（Planner Agent）。

你的职责：
1. 分析用户提出的任务需求
2. 将复杂任务分解为可执行的子任务步骤
3. 为每个子任务分配合适的执行策略
4. 确定子任务之间的依赖关系和执行顺序
5. 评估每个子任务所需的工具和资源

输出格式要求：
你必须以 JSON 格式输出任务计划，结构如下：
{
    "task_summary": "任务概述",
    "steps": [
        {
            "step_id": 1,
            "action": "子任务名称",
            "description": "详细描述",
            "tool": "使用的工具名称",
            "input": "输入参数说明",
            "expected_output": "预期输出",
            "depends_on": []  // 依赖的步骤 ID 列表
        }
    ],
    "estimated_steps": 3,
    "complexity": "low|medium|high"
}

注意：
- 每个步骤必须清晰、可执行
- 步骤之间的依赖关系要合理
- 尽量减少步骤数量，保持高效
- 输出必须是有效的 JSON
"""

PLANNER_REPLAN_PROMPT = """{task_summary}

之前的执行计划：
{previous_plan}

执行结果/问题：
{execution_result}

请根据执行结果重新规划或调整剩余步骤。如果任务已经成功完成，输出：
{{"status": "completed", "final_result": "任务完成总结"}}

如果需要调整计划，按标准格式输出新的计划。
"""


class PlannerAgent(BaseAgent):
    """规划 Agent：任务分析与计划制定"""

    def __init__(self):
        super().__init__(
            name="PlannerAgent",
            description="负责分析任务、制定执行计划、将复杂任务分解为子任务",
            system_prompt=PLANNER_SYSTEM_PROMPT,
            temperature=0.3,  # 较低温度以获得更稳定的规划
            max_tokens=4096,
        )

    async def execute(self, task: str, state: dict) -> dict:
        """
        制定任务执行计划

        Args:
            task: 任务描述
            state: 当前工作流状态

        Returns:
            dict: 包含计划的更新状态
        """
        logger.info(f"[PlannerAgent] 开始规划任务: {task[:100]}...")

        # 构建规划请求，包含已有状态信息
        context = {}
        if state.get("search_results"):
            context["搜索结果摘要"] = state["search_results"][:500]
        if state.get("collected_data"):
            context["已收集数据摘要"] = str(state["collected_data"])[:500]

        plan_text = await self.invoke(task, context=context if context else None)

        # 尝试解析 JSON 计划
        plan = self._parse_plan(plan_text)

        # 如果是重新规划模式
        if state.get("retry_count", 0) > 0:
            replan_prompt = PLANNER_REPLAN_PROMPT.format(
                task_summary=task,
                previous_plan=json.dumps(state.get("current_plan", {}), ensure_ascii=False),
                execution_result=state.get("last_error", "无具体错误"),
            )
            plan_text = await self.invoke(replan_prompt)
            plan = self._parse_plan(plan_text)

        return {
            "current_plan": plan,
            "current_step": 0,
            "plan_text": plan_text,
            "status": "planned",
        }

    def _parse_plan(self, plan_text: str) -> dict:
        """
        解析 Agent 输出为结构化计划

        Args:
            plan_text: Agent 的原始输出文本

        Returns:
            dict: 结构化的执行计划
        """
        # 尝试提取 JSON（可能在 markdown 代码块中）
        json_str = plan_text

        # 处理 markdown 代码块
        if "```json" in plan_text:
            json_str = plan_text.split("```json")[1].split("```")[0].strip()
        elif "```" in plan_text:
            json_str = plan_text.split("```")[1].split("```")[0].strip()

        # 尝试找到 JSON 对象
        if "{" in json_str and "}" in json_str:
            start = json_str.index("{")
            # 找到最后一个 }
            end = len(json_str) - 1 - json_str[::-1].index("}")
            json_str = json_str[start:end + 1]

        try:
            plan = json.loads(json_str)
            logger.info(f"[PlannerAgent] 成功解析计划，步骤数: {len(plan.get('steps', []))}")
            return plan
        except json.JSONDecodeError as e:
            logger.warning(f"[PlannerAgent] JSON 解析失败: {e}，使用原始文本")
            return {
                "task_summary": plan_text[:500],
                "steps": [],
                "plan_text": plan_text,
                "parse_error": str(e),
            }
