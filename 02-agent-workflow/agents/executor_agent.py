# ============================================
# Executor Agent - 执行 Agent
# 负责按照计划执行具体任务，调用工具完成操作
# ============================================

import json
from typing import Optional
from loguru import logger

from agents.base_agent import BaseAgent
from tools import ALL_TOOLS

# Executor Agent 的系统提示词
EXECUTOR_SYSTEM_PROMPT = """你是一个高效的任务执行者（Executor Agent）。

你的职责：
1. 根据计划中的步骤逐一执行子任务
2. 选择合适的工具完成每个步骤
3. 记录每一步的执行结果
4. 处理执行过程中的异常情况

你可以使用以下工具：
- web_search_tool: 网页搜索
- file_read_tool: 读取文件
- file_write_tool: 写入文件
- code_execute_tool: 执行代码
- calculator_tool: 数学计算
- api_call_tool: 调用 API

执行规则：
1. 严格按照计划中的步骤顺序执行
2. 每个步骤执行后报告结果
3. 如果某步骤失败，说明失败原因并建议下一步
4. 使用工具获取的实际数据作为输出基础
5. 输出格式为 JSON，包含 execution_results 字段

输出格式：
{
    "execution_results": [
        {
            "step_id": 1,
            "action": "步骤名称",
            "status": "success|failed",
            "result": "执行结果详情",
            "tool_used": "使用的工具",
            "data": {}  // 步骤产出的数据
        }
    ],
    "final_output": "最终执行结果摘要",
    "errors": []  // 错误列表（如果有）
}
"""


class ExecutorAgent(BaseAgent):
    """执行 Agent：工具调用与任务执行"""

    def __init__(self):
        super().__init__(
            name="ExecutorAgent",
            description="负责执行具体任务、调用工具、完成子任务",
            system_prompt=EXECUTOR_SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=4096,
        )
        # 绑定工具到 LLM
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)

    async def execute(self, task: str, state: dict) -> dict:
        """
        执行任务

        Args:
            task: 任务描述
            state: 当前工作流状态

        Returns:
            dict: 执行结果
        """
        logger.info(f"[ExecutorAgent] 开始执行任务: {task[:100]}...")

        plan = state.get("current_plan", {})
        steps = plan.get("steps", [])

        if not steps:
            # 如果没有结构化计划，直接执行任务
            return await self._execute_without_plan(task, state)

        # 按计划逐步执行
        all_results = []
        collected_data = state.get("collected_data", {})

        for step in steps:
            step_result = await self._execute_step(step, state, collected_data)
            all_results.append(step_result)

            # 收集步骤产出的数据
            if step_result.get("status") == "success" and step_result.get("data"):
                collected_data[f"step_{step.get('step_id', len(all_results))}"] = step_result["data"]

        return {
            "execution_results": all_results,
            "collected_data": collected_data,
            "final_output": self._summarize_results(all_results),
            "status": "executed",
        }

    async def _execute_without_plan(self, task: str, state: dict) -> dict:
        """无计划时直接执行任务"""
        context = {}
        if state.get("search_results"):
            context["搜索结果"] = state["search_results"]

        # 使用带工具的 LLM 执行
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"请执行以下任务:\n{task}"},
        ]

        # 尝试通过工具链执行（最多 5 轮工具调用）
        max_tool_rounds = 5
        for round_num in range(max_tool_rounds):
            response = await self.llm_with_tools.ainvoke(messages)
            messages.append({"role": "assistant", "content": response.content, "tool_calls": []})

            # 检查是否需要调用工具
            if not response.tool_calls:
                break

            # 执行工具调用
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                # 查找并执行对应工具
                tool_result = await self._invoke_tool(tool_name, tool_args)
                messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call["id"],
                })

        # 解析最终结果
        final_content = messages[-1]["content"] if messages else ""
        return {
            "execution_results": [{"step_id": 1, "action": "direct_execution", "status": "success", "result": final_content}],
            "collected_data": {},
            "final_output": final_content,
            "status": "executed",
        }

    async def _execute_step(self, step: dict, state: dict, collected_data: dict) -> dict:
        """执行单个步骤"""
        step_id = step.get("step_id", 0)
        action = step.get("action", "未知步骤")
        tool_name = step.get("tool", "")
        input_desc = step.get("input", "")

        logger.info(f"[ExecutorAgent] 执行步骤 {step_id}: {action}")

        # 构建步骤执行提示
        step_prompt = (
            f"请执行步骤 {step_id}: {action}\n"
            f"描述: {step.get('description', '')}\n"
            f"建议工具: {tool_name}\n"
            f"输入: {input_desc}\n"
            f"预期输出: {step.get('expected_output', '')}"
        )

        # 添加已有数据上下文
        if collected_data:
            step_prompt += f"\n\n已收集的数据:\n{json.dumps(collected_data, ensure_ascii=False, indent=2)[:2000]}"

        # 调用 LLM 执行步骤
        try:
            result = await self.invoke(step_prompt)
            return {
                "step_id": step_id,
                "action": action,
                "status": "success",
                "result": result,
                "tool_used": tool_name,
            }
        except Exception as e:
            logger.error(f"[ExecutorAgent] 步骤 {step_id} 执行失败: {e}")
            return {
                "step_id": step_id,
                "action": action,
                "status": "failed",
                "result": str(e),
                "tool_used": tool_name,
            }

    async def _invoke_tool(self, tool_name: str, args: dict) -> str:
        """通过名称调用对应工具"""
        tool_map = {t.name: t for t in ALL_TOOLS}

        if tool_name not in tool_map:
            return f"错误：未知工具 '{tool_name}'"

        try:
            tool = tool_map[tool_name]
            result = await tool.ainvoke(args)
            return str(result)
        except Exception as e:
            return f"工具调用失败: {str(e)}"

    def _summarize_results(self, results: list) -> str:
        """汇总所有步骤的执行结果"""
        success_count = sum(1 for r in results if r.get("status") == "success")
        failed_count = sum(1 for r in results if r.get("status") == "failed")

        summary = f"执行完成：{success_count} 个步骤成功"
        if failed_count > 0:
            summary += f"，{failed_count} 个步骤失败"

        return summary
