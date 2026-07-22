# ============================================
# 自动研究报告生成工作流
# 使用 LangGraph StateGraph 编排完整的 Agent 协作流程
#
# 流程：用户输入 → 规划(Planner) → 执行(Executor) → 审核(Reviewer) → 
#        [条件分支: 通过则生成报告 / 不通过则重新规划] → 总结(Summarizer) → 输出
# ============================================

import asyncio
from typing import TypedDict, Literal
from datetime import datetime
from loguru import logger

from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph

from workflows.state import WorkflowState, create_initial_state
from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
from agents.reviewer_agent import ReviewerAgent
from agents.summarizer_agent import SummarizerAgent
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from config import get_settings


class ResearchWorkflow:
    """
    自动研究报告生成工作流
    
    完整流程：
    1. 初始化：设置状态和记忆模块
    2. 规划：Planner Agent 分析任务，制定执行计划
    3. 执行：Executor Agent 按计划执行搜索、分析等操作
    4. 审核：Reviewer Agent 检查执行结果质量
    5. 条件分支：
       - 审核通过 → 进入总结阶段
       - 审核未通过且未超重试次数 → 回到规划阶段重新规划
       - 审核未通过且超重试次数 → 强制进入总结阶段
    6. 总结：Summarizer Agent 生成最终报告
    7. 完成：输出最终结果
    """

    def __init__(
        self,
        persist_dir: str = None,
        collection_name: str = None,
        memory_max_turns: int = None,
        max_retries: int = 3,
    ):
        """
        初始化研究报告工作流

        Args:
            persist_dir: ChromaDB 持久化目录
            collection_name: ChromaDB 集合名称
            memory_max_turns: 短期记忆最大轮数
            max_retries: 最大重试次数
        """
        settings = get_settings()

        # 初始化记忆模块
        self.short_term_memory = ShortTermMemory(
            max_turns=memory_max_turns or settings.memory_max_turns
        )
        self.long_term_memory = LongTermMemory(
            persist_dir=persist_dir or settings.chroma_persist_dir,
            collection_name=collection_name or settings.chroma_collection_name,
        )

        # 初始化 Agent
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.reviewer = ReviewerAgent()
        self.summarizer = SummarizerAgent()

        self.max_retries = max_retries

        # 构建 LangGraph 工作流
        self.graph = self._build_graph()

        logger.info(
            f"[ResearchWorkflow] 初始化完成, "
            f"max_retries={max_retries}, "
            f"长期记忆={self.long_term_memory.count()} 条"
        )

    def _build_graph(self) -> CompiledGraph:
        """
        构建 LangGraph StateGraph

        工作流结构：
        init → plan → execute → review → should_revise (条件分支)
                                           ├── revise → plan (循环)
                                           ├── force_summarize → summarize → end
        """
        # 创建状态图
        workflow = StateGraph(WorkflowState)

        # ---- 添加节点 ----
        workflow.add_node("init", self._node_init)
        workflow.add_node("plan", self._node_plan)
        workflow.add_node("execute", self._node_execute)
        workflow.add_node("review", self._node_review)
        workflow.add_node("summarize", self._node_summarize)

        # ---- 设置入口点 ----
        workflow.set_entry_point("init")

        # ---- 添加边 ----
        workflow.add_edge("init", "plan")          # 初始化 → 规划
        workflow.add_edge("plan", "execute")        # 规划 → 执行
        workflow.add_edge("execute", "review")      # 执行 → 审核

        # ---- 添加条件分支 ----
        # 审核 → 条件判断（通过/需要修改/强制完成）
        workflow.add_conditional_edges(
            "review",
            self._should_revise,
            {
                "revise": "plan",               # 需要修改 → 回到规划
                "accept": "summarize",          # 通过 → 总结
                "force_summarize": "summarize", # 超时强制 → 总结
            }
        )

        # 总结 → 结束
        workflow.add_edge("summarize", END)

        # 编译图
        compiled = workflow.compile()
        logger.info("[ResearchWorkflow] LangGraph StateGraph 编译完成")
        return compiled

    # ==========================================
    # 工作流节点函数
    # ==========================================

    async def _node_init(self, state: WorkflowState) -> dict:
        """初始化节点：设置记忆模块引用，记录任务开始"""
        task = state.get("task", "")
        task_id = state.get("task_id", "")

        logger.info(f"[工作流] 初始化任务: task_id={task_id}")

        # 保存任务到短期记忆
        self.short_term_memory.add_user_message(task)
        self.short_term_memory.set_metadata("task_id", task_id)

        # 搜索长期记忆中是否有相关经验
        relevant_memories = self.long_term_memory.search(query=task, top_k=3)
        memory_context = ""
        if relevant_memories:
            memory_context = "历史相关经验:\n"
            for mem in relevant_memories:
                memory_context += f"- {mem['content'][:200]}\n"

        return {
            "status": "initialized",
            "long_term_memory": self.long_term_memory,
            "short_term_memory": self.short_term_memory,
            "search_results": memory_context,
            "updated_at": datetime.now().isoformat(),
        }

    async def _node_plan(self, state: WorkflowState) -> dict:
        """规划节点：调用 Planner Agent 制定执行计划"""
        task = state.get("task", "")
        logger.info(f"[工作流] 规划阶段开始")

        try:
            result = await self.planner.execute(task, state)
            self.short_term_memory.add_assistant_message(
                f"已制定执行计划，共 {len(result.get('current_plan', {}).get('steps', []))} 个步骤"
            )
            return {
                **result,
                "updated_at": datetime.now().isoformat(),
            }
        except Exception as e:
            error_msg = f"规划阶段异常: {type(e).__name__}: {str(e)}"
            logger.error(f"[工作流] {error_msg}")
            return {
                "status": "failed",
                "last_error": error_msg,
                "error_log": state.get("error_log", []) + [error_msg],
                "updated_at": datetime.now().isoformat(),
            }

    async def _node_execute(self, state: WorkflowState) -> dict:
        """执行节点：调用 Executor Agent 执行计划"""
        task = state.get("task", "")
        logger.info(f"[工作流] 执行阶段开始")

        try:
            result = await self.executor.execute(task, state)
            self.short_term_memory.add_assistant_message(
                f"任务执行完成: {result.get('final_output', '无输出')[:200]}"
            )
            return {
                **result,
                "updated_at": datetime.now().isoformat(),
            }
        except Exception as e:
            error_msg = f"执行阶段异常: {type(e).__name__}: {str(e)}"
            logger.error(f"[工作流] {error_msg}")
            return {
                "status": "failed",
                "last_error": error_msg,
                "error_log": state.get("error_log", []) + [error_msg],
                "updated_at": datetime.now().isoformat(),
            }

    async def _node_review(self, state: WorkflowState) -> dict:
        """审核节点：调用 Reviewer Agent 审核执行结果"""
        task = state.get("task", "")
        logger.info(f"[工作流] 审核阶段开始")

        try:
            result = await self.reviewer.execute(task, state)

            self.short_term_memory.add_assistant_message(
                f"审核完成: 评分={result.get('score', 'N/A')}, "
                f"结论={result.get('verdict', '未知')}"
            )

            return {
                **result,
                "updated_at": datetime.now().isoformat(),
            }
        except Exception as e:
            error_msg = f"审核阶段异常: {type(e).__name__}: {str(e)}"
            logger.error(f"[工作流] {error_msg}")
            return {
                "passed": False,
                "verdict": "需重做",
                "score": 0,
                "review_result": {"error": error_msg},
                "last_error": error_msg,
                "error_log": state.get("error_log", []) + [error_msg],
                "updated_at": datetime.now().isoformat(),
            }

    async def _node_summarize(self, state: WorkflowState) -> dict:
        """总结节点：调用 Summarizer Agent 生成最终报告"""
        task = state.get("task", "")
        logger.info(f"[工作流] 总结阶段开始")

        try:
            result = await self.summarizer.execute(task, state)

            self.short_term_memory.add_assistant_message(
                f"已生成最终报告"
            )

            return {
                **result,
                "updated_at": datetime.now().isoformat(),
            }
        except Exception as e:
            error_msg = f"总结阶段异常: {type(e).__name__}: {str(e)}"
            logger.error(f"[工作流] {error_msg}")
            return {
                "status": "failed",
                "last_error": error_msg,
                "error_log": state.get("error_log", []) + [error_msg],
                "final_report": f"报告生成失败: {str(e)}",
                "updated_at": datetime.now().isoformat(),
            }

    # ==========================================
    # 条件分支函数
    # ==========================================

    def _should_revise(self, state: WorkflowState) -> Literal["revise", "accept", "force_summarize"]:
        """
        审核后的条件分支判断

        决定下一步：
        - revise: 审核未通过，需要修改 → 回到规划节点
        - accept: 审核通过 → 进入总结节点
        - force_summarize: 重试次数超限 → 强制进入总结节点
        """
        passed = state.get("passed", False)
        verdict = state.get("verdict", "需修改")
        retry_count = state.get("retry_count", 0)

        if passed and verdict == "通过":
            logger.info(f"[工作流] 审核通过，进入总结阶段")
            return "accept"

        if retry_count >= self.max_retries:
            logger.warning(
                f"[工作流] 重试次数已达上限 ({retry_count}/{self.max_retries})，强制进入总结"
            )
            return "force_summarize"

        # 需要修改，增加重试计数
        logger.info(
            f"[工作流] 审核未通过 (verdict={verdict})，重新规划 "
            f"(重试 {retry_count + 1}/{self.max_retries})"
        )
        return "revise"

    # ==========================================
    # 公共接口
    # ==========================================

    async def run(self, task: str, task_id: str = None) -> dict:
        """
        运行完整的研究报告工作流

        Args:
            task: 研究任务描述
            task_id: 任务 ID（可选）

        Returns:
            dict: 工作流最终状态，包含生成的报告和元数据
        """
        logger.info(f"[ResearchWorkflow] ===== 开始运行工作流 =====")
        logger.info(f"[ResearchWorkflow] 任务: {task[:100]}...")

        # 创建初始状态
        initial_state = create_initial_state(task=task, task_id=task_id)

        # 运行 LangGraph 工作流
        try:
            # 注意：LangGraph 的 StateGraph 每次调用节点函数时，
            # 需要更新 retry_count（在条件分支中隐式处理）
            # 这里通过多次迭代处理 revise 循环
            final_state = await self._run_with_retry_loop(initial_state)

            # 记录最终状态到短期记忆
            final_report = final_state.get("final_report", "")
            if final_report:
                self.short_term_memory.add_assistant_message(final_report[:500])

            # 保存到长期记忆
            try:
                self.long_term_memory.add_memory(
                    content=f"任务: {task}\n结果摘要: {final_report[:300]}",
                    metadata={"type": "workflow_result", "task": task[:200]},
                )
            except Exception as e:
                logger.warning(f"[ResearchWorkflow] 保存长期记忆失败: {e}")

            logger.info(f"[ResearchWorkflow] ===== 工作流完成 =====")
            return dict(final_state)

        except Exception as e:
            error_msg = f"工作流运行异常: {type(e).__name__}: {str(e)}"
            logger.error(f"[ResearchWorkflow] {error_msg}")
            return {
                **dict(initial_state),
                "status": "failed",
                "last_error": error_msg,
                "error_log": [error_msg],
            }

    async def _run_with_retry_loop(self, initial_state: WorkflowState) -> WorkflowState:
        """
        使用 StateGraph 运行工作流，处理 revise 循环

        由于 LangGraph 的条件分支会在编译后自动处理，
        我们通过检查状态来模拟循环行为
        """
        state = dict(initial_state)
        max_iterations = self.max_retries + 2  # 最多迭代次数（防止无限循环）
        iteration = 0

        # 逐步调用节点
        node_sequence = ["init", "plan", "execute", "review"]
        current_node_idx = 0

        while iteration < max_iterations:
            iteration += 1

            if current_node_idx < len(node_sequence):
                current_node = node_sequence[current_node_idx]
            elif current_node_idx == len(node_sequence):
                # review 之后判断
                branch = self._should_revise(state)
                if branch == "accept" or branch == "force_summarize":
                    # 进入总结
                    node_result = await self._node_summarize(state)
                    state.update(node_result)
                    break
                else:
                    # 回到规划
                    state["retry_count"] = state.get("retry_count", 0) + 1
                    current_node_idx = 1  # 回到 plan
                    continue
            else:
                break

            # 执行当前节点
            if current_node == "init":
                node_result = await self._node_init(state)
            elif current_node == "plan":
                node_result = await self._node_plan(state)
            elif current_node == "execute":
                node_result = await self._node_execute(state)
            elif current_node == "review":
                node_result = await self._node_review(state)
            else:
                break

            # 更新状态
            state.update(node_result)
            current_node_idx += 1

        return state

    def get_workflow_info(self) -> dict:
        """获取工作流信息（用于 API 展示）"""
        return {
            "name": "ResearchWorkflow",
            "description": "自动研究报告生成工作流",
            "flow": [
                {"step": 1, "name": "初始化", "agent": "system", "description": "初始化状态和记忆"},
                {"step": 2, "name": "规划", "agent": "PlannerAgent", "description": "分析任务，制定执行计划"},
                {"step": 3, "name": "执行", "agent": "ExecutorAgent", "description": "按计划执行搜索和分析"},
                {"step": 4, "name": "审核", "agent": "ReviewerAgent", "description": "审核执行结果质量"},
                {"step": 5, "name": "条件分支", "agent": "system", "description": "通过→总结 / 未通过→重新规划"},
                {"step": 6, "name": "总结", "agent": "SummarizerAgent", "description": "生成最终研究报告"},
            ],
            "max_retries": self.max_retries,
            "long_term_memory_count": self.long_term_memory.count(),
        }
