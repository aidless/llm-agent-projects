# ============================================
# 工作流状态定义
# 定义 LangGraph 工作流中传递的状态数据结构
# ============================================

from typing import TypedDict, Optional, List, Any
from datetime import datetime


class WorkflowState(TypedDict, total=False):
    """
    工作流状态类型定义
    所有 Agent 节点之间通过此状态传递数据
    """
    # ---- 基础信息 ----
    task_id: str                     # 任务唯一标识
    task: str                        # 原始任务描述
    created_at: str                  # 创建时间
    updated_at: str                  # 最后更新时间
    status: str                      # 当前状态（pending/planned/executed/reviewed/completed/failed）

    # ---- 规划阶段 ----
    current_plan: dict               # Planner 生成的执行计划
    plan_text: str                   # 计划原文
    current_step: int                # 当前执行到的步骤编号

    # ---- 执行阶段 ----
    execution_results: list          # 各步骤的执行结果列表
    collected_data: dict             # 执行过程中收集的数据
    final_output: str                # 执行阶段的最终输出

    # ---- 搜索数据 ----
    search_results: str             # 搜索工具返回的结果

    # ---- 审核阶段 ----
    review_result: dict              # Reviewer 的审核结果
    review_text: str                 # 审核原文
    passed: bool                     # 是否通过审核
    verdict: str                    # 审核结论
    score: int                      # 审核评分

    # ---- 输出阶段 ----
    final_report: str                # 最终生成的报告
    report_metadata: dict            # 报告元数据
    memory_id: str                   # 长期记忆 ID

    # ---- 异常处理 ----
    retry_count: int                 # 重试计数
    max_retries: int                 # 最大重试次数
    last_error: str                  # 最后一次错误信息
    error_log: list                  # 错误日志列表

    # ---- 记忆模块引用 ----
    long_term_memory: Any            # LongTermMemory 实例引用
    short_term_memory: Any           # ShortTermMemory 实例引用


def create_initial_state(task: str, task_id: str = None) -> WorkflowState:
    """
    创建工作流初始状态

    Args:
        task: 任务描述
        task_id: 任务 ID（可选，默认自动生成 UUID）

    Returns:
        WorkflowState: 初始化的工作流状态
    """
    if task_id is None:
        import uuid
        task_id = str(uuid.uuid4())

    return WorkflowState(
        task_id=task_id,
        task=task,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        status="pending",
        retry_count=0,
        max_retries=3,
        error_log=[],
        execution_results=[],
        collected_data={},
    )
