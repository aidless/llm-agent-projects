# ============================================
# 数据模型定义
# 定义 API 请求和响应的 Pydantic 模型
# ============================================

from typing import Optional, List, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ---- 请求模型 ----

class TaskRequest(BaseModel):
    """任务请求"""
    task: str = Field(..., description="任务描述", min_length=1, max_length=5000)
    task_id: Optional[str] = Field(None, description="任务 ID（可选，自动生成）")
    max_retries: Optional[int] = Field(3, description="最大重试次数", ge=0, le=10)


class ChatRequest(BaseModel):
    """对话请求"""
    message: str = Field(..., description="用户消息", min_length=1, max_length=5000)
    session_id: Optional[str] = Field(None, description="会话 ID")


class MemoryAddRequest(BaseModel):
    """添加记忆请求"""
    content: str = Field(..., description="记忆内容", min_length=1)
    metadata: Optional[dict] = Field(None, description="元数据")


class MemorySearchRequest(BaseModel):
    """搜索记忆请求"""
    query: str = Field(..., description="搜索查询", min_length=1)
    top_k: Optional[int] = Field(5, description="返回数量", ge=1, le=20)


class ToolCallRequest(BaseModel):
    """工具调用请求"""
    tool_name: str = Field(..., description="工具名称")
    arguments: dict = Field(default_factory=dict, description="工具参数")


# ---- 响应模型 ----

class TaskResponse(BaseModel):
    """任务响应"""
    task_id: str = Field(..., description="任务 ID")
    status: str = Field(..., description="任务状态")
    final_report: Optional[str] = Field(None, description="最终报告")
    score: Optional[int] = Field(None, description="审核评分")
    error: Optional[str] = Field(None, description="错误信息")
    created_at: str = Field(..., description="创建时间")
    completed_at: Optional[str] = Field(None, description="完成时间")


class ChatResponse(BaseModel):
    """对话响应"""
    message: str = Field(..., description="回复消息")
    session_id: str = Field(..., description="会话 ID")
    turn_count: int = Field(..., description="当前轮数")


class ToolResponse(BaseModel):
    """工具调用响应"""
    tool_name: str = Field(..., description="工具名称")
    result: str = Field(..., description="执行结果")
    success: bool = Field(..., description="是否成功")


class MemoryItem(BaseModel):
    """记忆条目"""
    id: str = Field(..., description="记忆 ID")
    content: str = Field(..., description="记忆内容")
    metadata: dict = Field(default_factory=dict, description="元数据")
    distance: Optional[float] = Field(None, description="相似度距离")


class MemorySearchResponse(BaseModel):
    """记忆搜索响应"""
    query: str = Field(..., description="搜索查询")
    results: List[MemoryItem] = Field(default_factory=list, description="搜索结果")
    total: int = Field(0, description="结果总数")


class WorkflowInfoResponse(BaseModel):
    """工作流信息响应"""
    name: str = Field(..., description="工作流名称")
    description: str = Field(..., description="工作流描述")
    flow: List[dict] = Field(default_factory=list, description="流程步骤")
    max_retries: int = Field(..., description="最大重试次数")
    long_term_memory_count: int = Field(0, description="长期记忆条数")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field("healthy", description="服务状态")
    version: str = Field("1.0.0", description="版本号")
    llm_provider: str = Field(..., description="LLM 提供商")
    llm_model: str = Field(..., description="LLM 模型")
    memory_count: int = Field(0, description="长期记忆数量")


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误类型")
    detail: str = Field(..., description="错误详情")
