"""Pydantic 请求/响应模型定义。"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """审查问题严重等级。"""

    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFO = "Info"


class Category(str, Enum):
    """审查问题类别。"""

    SECURITY = "Security"
    PERFORMANCE = "Performance"
    STYLE = "Style"
    LOGIC = "Logic"


class Finding(BaseModel):
    """单条审查发现。"""

    file: str = Field(..., description="文件路径")
    line: int = Field(..., description="行号")
    column: Optional[int] = Field(None, description="列号")
    severity: Severity = Field(..., description="严重等级")
    category: Category = Field(..., description="问题类别")
    message: str = Field(..., description="问题描述")
    suggestion: Optional[str] = Field(None, description="修复建议")
    code_snippet: Optional[str] = Field(None, description="相关代码片段")
    rule_id: Optional[str] = Field(None, description="规则标识")


class ReviewResult(BaseModel):
    """单个 Agent 的审查结果。"""

    agent_name: str = Field(..., description="Agent 名称")
    findings: list[Finding] = Field(default_factory=list, description="发现的问题列表")
    summary: str = Field("", description="该 Agent 的审查摘要")
    elapsed_time: float = Field(0.0, description="耗时(秒)")


class ReviewStatus(str, Enum):
    """审查状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CodeSubmitRequest(BaseModel):
    """代码提交审查请求。"""

    code: str = Field(..., description="待审查的代码内容")
    language: str = Field("python", description="编程语言")
    filename: str = Field("untitled.py", description="文件名")
    context: Optional[str] = Field(None, description="额外上下文信息")


class GitHubPRRequest(BaseModel):
    """GitHub PR 审查请求。"""

    repo_owner: str = Field(..., description="仓库所有者")
    repo_name: str = Field(..., description="仓库名称")
    pr_number: int = Field(..., description="PR 编号")
    github_token: Optional[str] = Field(None, description="GitHub Token")
    post_comments: bool = Field(False, description="是否自动发表审查评论")


class ReviewResponse(BaseModel):
    """审查响应。"""

    review_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: ReviewStatus = Field(ReviewStatus.COMPLETED)
    created_at: datetime = Field(default_factory=datetime.now)
    results: list[ReviewResult] = Field(default_factory=list)
    total_findings: int = Field(0)
    report_markdown: str = Field("")
    report_json: dict = Field(default_factory=dict)
    language: str = Field("python")
    filename: str = Field("")


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = "ok"
    version: str = "1.0.0"
    agents_available: list[str] = Field(default_factory=list)
