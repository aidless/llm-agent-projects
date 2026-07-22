"""Pydantic 数据模型定义"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 枚举类型
# ---------------------------------------------------------------------------

class MessageRole(str, Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class IntentCategory(str, Enum):
    """一级意图"""
    CONSULTATION = "consultation"      # 咨询
    COMPLAINT = "complaint"            # 投诉
    AFTER_SALE = "after_sale"          # 售后
    TRANSFER_HUMAN = "transfer_human"  # 转人工
    GREETING = "greeting"              # 问候
    FEEDBACK = "feedback"              # 反馈
    ORDER_QUERY = "order_query"        # 订单查询
    LOGISTICS = "logistics"            # 物流


class SubIntent(str, Enum):
    """二级意图"""
    # 咨询子意图
    PRODUCT_INFO = "product_info"
    PRICE_INQUIRY = "price_inquiry"
    STOCK_INQUIRY = "stock_inquiry"
    # 投诉子意图
    QUALITY_ISSUE = "quality_issue"
    SERVICE_ISSUE = "service_issue"
    DELIVERY_ISSUE = "delivery_issue"
    # 售后子意图
    RETURN_REQUEST = "return_request"
    EXCHANGE_REQUEST = "exchange_request"
    REFUND_REQUEST = "refund_request"
    # 订单子意图
    ORDER_STATUS = "order_status"
    ORDER_CANCEL = "order_cancel"
    ORDER_MODIFY = "order_modify"
    # 物流子意图
    TRACKING = "tracking"
    DELIVERY_TIME = "delivery_time"
    SHIPPING_FEE = "shipping_fee"
    # 反馈子意图
    POSITIVE_FEEDBACK = "positive_feedback"
    SUGGESTION = "suggestion"
    # 通用
    UNKNOWN = "unknown"


class TransferReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    NEGATIVE_SENTIMENT = "negative_sentiment"
    USER_REQUEST = "user_request"
    REPEATED_FAILURE = "repeated_failure"
    COMPLEX_QUERY = "complex_query"


class WebSocketMessageType(str, Enum):
    CHAT_MESSAGE = "chat_message"
    TYPING_INDICATOR = "typing_indicator"
    READ_RECEIPT = "read_receipt"
    TRANSFER_NOTICE = "transfer_notice"
    HEARTBEAT = "heartbeat"
    SYSTEM_NOTICE = "system_notice"


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """HTTP 聊天请求"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str = Field(..., min_length=1, max_length=2000, description="用户消息")
    user_id: Optional[str] = None
    context: Optional[dict[str, Any]] = None


class ChatResponse(BaseModel):
    """聊天响应"""
    session_id: str
    reply: str
    intent: IntentCategory
    sub_intent: SubIntent
    sentiment: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[dict[str, str]] = Field(default_factory=list)
    need_transfer: bool = False
    transfer_reason: Optional[TransferReason] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = 0
    is_active: bool = True
    sentiment_summary: list[dict[str, Any]] = Field(default_factory=list)
    assigned_agent: Optional[str] = None


class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 核心数据结构
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """对话消息"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntentResult(BaseModel):
    """意图识别结果"""
    intent: IntentCategory
    sub_intent: SubIntent
    confidence: float = Field(ge=0.0, le=1.0)
    raw_scores: dict[str, float] = Field(default_factory=dict)


class SentimentResult(BaseModel):
    """情感分析结果"""
    label: SentimentLabel
    score: float = Field(ge=0.0, le=1.0)
    is_angry: bool = False
    anger_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SlotValue(BaseModel):
    """槽位值"""
    name: str
    value: Optional[str] = None
    confirmed: bool = False


class DialogState(BaseModel):
    """对话状态 (DST)"""
    session_id: str
    current_intent: Optional[IntentCategory] = None
    sub_intent: Optional[SubIntent] = None
    slots: dict[str, SlotValue] = Field(default_factory=dict)
    turn_count: int = 0
    is_waiting_for_slot: bool = False
    pending_slot_name: Optional[str] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class TransferRequest(BaseModel):
    """转人工请求"""
    session_id: str
    reason: TransferReason
    context_summary: str
    sentiment_history: list[SentimentResult] = Field(default_factory=list)
    priority: int = Field(default=0, ge=0, le=10)


class QueueTicket(BaseModel):
    """排队票"""
    ticket_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    priority: int = Field(default=0, ge=0, le=10)
    transfer_reason: TransferReason = TransferReason.USER_REQUEST
    status: str = "waiting"  # waiting, serving, completed, cancelled


# ---------------------------------------------------------------------------
# WebSocket 消息
# ---------------------------------------------------------------------------

class WSIncomingMessage(BaseModel):
    """WebSocket 入站消息"""
    type: WebSocketMessageType
    session_id: str
    content: Optional[str] = None
    message_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WSOutgoingMessage(BaseModel):
    """WebSocket 出站消息"""
    type: WebSocketMessageType
    session_id: str
    content: Optional[str] = None
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 知识库相关
# ---------------------------------------------------------------------------

class FAQItem(BaseModel):
    """FAQ 条目"""
    question: str
    answer: str
    category: str
    sub_category: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    id: Optional[str] = None


class KnowledgeAddRequest(BaseModel):
    """知识库添加请求"""
    question: str
    answer: str
    category: str
    sub_category: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)


class KnowledgeSearchRequest(BaseModel):
    """知识库检索请求"""
    query: str
    top_k: int = Field(default=3, ge=1, le=10)
    category: Optional[str] = None


class KnowledgeSearchResult(BaseModel):
    """知识库检索结果"""
    id: str
    question: str
    answer: str
    category: str
    score: float
    source: str = "knowledge_base"


class KnowledgeBaseStats(BaseModel):
    """知识库统计"""
    total_count: int
    categories: dict[str, int]
    last_updated: Optional[datetime] = None
