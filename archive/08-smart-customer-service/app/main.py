"""FastAPI 应用入口"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.knowledge import router as knowledge_router
from app.api.session import router as session_router
from app.api.websocket import router as websocket_router
from core.context_manager import ContextManager
from core.entity_extractor import EntityExtractor
from core.intent_classifier import IntentClassifier
from core.response_generator import ResponseGenerator
from core.sentiment_analyzer import SentimentAnalyzer
from dialog.history import DialogHistory
from dialog.policy import DialogPolicy
from dialog.state_tracker import DialogStateTracker
from rag.knowledge_base import KnowledgeBase
from rag.retriever import Retriever
from transfer.escalator import Escalator
from transfer.queue import TransferQueue

# ---------------------------------------------------------------------------
# 项目根目录
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 初始化核心组件
# ---------------------------------------------------------------------------
intent_classifier = IntentClassifier()
sentiment_analyzer = SentimentAnalyzer()
entity_extractor = EntityExtractor()
response_generator = ResponseGenerator()
context_manager = ContextManager()
state_tracker = DialogStateTracker()
dialog_policy = DialogPolicy()
dialog_history = DialogHistory()
escalator = Escalator()
transfer_queue = TransferQueue()

# ---------------------------------------------------------------------------
# 初始化知识库
# ---------------------------------------------------------------------------
faq_data_path = str(PROJECT_ROOT / "rag" / "data" / "faq_data.json")
knowledge_base = KnowledgeBase(data_path=faq_data_path)
retriever = Retriever(knowledge_base=knowledge_base)

# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(
    title="智能客服对话管理系统",
    description="基于 LangChain + RAG + WebSocket 的多轮对话客服系统",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
app.include_router(session_router, prefix="/api/v1", tags=["Session"])
app.include_router(knowledge_router, prefix="/api/v1", tags=["Knowledge"])
app.include_router(websocket_router, tags=["WebSocket"])


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "faq_count": knowledge_base.count()}


@app.get("/")
async def root():
    return {
        "name": "智能客服对话管理系统",
        "version": "1.0.0",
        "docs": "/docs",
    }
