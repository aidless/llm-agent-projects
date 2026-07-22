"""会话管理 API"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from app.models import SessionCreateResponse, SessionInfo

router = APIRouter()

# 内存会话存储
_sessions: dict[str, dict] = {}


def _get_components():
    from app.main import (
        context_manager,
        dialog_history,
        sentiment_analyzer,
        state_tracker,
    )
    return {
        "context_manager": context_manager,
        "dialog_history": dialog_history,
        "sentiment_analyzer": sentiment_analyzer,
        "state_tracker": state_tracker,
    }


@router.post("/sessions", response_model=SessionCreateResponse)
async def create_session(user_id: str = None):
    """创建会话"""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "user_id": user_id,
        "is_active": True,
    }
    return SessionCreateResponse(session_id=session_id)


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """获取会话信息"""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    c = _get_components()
    session_data = _sessions[session_id]
    message_count = c["dialog_history"].get_message_count(session_id)

    return SessionInfo(
        session_id=session_id,
        user_id=session_data.get("user_id"),
        message_count=message_count,
        is_active=session_data.get("is_active", True),
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    c = _get_components()
    c["dialog_history"].clear_history(session_id)
    c["state_tracker"].delete_state(session_id)
    c["context_manager"].clear_session(session_id)
    c["sentiment_analyzer"].reset_history()

    del _sessions[session_id]
    return {"status": "ok", "message": "Session deleted"}


@router.get("/sessions/{session_id}/context")
async def get_session_context(session_id: str):
    """获取会话上下文"""
    c = _get_components()
    ctx = c["context_manager"].get_context_summary(session_id)
    history = c["dialog_history"].get_history_dict(session_id, last_n=10)
    return {
        "context": ctx,
        "recent_messages": history,
    }


@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str, last_n: int = 20):
    """获取会话历史"""
    c = _get_components()
    history = c["dialog_history"].get_history_dict(session_id, last_n=last_n)
    return {
        "session_id": session_id,
        "messages": history,
        "total_count": c["dialog_history"].get_message_count(session_id),
    }
