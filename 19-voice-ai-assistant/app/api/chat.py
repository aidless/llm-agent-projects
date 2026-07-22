"""文本聊天 API 路由"""

import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models import ChatRequest, ChatResponse, ChatStreamChunk, ErrorResponse

router = APIRouter(prefix="/chat", tags=["Chat"])

# 全局对话管理器（由 main.py 注入）
_dialog_manager = None


def set_dialog_manager(manager):
    """设置对话管理器"""
    global _dialog_manager
    _dialog_manager = manager


def get_dialog_manager():
    """获取对话管理器"""
    if _dialog_manager is None:
        raise HTTPException(status_code=503, detail="Dialog manager not initialized")
    return _dialog_manager


@router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest):
    """发送文本消息，获取 AI 回复"""
    manager = get_dialog_manager()

    session_id = request.session_id or str(uuid.uuid4())[:12]

    try:
        result = await manager.process_text_sync(
            session_id=session_id,
            user_text=request.message,
        )

        return ChatResponse(
            session_id=result["session_id"],
            response=result["text"],
            intent=result.get("intent", ""),
            confidence=result.get("confidence", 0.0),
            turn_count=result.get("turn_count", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message/stream")
async def chat_message_stream(request: ChatRequest):
    """发送文本消息，流式获取 AI 回复（SSE）"""
    manager = get_dialog_manager()

    session_id = request.session_id or str(uuid.uuid4())[:12]

    async def event_generator():
        try:
            async for chunk in manager.process_text_stream(
                session_id=session_id,
                user_text=request.message,
            ):
                yield f"data: {chunk.model_dump_json(by_alias=False)}\n\n"
        except Exception as e:
            error_chunk = ChatStreamChunk(
                session_id=session_id,
                text=f"Error: {str(e)}",
                is_final=True,
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    manager = get_dialog_manager()
    manager.remove_session(session_id)
    return {"status": "ok", "session_id": session_id}
