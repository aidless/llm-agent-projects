"""HTTP 聊天 API"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.models import ChatRequest, ChatResponse, TransferReason
from dialog.policy import DialogPolicy, PolicyDecision
from dialog.state_tracker import DialogStateTracker

router = APIRouter()


def _get_components():
    """延迟导入组件，避免循环依赖"""
    from app.main import (
        context_manager,
        dialog_history,
        dialog_policy,
        entity_extractor,
        intent_classifier,
        knowledge_base,
        response_generator,
        retriever,
        sentiment_analyzer,
        state_tracker,
        transfer_queue,
        escalator,
    )
    return {
        "intent_classifier": intent_classifier,
        "sentiment_analyzer": sentiment_analyzer,
        "entity_extractor": entity_extractor,
        "response_generator": response_generator,
        "context_manager": context_manager,
        "state_tracker": state_tracker,
        "dialog_policy": dialog_policy,
        "dialog_history": dialog_history,
        "retriever": retriever,
        "transfer_queue": transfer_queue,
        "escalator": escalator,
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理聊天请求"""
    c = _get_components()
    message = request.message

    # 1. 意图识别
    intent_result = c["intent_classifier"].classify(message)

    # 2. 情感分析
    sentiment_result = c["sentiment_analyzer"].analyze(message)

    # 3. 实体提取
    entities = c["entity_extractor"].extract(message)

    # 4. 更新对话状态
    c["state_tracker"].update_intent(
        request.session_id,
        intent_result.intent,
        intent_result.sub_intent,
    )
    for entity_type, values in entities.items():
        if values:
            c["state_tracker"].fill_slot(request.session_id, entity_type, values[0])

    dialog_state = c["state_tracker"].get_state(request.session_id)

    # 5. 策略决策
    decision = c["dialog_policy"].decide(intent_result, sentiment_result, dialog_state)

    # 6. 生成回复
    need_transfer = False
    transfer_reason = None
    knowledge_results = None
    pending_slot = None

    if decision.action.value == "transfer_human":
        need_transfer = True
        transfer_reason = decision.transfer_reason
        # 加入排队
        priority = 5 if sentiment_result.is_angry else 0
        c["transfer_queue"].enqueue(
            session_id=request.session_id,
            user_id=request.user_id,
            priority=priority,
            transfer_reason=transfer_reason or TransferReason.USER_REQUEST,
        )
    elif decision.action.value == "ask_slot":
        pending_slot = decision.pending_slot
    else:
        # RAG 检索
        knowledge_results = c["retriever"].retrieve(message, top_k=3)

    reply, sources = c["response_generator"].generate(
        intent=intent_result.intent,
        sub_intent=intent_result.sub_intent,
        sentiment=sentiment_result.label,
        is_angry=sentiment_result.is_angry,
        knowledge_results=knowledge_results,
        pending_slot=pending_slot,
    )

    # 7. 保存历史
    c["dialog_history"].add_message(
        request.session_id,
        role="user",
        content=message,
    )
    c["dialog_history"].add_message(
        request.session_id,
        role="agent",
        content=reply,
        metadata={"sources": sources},
    )
    c["state_tracker"].increment_turn(request.session_id)

    return ChatResponse(
        session_id=request.session_id,
        reply=reply,
        intent=intent_result.intent,
        sub_intent=intent_result.sub_intent,
        sentiment=sentiment_result.label,
        confidence=intent_result.confidence,
        sources=sources,
        need_transfer=need_transfer,
        transfer_reason=transfer_reason,
    )
