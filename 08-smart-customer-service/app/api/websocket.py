"""WebSocket 端点"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

# 连接管理
_active_connections: dict[str, WebSocket] = {}
# 心跳间隔
_HEARTBEAT_INTERVAL = 30
# 打字状态超时
_TYPING_TIMEOUT = 5


class ConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._session_to_conn: dict[str, str] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        self._connections[session_id] = websocket
        self._session_to_conn[session_id] = session_id

    def disconnect(self, session_id: str) -> None:
        self._connections.pop(session_id, None)
        self._session_to_conn.pop(session_id, None)

    async def send_message(self, session_id: str, message: dict) -> bool:
        conn = self._connections.get(session_id)
        if conn:
            try:
                await conn.send_json(message)
                return True
            except Exception:
                self.disconnect(session_id)
        return False

    async def broadcast(self, message: dict) -> None:
        disconnected = []
        for sid, conn in self._connections.items():
            try:
                await conn.send_json(message)
            except Exception:
                disconnected.append(sid)
        for sid in disconnected:
            self.disconnect(sid)

    def is_connected(self, session_id: str) -> bool:
        return session_id in self._connections

    def get_active_count(self) -> int:
        return len(self._connections)


manager = ConnectionManager()


def _get_components():
    from app.main import (
        dialog_history,
        entity_extractor,
        intent_classifier,
        knowledge_base,
        response_generator,
        retriever,
        sentiment_analyzer,
        state_tracker,
        transfer_queue,
        dialog_policy,
    )
    return {
        "intent_classifier": intent_classifier,
        "sentiment_analyzer": sentiment_analyzer,
        "entity_extractor": entity_extractor,
        "response_generator": response_generator,
        "state_tracker": state_tracker,
        "dialog_policy": dialog_policy,
        "dialog_history": dialog_history,
        "retriever": retriever,
        "transfer_queue": transfer_queue,
    }


def _make_outgoing(msg_type: str, session_id: str, content: Optional[str] = None, metadata: dict = None) -> dict:
    return {
        "type": msg_type,
        "session_id": session_id,
        "content": content,
        "message_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": metadata or {},
    }


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket 主端点"""
    await manager.connect(websocket, session_id)

    try:
        # 发送连接确认
        await manager.send_message(
            session_id,
            _make_outgoing("system_notice", session_id, "连接成功 / Connected"),
        )

        while True:
            # 接收消息（带超时，用于心跳检测）
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=_HEARTBEAT_INTERVAL + 5)
            except asyncio.TimeoutError:
                # 发送心跳探测
                await manager.send_message(
                    session_id,
                    _make_outgoing("heartbeat", session_id, "ping"),
                )
                continue

            try:
                incoming = json.loads(data)
            except json.JSONDecodeError:
                await manager.send_message(
                    session_id,
                    _make_outgoing("system_notice", session_id, "消息格式错误 / Invalid message format"),
                )
                continue

            msg_type = incoming.get("type", "chat_message")
            content = incoming.get("content", "")

            # 处理心跳响应
            if msg_type == "heartbeat":
                if content == "pong":
                    continue
                await manager.send_message(
                    session_id,
                    _make_outgoing("heartbeat", session_id, "pong"),
                )
                continue

            # 处理已读回执
            if msg_type == "read_receipt":
                await manager.send_message(
                    session_id,
                    _make_outgoing("read_receipt", session_id, metadata={"ack": True}),
                )
                continue

            # 处理聊天消息
            if msg_type == "chat_message" and content:
                # 发送打字状态
                await manager.send_message(
                    session_id,
                    _make_outgoing("typing_indicator", session_id, metadata={"typing": True}),
                )

                # 处理消息
                c = _get_components()
                intent_result = c["intent_classifier"].classify(content)
                sentiment_result = c["sentiment_analyzer"].analyze(content)
                entities = c["entity_extractor"].extract(content)

                c["state_tracker"].update_intent(
                    session_id, intent_result.intent, intent_result.sub_intent
                )
                for etype, vals in entities.items():
                    if vals:
                        c["state_tracker"].fill_slot(session_id, etype, vals[0])

                dialog_state = c["state_tracker"].get_state(session_id)
                decision = c["dialog_policy"].decide(intent_result, sentiment_result, dialog_state)

                need_transfer = False
                knowledge_results = None
                pending_slot = None
                transfer_reason = None

                if decision.action.value == "transfer_human":
                    need_transfer = True
                    transfer_reason = decision.transfer_reason
                    from app.models import TransferReason
                    c["transfer_queue"].enqueue(
                        session_id=session_id,
                        priority=5 if sentiment_result.is_angry else 0,
                        transfer_reason=transfer_reason or TransferReason.USER_REQUEST,
                    )
                    # 发送转接通知
                    await manager.send_message(
                        session_id,
                        _make_outgoing("transfer_notice", session_id, content=f"正在为您转接人工客服... / Transferring to human agent..."),
                    )
                elif decision.action.value == "ask_slot":
                    pending_slot = decision.pending_slot
                else:
                    knowledge_results = c["retriever"].retrieve(content, top_k=3)

                reply, sources = c["response_generator"].generate(
                    intent=intent_result.intent,
                    sub_intent=intent_result.sub_intent,
                    sentiment=sentiment_result.label,
                    is_angry=sentiment_result.is_angry,
                    knowledge_results=knowledge_results,
                    pending_slot=pending_slot,
                )

                c["dialog_history"].add_message(session_id, role="user", content=content)
                c["dialog_history"].add_message(session_id, role="agent", content=reply, metadata={"sources": sources})
                c["state_tracker"].increment_turn(session_id)

                # 发送回复
                await manager.send_message(
                    session_id,
                    _make_outgoing(
                        "chat_message", session_id, reply,
                        metadata={
                            "sources": sources,
                            "intent": intent_result.intent.value,
                            "confidence": intent_result.confidence,
                            "sentiment": sentiment_result.label.value,
                            "need_transfer": need_transfer,
                        },
                    ),
                )

                # 发送停止打字状态
                await manager.send_message(
                    session_id,
                    _make_outgoing("typing_indicator", session_id, metadata={"typing": False}),
                )

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        manager.disconnect(session_id)


@router.websocket("/ws/reconnect/{session_id}")
async def websocket_reconnect(websocket: WebSocket, session_id: str):
    """WebSocket 重连端点"""
    # 先断开旧连接
    manager.disconnect(session_id)
    # 建立新连接
    await websocket_endpoint(websocket, session_id)
