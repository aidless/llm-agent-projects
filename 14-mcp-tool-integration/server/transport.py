"""传输层抽象

提供 stdio / SSE / WebSocket 三种传输层的模拟实现。
所有通信在进程内完成，使用消息队列模拟数据传输。
"""

import asyncio
from collections import deque
from typing import Any, Callable, Optional

from protocol.jsonrpc import (
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    parse_message,
    encode_message,
)
from protocol.types import TransportType


class TransportMessage:
    """传输层消息封装"""

    def __init__(self, data: str, source: str = ""):
        self.data = data
        self.source = source


class BaseTransport:
    """传输层基类"""

    def __init__(self, transport_type: TransportType):
        self.transport_type = transport_type
        self._incoming: asyncio.Queue[TransportMessage] = asyncio.Queue()
        self._outgoing: asyncio.Queue[TransportMessage] = asyncio.Queue()
        self._connected = False
        self._message_handler: Optional[Callable] = None

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """建立连接"""
        self._connected = True

    async def disconnect(self) -> None:
        """断开连接"""
        self._connected = False

    async def send(self, data: str) -> None:
        """发送消息"""
        await self._outgoing.put(TransportMessage(data))

    async def receive(self) -> Optional[TransportMessage]:
        """接收消息"""
        try:
            return await asyncio.wait_for(self._incoming.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    def on_message(self, handler: Callable) -> None:
        """设置消息回调"""
        self._message_handler = handler

    async def _push_incoming(self, data: str, source: str = "") -> None:
        """推送消息到入站队列"""
        await self._incoming.put(TransportMessage(data, source))


class StdioTransport(BaseTransport):
    """模拟 stdio 传输层"""

    def __init__(self):
        super().__init__(TransportType.STDIO)


class SSETransport(BaseTransport):
    """模拟 SSE (Server-Sent Events) 传输层"""

    def __init__(self):
        super().__init__(TransportType.SSE)
        self._event_buffer: deque[str] = deque()

    async def send_event(self, event_type: str, data: Any) -> None:
        """发送 SSE 事件"""
        import json
        event_data = json.dumps({"event": event_type, "data": data}, ensure_ascii=False)
        self._event_buffer.append(event_data)

    def get_events(self) -> list[str]:
        """获取已发送的事件"""
        events = list(self._event_buffer)
        self._event_buffer.clear()
        return events


class WebSocketTransport(BaseTransport):
    """模拟 WebSocket 传输层"""

    def __init__(self):
        super().__init__(TransportType.WEBSOCKET)

    async def send_binary(self, data: bytes) -> None:
        """发送二进制数据"""
        await self._outgoing.put(TransportMessage(data.decode("utf-8"), "binary"))


class InMemoryTransport:
    """进程内内存传输 - 连接客户端和服务端

    用于测试和进程内模拟通信。
    """

    def __init__(self):
        self._server_queue: asyncio.Queue[str] = asyncio.Queue()
        self._client_queue: asyncio.Queue[str] = asyncio.Queue()
        self._server_connected = False
        self._client_connected = False

    async def connect(self) -> None:
        self._server_connected = True
        self._client_connected = True

    async def disconnect(self) -> None:
        self._server_connected = False
        self._client_connected = False

    @property
    def connected(self) -> bool:
        return self._server_connected and self._client_connected

    # --- 客户端 -> 服务端 ---
    async def client_send(self, data: str) -> None:
        """客户端发送消息到服务端"""
        await self._server_queue.put(data)

    async def server_receive(self) -> Optional[str]:
        """服务端接收来自客户端的消息"""
        try:
            return await asyncio.wait_for(self._server_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    # --- 服务端 -> 客户端 ---
    async def server_send(self, data: str) -> None:
        """服务端发送消息到客户端"""
        await self._client_queue.put(data)

    async def client_receive(self) -> Optional[str]:
        """客户端接收来自服务端的消息"""
        try:
            return await asyncio.wait_for(self._client_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None