"""MCP 客户端

连接到 MCP 服务端，支持:
- initialize 握手
- 工具发现和调用
- 资源读取
- 通过连接池管理多连接
- 超时和错误处理
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Optional

from protocol.jsonrpc import (
    create_request,
    encode_message,
    parse_message,
    JSONRPCResponse,
)
from protocol.messages import InitializeRequest
from protocol.types import MCPTool, ToolCallResult
from client.connection_pool import ConnectionPool
from client.tool_cache import ToolCache
from server.base import MCPServerBase
from server.transport import InMemoryTransport

logger = logging.getLogger(__name__)


class MCPClient:
    """MCP 协议客户端"""

    def __init__(
        self,
        name: str = "mcp-client",
        version: str = "1.0.0",
        default_timeout: float = 30.0,
    ):
        self.name = name
        self.version = version
        self._default_timeout = default_timeout

        # 连接池
        self.pool = ConnectionPool(default_timeout=default_timeout)

        # 工具缓存
        self.tool_cache = ToolCache()

        # 已初始化的服务器集合
        self._initialized_servers: set[str] = set()

    async def connect(self, server: MCPServerBase) -> dict[str, Any]:
        """连接并初始化一个 MCP 服务器

        Returns:
            initialize 响应内容
        """
        conn = await self.pool.register(server)

        # 发送 initialize 请求
        init_req = InitializeRequest()
        req = create_request("initialize", init_req.to_params())
        raw_req = encode_message(req)

        # 需要在同一事件循环中处理 server 端
        # 将消息发给 server，server 处理后回复
        await conn.transport.client_send(raw_req)
        # 让 server 处理
        await server.process_one()
        # 获取响应
        raw_resp = await conn.transport.client_receive()

        if raw_resp is None:
            raise ConnectionError(f"Failed to initialize server: {server.name}")

        resp = parse_message(raw_resp)
        if isinstance(resp, JSONRPCResponse) and resp.result:
            self._initialized_servers.add(server.name)
            return resp.result

        raise ConnectionError(f"Invalid initialize response from {server.name}")

    async def disconnect(self, server_name: str) -> bool:
        """断开与服务器的连接"""
        self._initialized_servers.discard(server_name)
        return await self.pool.unregister(server_name)

    async def list_tools(self, server_name: str, use_cache: bool = True) -> list[MCPTool]:
        """发现服务器上的工具

        Args:
            server_name: 服务器名称
            use_cache: 是否使用缓存

        Returns:
            工具列表
        """
        # 先检查缓存
        if use_cache:
            cached = self.tool_cache.search(server_name=server_name)
            if cached:
                return cached

        # 调用 tools/list
        result = await self._call_method(server_name, "tools/list", {})
        if result is None:
            return []

        tools_data = result.get("tools", [])
        tools = [MCPTool.from_dict(t) for t in tools_data]

        # 更新缓存
        self.tool_cache.put_all(tools)
        return tools

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: Optional[float] = None,
    ) -> ToolCallResult:
        """调用工具

        Args:
            server_name: 服务器名称
            tool_name: 工具名称
            arguments: 工具参数
            timeout: 超时时间

        Returns:
            工具调用结果
        """
        result = await self._call_method(
            server_name,
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            timeout=timeout,
        )

        if result is None:
            return ToolCallResult(
                content=[{"type": "text", "text": "No response from server"}],
                is_error=True,
            )

        return ToolCallResult(
            content=result.get("content", []),
            is_error=result.get("isError", False),
            metadata=result.get("metadata", {}),
        )

    async def list_resources(self, server_name: str) -> list[dict[str, Any]]:
        """列出服务器资源"""
        result = await self._call_method(server_name, "resources/list", {})
        if result is None:
            return []
        return result.get("resources", [])

    async def read_resource(self, server_name: str, uri: str) -> list[dict[str, Any]]:
        """读取资源"""
        result = await self._call_method(server_name, "resources/read", {"uri": uri})
        if result is None:
            return []
        return result.get("contents", [])

    async def ping(self, server_name: str) -> bool:
        """发送 ping"""
        result = await self._call_method(server_name, "ping", {}, timeout=5.0)
        return result is not None

    async def _call_method(
        self,
        server_name: str,
        method: str,
        params: dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """调用 MCP 方法（内部方法）"""
        conn = self.pool._connections.get(server_name)
        if conn is None:
            logger.error(f"No connection to server: {server_name}")
            return None

        req = create_request(method, params)
        raw_req = encode_message(req)

        await conn.transport.client_send(raw_req)
        await conn.server.process_one()

        timeout_val = timeout or self._default_timeout
        try:
            raw_resp = await asyncio.wait_for(
                conn.transport.client_receive(),
                timeout=timeout_val,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout calling {method} on {server_name}")
            return None

        if raw_resp is None:
            return None

        resp = parse_message(raw_resp)
        if isinstance(resp, JSONRPCResponse):
            if resp.error:
                logger.error(f"Error from {server_name}: {resp.error.message}")
                return None
            return resp.result

        return None

    def get_cached_tools(self, server_name: Optional[str] = None) -> list[MCPTool]:
        """获取缓存的工具列表"""
        return self.tool_cache.search(server_name=server_name)

    def find_tool(self, tool_name: str) -> Optional[tuple[str, MCPTool]]:
        """在缓存中查找工具

        Returns:
            (server_name, tool) 元组或 None
        """
        tool = self.tool_cache.get(tool_name)
        if tool:
            return (tool.server_name, tool)
        return None

    async def close(self) -> None:
        """关闭所有连接"""
        await self.pool.close_all()
        self.tool_cache.clear()
        self._initialized_servers.clear()