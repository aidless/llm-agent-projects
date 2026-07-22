"""MCP Server 基类

提供 MCP 协议服务端的完整实现，包括:
- JSON-RPC 消息路由和处理
- 工具注册装饰器
- 资源和 Prompt 管理
- 能力协商
"""

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Optional

from protocol.jsonrpc import (
    INVALID_PARAMS,
    INTERNAL_ERROR,
    METHOD_NOT_FOUND,
    SERVER_NOT_INITIALIZED,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    create_error_response,
    create_response,
    encode_message,
    parse_message,
)
from protocol.messages import InitializeResult
from protocol.types import (
    MCPCapabilities,
    MCPResource,
    MCPResourceContent,
    MCPPrompt,
    MCPTool,
    ServerInfo,
    ToolCallResult,
    ToolPermission,
    TransportType,
)
from server.transport import InMemoryTransport

logger = logging.getLogger(__name__)

# 类型别名
ToolHandler = Callable[[dict[str, Any]], Coroutine[Any, Any, ToolCallResult]]
ResourceHandler = Callable[[str], Coroutine[Any, Any, MCPResourceContent]]


class MCPServerBase:
    """MCP 服务端基类"""

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        transport: Optional[InMemoryTransport] = None,
    ):
        self.name = name
        self.version = version
        self.description = description
        self.transport = transport or InMemoryTransport()

        # 工具注册表: name -> {handler, tool_def}
        self._tools: dict[str, dict[str, Any]] = {}

        # 资源注册表: uri -> resource_def
        self._resources: dict[str, MCPResource] = {}

        # 资源读取处理器
        self._resource_handlers: dict[str, ResourceHandler] = {}

        # Prompt 模板
        self._prompts: dict[str, MCPPrompt] = {}

        # 方法路由
        self._method_handlers: dict[str, Callable] = {
            "initialize": self._handle_initialize,
            "ping": self._handle_ping,
            "tools/list": self._handle_list_tools,
            "tools/call": self._handle_call_tool,
            "resources/list": self._handle_list_resources,
            "resources/read": self._handle_read_resource,
            "prompts/list": self._handle_list_prompts,
        }

        # 服务器状态
        self._initialized = False
        self._running = False

        # 能力声明
        self.capabilities = MCPCapabilities()

    # ---- 装饰器 ----

    def tool(
        self,
        name: str,
        description: str = "",
        input_schema: Optional[dict] = None,
        permission: ToolPermission = ToolPermission.PUBLIC,
        tags: Optional[list[str]] = None,
    ) -> Callable:
        """工具注册装饰器

        用法:
            @server.tool("add", description="加法运算", input_schema={...})
            async def add_handler(args):
                return ToolCallResult(content=[{"type": "text", "text": str(a+b)}])
        """
        def decorator(func: ToolHandler) -> ToolHandler:
            tool_def = MCPTool(
                name=name,
                description=description or func.__doc__ or "",
                input_schema=input_schema or {
                    "type": "object",
                    "properties": {},
                },
                server_name=self.name,
                permission=permission,
                tags=tags or [],
            )
            self._tools[name] = {
                "handler": func,
                "tool_def": tool_def,
            }
            # 如果工具已注册，更新 capabilities
            if not self.capabilities.tools:
                self.capabilities.tools = {}
            return func

        return decorator

    def resource(self, uri: str, name: str, description: str = "", mime_type: str = "text/plain") -> Callable:
        """资源注册装饰器"""
        def decorator(func: ResourceHandler) -> ResourceHandler:
            res = MCPResource(
                uri=uri,
                name=name,
                description=description,
                mime_type=mime_type,
            )
            self._resources[uri] = res
            self._resource_handlers[uri] = func
            if not self.capabilities.resources:
                self.capabilities.resources = {}
            return func

        return decorator

    def prompt(self, name: str, description: str = "", arguments: Optional[list[dict]] = None) -> Callable:
        """Prompt 模板注册装饰器"""
        def decorator(func):
            p = MCPPrompt(
                name=name,
                description=description or func.__doc__ or "",
                arguments=arguments or [],
            )
            self._prompts[name] = p
            if not self.capabilities.prompts:
                self.capabilities.prompts = {}
            return func

        return decorator

    # ---- 方法处理器 ----

    async def _handle_initialize(self, params: dict, req_id) -> dict:
        """处理 initialize 请求"""
        self._initialized = True
        result = InitializeResult(
            capabilities=self.capabilities,
            server_info=ServerInfo(
                name=self.name,
                version=self.version,
                description=self.description,
                transport=TransportType.STDIO,
                capabilities=self.capabilities,
            ),
        )
        return result.to_dict()

    async def _handle_ping(self, params: dict, req_id) -> dict:
        """处理 ping 请求"""
        return {}

    async def _handle_list_tools(self, params: dict, req_id) -> dict:
        """处理 tools/list 请求"""
        tools = [v["tool_def"].to_dict() for v in self._tools.values()]
        return {"tools": tools}

    async def _handle_call_tool(self, params: dict, req_id) -> dict:
        """处理 tools/call 请求"""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        handler = self._tools[tool_name]["handler"]
        result: ToolCallResult = await handler(arguments)
        return result.to_dict()

    async def _handle_list_resources(self, params: dict, req_id) -> dict:
        """处理 resources/list 请求"""
        resources = [r.to_dict() for r in self._resources.values()]
        return {"resources": resources}

    async def _handle_read_resource(self, params: dict, req_id) -> dict:
        """处理 resources/read 请求"""
        uri = params.get("uri", "")
        if uri not in self._resource_handlers:
            raise ValueError(f"Unknown resource: {uri}")

        handler = self._resource_handlers[uri]
        content: MCPResourceContent = await handler(uri)
        return {"contents": [content.to_dict()]}

    async def _handle_list_prompts(self, params: dict, req_id) -> dict:
        """处理 prompts/list 请求"""
        prompts = [p.to_dict() for p in self._prompts.values()]
        return {"prompts": prompts}

    # ---- 消息处理 ----

    async def handle_message(self, raw_message: str) -> Optional[str]:
        """处理单条消息，返回响应 JSON 字符串（通知返回 None）"""
        try:
            msg = parse_message(raw_message)
        except ValueError as e:
            resp = create_error_response(-32700, str(e))
            return encode_message(resp)

        if isinstance(msg, JSONRPCNotification):
            # 通知不返回响应
            if msg.method in self._method_handlers:
                try:
                    await self._method_handlers[msg.method](msg.params, None)
                except Exception as e:
                    logger.warning(f"Notification handler error: {e}")
            return None

        if isinstance(msg, JSONRPCRequest):
            if not self._initialized and msg.method != "initialize" and msg.method != "ping":
                resp = create_error_response(SERVER_NOT_INITIALIZED, "Server not initialized", req_id=msg.id)
                return encode_message(resp)

            handler = self._method_handlers.get(msg.method)
            if handler is None:
                resp = create_error_response(METHOD_NOT_FOUND, f"Method not found: {msg.method}", req_id=msg.id)
                return encode_message(resp)

            try:
                result = await handler(msg.params, msg.id)
                resp = create_response(result, req_id=msg.id)
                return encode_message(resp)
            except ValueError as e:
                resp = create_error_response(INVALID_PARAMS, str(e), req_id=msg.id)
                return encode_message(resp)
            except Exception as e:
                resp = create_error_response(INTERNAL_ERROR, str(e), req_id=msg.id)
                return encode_message(resp)

        if isinstance(msg, JSONRPCResponse):
            return None

        return None

    # ---- 服务器生命周期 ----

    async def start(self) -> None:
        """启动服务器"""
        self._running = True
        await self.transport.connect()
        logger.info(f"MCP Server '{self.name}' started")

    async def stop(self) -> None:
        """停止服务器"""
        self._running = False
        self._initialized = False
        await self.transport.disconnect()
        logger.info(f"MCP Server '{self.name}' stopped")

    async def process_one(self) -> bool:
        """从传输层读取并处理一条消息

        Returns:
            True 表示处理了消息, False 表示超时无消息
        """
        raw = await self.transport.server_receive()
        if raw is None:
            return False
        response = await self.handle_message(raw)
        if response is not None:
            await self.transport.server_send(response)
        return True

    def get_tool_names(self) -> list[str]:
        """获取已注册的工具名称列表"""
        return list(self._tools.keys())

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """获取工具定义"""
        entry = self._tools.get(name)
        if entry:
            return entry["tool_def"]
        return None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def is_running(self) -> bool:
        return self._running

    def get_server_info(self) -> ServerInfo:
        """获取服务器信息"""
        return ServerInfo(
            name=self.name,
            version=self.version,
            description=self.description,
            transport=TransportType.STDIO,
            capabilities=self.capabilities,
        )