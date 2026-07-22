"""MCP 消息类型定义"""

from dataclasses import dataclass, field
from typing import Any, Optional

from protocol.types import MCPCapabilities, ServerInfo


@dataclass
class InitializeRequest:
    """initialize 请求"""
    protocol_version: str = "2024-11-05"
    capabilities: dict[str, Any] = field(default_factory=dict)
    client_info: dict[str, str] = field(default_factory=lambda: {"name": "mcp-client", "version": "1.0.0"})

    def to_params(self) -> dict[str, Any]:
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": self.capabilities,
            "clientInfo": self.client_info,
        }


@dataclass
class InitializeResult:
    """initialize 响应"""
    protocol_version: str = "2024-11-05"
    capabilities: MCPCapabilities = field(default_factory=MCPCapabilities)
    server_info: ServerInfo = field(default_factory=lambda: ServerInfo(name="mcp-server", version="1.0.0"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": self.capabilities.to_dict(),
            "serverInfo": self.server_info.to_dict(),
        }


@dataclass
class ListToolsRequest:
    """list_tools 请求"""
    cursor: Optional[str] = None


@dataclass
class ListToolsResult:
    """list_tools 响应"""
    tools: list[dict[str, Any]] = field(default_factory=list)
    next_cursor: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"tools": self.tools}
        if self.next_cursor is not None:
            result["nextCursor"] = self.next_cursor
        return result


@dataclass
class CallToolRequest:
    """call_tool 请求"""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class ListResourcesRequest:
    """list_resources 请求"""
    cursor: Optional[str] = None


@dataclass
class ListResourcesResult:
    """list_resources 响应"""
    resources: list[dict[str, Any]] = field(default_factory=list)
    next_cursor: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"resources": self.resources}
        if self.next_cursor is not None:
            result["nextCursor"] = self.next_cursor
        return result


@dataclass
class ReadResourceRequest:
    """read_resource 请求"""
    uri: str


@dataclass
class ReadResourceResult:
    """read_resource 响应"""
    contents: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"contents": self.contents}


@dataclass
class ListPromptsRequest:
    """list_prompts 请求"""
    cursor: Optional[str] = None


@dataclass
class ListPromptsResult:
    """list_prompts 响应"""
    prompts: list[dict[str, Any]] = field(default_factory=list)
    next_cursor: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"prompts": self.prompts}
        if self.next_cursor is not None:
            result["nextCursor"] = self.next_cursor
        return result


@dataclass
class GetPromptRequest:
    """get_prompt 请求"""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class GetPromptResult:
    """get_prompt 响应"""
    description: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"messages": self.messages}
        if self.description:
            result["description"] = self.description
        return result


@dataclass
class PingRequest:
    """ping 请求 - 心跳检测"""
    pass