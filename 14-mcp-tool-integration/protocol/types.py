"""MCP 协议类型定义"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TransportType(str, Enum):
    """传输层类型"""
    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"


class ToolPermission(str, Enum):
    """工具权限级别"""
    PUBLIC = "public"
    PRIVATE = "private"
    RESTRICTED = "restricted"


@dataclass
class MCPTool:
    """MCP 工具定义"""
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    server_name: str = ""
    permission: ToolPermission = ToolPermission.PUBLIC
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "serverName": self.server_name,
            "permission": self.permission.value,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPTool":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            input_schema=data.get("inputSchema", {}),
            server_name=data.get("serverName", ""),
            permission=ToolPermission(data.get("permission", "public")),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MCPResource:
    """MCP 资源定义"""
    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
            "metadata": self.metadata,
        }


@dataclass
class MCPResourceContent:
    """MCP 资源内容"""
    uri: str
    mime_type: str = "text/plain"
    text: Optional[str] = None
    blob: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "uri": self.uri,
            "mimeType": self.mime_type,
        }
        if self.text is not None:
            result["text"] = self.text
        if self.blob is not None:
            result["blob"] = self.blob
        return result


@dataclass
class MCPPrompt:
    """MCP Prompt 模板定义"""
    name: str
    description: str = ""
    arguments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }


@dataclass
class MCPCapabilities:
    """MCP 服务器能力声明"""
    tools: Optional[dict[str, Any]] = None
    resources: Optional[dict[str, Any]] = None
    prompts: Optional[dict[str, Any]] = None
    logging: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.tools is not None:
            result["tools"] = self.tools
        if self.resources is not None:
            result["resources"] = self.resources
        if self.prompts is not None:
            result["prompts"] = self.prompts
        if self.logging is not None:
            result["logging"] = self.logging
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPCapabilities":
        return cls(
            tools=data.get("tools"),
            resources=data.get("resources"),
            prompts=data.get("prompts"),
            logging=data.get("logging"),
        )


@dataclass
class ServerInfo:
    """MCP 服务器信息"""
    name: str
    version: str
    description: str = ""
    transport: TransportType = TransportType.STDIO
    capabilities: MCPCapabilities = field(default_factory=MCPCapabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "protocolVersion": "2024-11-05",
            "transport": self.transport.value,
            "capabilities": self.capabilities.to_dict(),
        }


@dataclass
class ToolCallResult:
    """工具调用结果"""
    content: list[dict[str, Any]]
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "isError": self.is_error,
            "metadata": self.metadata,
        }


@dataclass
class HeartbeatStatus:
    """心跳状态"""
    server_name: str
    alive: bool
    last_heartbeat: float
    response_time_ms: Optional[float] = None