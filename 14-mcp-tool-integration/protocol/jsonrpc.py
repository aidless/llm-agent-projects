"""JSON-RPC 2.0 完整实现

支持:
- Request / Response / Notification / Batch
- 内部 ID 生成和映射
- 错误码定义
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Union

# JSON-RPC 2.0 标准错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# MCP 自定义错误码
SERVER_NOT_INITIALIZED = -32002
UNKNOWN_ERROR = -32001


@dataclass
class JSONRPCError:
    """JSON-RPC 错误对象"""
    code: int
    message: str
    data: Any = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.data is not None:
            result["data"] = self.data
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JSONRPCError":
        return cls(
            code=data["code"],
            message=data["message"],
            data=data.get("data"),
        )


@dataclass
class JSONRPCRequest:
    """JSON-RPC 请求对象"""
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: Union[str, int] = ""

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
            "id": self.id,
        }
        return result

    def is_notification(self) -> bool:
        """判断是否为通知（无 id）"""
        return not self.id


@dataclass
class JSONRPCResponse:
    """JSON-RPC 响应对象"""
    id: Union[str, int, None] = None
    result: Any = None
    error: Optional[JSONRPCError] = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"jsonrpc": "2.0", "id": self.id}
        if self.error is not None:
            result["error"] = self.error.to_dict()
        else:
            result["result"] = self.result
        return result


@dataclass
class JSONRPCNotification:
    """JSON-RPC 通知对象（无 id，不期望响应）"""
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
        }


def create_request(method: str, params: Optional[dict] = None, req_id: Optional[Union[str, int]] = None) -> JSONRPCRequest:
    """创建 JSON-RPC 请求"""
    return JSONRPCRequest(
        method=method,
        params=params or {},
        id=req_id if req_id is not None else str(uuid.uuid4())[:8],
    )


def create_notification(method: str, params: Optional[dict] = None) -> JSONRPCNotification:
    """创建 JSON-RPC 通知"""
    return JSONRPCNotification(
        method=method,
        params=params or {},
    )


def create_response(result: Any, req_id: Union[str, int, None] = None) -> JSONRPCResponse:
    """创建成功响应"""
    return JSONRPCResponse(id=req_id, result=result)


def create_error_response(code: int, message: str, data: Any = None, req_id: Union[str, int, None] = None) -> JSONRPCResponse:
    """创建错误响应"""
    return JSONRPCResponse(
        id=req_id,
        error=JSONRPCError(code=code, message=message, data=data),
    )


def create_batch_request(requests: list[Union[JSONRPCRequest, JSONRPCNotification]]) -> list[dict[str, Any]]:
    """创建批量请求"""
    return [r.to_dict() for r in requests]


def create_batch_response(responses: list[JSONRPCResponse]) -> list[dict[str, Any]]:
    """创建批量响应"""
    return [r.to_dict() for r in responses]


def parse_message(data: Union[str, dict, list]) -> Union[JSONRPCRequest, JSONRPCNotification, JSONRPCResponse, list]:
    """解析 JSON-RPC 消息

    Args:
        data: 可以是 JSON 字符串、字典（单条）或列表（批量）

    Returns:
        解析后的消息对象或消息列表

    Raises:
        ValueError: 消息格式无效
    """
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
    else:
        parsed = data

    if isinstance(parsed, list):
        if len(parsed) == 0:
            raise ValueError("Empty batch request")
        return [parse_message(item) for item in parsed]

    if not isinstance(parsed, dict):
        raise ValueError("Message must be a JSON object")

    if "jsonrpc" not in parsed or parsed["jsonrpc"] != "2.0":
        raise ValueError("Invalid or missing 'jsonrpc' version")

    if "method" in parsed:
        params = parsed.get("params", {})
        method = parsed["method"]
        msg_id = parsed.get("id")

        if msg_id is None:
            return JSONRPCNotification(method=method, params=params)
        return JSONRPCRequest(method=method, params=params, id=msg_id)

    if "result" in parsed or "error" in parsed:
        error = None
        if "error" in parsed:
            error = JSONRPCError.from_dict(parsed["error"])
        return JSONRPCResponse(
            id=parsed.get("id"),
            result=parsed.get("result"),
            error=error,
        )

    raise ValueError("Invalid JSON-RPC message: missing method, result, or error")


def encode_message(msg: Union[JSONRPCRequest, JSONRPCResponse, JSONRPCNotification]) -> str:
    """将消息对象编码为 JSON 字符串"""
    return json.dumps(msg.to_dict(), ensure_ascii=False)


def encode_batch(messages: list[Union[JSONRPCRequest, JSONRPCResponse, JSONRPCNotification]]) -> str:
    """将批量消息编码为 JSON 字符串"""
    return json.dumps([m.to_dict() for m in messages], ensure_ascii=False)