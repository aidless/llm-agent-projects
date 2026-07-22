"""HTTP 请求节点 - GET/POST/PUT/DELETE"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from engine.errors import NodeExecutionError

from .base import BaseNode

logger = logging.getLogger(__name__)


class HTTPNode(BaseNode):
    """HTTP 请求节点"""

    node_type = "http"

    def execute(
        self,
        inputs: Dict[str, Any],
        context: "ExecutionContext",
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        url = inputs.get("url", "")
        method = inputs.get("method", "GET").upper()
        headers = inputs.get("headers", {})
        body = inputs.get("body")
        timeout = inputs.get("timeout", 30)

        if not url:
            raise NodeExecutionError("HTTP 节点缺少 url 参数")

        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8") if isinstance(body, dict) else str(body).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")

        req = Request(url, data=data, method=method, headers=headers)

        try:
            with urlopen(req, timeout=timeout) as resp:
                resp_body = resp.read().decode("utf-8")
                try:
                    resp_json = json.loads(resp_body)
                except (json.JSONDecodeError, ValueError):
                    resp_json = None
                return {
                    "status_code": resp.status,
                    "body": resp_json if resp_json is not None else resp_body,
                    "headers": dict(resp.headers),
                }
        except HTTPError as e:
            raise NodeExecutionError(
                f"HTTP {method} {url} 失败: {e.code} {e.reason}"
            )
        except URLError as e:
            raise NodeExecutionError(
                f"HTTP {method} {url} 连接失败: {e.reason}"
            )


def create_http_handler():
    """工厂函数"""
    node = HTTPNode()

    def handler(inputs, context, config):
        return node.execute(inputs, context, config)

    handler.node_type = HTTPNode.node_type
    return handler
