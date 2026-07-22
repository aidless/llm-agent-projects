"""内置 MCP 工具服务器

提供三个内置服务器:
1. FileSystemServer - 模拟文件系统操作
2. CalculatorServer - 数学计算
3. DatabaseServer - 数据库查询
"""

import json
import math
import os
import time
from typing import Any

from protocol.types import ToolCallResult, MCPResourceContent
from server.base import MCPServerBase


class FileSystemServer(MCPServerBase):
    """文件系统操作服务器（模拟）"""

    def __init__(self):
        super().__init__(
            name="filesystem",
            version="1.0.0",
            description="模拟文件系统操作服务器",
        )
        # 模拟文件系统
        self._files: dict[str, str] = {
            "/hello.txt": "Hello, MCP World!",
            "/config.json": json.dumps({"debug": True, "port": 8080}, ensure_ascii=False),
            "/notes.txt": "这是一份笔记文件。",
        }
        self._register_tools()
        self._register_resources()

    def _register_tools(self):
        @self.tool(
            name="read_file",
            description="读取文件内容",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                },
                "required": ["path"],
            },
            tags=["file", "read"],
        )
        async def read_file(args: dict) -> ToolCallResult:
            path = args.get("path", "")
            if path not in self._files:
                return ToolCallResult(
                    content=[{"type": "text", "text": f"Error: File not found: {path}"}],
                    is_error=True,
                )
            return ToolCallResult(content=[{"type": "text", "text": self._files[path]}])

        @self.tool(
            name="write_file",
            description="写入文件内容",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "文件内容"},
                },
                "required": ["path", "content"],
            },
            tags=["file", "write"],
        )
        async def write_file(args: dict) -> ToolCallResult:
            path = args.get("path", "")
            content = args.get("content", "")
            self._files[path] = content
            return ToolCallResult(content=[{"type": "text", "text": f"Written {len(content)} bytes to {path}"}])

        @self.tool(
            name="list_files",
            description="列出文件目录",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径", "default": "/"},
                },
            },
            tags=["file", "list"],
        )
        async def list_files(args: dict) -> ToolCallResult:
            path = args.get("path", "/")
            files = [p for p in self._files if p.startswith(path)]
            result = "\n".join(files) if files else "(empty directory)"
            return ToolCallResult(content=[{"type": "text", "text": result}])

        @self.tool(
            name="delete_file",
            description="删除文件",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                },
                "required": ["path"],
            },
            tags=["file", "delete"],
        )
        async def delete_file(args: dict) -> ToolCallResult:
            path = args.get("path", "")
            if path in self._files:
                del self._files[path]
                return ToolCallResult(content=[{"type": "text", "text": f"Deleted {path}"}])
            return ToolCallResult(
                content=[{"type": "text", "text": f"Error: File not found: {path}"}],
                is_error=True,
            )

    def _register_resources(self):
        @self.resource(uri="file:///hello.txt", name="hello.txt", description="示例文本文件")
        async def hello_resource(uri: str) -> MCPResourceContent:
            return MCPResourceContent(
                uri=uri,
                mime_type="text/plain",
                text=self._files.get("/hello.txt", ""),
            )


class CalculatorServer(MCPServerBase):
    """数学计算服务器"""

    def __init__(self):
        super().__init__(
            name="calculator",
            version="1.0.0",
            description="数学计算服务器",
        )
        self._register_tools()

    def _register_tools(self):
        @self.tool(
            name="add",
            description="计算两个数的和",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个数"},
                    "b": {"type": "number", "description": "第二个数"},
                },
                "required": ["a", "b"],
            },
            tags=["math", "arithmetic"],
        )
        async def add(args: dict) -> ToolCallResult:
            a, b = args["a"], args["b"]
            return ToolCallResult(content=[{"type": "text", "text": str(a + b)}])

        @self.tool(
            name="subtract",
            description="计算两个数的差",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "被减数"},
                    "b": {"type": "number", "description": "减数"},
                },
                "required": ["a", "b"],
            },
            tags=["math", "arithmetic"],
        )
        async def subtract(args: dict) -> ToolCallResult:
            a, b = args["a"], args["b"]
            return ToolCallResult(content=[{"type": "text", "text": str(a - b)}])

        @self.tool(
            name="multiply",
            description="计算两个数的积",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个数"},
                    "b": {"type": "number", "description": "第二个数"},
                },
                "required": ["a", "b"],
            },
            tags=["math", "arithmetic"],
        )
        async def multiply(args: dict) -> ToolCallResult:
            a, b = args["a"], args["b"]
            return ToolCallResult(content=[{"type": "text", "text": str(a * b)}])

        @self.tool(
            name="divide",
            description="计算两个数的商",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "被除数"},
                    "b": {"type": "number", "description": "除数"},
                },
                "required": ["a", "b"],
            },
            tags=["math", "arithmetic"],
        )
        async def divide(args: dict) -> ToolCallResult:
            a, b = args["a"], args["b"]
            if b == 0:
                return ToolCallResult(
                    content=[{"type": "text", "text": "Error: Division by zero"}],
                    is_error=True,
                )
            return ToolCallResult(content=[{"type": "text", "text": str(a / b)}])

        @self.tool(
            name="sqrt",
            description="计算平方根",
            input_schema={
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "数值"},
                },
                "required": ["x"],
            },
            tags=["math"],
        )
        async def sqrt(args: dict) -> ToolCallResult:
            x = args["x"]
            if x < 0:
                return ToolCallResult(
                    content=[{"type": "text", "text": "Error: Negative number"}],
                    is_error=True,
                )
            return ToolCallResult(content=[{"type": "text", "text": str(math.sqrt(x))}])


class DatabaseServer(MCPServerBase):
    """数据库查询服务器（模拟）"""

    def __init__(self):
        super().__init__(
            name="database",
            version="1.0.0",
            description="模拟数据库查询服务器",
        )
        # 模拟数据库表
        self._tables: dict[str, list[dict[str, Any]]] = {
            "users": [
                {"id": 1, "name": "Alice", "age": 30, "role": "admin"},
                {"id": 2, "name": "Bob", "age": 25, "role": "user"},
                {"id": 3, "name": "Charlie", "age": 35, "role": "user"},
            ],
            "products": [
                {"id": 1, "name": "Laptop", "price": 999.99, "stock": 50},
                {"id": 2, "name": "Mouse", "price": 29.99, "stock": 200},
                {"id": 3, "name": "Keyboard", "price": 79.99, "stock": 100},
            ],
        }
        self._register_tools()

    def _register_tools(self):
        @self.tool(
            name="query",
            description="执行SQL查询（模拟）",
            input_schema={
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "表名"},
                    "filter": {"type": "object", "description": "过滤条件（可选）"},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "返回的列（可选，默认全部）",
                    },
                },
                "required": ["table"],
            },
            tags=["database", "query"],
        )
        async def query(args: dict) -> ToolCallResult:
            table = args.get("table", "")
            if table not in self._tables:
                return ToolCallResult(
                    content=[{"type": "text", "text": f"Error: Table '{table}' not found"}],
                    is_error=True,
                )

            rows = self._tables[table]
            filter_cond = args.get("filter")
            if filter_cond:
                rows = [r for r in rows if all(r.get(k) == v for k, v in filter_cond.items())]

            columns = args.get("columns")
            if columns:
                rows = [{k: r.get(k) for k in columns} for r in rows]

            return ToolCallResult(content=[{
                "type": "text",
                "text": json.dumps(rows, ensure_ascii=False),
            }])

        @self.tool(
            name="insert",
            description="插入数据（模拟）",
            input_schema={
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "表名"},
                    "data": {"type": "object", "description": "插入的数据"},
                },
                "required": ["table", "data"],
            },
            tags=["database", "write"],
        )
        async def insert(args: dict) -> ToolCallResult:
            table = args.get("table", "")
            data = args.get("data", {})
            if table not in self._tables:
                return ToolCallResult(
                    content=[{"type": "text", "text": f"Error: Table '{table}' not found"}],
                    is_error=True,
                )

            row_id = max(r.get("id", 0) for r in self._tables[table]) + 1
            data["id"] = row_id
            self._tables[table].append(data)
            return ToolCallResult(content=[{"type": "text", "text": f"Inserted row with id={row_id}"}])

        @self.tool(
            name="list_tables",
            description="列出所有表",
            input_schema={"type": "object", "properties": {}},
            tags=["database"],
        )
        async def list_tables(args: dict) -> ToolCallResult:
            tables = list(self._tables.keys())
            info = []
            for t in tables:
                info.append(f"{t}: {len(self._tables[t])} rows")
            return ToolCallResult(content=[{"type": "text", "text": "\n".join(info)}])

        @self.tool(
            name="count",
            description="统计表行数",
            input_schema={
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "表名"},
                },
                "required": ["table"],
            },
            tags=["database", "query"],
        )
        async def count(args: dict) -> ToolCallResult:
            table = args.get("table", "")
            if table not in self._tables:
                return ToolCallResult(
                    content=[{"type": "text", "text": f"Error: Table '{table}' not found"}],
                    is_error=True,
                )
            n = len(self._tables[table])
            return ToolCallResult(content=[{"type": "text", "text": str(n)}])
