"""MCP Server 测试"""

import pytest
import pytest_asyncio

from server.base import MCPServerBase
from server.builtin_servers import CalculatorServer, DatabaseServer, FileSystemServer
from server.transport import InMemoryTransport, SSETransport, WebSocketTransport, StdioTransport
from protocol.types import ToolCallResult, TransportType


class TestMCPServerBase:
    """MCP Server 基类测试"""

    @pytest.mark.asyncio
    async def test_server_initialization(self):
        server = MCPServerBase(name="test", version="0.1.0")
        assert server.name == "test"
        assert server.version == "0.1.0"
        assert not server.is_initialized
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_register_tool_with_decorator(self):
        server = MCPServerBase(name="test")

        @server.tool("echo", description="Echo tool")
        async def echo(args):
            return ToolCallResult(content=[{"type": "text", "text": args.get("msg", "")}])

        assert "echo" in server.get_tool_names()
        tool = server.get_tool("echo")
        assert tool is not None
        assert tool.description == "Echo tool"

    @pytest.mark.asyncio
    async def test_handle_initialize(self):
        server = MCPServerBase(name="test-srv")
        raw = '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"c","version":"1.0"}},"id":"init-1"}'
        resp = await server.handle_message(raw)
        assert resp is not None
        import json
        data = json.loads(resp)
        assert "result" in data
        assert data["result"]["serverInfo"]["name"] == "test-srv"
        assert server.is_initialized

    @pytest.mark.asyncio
    async def test_handle_ping(self):
        server = MCPServerBase(name="ping-srv")
        await server.start()
        raw = '{"jsonrpc":"2.0","method":"ping","params":{},"id":"p1"}'
        resp = await server.handle_message(raw)
        assert resp is not None
        import json
        data = json.loads(resp)
        assert data.get("result") is not None or data.get("error") is None
        await server.stop()

    @pytest.mark.asyncio
    async def test_handle_list_tools(self):
        server = MCPServerBase(name="tools-srv")

        @server.tool("my_tool", description="A tool")
        async def my_tool(args):
            return ToolCallResult(content=[{"type": "text", "text": "ok"}])

        raw = '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":"t1"}'
        # 需要先初始化
        init_raw = '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"c","version":"1.0"}},"id":"init"}'
        await server.handle_message(init_raw)
        resp = await server.handle_message(raw)
        import json
        data = json.loads(resp)
        tools = data["result"]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_handle_call_tool(self):
        server = MCPServerBase(name="call-srv")

        @server.tool("greet", description="Greet")
        async def greet(args):
            name = args.get("name", "World")
            return ToolCallResult(content=[{"type": "text", "text": f"Hello, {name}!"}])

        init_raw = '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"c","version":"1.0"}},"id":"init"}'
        await server.handle_message(init_raw)

        call_raw = '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"greet","arguments":{"name":"Test"}},"id":"c1"}'
        resp = await server.handle_message(call_raw)
        import json
        data = json.loads(resp)
        assert data["result"]["content"][0]["text"] == "Hello, Test!"

    @pytest.mark.asyncio
    async def test_handle_unknown_tool(self):
        server = MCPServerBase(name="unknown-srv")
        init_raw = '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"c","version":"1.0"}},"id":"init"}'
        await server.handle_message(init_raw)

        call_raw = '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"nonexistent","arguments":{}},"id":"c2"}'
        resp = await server.handle_message(call_raw)
        import json
        data = json.loads(resp)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_handle_notification_no_response(self):
        server = MCPServerBase(name="notif-srv")
        raw = '{"jsonrpc":"2.0","method":"ping","params":{}}'
        resp = await server.handle_message(raw)
        assert resp is None

    @pytest.mark.asyncio
    async def test_server_not_initialized_error(self):
        server = MCPServerBase(name="uninit-srv")
        raw = '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":"t1"}'
        resp = await server.handle_message(raw)
        import json
        data = json.loads(resp)
        assert "error" in data
        assert data["error"]["code"] == -32002

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        server = MCPServerBase(name="bad-json")
        resp = await server.handle_message("not json")
        assert resp is not None
        import json
        data = json.loads(resp)
        assert data["error"]["code"] == -32700


class TestBuiltinServers:
    """内置服务器测试"""

    @pytest.mark.asyncio
    async def test_calculator_add(self):
        server = CalculatorServer()
        init_raw = '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"c","version":"1.0"}},"id":"init"}'
        await server.handle_message(init_raw)

        raw = '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"add","arguments":{"a":3,"b":5}},"id":"c1"}'
        resp = await server.handle_message(raw)
        import json
        data = json.loads(resp)
        assert data["result"]["content"][0]["text"] == "8"

    @pytest.mark.asyncio
    async def test_calculator_divide_by_zero(self):
        server = CalculatorServer()
        init_raw = '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"c","version":"1.0"}},"id":"init"}'
        await server.handle_message(init_raw)

        raw = '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"divide","arguments":{"a":10,"b":0}},"id":"c2"}'
        resp = await server.handle_message(raw)
        import json
        data = json.loads(resp)
        assert data["result"]["isError"] is True

    @pytest.mark.asyncio
    async def test_filesystem_read_file(self):
        server = FileSystemServer()
        init_raw = '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"c","version":"1.0"}},"id":"init"}'
        await server.handle_message(init_raw)

        raw = '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"read_file","arguments":{"path":"/hello.txt"}},"id":"c1"}'
        resp = await server.handle_message(raw)
        import json
        data = json.loads(resp)
        assert "Hello, MCP World!" in data["result"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_database_query(self):
        server = DatabaseServer()
        init_raw = '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"c","version":"1.0"}},"id":"init"}'
        await server.handle_message(init_raw)

        raw = '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"query","arguments":{"table":"users","filter":{"role":"admin"}}},"id":"c1"}'
        resp = await server.handle_message(raw)
        import json
        data = json.loads(resp)
        result_text = data["result"]["content"][0]["text"]
        users = json.loads(result_text)
        assert len(users) == 1
        assert users[0]["name"] == "Alice"


class TestTransport:
    """传输层测试"""

    def test_stdio_transport_type(self):
        t = StdioTransport()
        assert t.transport_type == TransportType.STDIO

    def test_sse_transport_type(self):
        t = SSETransport()
        assert t.transport_type == TransportType.SSE

    def test_websocket_transport_type(self):
        t = WebSocketTransport()
        assert t.transport_type == TransportType.WEBSOCKET

    @pytest.mark.asyncio
    async def test_in_memory_transport(self):
        transport = InMemoryTransport()
        await transport.connect()
        assert transport.connected

        await transport.client_send('{"test": true}')
        msg = await transport.server_receive()
        assert msg == '{"test": true}'

        await transport.server_send('{"ok": true}')
        msg = await transport.client_receive()
        assert msg == '{"ok": true}'

        await transport.disconnect()
        assert not transport.connected