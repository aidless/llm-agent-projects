"""MCP Client 测试"""

import pytest
import pytest_asyncio
import time

from client.mcp_client import MCPClient
from client.connection_pool import ConnectionPool, PooledConnection
from client.tool_cache import ToolCache, CachedTool
from protocol.types import MCPTool, ToolPermission
from server.builtin_servers import CalculatorServer, FileSystemServer, DatabaseServer
from server.base import MCPServerBase


class TestToolCache:
    """工具缓存测试"""

    def test_put_and_get(self):
        cache = ToolCache()
        tool = MCPTool(name="add", description="Add numbers", server_name="calc")
        cache.put(tool)
        retrieved = cache.get("add")
        assert retrieved is not None
        assert retrieved.name == "add"
        assert retrieved.server_name == "calc"

    def test_get_miss(self):
        cache = ToolCache()
        assert cache.get("nonexistent") is None

    def test_put_all_and_list(self):
        cache = ToolCache()
        tools = [
            MCPTool(name="a", description="Tool A", server_name="s1"),
            MCPTool(name="b", description="Tool B", server_name="s1"),
        ]
        cache.put_all(tools)
        assert cache.size == 2
        all_tools = cache.list_all()
        assert len(all_tools) == 2

    def test_invalidate(self):
        cache = ToolCache()
        cache.put(MCPTool(name="x", description="X", server_name="s1"))
        cache.invalidate("x")
        assert cache.get("x") is None

    def test_invalidate_server(self):
        cache = ToolCache()
        cache.put(MCPTool(name="a", description="A", server_name="s1"))
        cache.put(MCPTool(name="b", description="B", server_name="s1"))
        cache.put(MCPTool(name="c", description="C", server_name="s2"))
        count = cache.invalidate_server("s1")
        assert count == 2
        assert cache.get("c") is not None

    def test_search_by_keyword(self):
        cache = ToolCache()
        cache.put(MCPTool(name="add_numbers", description="Add two numbers", server_name="calc", tags=["math"]))
        cache.put(MCPTool(name="read_file", description="Read a file", server_name="fs", tags=["file"]))
        results = cache.search(keyword="add")
        assert len(results) == 1
        assert results[0].name == "add_numbers"

    def test_search_by_tag(self):
        cache = ToolCache()
        cache.put(MCPTool(name="add", description="Add", server_name="s", tags=["math"]))
        cache.put(MCPTool(name="read", description="Read", server_name="s", tags=["file"]))
        results = cache.search(tag="math")
        assert len(results) == 1

    def test_clear(self):
        cache = ToolCache()
        cache.put(MCPTool(name="a", description="A", server_name="s"))
        cache.clear()
        assert cache.size == 0

    def test_expired_entry(self):
        cache = ToolCache(default_ttl=0.01)  # 10ms TTL
        cache.put(MCPTool(name="exp", description="Expires", server_name="s"))
        time.sleep(0.02)
        assert cache.get("exp") is None

    def test_cleanup(self):
        cache = ToolCache(default_ttl=0.01)
        cache.put(MCPTool(name="exp", description="Expires", server_name="s"))
        cache.put(MCPTool(name="keep", description="Keep", server_name="s"), ttl=300)
        time.sleep(0.02)
        removed = cache.cleanup()
        assert removed == 1
        assert cache.get("keep") is not None


class TestConnectionPool:
    """连接池测试"""

    @pytest.mark.asyncio
    async def test_register_and_get(self):
        pool = ConnectionPool()
        server = CalculatorServer()
        conn = await pool.register(server)
        assert conn.server_name == "calculator"
        assert conn.connected
        assert pool.size == 1

        fetched = await pool.get_connection("calculator")
        assert fetched is not None

        await pool.close_all()

    @pytest.mark.asyncio
    async def test_unregister(self):
        pool = ConnectionPool()
        server = CalculatorServer()
        await pool.register(server)
        success = await pool.unregister("calculator")
        assert success
        assert pool.size == 0

    @pytest.mark.asyncio
    async def test_list_connections(self):
        pool = ConnectionPool()
        s1 = CalculatorServer()
        s2 = DatabaseServer()
        await pool.register(s1)
        await pool.register(s2)
        conns = pool.list_connections()
        assert len(conns) == 2
        names = {c["server_name"] for c in conns}
        assert "calculator" in names
        assert "database" in names
        await pool.close_all()


class TestMCPClient:
    """MCP Client 测试"""

    @pytest.mark.asyncio
    async def test_connect_and_list_tools(self):
        client = MCPClient()
        server = CalculatorServer()
        await client.connect(server)

        tools = await client.list_tools("calculator", use_cache=False)
        tool_names = [t.name for t in tools]
        assert "add" in tool_names
        assert "multiply" in tool_names

        await client.close()

    @pytest.mark.asyncio
    async def test_call_tool(self):
        client = MCPClient()
        server = CalculatorServer()
        await client.connect(server)

        result = await client.call_tool("calculator", "add", {"a": 10, "b": 20})
        assert not result.is_error
        assert result.content[0]["text"] == "30"

        await client.close()

    @pytest.mark.asyncio
    async def test_call_tool_error(self):
        client = MCPClient()
        server = CalculatorServer()
        await client.connect(server)

        result = await client.call_tool("calculator", "divide", {"a": 5, "b": 0})
        assert result.is_error

        await client.close()

    @pytest.mark.asyncio
    async def test_ping(self):
        client = MCPClient()
        server = CalculatorServer()
        await client.connect(server)

        alive = await client.ping("calculator")
        assert alive

        await client.close()

    @pytest.mark.asyncio
    async def test_list_resources(self):
        client = MCPClient()
        server = FileSystemServer()
        await client.connect(server)

        resources = await client.list_resources("filesystem")
        assert len(resources) >= 1
        assert resources[0]["uri"] == "file:///hello.txt"

        await client.close()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        client = MCPClient()
        server = CalculatorServer()
        await client.connect(server)
        success = await client.disconnect("calculator")
        assert success