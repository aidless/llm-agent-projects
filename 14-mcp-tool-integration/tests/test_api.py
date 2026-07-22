"""FastAPI API 测试"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app, registry, client, agent
from server.builtin_servers import FileSystemServer, CalculatorServer, DatabaseServer


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture(autouse=True)
async def setup_builtins():
    """自动注册内置服务器用于 API 测试"""
    # 如果还没有注册服务器，则注册
    if registry.server_count == 0:
        for server_cls in [FileSystemServer, CalculatorServer, DatabaseServer]:
            srv = server_cls()
            await registry.register_server(srv)
            try:
                await client.connect(srv)
            except Exception:
                pass
    yield


class TestRootEndpoints:
    """根端点测试"""

    @pytest.mark.asyncio
    async def test_root(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "MCP Tool Integration System"

    @pytest.mark.asyncio
    async def test_health(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestRegistryAPI:
    """注册中心 API 测试"""

    @pytest.mark.asyncio
    async def test_list_servers(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/registry/servers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"]
        assert len(data["data"]) >= 3  # 3 builtin servers

    @pytest.mark.asyncio
    async def test_search_tools(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/registry/tools?keyword=add")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"]
        names = [t["name"] for t in data["data"]["tools"]]
        assert "add" in names

    @pytest.mark.asyncio
    async def test_set_permission(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/registry/permissions", json={
                "tool_name": "test_tool",
                "level": "deny",
                "roles": ["*"],
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"]

    @pytest.mark.asyncio
    async def test_list_permissions(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/registry/permissions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"]


class TestServersAPI:
    """服务器管理 API 测试"""

    @pytest.mark.asyncio
    async def test_list_servers(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/servers/list")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["count"] >= 3

    @pytest.mark.asyncio
    async def test_server_info(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/servers/calculator/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["name"] == "calculator"

    @pytest.mark.asyncio
    async def test_server_tools(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/servers/calculator/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["count"] >= 4