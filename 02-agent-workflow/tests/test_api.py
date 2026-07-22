# ============================================
# API 集成测试
# 使用 httpx AsyncClient 测试 FastAPI 路由
# ============================================

import pytest
from httpx import AsyncClient, ASGITransport

# 注意：集成测试需要实际的环境配置
# 如果没有 API Key，某些测试会被跳过


@pytest.fixture
def anyio_backend():
    """anyio 后端配置"""
    return "asyncio"


class TestHealthEndpoint:
    """健康检查接口测试"""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """测试健康检查端点"""
        # 延迟导入以避免在没有配置时初始化失败
        try:
            from app.main import app
        except Exception:
            pytest.skip("应用初始化需要 .env 配置")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "version" in data
            assert "llm_provider" in data


class TestToolsEndpoint:
    """工具接口测试"""

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """测试获取工具列表"""
        try:
            from app.main import app
        except Exception:
            pytest.skip("应用初始化需要 .env 配置")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/tools")
            assert response.status_code == 200
            data = response.json()
            assert "tools" in data
            assert "total" in data
            assert data["total"] > 0

    @pytest.mark.asyncio
    async def test_calculator_tool(self):
        """测试通过 API 调用计算器工具"""
        try:
            from app.main import app
        except Exception:
            pytest.skip("应用初始化需要 .env 配置")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/tools/call", json={
                "tool_name": "calculator_tool",
                "arguments": {"expression": "2 + 3"},
            })
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "5" in data["result"]


class TestWorkflowEndpoint:
    """工作流接口测试"""

    @pytest.mark.asyncio
    async def test_workflow_info(self):
        """测试获取工作流信息"""
        try:
            from app.main import app
        except Exception:
            pytest.skip("应用初始化需要 .env 配置")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/workflow/info")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "ResearchWorkflow"
            assert "flow" in data
