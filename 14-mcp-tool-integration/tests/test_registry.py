"""Registry 测试"""

import pytest
import pytest_asyncio

from registry.tool_registry import ToolRegistry, ServerRegistration
from registry.permission import PermissionManager, PermissionLevel, PermissionRule
from protocol.types import ToolPermission
from server.builtin_servers import CalculatorServer, DatabaseServer, FileSystemServer


class TestPermissionManager:
    """权限管理测试"""

    def test_default_allow(self):
        mgr = PermissionManager()
        level = mgr.check_permission("any_tool")
        assert level == PermissionLevel.ALLOW

    def test_deny_tool(self):
        mgr = PermissionManager()
        mgr.deny("dangerous_tool")
        level = mgr.check_permission("dangerous_tool")
        assert level == PermissionLevel.DENY

    def test_grant_tool(self):
        mgr = PermissionManager()
        mgr.grant("safe_tool")
        level = mgr.check_permission("safe_tool")
        assert level == PermissionLevel.ALLOW

    def test_require_approval(self):
        mgr = PermissionManager()
        mgr.set_require_approval("sensitive_tool")
        level = mgr.check_permission("sensitive_tool")
        assert level == PermissionLevel.REQUIRE_APPROVAL

    def test_role_based_deny(self):
        mgr = PermissionManager()
        mgr.deny("admin_tool", roles=["guest"])
        assert mgr.check_permission("admin_tool", role="guest") == PermissionLevel.DENY
        assert mgr.check_permission("admin_tool", role="admin") == PermissionLevel.ALLOW

    def test_wildcard_rule(self):
        mgr = PermissionManager()
        mgr.deny("*", roles=["restricted"])
        assert mgr.check_permission("any_tool", role="restricted") == PermissionLevel.DENY
        assert mgr.check_permission("any_tool", role="default") == PermissionLevel.ALLOW

    def test_expired_rule(self):
        import time
        mgr = PermissionManager()
        mgr.deny("temp_deny", expires_in=0.01)
        time.sleep(0.02)
        level = mgr.check_permission("temp_deny")
        assert level == PermissionLevel.ALLOW

    def test_remove_rule(self):
        mgr = PermissionManager()
        mgr.deny("tool_x")
        mgr.deny("tool_x", roles=["admin"])
        removed = mgr.remove_rule("tool_x")
        assert removed == 2
        assert mgr.check_permission("tool_x") == PermissionLevel.ALLOW

    def test_list_rules(self):
        mgr = PermissionManager()
        mgr.grant("a")
        mgr.deny("b")
        rules = mgr.list_rules()
        assert len(rules) == 2

    def test_get_tool_permissions(self):
        mgr = PermissionManager()
        mgr.grant("x")
        mgr.deny("y")
        perms = mgr.get_tool_permissions("x")
        assert len(perms) == 1
        assert perms[0]["level"] == "allow"


class TestToolRegistry:
    """工具注册中心测试"""

    @pytest.mark.asyncio
    async def test_register_server(self):
        reg = ToolRegistry()
        server = CalculatorServer()
        result = await reg.register_server(server)
        assert result["status"] == "registered"
        assert reg.server_count == 1
        assert reg.tool_count == 5  # add, subtract, multiply, divide, sqrt

    @pytest.mark.asyncio
    async def test_register_duplicate(self):
        reg = ToolRegistry()
        server = CalculatorServer()
        await reg.register_server(server)
        result = await reg.register_server(server)
        assert result["status"] == "already_registered"

    @pytest.mark.asyncio
    async def test_unregister_server(self):
        reg = ToolRegistry()
        server = CalculatorServer()
        await reg.register_server(server)
        success = await reg.unregister_server("calculator")
        assert success
        assert reg.server_count == 0

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self):
        reg = ToolRegistry()
        success = await reg.unregister_server("nonexistent")
        assert not success

    @pytest.mark.asyncio
    async def test_list_tools(self):
        reg = ToolRegistry()
        await reg.register_server(CalculatorServer())
        await reg.register_server(FileSystemServer())
        tools = reg.list_tools()
        names = [t["name"] for t in tools]
        assert "add" in names
        assert "read_file" in names

    @pytest.mark.asyncio
    async def test_search_tools_by_keyword(self):
        reg = ToolRegistry()
        await reg.register_server(CalculatorServer())
        tools = reg.list_tools(keyword="add")
        names = [t["name"] for t in tools]
        assert "add" in names
        assert "multiply" not in names

    @pytest.mark.asyncio
    async def test_search_tools_by_tag(self):
        reg = ToolRegistry()
        await reg.register_server(CalculatorServer())
        tools = reg.list_tools(tag="math")
        names = [t["name"] for t in tools]
        assert len(names) >= 1

    @pytest.mark.asyncio
    async def test_search_tools_by_server(self):
        reg = ToolRegistry()
        await reg.register_server(CalculatorServer())
        await reg.register_server(DatabaseServer())
        tools = reg.list_tools(server_name="calculator")
        for t in tools:
            assert t["serverName"] == "calculator"

    @pytest.mark.asyncio
    async def test_permission_filter_in_list(self):
        reg = ToolRegistry()
        await reg.register_server(CalculatorServer())
        reg.permissions.deny("add", roles=["restricted"])
        tools = reg.list_tools(role="restricted")
        names = [t["name"] for t in tools]
        assert "add" not in names
        # Other tools should still be available
        assert "subtract" in names

    @pytest.mark.asyncio
    async def test_get_tool_server(self):
        reg = ToolRegistry()
        await reg.register_server(CalculatorServer())
        srv = reg.get_tool_server("add")
        assert srv == "calculator"
        assert reg.get_tool_server("nonexistent") is None

    @pytest.mark.asyncio
    async def test_usage_stats(self):
        reg = ToolRegistry()
        await reg.register_server(CalculatorServer())
        reg.record_usage("add", "calculator", success=True, latency_ms=10.0)
        reg.record_usage("add", "calculator", success=True, latency_ms=20.0)
        reg.record_usage("add", "calculator", success=False, latency_ms=5.0)
        stats = reg.get_usage_stats(tool_name="add")
        assert stats["total_calls"] == 3
        assert stats["success_count"] == 2
        assert stats["error_count"] == 1

    @pytest.mark.asyncio
    async def test_heartbeat(self):
        reg = ToolRegistry()
        server = CalculatorServer()
        await server.start()
        await reg.register_server(server)
        status = await reg.check_heartbeat("calculator")
        assert status.alive
        await reg.shutdown()

    @pytest.mark.asyncio
    async def test_list_servers(self):
        reg = ToolRegistry()
        await reg.register_server(CalculatorServer())
        await reg.register_server(DatabaseServer())
        servers = reg.list_servers()
        assert len(servers) == 2

    @pytest.mark.asyncio
    async def test_shutdown(self):
        reg = ToolRegistry()
        await reg.register_server(CalculatorServer())
        await reg.shutdown()
        assert reg.server_count == 0