"""工具注册中心

核心功能:
- 工具服务器注册/注销
- 心跳检测
- 工具搜索和过滤
- 权限管理
- 使用统计
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from protocol.types import (
    HeartbeatStatus,
    MCPTool,
    ServerInfo,
    ToolPermission,
)
from server.base import MCPServerBase
from registry.permission import PermissionLevel, PermissionManager

logger = logging.getLogger(__name__)


@dataclass
class ServerRegistration:
    """服务器注册信息"""
    server: MCPServerBase
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    alive: bool = True
    heartbeat_count: int = 0
    total_calls: int = 0
    error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.server.name,
            "version": self.server.version,
            "description": self.server.description,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "alive": self.alive,
            "heartbeat_count": self.heartbeat_count,
            "total_calls": self.total_calls,
            "error_count": self.error_count,
        }


@dataclass
class ToolUsageRecord:
    """工具使用记录"""
    tool_name: str
    server_name: str
    called_at: float = field(default_factory=time.time)
    success: bool = True
    latency_ms: float = 0.0


class ToolRegistry:
    """工具注册中心"""

    def __init__(self, heartbeat_interval: float = 30.0):
        # 服务器注册表
        self._servers: dict[str, ServerRegistration] = {}

        # 工具注册表: tool_name -> server_name
        self._tool_server_map: dict[str, str] = {}

        # 权限管理
        self.permissions = PermissionManager()

        # 使用统计
        self._usage_history: list[ToolUsageRecord] = []
        self._max_history = 1000

        # 心跳配置
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task] = None

    # ---- 服务器管理 ----

    async def register_server(self, server: MCPServerBase) -> dict[str, Any]:
        """注册一个工具服务器"""
        name = server.name
        if name in self._servers:
            return {"status": "already_registered", "name": name}

        reg = ServerRegistration(server=server)
        self._servers[name] = reg

        # 发现并注册该服务器的工具
        for tool_name in server.get_tool_names():
            self._tool_server_map[tool_name] = name

        logger.info(f"Server registered: {name} with {len(server.get_tool_names())} tools")
        return {"status": "registered", "name": name, "tools_count": len(server.get_tool_names())}

    async def unregister_server(self, server_name: str) -> bool:
        """注销服务器"""
        reg = self._servers.pop(server_name, None)
        if reg is None:
            return False

        # 清理工具映射
        self._tool_server_map = {
            k: v for k, v in self._tool_server_map.items() if v != server_name
        }

        try:
            await reg.server.stop()
        except Exception as e:
            logger.warning(f"Error stopping server {server_name}: {e}")

        return True

    def get_server(self, server_name: str) -> Optional[ServerRegistration]:
        """获取服务器注册信息"""
        return self._servers.get(server_name)

    def list_servers(self) -> list[dict[str, Any]]:
        """列出所有已注册的服务器"""
        return [reg.to_dict() for reg in self._servers.values()]

    # ---- 工具管理 ----

    def list_tools(
        self,
        server_name: Optional[str] = None,
        tag: Optional[str] = None,
        keyword: Optional[str] = None,
        role: str = "default",
    ) -> list[dict[str, Any]]:
        """列出/搜索工具

        Args:
            server_name: 按服务器过滤
            tag: 按标签过滤
            keyword: 关键词搜索
            role: 用户角色（用于权限过滤）

        Returns:
            匹配的工具列表
        """
        results = []

        for sname, reg in self._servers.items():
            if server_name and sname != server_name:
                continue

            for tool_name in reg.server.get_tool_names():
                tool = reg.server.get_tool(tool_name)
                if tool is None:
                    continue

                # 标签过滤
                if tag and tag not in tool.tags:
                    continue

                # 关键词过滤
                if keyword:
                    kw = keyword.lower()
                    if kw not in tool.name.lower() and kw not in tool.description.lower():
                        continue

                # 权限检查
                perm = self.permissions.check_permission(tool_name, role)
                if perm == PermissionLevel.DENY:
                    continue

                tool_data = tool.to_dict()
                tool_data["permission_level"] = perm.value
                tool_data["server_alive"] = reg.alive
                results.append(tool_data)

        return results

    def get_tool_server(self, tool_name: str) -> Optional[str]:
        """获取工具所属的服务器名"""
        return self._tool_server_map.get(tool_name)

    # ---- 心跳检测 ----

    async def check_heartbeat(self, server_name: str) -> HeartbeatStatus:
        """检查服务器心跳"""
        reg = self._servers.get(server_name)
        if reg is None:
            return HeartbeatStatus(
                server_name=server_name,
                alive=False,
                last_heartbeat=0,
            )

        start = time.time()
        try:
            if reg.server.is_running:
                # 通过 ping 检查
                import json
                from protocol.jsonrpc import encode_message, create_request
                req = create_request("ping", {})
                raw = encode_message(req)
                await reg.server.transport.client_send(raw)
                await reg.server.process_one()
                resp_raw = await reg.server.transport.client_receive()

                elapsed = (time.time() - start) * 1000
                if resp_raw:
                    reg.alive = True
                    reg.last_heartbeat = time.time()
                    reg.heartbeat_count += 1
                    return HeartbeatStatus(
                        server_name=server_name,
                        alive=True,
                        last_heartbeat=reg.last_heartbeat,
                        response_time_ms=elapsed,
                    )
        except Exception:
            pass

        reg.alive = False
        return HeartbeatStatus(
            server_name=server_name,
            alive=False,
            last_heartbeat=reg.last_heartbeat,
        )

    async def check_all_heartbeats(self) -> list[HeartbeatStatus]:
        """检查所有服务器心跳"""
        tasks = [self.check_heartbeat(name) for name in self._servers]
        return await asyncio.gather(*tasks)

    async def start_heartbeat_monitor(self) -> None:
        """启动心跳监控任务"""
        async def _monitor():
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                try:
                    await self.check_all_heartbeats()
                except Exception as e:
                    logger.error(f"Heartbeat monitor error: {e}")

        self._heartbeat_task = asyncio.create_task(_monitor())

    async def stop_heartbeat_monitor(self) -> None:
        """停止心跳监控"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    # ---- 使用统计 ----

    def record_usage(
        self,
        tool_name: str,
        server_name: str,
        success: bool = True,
        latency_ms: float = 0.0,
    ) -> None:
        """记录工具使用"""
        record = ToolUsageRecord(
            tool_name=tool_name,
            server_name=server_name,
            success=success,
            latency_ms=latency_ms,
        )
        self._usage_history.append(record)

        # 限制历史大小
        if len(self._usage_history) > self._max_history:
            self._usage_history = self._usage_history[-self._max_history:]

        # 更新服务器统计
        reg = self._servers.get(server_name)
        if reg:
            reg.total_calls += 1
            if not success:
                reg.error_count += 1

    def get_usage_stats(
        self,
        server_name: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """获取使用统计"""
        records = self._usage_history
        if server_name:
            records = [r for r in records if r.server_name == server_name]
        if tool_name:
            records = [r for r in records if r.tool_name == tool_name]

        if not records:
            return {
                "total_calls": 0,
                "success_count": 0,
                "error_count": 0,
                "avg_latency_ms": 0,
            }

        total = len(records)
        success = sum(1 for r in records if r.success)
        errors = total - success
        avg_latency = sum(r.latency_ms for r in records) / total

        return {
            "total_calls": total,
            "success_count": success,
            "error_count": errors,
            "avg_latency_ms": round(avg_latency, 2),
        }

    # ---- 生命周期 ----

    async def shutdown(self) -> None:
        """关闭注册中心"""
        await self.stop_heartbeat_monitor()
        for name in list(self._servers.keys()):
            await self.unregister_server(name)

    @property
    def server_count(self) -> int:
        return len(self._servers)

    @property
    def tool_count(self) -> int:
        return len(self._tool_server_map)