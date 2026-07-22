"""连接池

管理多个 MCP 服务端的连接，支持:
- 连接复用
- 连接健康检查
- 自动重连
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from server.base import MCPServerBase
from server.transport import InMemoryTransport

logger = logging.getLogger(__name__)


@dataclass
class PooledConnection:
    """连接池中的连接条目"""
    server_name: str
    server: MCPServerBase
    transport: InMemoryTransport
    connected: bool = False
    last_used: float = field(default_factory=time.time)
    reconnect_attempts: int = 0
    max_reconnect: int = 3
    reconnect_delay: float = 1.0

    def touch(self) -> None:
        self.last_used = time.time()

    @property
    def is_stale(self) -> bool:
        """连接是否过期（超过 60 秒未使用）"""
        return (time.time() - self.last_used) > 60.0


class ConnectionPool:
    """MCP 服务器连接池"""

    def __init__(
        self,
        max_size: int = 10,
        default_timeout: float = 30.0,
        auto_reconnect: bool = True,
    ):
        self._connections: dict[str, PooledConnection] = {}
        self._max_size = max_size
        self._default_timeout = default_timeout
        self._auto_reconnect = auto_reconnect
        self._lock = asyncio.Lock()

    async def register(
        self,
        server: MCPServerBase,
        transport: Optional[InMemoryTransport] = None,
    ) -> PooledConnection:
        """注册一个服务器连接

        Args:
            server: MCP 服务器实例
            transport: 传输层（如不提供则使用服务器自带的）

        Returns:
            连接池条目
        """
        async with self._lock:
            name = server.name
            if name in self._connections:
                return self._connections[name]

            if len(self._connections) >= self._max_size:
                # 驱逐最久未使用的连接
                self._evict_stale()

            transport = transport or server.transport
            conn = PooledConnection(
                server_name=name,
                server=server,
                transport=transport,
            )
            # 启动服务器
            await server.start()
            conn.connected = True
            self._connections[name] = conn
            logger.info(f"Connection registered: {name}")
            return conn

    async def unregister(self, server_name: str) -> bool:
        """注销连接"""
        async with self._lock:
            conn = self._connections.pop(server_name, None)
            if conn is None:
                return False
            try:
                await conn.server.stop()
            except Exception as e:
                logger.warning(f"Error stopping server {server_name}: {e}")
            conn.connected = False
            return True

    async def get_connection(self, server_name: str) -> Optional[PooledConnection]:
        """获取连接，支持自动重连"""
        conn = self._connections.get(server_name)
        if conn is None:
            return None

        if not conn.connected:
            if self._auto_reconnect and conn.reconnect_attempts < conn.max_reconnect:
                logger.info(f"Reconnecting to {server_name} (attempt {conn.reconnect_attempts + 1})")
                await asyncio.sleep(conn.reconnect_delay * (conn.reconnect_attempts + 1))
                try:
                    await conn.server.start()
                    conn.connected = True
                    conn.reconnect_attempts = 0
                    logger.info(f"Reconnected to {server_name}")
                except Exception as e:
                    conn.reconnect_attempts += 1
                    logger.warning(f"Reconnect failed for {server_name}: {e}")
                    return None
            else:
                return None

        conn.touch()
        return conn

    async def send_and_receive(
        self,
        server_name: str,
        message: str,
        timeout: Optional[float] = None,
    ) -> Optional[str]:
        """发送消息并等待响应"""
        conn = await self.get_connection(server_name)
        if conn is None:
            return None

        await conn.transport.client_send(message)

        timeout_val = timeout or self._default_timeout
        try:
            response = await asyncio.wait_for(
                conn.transport.client_receive(),
                timeout=timeout_val,
            )
            return response
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for response from {server_name}")
            return None

    async def check_health(self, server_name: str) -> bool:
        """检查连接健康状态"""
        conn = self._connections.get(server_name)
        if conn is None or not conn.connected:
            return False

        # 发送 ping
        try:
            response = await self.send_and_receive(
                server_name,
                '{"jsonrpc":"2.0","method":"ping","params":{},"id":"health"}',
                timeout=5.0,
            )
            return response is not None
        except Exception:
            return False

    async def _evict_stale(self) -> None:
        """驱逐过期连接"""
        stale_names = [
            name for name, conn in self._connections.items()
            if conn.is_stale
        ]
        for name in stale_names[:1]:  # 每次只驱逐一个
            await self.unregister(name)

    def list_connections(self) -> list[dict[str, Any]]:
        """列出所有连接状态"""
        return [
            {
                "server_name": conn.server_name,
                "connected": conn.connected,
                "last_used": conn.last_used,
                "reconnect_attempts": conn.reconnect_attempts,
            }
            for conn in self._connections.values()
        ]

    @property
    def size(self) -> int:
        return len(self._connections)

    async def close_all(self) -> None:
        """关闭所有连接"""
        names = list(self._connections.keys())
        for name in names:
            await self.unregister(name)