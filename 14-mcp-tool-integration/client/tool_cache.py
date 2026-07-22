"""工具缓存

缓存已发现的工具信息，减少重复的 list_tools 调用。
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from protocol.types import MCPTool


@dataclass
class CachedTool:
    """缓存中的工具条目"""
    tool: MCPTool
    cached_at: float = field(default_factory=time.time)
    ttl: float = 300.0  # 默认缓存 5 分钟

    def is_expired(self) -> bool:
        return (time.time() - self.cached_at) > self.ttl


class ToolCache:
    """工具发现缓存"""

    def __init__(self, default_ttl: float = 300.0):
        self._cache: dict[str, CachedTool] = {}
        self._default_ttl = default_ttl
        self._server_tool_map: dict[str, list[str]] = {}  # server_name -> [tool_names]

    def get(self, tool_name: str) -> Optional[MCPTool]:
        """获取缓存中的工具"""
        entry = self._cache.get(tool_name)
        if entry is None:
            return None
        if entry.is_expired():
            del self._cache[tool_name]
            return None
        return entry.tool

    def put(self, tool: MCPTool, ttl: Optional[float] = None) -> None:
        """缓存一个工具"""
        self._cache[tool.name] = CachedTool(
            tool=tool,
            ttl=ttl or self._default_ttl,
        )
        # 更新 server -> tools 映射
        server = tool.server_name
        if server not in self._server_tool_map:
            self._server_tool_map[server] = []
        if tool.name not in self._server_tool_map[server]:
            self._server_tool_map[server].append(tool.name)

    def put_all(self, tools: list[MCPTool], ttl: Optional[float] = None) -> None:
        """批量缓存工具"""
        for t in tools:
            self.put(t, ttl)

    def invalidate(self, tool_name: str) -> None:
        """使某个工具缓存失效"""
        self._cache.pop(tool_name, None)

    def invalidate_server(self, server_name: str) -> int:
        """使某个服务器下所有工具缓存失效

        Returns:
            失效的工具数量
        """
        tool_names = self._server_tool_map.pop(server_name, [])
        count = 0
        for name in tool_names:
            if name in self._cache:
                del self._cache[name]
                count += 1
        return count

    def clear(self) -> None:
        """清空所有缓存"""
        self._cache.clear()
        self._server_tool_map.clear()

    def search(
        self,
        keyword: Optional[str] = None,
        tag: Optional[str] = None,
        server_name: Optional[str] = None,
    ) -> list[MCPTool]:
        """搜索缓存中的工具"""
        results = []
        for entry in self._cache.values():
            if entry.is_expired():
                continue
            tool = entry.tool
            if server_name and tool.server_name != server_name:
                continue
            if tag and tag not in tool.tags:
                continue
            if keyword:
                kw = keyword.lower()
                if kw not in tool.name.lower() and kw not in tool.description.lower():
                    continue
            results.append(tool)
        return results

    def list_all(self) -> list[MCPTool]:
        """列出所有未过期的缓存工具"""
        return [
            e.tool for e in self._cache.values() if not e.is_expired()
        ]

    @property
    def size(self) -> int:
        """缓存中的工具数量（含已过期）"""
        return len(self._cache)

    def cleanup(self) -> int:
        """清理过期缓存

        Returns:
            清理的条目数量
        """
        expired = [k for k, v in self._cache.items() if v.is_expired()]
        for k in expired:
            del self._cache[k]
        return len(expired)
