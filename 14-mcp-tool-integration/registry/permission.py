"""权限管理

支持工具级别的权限控制:
- 角色定义和分配
- 权限检查
- 访问控制列表
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class PermissionLevel(str, Enum):
    """权限级别"""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class PermissionRule:
    """权限规则"""
    tool_name: str
    level: PermissionLevel
    roles: list[str] = field(default_factory=list)
    expires_at: Optional[float] = None

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def matches_role(self, role: str) -> bool:
        """角色是否匹配此规则"""
        if not self.roles or "*" in self.roles:
            return True
        return role in self.roles


class PermissionManager:
    """权限管理器"""

    def __init__(self):
        self._rules: list[PermissionRule] = []

    def add_rule(self, rule: PermissionRule) -> None:
        """添加权限规则"""
        self._rules.append(rule)

    def remove_rule(self, tool_name: str) -> int:
        """移除某个工具的所有规则"""
        original_len = len(self._rules)
        self._rules = [r for r in self._rules if r.tool_name != tool_name]
        return original_len - len(self._rules)

    def check_permission(
        self,
        tool_name: str,
        role: str = "default",
    ) -> PermissionLevel:
        """检查工具权限

        优先级:
        1. 精确匹配的工具规则
        2. 通配符规则 (*)
        3. 默认: ALLOW

        Args:
            tool_name: 工具名称
            role: 用户角色

        Returns:
            权限级别
        """
        # 清理过期规则
        self._rules = [r for r in self._rules if not r.is_expired()]

        # 精确匹配
        for rule in self._rules:
            if rule.tool_name == tool_name and rule.matches_role(role):
                return rule.level

        # 通配符匹配
        for rule in self._rules:
            if rule.tool_name == "*" and rule.matches_role(role):
                return rule.level

        return PermissionLevel.ALLOW

    def grant(
        self,
        tool_name: str,
        roles: Optional[list[str]] = None,
        expires_in: Optional[float] = None,
    ) -> None:
        """授予工具访问权限"""
        expires_at = time.time() + expires_in if expires_in else None
        self.add_rule(PermissionRule(
            tool_name=tool_name,
            level=PermissionLevel.ALLOW,
            roles=roles or ["*"],
            expires_at=expires_at,
        ))

    def deny(
        self,
        tool_name: str,
        roles: Optional[list[str]] = None,
        expires_in: Optional[float] = None,
    ) -> None:
        """拒绝工具访问"""
        expires_at = time.time() + expires_in if expires_in else None
        self.add_rule(PermissionRule(
            tool_name=tool_name,
            level=PermissionLevel.DENY,
            roles=roles or ["*"],
            expires_at=expires_at,
        ))

    def set_require_approval(
        self,
        tool_name: str,
        roles: Optional[list[str]] = None,
        expires_in: Optional[float] = None,
    ) -> None:
        """设置工具需要审批"""
        expires_at = time.time() + expires_in if expires_in else None
        self.add_rule(PermissionRule(
            tool_name=tool_name,
            level=PermissionLevel.REQUIRE_APPROVAL,
            roles=roles or ["*"],
            expires_at=expires_at,
        ))

    def list_rules(self) -> list[dict[str, Any]]:
        """列出所有规则"""
        return [
            {
                "tool_name": r.tool_name,
                "level": r.level.value,
                "roles": r.roles,
                "expires_at": r.expires_at,
                "expired": r.is_expired(),
            }
            for r in self._rules
        ]

    def get_tool_permissions(self, tool_name: str) -> list[dict[str, Any]]:
        """获取工具的所有权限规则"""
        return [
            {
                "tool_name": r.tool_name,
                "level": r.level.value,
                "roles": r.roles,
                "expires_at": r.expires_at,
            }
            for r in self._rules
            if r.tool_name == tool_name and not r.is_expired()
        ]