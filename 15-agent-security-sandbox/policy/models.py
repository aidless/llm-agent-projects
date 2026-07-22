"""策略模型定义"""

from enum import Enum
from typing import Set, Optional, Dict, Any, List
from dataclasses import dataclass, field


class SecurityLevel(str, Enum):
    """安全等级"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    STRICT = "STRICT"


@dataclass
class ResourceLimits:
    """资源限制配置"""
    max_execution_time: float = 30.0       # 秒
    max_memory_mb: int = 256                # MB
    max_output_size: int = 1024 * 1024      # 1MB
    max_file_size: int = 10 * 1024 * 1024   # 10MB
    max_concurrent: int = 10                # 最大并发执行数
    cpu_time_limit: float = 10.0            # CPU 时间秒数

    def to_dict(self) -> dict:
        return {
            "max_execution_time": self.max_execution_time,
            "max_memory_mb": self.max_memory_mb,
            "max_output_size": self.max_output_size,
            "max_file_size": self.max_file_size,
            "max_concurrent": self.max_concurrent,
            "cpu_time_limit": self.cpu_time_limit,
        }


@dataclass
class ModulePolicy:
    """模块策略"""
    allowed_modules: Set[str] = field(default_factory=set)
    blocked_modules: Set[str] = field(default_factory=set)
    allowed_builtins: Set[str] = field(default_factory=set)
    blocked_builtins: Set[str] = field(default_factory=set)
    allow_star_import: bool = False

    def to_dict(self) -> dict:
        return {
            "allowed_modules": sorted(self.allowed_modules),
            "blocked_modules": sorted(self.blocked_modules),
            "allowed_builtins": sorted(self.allowed_builtins),
            "blocked_builtins": sorted(self.blocked_builtins),
            "allow_star_import": self.allow_star_import,
        }


@dataclass
class FilesystemPolicy:
    """文件系统策略"""
    allowed_read_dirs: Set[str] = field(default_factory=set)
    allowed_write_dirs: Set[str] = field(default_factory=set)
    block_all_writes: bool = True
    block_all_reads: bool = False

    def to_dict(self) -> dict:
        return {
            "allowed_read_dirs": sorted(self.allowed_read_dirs),
            "allowed_write_dirs": sorted(self.allowed_write_dirs),
            "block_all_writes": self.block_all_writes,
            "block_all_reads": self.block_all_reads,
        }


@dataclass
class NetworkPolicy:
    """网络策略"""
    allowed_domains: Set[str] = field(default_factory=set)
    blocked_domains: Set[str] = field(default_factory=set)
    allowed_ports: Set[int] = field(default_factory=set)
    blocked_ports: Set[int] = field(default_factory=set)
    block_all_network: bool = True

    def to_dict(self) -> dict:
        return {
            "allowed_domains": sorted(self.allowed_domains),
            "blocked_domains": sorted(self.blocked_domains),
            "allowed_ports": sorted(self.allowed_ports),
            "blocked_ports": sorted(self.blocked_ports),
            "block_all_network": self.block_all_network,
        }


@dataclass
class SecurityPolicy:
    """完整安全策略"""
    name: str
    level: SecurityLevel = SecurityLevel.MEDIUM
    description: str = ""
    enabled: bool = True
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    module_policy: ModulePolicy = field(default_factory=ModulePolicy)
    filesystem_policy: FilesystemPolicy = field(default_factory=FilesystemPolicy)
    network_policy: NetworkPolicy = field(default_factory=NetworkPolicy)
    custom_rules: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "level": self.level.value,
            "description": self.description,
            "enabled": self.enabled,
            "resource_limits": self.resource_limits.to_dict(),
            "module_policy": self.module_policy.to_dict(),
            "filesystem_policy": self.filesystem_policy.to_dict(),
            "network_policy": self.network_policy.to_dict(),
            "custom_rules": self.custom_rules,
        }