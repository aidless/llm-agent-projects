"""网络守卫 - 控制沙箱中的网络访问"""

from typing import Set, Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class NetworkGuardConfig:
    """网络守卫配置"""
    allowed_domains: Set[str] = field(default_factory=set)
    blocked_domains: Set[str] = field(default_factory=lambda: {
        "localhost", "127.0.0.1", "0.0.0.0", "::1",
        "169.254.169.254",  # AWS metadata
        "metadata.google.internal",  # GCP metadata
    })
    allowed_ports: Set[int] = field(default_factory=set)
    blocked_ports: Set[int] = field(default_factory=lambda: {
        22, 23, 25, 53, 80, 443, 3306, 5432, 6379, 8080, 8443, 27017,
    })
    block_all_network: bool = True
    allow_dns: bool = False


class NetworkGuard:
    """网络访问守卫"""

    def __init__(self, config: Optional[NetworkGuardConfig] = None):
        self.config = config or NetworkGuardConfig()

    def check_domain(self, domain: str) -> bool:
        """检查是否允许访问指定域名"""
        if self.config.block_all_network:
            return False
        if domain in self.config.blocked_domains:
            return False
        if self.config.allowed_domains:
            return domain in self.config.allowed_domains
        return False  # 默认不允许

    def check_port(self, port: int) -> bool:
        """检查是否允许访问指定端口"""
        if self.config.block_all_network:
            return False
        if port in self.config.blocked_ports:
            return False
        if self.config.allowed_ports:
            return port in self.config.allowed_ports
        return False  # 默认不允许

    def check_connection(self, domain: str, port: int) -> Tuple[bool, str]:
        """检查是否允许网络连接"""
        if self.config.block_all_network:
            return False, "Network access is completely blocked by policy"
        if not self.check_domain(domain):
            return False, f"Domain '{domain}' is not in allowed list"
        if not self.check_port(port):
            return False, f"Port {port} is not in allowed list"
        return True, "Connection allowed"