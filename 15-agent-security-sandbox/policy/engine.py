"""策略引擎 - 策略管理、继承和组合"""

from typing import Dict, Optional, List
from policy.models import SecurityPolicy, SecurityLevel
from policy.presets import get_preset_policy, PRESET_POLICIES


class PolicyEngine:
    """策略引擎"""

    def __init__(self):
        self._policies: Dict[str, SecurityPolicy] = {}
        self._load_presets()

    def _load_presets(self):
        """加载预设策略"""
        for level, factory in PRESET_POLICIES.items():
            policy = factory()
            self._policies[policy.name] = policy

    def get_policy(self, name: str) -> Optional[SecurityPolicy]:
        """获取指定名称的策略"""
        return self._policies.get(name)

    def get_policy_by_level(self, level: SecurityLevel) -> Optional[SecurityPolicy]:
        """根据安全等级获取策略"""
        for policy in self._policies.values():
            if policy.level == level:
                return policy
        return get_preset_policy(level)

    def add_policy(self, policy: SecurityPolicy) -> bool:
        """添加自定义策略"""
        if not policy.name:
            return False
        self._policies[policy.name] = policy
        return True

    def remove_policy(self, name: str) -> bool:
        """移除策略（不允许移除预设策略）"""
        if name not in self._policies:
            return False
        # 预设策略不能删除 - 检查 name 是否在预设策略名称中
        preset_names = {factory().name for factory in PRESET_POLICIES.values()}
        if name in preset_names:
            return False
        del self._policies[name]
        return True

    def list_policies(self) -> List[dict]:
        """列出所有策略"""
        return [p.to_dict() for p in self._policies.values()]

    def inherit_policy(
        self,
        base_name: str,
        overrides: dict,
        new_name: str,
        description: str = "",
    ) -> Optional[SecurityPolicy]:
        """基于已有策略创建继承策略"""
        base = self._policies.get(base_name)
        if not base:
            return None

        from dataclasses import replace
        new_policy = replace(base, name=new_name, description=description or f"Inherited from {base_name}")

        # 应用覆盖
        if "resource_limits" in overrides:
            rl = overrides["resource_limits"]
            if isinstance(rl, dict):
                current_rl = new_policy.resource_limits
                for k, v in rl.items():
                    if hasattr(current_rl, k):
                        setattr(current_rl, k, v)

        if "module_policy" in overrides:
            mp = overrides["module_policy"]
            if isinstance(mp, dict):
                current_mp = new_policy.module_policy
                for k, v in mp.items():
                    if k in ("allowed_modules", "blocked_modules", "allowed_builtins", "blocked_builtins"):
                        if isinstance(v, (list, set)):
                            v = set(v)
                        setattr(current_mp, k, v)
                    elif hasattr(current_mp, k):
                        setattr(current_mp, k, v)

        if "filesystem_policy" in overrides:
            fp = overrides["filesystem_policy"]
            if isinstance(fp, dict):
                current_fp = new_policy.filesystem_policy
                for k, v in fp.items():
                    if k in ("allowed_read_dirs", "allowed_write_dirs"):
                        if isinstance(v, (list, set)):
                            v = set(v)
                        setattr(current_fp, k, v)
                    elif hasattr(current_fp, k):
                        setattr(current_fp, k, v)

        if "network_policy" in overrides:
            np_ = overrides["network_policy"]
            if isinstance(np_, dict):
                current_np = new_policy.network_policy
                for k, v in np_.items():
                    if k in ("allowed_domains", "blocked_domains", "allowed_ports", "blocked_ports"):
                        if isinstance(v, (list, set)):
                            v = set(v)
                        setattr(current_np, k, v)
                    elif hasattr(current_np, k):
                        setattr(current_np, k, v)

        if "level" in overrides:
            new_policy.level = SecurityLevel(overrides["level"])

        self._policies[new_name] = new_policy
        return new_policy

    def combine_policies(self, names: List[str], combined_name: str) -> Optional[SecurityPolicy]:
        """组合多个策略（取最严格的配置）"""
        policies = [self._policies.get(n) for n in names]
        policies = [p for p in policies if p is not None]

        if not policies:
            return None

        # 以第一个策略为基础
        from dataclasses import replace
        result = replace(policies[0], name=combined_name, description=f"Combined from {names}")

        # 合并资源限制 - 取最严格
        for p in policies[1:]:
            result.resource_limits.max_execution_time = min(
                result.resource_limits.max_execution_time, p.resource_limits.max_execution_time
            )
            result.resource_limits.max_memory_mb = min(
                result.resource_limits.max_memory_mb, p.resource_limits.max_memory_mb
            )
            result.resource_limits.max_output_size = min(
                result.resource_limits.max_output_size, p.resource_limits.max_output_size
            )
            result.resource_limits.max_file_size = min(
                result.resource_limits.max_file_size, p.resource_limits.max_file_size
            )
            result.resource_limits.max_concurrent = min(
                result.resource_limits.max_concurrent, p.resource_limits.max_concurrent
            )
            result.resource_limits.cpu_time_limit = min(
                result.resource_limits.cpu_time_limit, p.resource_limits.cpu_time_limit
            )

            # 合并模块策略 - 取交集
            if p.module_policy.allowed_modules:
                if result.module_policy.allowed_modules:
                    result.module_policy.allowed_modules &= p.module_policy.allowed_modules
                else:
                    result.module_policy.allowed_modules = set(p.module_policy.allowed_modules)
            result.module_policy.blocked_modules |= p.module_policy.blocked_modules
            result.module_policy.blocked_builtins |= p.module_policy.blocked_builtins

            # 合并文件策略 - 取最严格
            result.filesystem_policy.block_all_writes = (
                result.filesystem_policy.block_all_writes or p.filesystem_policy.block_all_writes
            )
            result.filesystem_policy.block_all_reads = (
                result.filesystem_policy.block_all_reads or p.filesystem_policy.block_all_reads
            )

            # 合并网络策略 - 取最严格
            result.network_policy.block_all_network = (
                result.network_policy.block_all_network or p.network_policy.block_all_network
            )

        # 取最高安全等级
        level_order = [SecurityLevel.LOW, SecurityLevel.MEDIUM, SecurityLevel.HIGH, SecurityLevel.STRICT]
        max_level_idx = 0
        for p in policies:
            idx = level_order.index(p.level)
            if idx > max_level_idx:
                max_level_idx = idx
        result.level = level_order[max_level_idx]

        self._policies[combined_name] = result
        return result


# 全局策略引擎实例
engine = PolicyEngine()