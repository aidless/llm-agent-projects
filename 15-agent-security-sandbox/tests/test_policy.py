"""策略引擎测试"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy.engine import PolicyEngine
from policy.models import SecurityLevel, SecurityPolicy, ResourceLimits
from policy.presets import get_preset_policy, list_preset_policies


@pytest.fixture
def engine():
    return PolicyEngine()


class TestPresetPolicies:
    """预设策略测试"""

    def test_four_levels_exist(self):
        """测试4个安全等级都存在"""
        policies = list_preset_policies()
        assert len(policies) == 4
        levels = {p["level"] for p in policies}
        assert levels == {"LOW", "MEDIUM", "HIGH", "STRICT"}

    def test_get_preset_by_level(self):
        """测试按等级获取策略"""
        for level in SecurityLevel:
            policy = get_preset_policy(level)
            assert policy is not None
            assert policy.level == level

    def test_low_most_permissive(self):
        """测试 LOW 级别最宽松"""
        low = get_preset_policy(SecurityLevel.LOW)
        strict = get_preset_policy(SecurityLevel.STRICT)
        assert low.resource_limits.max_execution_time > strict.resource_limits.max_execution_time
        assert low.resource_limits.max_memory_mb > strict.resource_limits.max_memory_mb

    def test_strict_most_restrictive(self):
        """测试 STRICT 级别最严格"""
        strict = get_preset_policy(SecurityLevel.STRICT)
        assert strict.filesystem_policy.block_all_writes is True
        assert strict.filesystem_policy.block_all_reads is True
        assert strict.network_policy.block_all_network is True
        assert strict.resource_limits.max_execution_time <= 5.0
        assert strict.resource_limits.max_memory_mb <= 512

    def test_strict_blocks_open(self):
        """测试 STRICT 级别拦截 open"""
        strict = get_preset_policy(SecurityLevel.STRICT)
        assert "open" in strict.module_policy.blocked_builtins

    def test_policies_have_names(self):
        """测试策略都有名称"""
        for level in SecurityLevel:
            policy = get_preset_policy(level)
            assert policy.name
            assert policy.description

    def test_policy_serialization(self):
        """测试策略序列化"""
        policy = get_preset_policy(SecurityLevel.MEDIUM)
        d = policy.to_dict()
        assert "name" in d
        assert "level" in d
        assert "resource_limits" in d
        assert "module_policy" in d


class TestPolicyEngine:
    """策略引擎测试"""

    def test_list_policies(self, engine):
        """测试列出策略"""
        policies = engine.list_policies()
        assert len(policies) >= 4

    def test_get_policy(self, engine):
        """测试获取策略"""
        policy = engine.get_policy("medium")
        assert policy is not None
        assert policy.name == "medium"

    def test_get_nonexistent_policy(self, engine):
        """测试获取不存在的策略"""
        policy = engine.get_policy("nonexistent")
        assert policy is None

    def test_add_custom_policy(self, engine):
        """测试添加自定义策略"""
        policy = SecurityPolicy(name="custom_test", level=SecurityLevel.MEDIUM)
        result = engine.add_policy(policy)
        assert result is True
        assert engine.get_policy("custom_test") is not None

    def test_remove_custom_policy(self, engine):
        """测试删除自定义策略"""
        engine.add_policy(SecurityPolicy(name="to_remove", level=SecurityLevel.LOW))
        result = engine.remove_policy("to_remove")
        assert result is True
        assert engine.get_policy("to_remove") is None

    def test_cannot_remove_preset(self, engine):
        """测试不能删除预设策略"""
        result = engine.remove_policy("medium")
        assert result is False

    def test_inherit_policy(self, engine):
        """测试继承策略"""
        policy = engine.inherit_policy(
            base_name="medium",
            overrides={"resource_limits": {"max_execution_time": 1.0}},
            new_name="inherited_test",
        )
        assert policy is not None
        assert policy.name == "inherited_test"
        assert policy.resource_limits.max_execution_time == 1.0

    def test_inherit_nonexistent_policy(self, engine):
        """测试继承不存在的策略"""
        policy = engine.inherit_policy(
            base_name="nonexistent",
            overrides={},
            new_name="should_fail",
        )
        assert policy is None

    def test_combine_policies(self, engine):
        """测试组合策略"""
        policy = engine.combine_policies(
            names=["low", "strict"],
            combined_name="combined_test",
        )
        assert policy is not None
        assert policy.name == "combined_test"
        # 应该取 STRICT 等级
        assert policy.level == SecurityLevel.STRICT
        # 资源限制应该取最严格
        assert policy.resource_limits.max_execution_time == 5.0  # STRICT 的时间限制

    def test_combine_nonexistent_policies(self, engine):
        """测试组合不存在的策略"""
        policy = engine.combine_policies(
            names=["nonexistent1", "nonexistent2"],
            combined_name="should_fail",
        )
        assert policy is None