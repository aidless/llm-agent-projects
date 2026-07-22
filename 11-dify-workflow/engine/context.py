"""执行上下文 - 管理节点间变量传递和模板解析"""

from __future__ import annotations

import os
import re
import copy
from typing import Any, Dict, Optional

from .errors import VariableResolutionError


# 模板变量正则: {{node_id.output.key}} 或 {{node_id.output}} 或 {{env.VAR}}
_VAR_PATTERN = re.compile(r"\{\{(\w+(?:\.\w+)*)\}\}")


class ExecutionContext:
    """工作流执行上下文，负责变量存储和模板解析"""

    def __init__(self, env_vars: Optional[Dict[str, str]] = None):
        # node_id -> {输出key: 值}
        self._node_outputs: Dict[str, Dict[str, Any]] = {}
        # node_id -> 完整输入 (调试用)
        self._node_inputs: Dict[str, Dict[str, Any]] = {}
        # 环境变量
        self._env: Dict[str, str] = dict(env_vars or os.environ)

    # ── 节点输出 ────────────────────────────────────────────

    def set_output(self, node_id: str, key: str, value: Any) -> None:
        """设置节点输出"""
        if node_id not in self._node_outputs:
            self._node_outputs[node_id] = {}
        self._node_outputs[node_id][key] = value

    def set_outputs(self, node_id: str, outputs: Dict[str, Any]) -> None:
        """批量设置节点输出"""
        if node_id not in self._node_outputs:
            self._node_outputs[node_id] = {}
        self._node_outputs[node_id].update(outputs)

    def get_output(self, node_id: str, key: str = None) -> Any:
        """获取节点输出。若 key 为 None 则返回整个输出字典"""
        if node_id not in self._node_outputs:
            raise VariableResolutionError(
                f"节点 '{node_id}' 尚未产生输出", node_id=node_id
            )
        if key is None:
            return self._node_outputs[node_id]
        if key not in self._node_outputs[node_id]:
            raise VariableResolutionError(
                f"节点 '{node_id}' 没有输出 '{key}'，可用: {list(self._node_outputs[node_id].keys())}",
                node_id=node_id,
            )
        return self._node_outputs[node_id][key]

    def get_node_outputs(self, node_id: str) -> Dict[str, Any]:
        return dict(self._node_outputs.get(node_id, {}))

    # ── 节点输入 (快照) ─────────────────────────────────────

    def set_input(self, node_id: str, inputs: Dict[str, Any]) -> None:
        self._node_inputs[node_id] = copy.deepcopy(inputs)

    def get_input(self, node_id: str) -> Dict[str, Any]:
        return copy.deepcopy(self._node_inputs.get(node_id, {}))

    # ── 模板解析 ────────────────────────────────────────────

    def resolve(self, template: str) -> Any:
        """
        解析模板字符串。
        - 若整个字符串只有一个 {{}} 且无其他内容，返回原始类型 (不强制 str)
        - 若有多个变量或混合文本，返回拼接字符串
        - 支持 {{env.VAR_NAME}} 引用环境变量
        """
        matches = list(_VAR_PATTERN.finditer(template))

        if not matches:
            return template  # 无变量，原样返回

        # 整个字符串就是一个变量
        if len(matches) == 1 and matches[0].group(0) == template.strip():
            ref = matches[0].group(1)
            return self._resolve_reference(ref)

        # 多变量或混合文本 -> 拼接为字符串
        result = template
        for m in matches:
            ref = m.group(1)
            val = self._resolve_reference(ref)
            result = result.replace(m.group(0), str(val))
        return result

    def resolve_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """递归解析字典中所有模板变量"""
        return {k: self._resolve_value(v) for k, v in data.items()}

    def _resolve_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.resolve(value)
        if isinstance(value, dict):
            return self.resolve_dict(value)
        if isinstance(value, list):
            return [self._resolve_value(item) for item in value]
        return value

    def _resolve_reference(self, ref: str) -> Any:
        """解析单个引用: node_id.output.key / env.VAR"""
        parts = ref.split(".")

        # 环境变量
        if parts[0] == "env" and len(parts) == 2:
            var_name = parts[1]
            if var_name in self._env:
                return self._env[var_name]
            raise VariableResolutionError(
                f"环境变量 '{var_name}' 未定义"
            )

        # 节点输出引用
        if len(parts) >= 2:
            node_id = parts[0]
            if parts[1] != "output":
                raise VariableResolutionError(
                    f"无效的引用格式 '{ref}'，期望 {{node_id.output.key}}"
                )
            if len(parts) == 2:
                return self.get_output(node_id)
            # parts[2:] 是嵌套 key
            output = self.get_output(node_id)
            for key in parts[2:]:
                if isinstance(output, dict) and key in output:
                    output = output[key]
                else:
                    raise VariableResolutionError(
                        f"无法解析 '{ref}'，key '{key}' 不存在于节点 '{node_id}' 的输出中",
                        node_id=node_id,
                    )
            return output

        raise VariableResolutionError(
            f"无效的引用格式 '{ref}'，期望 {{node_id.output.key}} 或 {{env.VAR}}"
        )

    # ── 快照 / 导出 ─────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        return {
            "node_outputs": copy.deepcopy(self._node_outputs),
            "node_inputs": copy.deepcopy(self._node_inputs),
        }

    def clear(self) -> None:
        self._node_outputs.clear()
        self._node_inputs.clear()
