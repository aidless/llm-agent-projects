"""LLM 节点 - 调用 LLM 生成文本"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .base import BaseNode

logger = logging.getLogger(__name__)


class LLMNode(BaseNode):
    """LLM 文本生成节点，支持模板变量"""

    node_type = "llm"

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: 可选的 LLM 客户端，签名: client(prompt, **kwargs) -> str
                        若为 None 则使用内置的 mock 实现
        """
        self._client = llm_client

    def execute(
        self,
        inputs: Dict[str, Any],
        context: "ExecutionContext",
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = inputs.get("prompt", "")
        system_prompt = inputs.get("system_prompt", "")
        model = inputs.get("model", "default")
        temperature = inputs.get("temperature", 0.7)
        max_tokens = inputs.get("max_tokens", 1024)

        if self._client:
            result = self._client(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            # Mock: 返回提示词的简单变换
            result = f"[LLM Mock] 基于提示生成的文本: {prompt[:100]}"

        return {
            "text": result,
            "model": model,
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(result)},
        }


def create_llm_handler(llm_client=None):
    """工厂函数，返回兼容 executor.register_handler 的处理器"""
    node = LLMNode(llm_client=llm_client)

    def handler(inputs, context, config):
        return node.execute(inputs, context, config)

    handler.node_type = LLMNode.node_type
    return handler
