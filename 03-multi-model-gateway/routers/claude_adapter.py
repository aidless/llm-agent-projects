"""
Claude 适配器
Claude 使用 Anthropic API，请求/响应格式与 OpenAI 不同，需要适配转换
"""
import logging
from typing import Optional

import httpx

from app.config import ModelConfig
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from routers.base import BaseAdapter

logger = logging.getLogger(__name__)


class ClaudeAdapter(BaseAdapter):
    """Claude 适配器 - 将 OpenAI 格式转换为 Anthropic API 格式"""

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        timeout: float = 60.0,
        api_key: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """调用 Anthropic Messages API"""
        key = api_key or self.get_next_api_key()
        url = f"{self.config.base_url}/messages"

        # 将 OpenAI 格式的 messages 转换为 Anthropic 格式
        system_prompt, claude_messages = self._convert_messages(request.messages)

        # 构建 Anthropic API 请求体
        payload: dict = {
            "model": self.config.model_id,
            "messages": claude_messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        # 如果有系统提示，添加到顶层
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        logger.info(f"[Claude] 发送请求到 {url}, 模型: {self.config.model_id}")

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        return self._parse_response(data)

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str, list[dict]]:
        """
        将 OpenAI 格式的消息列表转换为 Anthropic 格式

        OpenAI 格式: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        Anthropic 格式: system 在顶层，messages 只有 user/assistant 交替

        返回:
            (system_prompt, claude_messages)
        """
        system_parts = []
        claude_messages = []

        for msg in messages:
            if msg.role == "system":
                # Anthropic 的 system 是顶层字段，单独收集
                system_parts.append(msg.content)
            elif msg.role in ("user", "assistant"):
                claude_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        system_prompt = "\n".join(system_parts) if system_parts else ""
        return system_prompt, claude_messages

    def _parse_response(self, data: dict) -> ChatCompletionResponse:
        """解析 Anthropic API 响应为统一格式"""
        # Anthropic 响应格式:
        # {
        #   "content": [{"type": "text", "text": "..."}],
        #   "usage": {"input_tokens": 10, "output_tokens": 20}
        # }

        content_blocks = data.get("content", [])
        text_content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")

        usage_data = data.get("usage", {})
        prompt_tokens = usage_data.get("input_tokens", 0)
        completion_tokens = usage_data.get("output_tokens", 0)

        # 判断停止原因
        stop_reason = data.get("stop_reason", "end_turn")
        finish_reason = "stop" if stop_reason in ("end_turn", "max_tokens") else stop_reason

        response = self._build_default_response(
            content=text_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )
        response.id = data.get("id", response.id)

        return response