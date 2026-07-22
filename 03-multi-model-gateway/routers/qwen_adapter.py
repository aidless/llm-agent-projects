"""
Qwen (通义千问) 适配器
通过阿里云 DashScope OpenAI 兼容接口调用
"""
import logging
from typing import Optional

import httpx

from app.config import ModelConfig
from app.models import ChatCompletionRequest, ChatCompletionResponse
from routers.base import BaseAdapter

logger = logging.getLogger(__name__)


class QwenAdapter(BaseAdapter):
    """Qwen (通义千问) 适配器 - 通过 DashScope OpenAI 兼容接口"""

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        timeout: float = 60.0,
        api_key: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """调用 Qwen Chat Completions API（OpenAI 兼容格式）"""
        key = api_key or self.get_next_api_key()
        url = f"{self.config.base_url}/chat/completions"

        # Qwen DashScope 兼容 OpenAI 格式
        payload = {
            "model": self.config.model_id,
            "messages": [msg.to_openai_dict() for msg in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        logger.info(f"[Qwen] 发送请求到 {url}, 模型: {self.config.model_id}")

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        return self._parse_response(data)

    def _parse_response(self, data: dict) -> ChatCompletionResponse:
        """解析 Qwen API 响应（OpenAI 兼容格式）"""
        choices = data.get("choices", [])
        usage_data = data.get("usage", {})

        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)

        response = self._build_default_response(
            content=choices[0]["message"]["content"] if choices else "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=choices[0].get("finish_reason", "stop") if choices else "stop",
        )
        response.id = data.get("id", response.id)
        response.created = data.get("created", response.created)

        return response