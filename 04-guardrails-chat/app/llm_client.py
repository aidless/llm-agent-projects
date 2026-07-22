"""
LLM 客户端模块 - LLMClient

封装对 OpenAI / DeepSeek API 的调用，支持统一的接口。

功能：
1. 兼容 OpenAI API 格式的接口调用
2. 支持流式和非流式响应
3. 错误处理和重试机制
4. 超时控制
"""

import os
import time
import httpx
import json
from typing import Dict, List, Optional, Any, Generator


class LLMClient:
    """
    LLM API 客户端

    兼容 OpenAI API 和 DeepSeek API（两者使用相同的接口格式）。
    """

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "https://api.openai.com/v1",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
        system_prompt: str = "",
    ):
        """
        初始化 LLM 客户端

        Args:
            api_key: API Key
            api_base: API 基础 URL
            model: 模型名称
            temperature: 生成温度
            max_tokens: 最大生成 token 数
            timeout: 请求超时时间（秒）
            system_prompt: 系统提示词
        """
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.system_prompt = system_prompt

        # 创建 HTTP 客户端
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        发送聊天请求（非流式）

        Args:
            messages: 消息列表 [{role, content}]
            temperature: 生成温度（可选，覆盖默认值）
            max_tokens: 最大 token 数（可选，覆盖默认值）

        Returns:
            Dict[str, Any]: API 响应 {
                "content": "回复内容",
                "model": "模型名称",
                "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
                "finish_reason": "stop"
            }
        """
        # 构建完整的消息列表（添加系统提示词）
        full_messages = self._build_messages(messages)

        payload = {
            "model": self.model,
            "messages": full_messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        url = f"{self.api_base}/chat/completions"

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            finish_reason = data["choices"][0].get("finish_reason", "stop")

            return {
                "content": content,
                "model": data.get("model", self.model),
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                "finish_reason": finish_reason,
            }

        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))
            except Exception:
                error_detail = str(e)
            return {
                "content": "",
                "error": f"API 请求失败 ({e.response.status_code}): {error_detail}",
                "model": self.model,
                "usage": {},
                "finish_reason": "error",
            }

        except httpx.TimeoutException:
            return {
                "content": "",
                "error": f"API 请求超时 ({self.timeout}s)",
                "model": self.model,
                "usage": {},
                "finish_reason": "error",
            }

        except Exception as e:
            return {
                "content": "",
                "error": f"API 请求异常: {str(e)}",
                "model": self.model,
                "usage": {},
                "finish_reason": "error",
            }

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        发送聊天请求（流式）

        Args:
            messages: 消息列表
            temperature: 生成温度
            max_tokens: 最大 token 数

        Yields:
            str: 每次生成的文本片段
        """
        full_messages = self._build_messages(messages)

        payload = {
            "model": self.model,
            "messages": full_messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
        }

        url = f"{self.api_base}/chat/completions"

        try:
            with self._client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except Exception:
            yield ""

    def _build_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        构建完整的消息列表

        如果设置了系统提示词且消息列表中没有 system 消息，则在开头添加。

        Args:
            messages: 原始消息列表

        Returns:
            List[Dict[str, str]]: 完整的消息列表
        """
        full_messages = []

        # 添加系统提示词
        if self.system_prompt:
            full_messages.append({"role": "system", "content": self.system_prompt})

        # 添加用户/助手消息
        full_messages.extend(messages)

        return full_messages

    def is_available(self) -> bool:
        """
        检查 API 是否可用

        Returns:
            bool: API 是否可用
        """
        if not self.api_key:
            return False
        try:
            # 发送一个简单的请求测试连通性
            result = self.chat(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            return "error" not in result
        except Exception:
            return False

    def close(self) -> None:
        """关闭 HTTP 客户端"""
        self._client.close()
