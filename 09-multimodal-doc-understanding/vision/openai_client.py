"""OpenAI Vision 客户端 - 统一实现描述、问答和图表理解功能。"""

import base64
import io
import json
import logging
import os
from typing import Optional

import openai
from PIL import Image

from app.models import ChartUnderstanding, VisionDescription, VQAResult
from utils.image_utils import image_to_base64
from vision.chart_understanding import BaseChartClient
from vision.describer import BaseVisionClient
from vision.vqa import BaseVQAClient

logger = logging.getLogger(__name__)

# 图表理解专用提示
_CHART_SYSTEM_PROMPT = (
    "You are a data visualization expert. Analyze the chart image and respond "
    "in JSON with these exact keys: chart_type (bar/line/pie/scatter/area/mixed/unknown), "
    "description (detailed chart description), data_summary (dict with extracted data points, "
    "categories, values, max, min), key_insights (list of key observations). "
    "Respond ONLY with valid JSON, no extra text."
)


def _image_to_data_url(image: Image.Image, format: str = "PNG") -> str:
    """将 PIL Image 转换为 base64 data URL。"""
    mime = "image/png" if format == "PNG" else "image/jpeg"
    b64 = image_to_base64(image, format=format)
    return f"data:{mime};base64,{b64}"


class OpenAIVisionClient(BaseVisionClient, BaseVQAClient, BaseChartClient):
    """基于 OpenAI Vision API 的统一客户端。

    同时实现 BaseVisionClient、BaseVQAClient 和 BaseChartClient 三个接口。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
    ):
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        logger.info(f"OpenAIVisionClient 初始化: model={model}, base_url={base_url}")

    # ---- BaseVisionClient ----

    def describe(self, image: Image.Image, prompt: str = "") -> VisionDescription:
        """使用 Vision API 生成图片描述。"""
        data_url = _image_to_data_url(image)
        text_content = prompt or "请详细描述这张图片的内容。"
        try:
            response = self._call_vision(text_content, data_url)
            return VisionDescription(
                description=response,
                confidence=0.85,
                labels=self._extract_labels(response),
            )
        except openai.RateLimitError:
            logger.warning("OpenAI API 速率限制，请稍后重试")
            raise
        except openai.APIError as e:
            logger.error(f"OpenAI API 错误: {e}")
            raise

    def chat(self, image: Image.Image, question: str) -> str:
        """使用 Vision API 进行对话式问答。"""
        data_url = _image_to_data_url(image)
        try:
            return self._call_vision(question, data_url)
        except openai.RateLimitError:
            logger.warning("OpenAI API 速率限制，请稍后重试")
            raise
        except openai.APIError as e:
            logger.error(f"OpenAI API 错误: {e}")
            raise

    # ---- BaseVQAClient ----

    def answer(self, image: Image.Image, question: str) -> VQAResult:
        """使用 Vision API 回答关于图片的问题。"""
        data_url = _image_to_data_url(image)
        prompt = f"Answer the following question about this image. Be specific and concise.\n\nQuestion: {question}"
        try:
            answer_text = self._call_vision(prompt, data_url)
            return VQAResult(
                question=question,
                answer=answer_text,
                confidence=0.85,
                sources=["openai-vision-api"],
            )
        except openai.RateLimitError:
            logger.warning("OpenAI API 速率限制，请稍后重试")
            raise
        except openai.APIError as e:
            logger.error(f"OpenAI API 错误: {e}")
            raise

    # ---- BaseChartClient ----

    def understand(self, image: Image.Image) -> ChartUnderstanding:
        """使用 Vision API 分析图表。"""
        data_url = _image_to_data_url(image)
        try:
            raw = self._call_vision(
                "Analyze this chart image.",
                data_url,
                system_prompt=_CHART_SYSTEM_PROMPT,
            )
            # 尝试从响应中解析 JSON
            parsed = self._parse_chart_json(raw)
            return ChartUnderstanding(**parsed)
        except openai.RateLimitError:
            logger.warning("OpenAI API 速率限制，请稍后重试")
            raise
        except openai.APIError as e:
            logger.error(f"OpenAI API 错误: {e}")
            raise

    # ---- 工厂方法 ----

    @classmethod
    def create(cls) -> Optional["OpenAIVisionClient"]:
        """从环境变量创建客户端实例。

        读取 OPENAI_API_KEY、OPENAI_BASE_URL、VISION_MODEL。
        如果 OPENAI_API_KEY 未配置则返回 None。
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY 未配置，无法创建 OpenAIVisionClient")
            return None
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("VISION_MODEL", "gpt-4o")
        return cls(api_key=api_key, base_url=base_url, model=model)

    # ---- 内部方法 ----

    def _call_vision(
        self,
        text: str,
        data_url: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """调用 OpenAI Vision API。"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        })
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""

    def _extract_labels(self, description: str) -> list[str]:
        """从描述文本中提取简单标签（取前5个关键词）。"""
        # 简单提取：取描述中的中文/英文词组
        words = description.replace(",", " ").replace(".", " ").replace("，", " ").replace("。", " ").split()
        # 过滤短词，取前5个
        labels = [w for w in words if len(w) >= 2][:5]
        return labels

    def _parse_chart_json(self, raw: str) -> dict:
        """从 Vision API 响应中解析 JSON。"""
        # 尝试直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # 尝试从 markdown code block 中提取
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # 解析失败，构造默认结构
        logger.warning(f"图表 JSON 解析失败，原始响应: {raw[:200]}")
        return {
            "chart_type": "unknown",
            "description": raw,
            "data_summary": {},
            "key_insights": [],
        }
