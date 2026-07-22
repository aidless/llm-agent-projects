"""图片描述生成器 - 使用 Vision LLM API 生成图片描述。"""

import abc
import base64
import io
import json
import logging
from typing import Dict, List, Optional

from PIL import Image

from app.models import VisionDescription
from utils.image_utils import image_to_base64

logger = logging.getLogger(__name__)


class BaseVisionClient(abc.ABC):
    """Vision LLM 客户端抽象基类。"""

    @abc.abstractmethod
    def describe(self, image: Image.Image, prompt: str = "") -> VisionDescription:
        ...

    @abc.abstractmethod
    def chat(self, image: Image.Image, question: str) -> str:
        ...


class MockVisionClient(BaseVisionClient):
    """Mock Vision 客户端 - 用于测试。"""

    def describe(self, image: Image.Image, prompt: str = "") -> VisionDescription:
        w, h = image.size
        return VisionDescription(
            description=f"这是一张 {w}x{h} 像素的图片 (mock 描述)",
            confidence=0.95,
            labels=["mock", "test", f"{w}x{h}"],
        )

    def chat(self, image: Image.Image, question: str) -> str:
        w, h = image.size
        return f"这是针对 {w}x{h} 图片的回答 (mock): {question}"


class ImageDescriber:
    """图片描述生成器。

    使用 Vision LLM API 生成图片描述，支持多种提示模板。
    """

    DEFAULT_PROMPTS = {
        "general": "请详细描述这张图片的内容。",
        "document": "请描述这张文档图片的内容，包括标题、正文、表格等信息。",
        "chart": "请描述这张图表，包括图表类型、数据趋势和关键数值。",
        "receipt": "请提取这张收据/发票中的所有文字信息。",
    }

    def __init__(self, client: Optional[BaseVisionClient] = None):
        """初始化描述生成器。

        Args:
            client: Vision LLM 客户端。默认使用 MockVisionClient。
        """
        self.client = client or MockVisionClient()

    def describe(
        self,
        image: Image.Image,
        prompt_type: str = "general",
        custom_prompt: str = "",
    ) -> VisionDescription:
        """生成图片描述。

        Args:
            image: PIL Image 对象。
            prompt_type: 预设提示类型 (general/document/chart/receipt)。
            custom_prompt: 自定义提示 (优先级高于 prompt_type)。

        Returns:
            VisionDescription: 描述结果。
        """
        prompt = custom_prompt or self.DEFAULT_PROMPTS.get(
            prompt_type, self.DEFAULT_PROMPTS["general"]
        )
        return self.client.describe(image, prompt)

    def describe_multi_page(
        self,
        images: List[Image.Image],
        prompt_type: str = "document",
    ) -> List[VisionDescription]:
        """描述多页文档。

        Args:
            images: 图片列表。
            prompt_type: 提示类型。

        Returns:
            List[VisionDescription]: 每页的描述结果。
        """
        results = []
        for i, img in enumerate(images):
            try:
                desc = self.describe(img, prompt_type)
                results.append(desc)
            except Exception as e:
                logger.warning(f"第{i + 1}页描述生成失败: {e}")
                results.append(
                    VisionDescription(
                        description=f"[描述生成失败: {e}]",
                        confidence=0.0,
                        labels=[],
                    )
                )
        return results
