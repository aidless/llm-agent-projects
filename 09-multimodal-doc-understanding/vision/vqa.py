"""视觉问答 (VQA) 模块。"""

import abc
import logging
from typing import List, Optional

from PIL import Image

from app.models import VQAResult

logger = logging.getLogger(__name__)


class BaseVQAClient(abc.ABC):
    """VQA 客户端抽象基类。"""

    @abc.abstractmethod
    def answer(self, image: Image.Image, question: str) -> VQAResult:
        ...


class MockVQAClient(BaseVQAClient):
    """Mock VQA 客户端。"""

    def answer(self, image: Image.Image, question: str) -> VQAResult:
        w, h = image.size
        return VQAResult(
            question=question,
            answer=f"关于 {w}x{h} 图片的问题 '{question}' 的回答 (mock)。",
            confidence=0.9,
            sources=["mock-source-1"],
        )


class VQAEngine:
    """视觉问答引擎。

    支持对图片进行提问并获取答案。
    """

    def __init__(self, client: Optional[BaseVQAClient] = None):
        """初始化 VQA 引擎。

        Args:
            client: VQA 客户端。默认 MockVQAClient。
        """
        self.client = client or MockVQAClient()

    def answer(self, image: Image.Image, question: str) -> VQAResult:
        """对图片进行提问。

        Args:
            image: PIL Image 对象。
            question: 问题文本。

        Returns:
            VQAResult: 问答结果。
        """
        return self.client.answer(image, question)

    def batch_answer(
        self,
        images: List[Image.Image],
        question: str,
    ) -> List[VQAResult]:
        """批量对多张图片提问。

        Args:
            images: 图片列表。
            question: 问题文本。

        Returns:
            List[VQAResult]: 每张图片的回答。
        """
        results = []
        for i, img in enumerate(images):
            try:
                result = self.answer(img, question)
                results.append(result)
            except Exception as e:
                logger.warning(f"第{i + 1}张图片问答失败: {e}")
                results.append(
                    VQAResult(
                        question=question,
                        answer=f"[问答失败: {e}]",
                        confidence=0.0,
                    )
                )
        return results

    def multi_turn_qa(
        self,
        image: Image.Image,
        questions: List[str],
    ) -> List[VQAResult]:
        """多轮问答。

        Args:
            image: PIL Image 对象。
            questions: 问题列表 (有序)。

        Returns:
            List[VQAResult]: 每个问题的回答。
        """
        results = []
        context = []

        for q in questions:
            # 构建带上下文的问题
            if context:
                full_question = f"[上下文: 之前讨论了 {context[-1]}] {q}"
            else:
                full_question = q

            result = self.answer(image, full_question)
            # 保留原始问题
            result = VQAResult(
                question=q,
                answer=result.answer,
                confidence=result.confidence,
                sources=result.sources,
            )
            context.append(result.answer[:50])
            results.append(result)

        return results
