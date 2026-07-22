"""OCR 引擎封装 - 支持 Tesseract 和 API 两种模式。"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from PIL import Image

from app.models import BoundingBox, OCRResult, OCRWord
from ocr.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


class BaseOREngine(ABC):
    """OCR 引擎抽象基类。"""

    @abstractmethod
    def recognize(
        self,
        image: Image.Image,
        languages: Optional[List[str]] = None,
    ) -> OCRResult:
        """识别图片中的文字。

        Args:
            image: PIL Image 对象。
            languages: 语言列表。

        Returns:
            OCRResult: 识别结果。
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """检查引擎是否可用。"""
        ...


class TesseractEngine(BaseOREngine):
    """Tesseract OCR 引擎封装。"""

    def __init__(self, preprocessor: Optional[ImagePreprocessor] = None):
        """初始化 Tesseract 引擎。

        Args:
            preprocessor: 图片预处理器 (可选)。
        """
        self.preprocessor = preprocessor or ImagePreprocessor()
        self._available = self._check_tesseract()

    @staticmethod
    def _check_tesseract() -> bool:
        try:
            import pytesseract  # noqa: F401
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._available

    def recognize(
        self,
        image: Image.Image,
        languages: Optional[List[str]] = None,
    ) -> OCRResult:
        """使用 Tesseract 识别图片文字。

        Args:
            image: PIL Image 对象。
            languages: 语言列表，如 ['chi_sim', 'eng']。

        Returns:
            OCRResult: 识别结果。
        """
        if not self._available:
            raise ImportError(
                "pytesseract 未安装。请运行 pip install pytesseract，"
                "并确保系统已安装 tesseract-ocr。"
            )

        import pytesseract

        # 预处理
        processed = self.preprocessor.process(image)

        # 构建语言参数
        lang = "+".join(languages) if languages else "eng"

        try:
            # 获取详细数据
            data = pytesseract.image_to_data(
                processed,
                lang=lang,
                output_type=pytesseract.Output.DICT,
            )

            words = []
            confidences = []

            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                if not text:
                    continue
                conf = int(data["conf"][i])
                if conf < 0:
                    conf = 0
                confidence = conf / 100.0

                words.append(
                    OCRWord(
                        text=text,
                        confidence=confidence,
                        bbox=BoundingBox(
                            x=data["left"][i],
                            y=data["top"][i],
                            width=data["width"][i],
                            height=data["height"][i],
                        ),
                    )
                )
                confidences.append(confidence)

            # 获取完整文本
            full_text = pytesseract.image_to_string(processed, lang=lang)
            full_text = full_text.strip()

            avg_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

            return OCRResult(
                text=full_text,
                words=words,
                confidence=avg_confidence,
                engine="tesseract",
            )

        except Exception as e:
            logger.error(f"Tesseract OCR 失败: {e}")
            return OCRResult(
                text="",
                words=[],
                confidence=0.0,
                engine="tesseract",
            )


class APIEngine(BaseOREngine):
    """基于 API 的 OCR 引擎 (抽象，用于对接多模态 LLM API)。

    实际调用使用 mock，可通过子类化实现真实 API 调用。
    """

    def __init__(
        self,
        api_url: str = "",
        api_key: str = "",
        model: str = "glm-4v",
        preprocessor: Optional[ImagePreprocessor] = None,
    ):
        """初始化 API OCR 引擎。

        Args:
            api_url: API 地址。
            api_key: API 密钥。
            model: 模型名称。
            preprocessor: 图片预处理器。
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.preprocessor = preprocessor

    def is_available(self) -> bool:
        return bool(self.api_url and self.api_key)

    def recognize(
        self,
        image: Image.Image,
        languages: Optional[List[str]] = None,
    ) -> OCRResult:
        """使用 Vision LLM API 识别图片文字。

        Args:
            image: PIL Image 对象。
            languages: 语言列表 (对 API 模式可选)。

        Returns:
            OCRResult: 识别结果。
        """
        # 实际项目中这里会调用 Vision LLM API
        # 这里返回空结果，测试中通过 mock 覆盖
        return OCRResult(
            text="",
            words=[],
            confidence=0.0,
            engine=f"api-{self.model}",
        )


class OCREngine:
    """OCR 引擎统一封装，根据配置选择实际引擎。"""

    def __init__(
        self,
        engine_type: str = "tesseract",
        preprocessor: Optional[ImagePreprocessor] = None,
        **kwargs,
    ):
        """初始化 OCR 引擎。

        Args:
            engine_type: 引擎类型 (tesseract / api)。
            preprocessor: 图片预处理器。
            **kwargs: 传递给具体引擎的参数。
        """
        self.preprocessor = preprocessor or ImagePreprocessor(
            steps=["grayscale", "denoise", "binarize"]
        )

        if engine_type == "tesseract":
            self._engine = TesseractEngine(self.preprocessor)
        elif engine_type == "api":
            self._engine = APIEngine(preprocessor=self.preprocessor, **kwargs)
        else:
            raise ValueError(f"不支持的 OCR 引擎类型: {engine_type}")

        self.engine_type = engine_type

    def recognize(
        self,
        image: Image.Image,
        languages: Optional[List[str]] = None,
    ) -> OCRResult:
        """识别图片文字。

        Args:
            image: PIL Image 对象。
            languages: 语言列表。

        Returns:
            OCRResult: 识别结果。
        """
        return self._engine.recognize(image, languages)

    def is_available(self) -> bool:
        """检查引擎是否可用。"""
        return self._engine.is_available()