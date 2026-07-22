"""图片文档解析器 - 支持发票/证件/表格截图等图片文档。"""

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from app.models import (
    DocumentFormat,
    ParsedDocument,
    ParsedPage,
)
from utils.image_utils import get_image_info, image_to_base64, load_image

logger = logging.getLogger(__name__)


class ImageParser:
    """图片文档解析器。

    支持:
    - 常见图片格式 (PNG, JPG, BMP, TIFF, WebP)
    - OCR 文字识别
    - 表格检测
    - 版面分析
    """

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

    def __init__(self, ocr_engine=None, table_extractor=None, layout_analyzer=None):
        """初始化图片解析器。

        Args:
            ocr_engine: OCR 引擎实例。
            table_extractor: 表格提取器实例。
            layout_analyzer: 版面分析器实例。
        """
        self.ocr_engine = ocr_engine
        self.table_extractor = table_extractor
        self.layout_analyzer = layout_analyzer

    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """解析图片文件。

        Args:
            file_path: 图片文件路径。
            **kwargs: 额外参数。

        Returns:
            ParsedDocument: 解析结果。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的图片格式: {suffix}")

        image = load_image(file_path)
        return self.parse_image(image, filename=path.name, **kwargs)

    def parse_image(
        self,
        image: Image.Image,
        filename: str = "image.png",
        **kwargs,
    ) -> ParsedDocument:
        """直接解析 PIL Image 对象。

        Args:
            image: PIL Image 对象。
            filename: 文件名。
            **kwargs: 额外参数。

        Returns:
            ParsedDocument: 解析结果。
        """
        # 确保 RGB 模式
        if image.mode != "RGB":
            image = image.convert("RGB")

        page = ParsedPage(page_num=1, text="")
        page.images.append(image_to_base64(image))

        # OCR
        if self.ocr_engine and kwargs.get("ocr_enabled", True):
            try:
                ocr_result = self.ocr_engine.recognize(image)
                page.text = ocr_result.text
                page.ocr_result = ocr_result
            except Exception as e:
                logger.warning(f"OCR 识别失败: {e}")

        # 表格检测
        if self.table_extractor and kwargs.get("table_extraction", True):
            try:
                table_result = self.table_extractor.extract(image)
                page.tables = table_result.tables
            except Exception as e:
                logger.warning(f"表格提取失败: {e}")

        # 版面分析
        if self.layout_analyzer and kwargs.get("layout_analysis", False):
            try:
                layout = self.layout_analyzer.analyze(image)
                page.layout = layout
            except Exception as e:
                logger.warning(f"版面分析失败: {e}")

        info = get_image_info(image)

        return ParsedDocument(
            filename=filename,
            format=DocumentFormat.IMAGE,
            pages=[page],
            full_text=page.text,
            metadata={
                "width": info["width"],
                "height": info["height"],
                "mode": info["mode"],
                "size_bytes": info["size_bytes"],
                "parser": "ImageParser",
            },
        )

    def parse_multi_page(self, file_paths: List[str], **kwargs) -> ParsedDocument:
        """解析多张图片为多页文档。

        Args:
            file_paths: 图片文件路径列表。
            **kwargs: 额外参数。

        Returns:
            ParsedDocument: 合并后的文档结果。
        """
        all_pages = []
        all_text_parts = []

        for i, fp in enumerate(file_paths, 1):
            image = load_image(fp)
            if image.mode != "RGB":
                image = image.convert("RGB")

            page = ParsedPage(page_num=i, text="")
            page.images.append(image_to_base64(image))

            if self.ocr_engine and kwargs.get("ocr_enabled", True):
                try:
                    ocr_result = self.ocr_engine.recognize(image)
                    page.text = ocr_result.text
                    page.ocr_result = ocr_result
                except Exception as e:
                    logger.warning(f"第{i}页 OCR 失败: {e}")

            all_pages.append(page)
            all_text_parts.append(page.text)

        return ParsedDocument(
            filename=f"multi_page_{len(file_paths)}pages",
            format=DocumentFormat.IMAGE,
            pages=all_pages,
            full_text="\n\n".join(all_text_parts),
            metadata={"page_count": len(all_pages), "parser": "ImageParser"},
        )