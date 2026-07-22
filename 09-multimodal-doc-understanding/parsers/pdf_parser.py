"""PDF 解析器 - 支持文本层提取、图片提取、OCR fallback。"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from app.models import (
    DocumentFormat,
    OCRResult,
    ParsedDocument,
    ParsedPage,
    TableData,
)
from utils.image_utils import image_to_base64

logger = logging.getLogger(__name__)


class PDFParser:
    """PDF 文档解析器。

    支持:
    - 文本层提取 (PyPDF2)
    - 页面转图片 (pdf2image, 可选)
    - 图片提取
    - OCR fallback (当文本层为空时)
    """

    def __init__(
        self,
        ocr_engine=None,
        table_extractor=None,
        enable_ocr_fallback: bool = True,
        dpi: int = 200,
    ):
        """初始化 PDF 解析器。

        Args:
            ocr_engine: OCR 引擎实例 (可选)。
            table_extractor: 表格提取器实例 (可选)。
            enable_ocr_fallback: 文本层为空时是否使用 OCR。
            dpi: 页面转图片的 DPI。
        """
        self.ocr_engine = ocr_engine
        self.table_extractor = table_extractor
        self.enable_ocr_fallback = enable_ocr_fallback
        self.dpi = dpi
        self._pypdf2_available = self._check_pypdf2()
        self._pdf2image_available = self._check_pdf2image()

    @staticmethod
    def _check_pypdf2() -> bool:
        try:
            import pypdf  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_pdf2image() -> bool:
        try:
            from pdf2image import convert_from_path  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """解析 PDF 文件。

        Args:
            file_path: PDF 文件路径。
            **kwargs: 额外参数。

        Returns:
            ParsedDocument: 解析结果。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {file_path}")

        filename = path.name
        pages: List[ParsedPage] = []
        all_text_parts: List[str] = []

        # 提取文本层
        text_by_page = self._extract_text(file_path)

        # 提取图片
        extracted_images = self._extract_images(file_path)

        # 尝试 pdf2image 转页面图片
        page_images = self._convert_pages_to_images(file_path)

        for page_num, text in enumerate(text_by_page, 1):
            page = ParsedPage(page_num=page_num, text=text or "")

            # OCR fallback
            if not text.strip() and self.enable_ocr_fallback and self.ocr_engine:
                if page_images and page_num <= len(page_images):
                    ocr_result = self.ocr_engine.recognize(page_images[page_num - 1])
                    page.text = ocr_result.text
                    page.ocr_result = ocr_result

            # 附加页面图片
            if page_images and page_num <= len(page_images):
                page_b64 = image_to_base64(page_images[page_num - 1])
                page.images.append(page_b64)

            # 附加提取的图片
            if page_num <= len(extracted_images):
                for img in extracted_images[page_num - 1]:
                    page.images.append(img)

            # 表格提取
            if self.table_extractor and page_images and page_num <= len(page_images):
                try:
                    table_result = self.table_extractor.extract(
                        page_images[page_num - 1]
                    )
                    page.tables = table_result.tables
                except Exception as e:
                    logger.warning(f"表格提取失败 (第{page_num}页): {e}")

            pages.append(page)
            all_text_parts.append(page.text)

        full_text = "\n\n".join(all_text_parts)

        metadata: Dict[str, Any] = {
            "page_count": len(pages),
            "file_size": path.stat().st_size if path.exists() else 0,
            "parser": "PDFParser",
        }

        return ParsedDocument(
            filename=filename,
            format=DocumentFormat.PDF,
            pages=pages,
            full_text=full_text,
            metadata=metadata,
        )

    def _extract_text(self, file_path: str) -> List[str]:
        """使用 PyPDF2 提取文本层。"""
        if not self._pypdf2_available:
            return []

        import pypdf

        texts = []
        try:
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                texts.append(page.extract_text() or "")
        except Exception as e:
            logger.error(f"PDF 文本提取失败: {e}")
            texts = []

        return texts

    def _convert_pages_to_images(self, file_path: str) -> List[Image.Image]:
        """将 PDF 页面转换为图片。"""
        if not self._pdf2image_available:
            return []

        from pdf2image import convert_from_path

        try:
            images = convert_from_path(file_path, dpi=self.dpi)
            return images
        except Exception as e:
            logger.warning(f"PDF 转图片失败: {e}")
            return []

    def _extract_images(self, file_path: str) -> List[List[str]]:
        """从 PDF 中提取嵌入图片。"""
        if not self._pypdf2_available:
            return []

        import pypdf

        page_images: List[List[str]] = []
        try:
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                images_on_page = []
                if "/XObject" in (page.get("/Resources") or {}):
                    x_objects = page["/Resources"]["/XObject"].get_object()
                    for obj_name in x_objects:
                        obj = x_objects[obj_name].get_object()
                        if obj.get("/Subtype") == "/Image":
                            try:
                                width = obj["/Width"]
                                height = obj["/Height"]
                                color_space = obj.get("/ColorSpace", "/DeviceRGB")
                                data = obj.get_data()

                                if color_space == "/DeviceRGB":
                                    mode = "RGB"
                                elif color_space == "/DeviceGray":
                                    mode = "L"
                                else:
                                    mode = "RGB"

                                img = Image.frombytes(mode, (width, height), data)
                                images_on_page.append(image_to_base64(img))
                            except Exception:
                                continue
                page_images.append(images_on_page)
        except Exception as e:
            logger.warning(f"PDF 图片提取失败: {e}")
            page_images = []

        return page_images