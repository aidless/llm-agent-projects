"""表格提取器 - 从图片中提取表格数据。"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from app.models import TableCell, TableData, TableExtractionResult
from table.detector import TableDetector

logger = logging.getLogger(__name__)


class TableExtractor:
    """表格提取器。

    支持:
    - 表格区域检测
    - 行列结构识别
    - 单元格文本提取
    """

    def __init__(
        self,
        detector: Optional[TableDetector] = None,
        ocr_engine=None,
    ):
        """初始化表格提取器。

        Args:
            detector: 表格检测器。
            ocr_engine: OCR 引擎 (用于提取单元格文本)。
        """
        self.detector = detector or TableDetector()
        self.ocr_engine = ocr_engine

    def extract(self, image: Image.Image) -> TableExtractionResult:
        """从图片中提取表格。

        Args:
            image: PIL Image 对象。

        Returns:
            TableExtractionResult: 表格提取结果。
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 检测表格区域
        table_bboxes = self.detector.detect_multiple(image)

        tables: List[TableData] = []

        for bbox in table_bboxes:
            table_img = image.crop(
                (bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)
            )
            table_data = self._extract_table_structure(table_img)
            tables.append(table_data)

        return TableExtractionResult(tables=tables, page=1)

    def _extract_table_structure(self, table_image: Image.Image) -> TableData:
        """提取单个表格的结构。

        Args:
            table_image: 裁剪后的表格图片。

        Returns:
            TableData: 表格数据。
        """
        width, height = table_image.size
        gray = table_image.convert("L")
        arr = np.array(gray)
        binary = (arr < 180).astype(np.uint8)

        # 水平投影找行
        h_proj = np.sum(binary, axis=1)
        row_boundaries = self._find_cell_boundaries(h_proj, height)

        # 垂直投影找列
        v_proj = np.sum(binary, axis=0)
        col_boundaries = self._find_cell_boundaries(v_proj, width)

        num_rows = max(len(row_boundaries) - 1, 0)
        num_cols = max(len(col_boundaries) - 1, 0)

        cells: List[TableCell] = []
        headers: List[str] = []

        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                y1 = row_boundaries[row_idx]
                y2 = row_boundaries[row_idx + 1]
                x1 = col_boundaries[col_idx]
                x2 = col_boundaries[col_idx + 1]

                # 跳过过小的单元格
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    continue

                cell_img = table_image.crop((x1, y1, x2, y2))
                cell_text = self._extract_cell_text(cell_img)

                cell = TableCell(
                    row=row_idx,
                    col=col_idx,
                    text=cell_text,
                )
                cells.append(cell)

                # 第一行作为表头
                if row_idx == 0:
                    headers.append(cell_text)

        return TableData(
            rows=num_rows,
            cols=num_cols,
            cells=cells,
            headers=headers,
        )

    @staticmethod
    def _find_cell_boundaries(
        projection: np.ndarray, dimension: int
    ) -> List[int]:
        """从投影中找到单元格边界位置。

        Args:
            projection: 投影数组。
            dimension: 维度 (宽度或高度)。

        Returns:
            List[int]: 边界位置列表。
        """
        threshold = max(np.mean(projection) * 0.4, 5)

        boundaries = [0]
        in_cell = projection[0] > threshold

        for i in range(1, dimension):
            is_cell = projection[i] > threshold
            if is_cell != in_cell:
                boundaries.append(i)
                in_cell = is_cell

        boundaries.append(dimension)

        # 过滤太近的边界
        filtered = [boundaries[0]]
        for b in boundaries[1:]:
            if b - filtered[-1] >= 3:
                filtered.append(b)

        if filtered[-1] != dimension:
            filtered.append(dimension)

        return filtered

    def _extract_cell_text(self, cell_image: Image.Image) -> str:
        """提取单元格文本。

        Args:
            cell_image: 单元格图片。

        Returns:
            str: 单元格文本。
        """
        if self.ocr_engine:
            try:
                result = self.ocr_engine.recognize(cell_image)
                return result.text.strip()
            except Exception:
                pass

        # Fallback: 像素分析判断是否为空白
        import numpy as np

        gray = cell_image.convert("L")
        arr = np.array(gray)
        white_ratio = np.sum(arr > 200) / arr.size

        return "" if white_ratio > 0.95 else "[content]"