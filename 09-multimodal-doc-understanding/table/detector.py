"""表格检测器 - 检测图片/PDF 中的表格区域。"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from app.models import BoundingBox

logger = logging.getLogger(__name__)


class TableDetector:
    """表格检测器。

    基于线段检测和网格分析的表格区域检测:
    - 水平/垂直线段检测
    - 网格结构识别
    - 表格区域定位
    """

    def __init__(
        self,
        min_lines: int = 3,
        min_area_ratio: float = 0.05,
    ):
        """初始化表格检测器。

        Args:
            min_lines: 最少线段数量 (水平或垂直)。
            min_area_ratio: 最小面积占比。
        """
        self.min_lines = min_lines
        self.min_area_ratio = min_area_ratio

    def detect(self, image: Image.Image) -> List[BoundingBox]:
        """检测图片中的表格区域。

        Args:
            image: PIL Image 对象。

        Returns:
            List[BoundingBox]: 检测到的表格区域列表。
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size
        gray = image.convert("L")
        arr = np.array(gray)

        # 边缘检测 (简单差分)
        binary = (arr < 180).astype(np.uint8)

        # 水平投影
        h_proj = np.sum(binary, axis=1)
        h_lines = self._find_lines(h_proj, min_length=width * 0.3)

        # 垂直投影
        v_proj = np.sum(binary, axis=0)
        v_lines = self._find_lines(v_proj, min_length=height * 0.2)

        # 如果检测到足够的横线和竖线，认为存在表格
        if len(h_lines) >= self.min_lines and len(v_lines) >= self.min_lines:
            # 合并横线和竖线确定表格区域
            y_min = h_lines[0][0]
            y_max = h_lines[-1][1]
            x_min = v_lines[0][0]
            x_max = v_lines[-1][1]

            area = (x_max - x_min) * (y_max - y_min)
            img_area = width * height

            if area / img_area >= self.min_area_ratio:
                return [
                    BoundingBox(
                        x=x_min,
                        y=y_min,
                        width=x_max - x_min,
                        height=y_max - y_min,
                    )
                ]

        return []

    @staticmethod
    def _find_lines(
        projection: np.ndarray,
        min_length: int = 50,
        gap_threshold: int = 3,
    ) -> List[Tuple[int, int]]:
        """从投影中查找线段位置。

        线段特征: 投影值连续大于阈值的区域。

        Args:
            projection: 投影数组。
            min_length: 最小线段长度。
            gap_threshold: 间隔阈值。

        Returns:
            List[Tuple[int, int]]: 线段起止位置列表。
        """
        threshold = max(np.mean(projection) * 0.5, 10)
        length = len(projection)
        in_line = False
        start = 0
        lines = []

        for i in range(length):
            if projection[i] > threshold and not in_line:
                in_line = True
                start = i
            elif projection[i] <= threshold and in_line:
                in_line = False
                if i - start >= min_length:
                    lines.append((start, i))

        if in_line and length - start >= min_length:
            lines.append((start, length))

        return lines

    def detect_multiple(self, image: Image.Image) -> List[BoundingBox]:
        """检测多个表格区域。

        将图片分块后分别检测。

        Args:
            image: PIL Image 对象。

        Returns:
            List[BoundingBox]: 表格区域列表。
        """
        width, height = image.size

        # 水平分块
        block_height = height // 2
        table_regions = []

        for block_idx in range(2):
            y_start = block_idx * block_height
            y_end = min((block_idx + 1) * block_height + 50, height)
            block = image.crop((0, y_start, width, y_end))

            detections = self.detect(block)
            for det in detections:
                # 调整坐标到原图
                table_regions.append(
                    BoundingBox(
                        x=det.x,
                        y=det.y + y_start,
                        width=det.width,
                        height=min(det.height, height - y_start),
                    )
                )

        return table_regions