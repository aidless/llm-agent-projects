"""版面分析器 - 文档区域检测与阅读顺序排序。"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from app.models import BoundingBox, DocumentRegion, LayoutAnalysisResult, RegionType

logger = logging.getLogger(__name__)


class LayoutAnalyzer:
    """文档版面分析器。

    基于启发式规则的版面分析:
    - 区域检测 (标题/段落/表格/图片/页眉页脚)
    - 阅读顺序排序
    - 文档结构树生成
    """

    # 最小区域面积占比
    MIN_REGION_RATIO = 0.01

    def __init__(
        self,
        ocr_engine=None,
        min_region_area_ratio: float = 0.01,
    ):
        """初始化版面分析器。

        Args:
            ocr_engine: OCR 引擎 (用于文字区域识别)。
            min_region_area_ratio: 最小区域面积占图片面积比例。
        """
        self.ocr_engine = ocr_engine
        self.min_region_area_ratio = min_region_area_ratio

    def analyze(self, image: Image.Image) -> LayoutAnalysisResult:
        """分析文档版面。

        Args:
            image: PIL Image 对象。

        Returns:
            LayoutAnalysisResult: 版面分析结果。
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size
        regions = self._detect_regions(image)

        # 过滤过小区域
        img_area = width * height
        regions = [
            r for r in regions
            if (r.bbox.width * r.bbox.height) / img_area >= self.min_region_area_ratio
        ]

        # 计算阅读顺序
        reading_order = self._sort_reading_order(regions)

        return LayoutAnalysisResult(
            regions=regions,
            page_count=1,
            reading_order=reading_order,
        )

    def _detect_regions(self, image: Image.Image) -> List[DocumentRegion]:
        """检测文档中的区域。

        使用基于连通域的启发式方法:
        1. 灰度化 -> 二值化
        2. 查找文本块 (连通的黑色像素区域)
        3. 根据位置和大小分类区域类型
        """
        import numpy as np

        width, height = image.size

        # 转灰度
        gray = image.convert("L")

        # 简单二值化
        arr = np.array(gray)
        threshold = 200
        binary = (arr < threshold).astype(np.uint8)

        # 投影法分割 - 水平投影找行
        h_projection = np.sum(binary, axis=1)
        row_regions = self._find_text_rows(h_projection, height)

        # 在每行内用垂直投影找列
        regions = []
        for y_start, y_end in row_regions:
            row_data = binary[y_start:y_end, :]
            v_projection = np.sum(row_data, axis=0)
            col_regions = self._find_text_cols(v_projection, width)

            for x_start, x_end in col_regions:
                bbox = BoundingBox(
                    x=x_start,
                    y=y_start,
                    width=max(x_end - x_start, 1),
                    height=max(y_end - y_start, 1),
                )
                region_type = self._classify_region(bbox, width, height)

                # 提取区域文本
                content = ""
                if self.ocr_engine:
                    try:
                        region_img = image.crop(
                            (x_start, y_start, x_end, y_end)
                        )
                        ocr_result = self.ocr_engine.recognize(region_img)
                        content = ocr_result.text
                    except Exception:
                        pass

                regions.append(
                    DocumentRegion(
                        region_type=region_type,
                        bbox=bbox,
                        content=content,
                        confidence=0.8,
                        page=1,
                    )
                )

        return regions

    @staticmethod
    def _find_text_rows(
        projection: np.ndarray, height: int, min_gap: int = 5
    ) -> List[Tuple[int, int]]:
        """从水平投影中查找文本行。"""
        threshold = max(np.mean(projection) * 0.3, 5)
        in_text = False
        start = 0
        regions = []

        for y in range(height):
            if projection[y] > threshold and not in_text:
                in_text = True
                start = y
            elif projection[y] <= threshold and in_text:
                in_text = False
                if y - start >= 3:
                    regions.append((start, y))

        if in_text:
            regions.append((start, height))

        # 合并间隔过近的行
        merged = []
        for region in regions:
            if merged and region[0] - merged[-1][1] < min_gap:
                merged[-1] = (merged[-1][0], region[1])
            else:
                merged.append(region)

        return merged

    @staticmethod
    def _find_text_cols(
        projection: np.ndarray, width: int, min_gap: int = 10
    ) -> List[Tuple[int, int]]:
        """从垂直投影中查找文本列。"""
        threshold = max(np.mean(projection) * 0.3, 5)
        in_text = False
        start = 0
        regions = []

        for x in range(width):
            if projection[x] > threshold and not in_text:
                in_text = True
                start = x
            elif projection[x] <= threshold and in_text:
                in_text = False
                if x - start >= 3:
                    regions.append((start, x))

        if in_text:
            regions.append((start, width))

        # 合并间隔过近的列
        merged = []
        for region in regions:
            if merged and region[0] - merged[-1][1] < min_gap:
                merged[-1] = (merged[-1][0], region[1])
            else:
                merged.append(region)

        return merged

    @staticmethod
    def _classify_region(
        bbox: BoundingBox, page_width: int, page_height: int
    ) -> RegionType:
        """根据位置和大小分类区域类型。

        规则:
        - 页面顶部区域 -> 页眉 (header)
        - 页面底部区域 -> 页脚 (footer)
        - 占据大部分宽度 -> 标题 (title) 或段落
        - 小而宽 -> 标题
        - 大面积 -> 段落
        """
        y_ratio = bbox.y / page_height if page_height > 0 else 0
        width_ratio = bbox.width / page_width if page_width > 0 else 0
        height_ratio = bbox.height / page_height if page_height > 0 else 0

        # 页眉 (顶部 10%)
        if y_ratio < 0.10 and height_ratio < 0.05:
            return RegionType.HEADER

        # 页脚 (底部 10%)
        if y_ratio > 0.90 and height_ratio < 0.05:
            return RegionType.FOOTER

        # 标题: 较短但占满宽度
        if height_ratio < 0.03 and width_ratio > 0.5:
            return RegionType.TITLE

        # 默认为段落
        return RegionType.PARAGRAPH

    @staticmethod
    def _sort_reading_order(regions: List[DocumentRegion]) -> List[int]:
        """对区域按阅读顺序排序 (从上到下，从左到右)。

        Returns:
            List[int]: 排序后的原始索引列表。
        """
        indexed = list(enumerate(regions))
        indexed.sort(key=lambda x: (x[1].bbox.y, x[1].bbox.x))
        return [i for i, _ in indexed]

    def generate_structure_tree(
        self, layout_result: LayoutAnalysisResult
    ) -> dict:
        """根据版面分析结果生成文档结构树。

        Args:
            layout_result: 版面分析结果。

        Returns:
            dict: 结构树。
        """
        if not layout_result.reading_order:
            return {"type": "document", "children": []}

        ordered_regions = [
            layout_result.regions[i] for i in layout_result.reading_order
        ]

        children = []
        current_section = None

        for region in ordered_regions:
            node = {
                "type": region.region_type.value,
                "content": region.content[:200] if region.content else "",
                "bbox": {
                    "x": region.bbox.x,
                    "y": region.bbox.y,
                    "width": region.bbox.width,
                    "height": region.bbox.height,
                },
            }

            if region.region_type == RegionType.TITLE:
                if current_section:
                    children.append(current_section)
                current_section = {
                    "type": "section",
                    "title": region.content,
                    "children": [node],
                }
            else:
                if current_section:
                    current_section["children"].append(node)
                else:
                    children.append(node)

        if current_section:
            children.append(current_section)

        return {
            "type": "document",
            "children": children,
            "region_count": len(ordered_regions),
        }

    def extract_key_info(self, text: str) -> dict:
        """从文本中提取关键信息 (简单实现)。

        Args:
            text: 文档文本。

        Returns:
            dict: 关键信息 (摘要、关键词)。
        """
        import re

        # 提取关键词 (简单词频统计)
        words = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]{2,}", text)
        word_freq = {}
        for w in words:
            w_lower = w.lower()
            word_freq[w_lower] = word_freq.get(w_lower, 0) + 1

        # 取高频词 (过滤常见停用词)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "but", "with", "by", "from",
            "的", "了", "是", "在", "和", "与", "或", "不", "有", "也",
        }
        keywords = [
            (w, c)
            for w, c in sorted(word_freq.items(), key=lambda x: -x[1])
            if w not in stop_words and c >= 2
        ][:10]

        # 生成简单摘要 (取前几句话)
        sentences = re.split(r"[。！？.!?]", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        summary = "".join(sentences[:3]) + ("..." if len(sentences) > 3 else "")

        return {
            "summary": summary,
            "keywords": [w for w, _ in keywords],
            "word_count": len(words),
            "char_count": len(text),
        }