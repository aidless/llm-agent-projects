"""图表理解模块 - 理解柱状图/折线图/饼图等图表。"""

import abc
import logging
from typing import Any, Dict, List, Optional

from PIL import Image

from app.models import ChartUnderstanding

logger = logging.getLogger(__name__)


class BaseChartClient(abc.ABC):
    """图表理解客户端抽象基类。"""

    @abc.abstractmethod
    def understand(self, image: Image.Image) -> ChartUnderstanding:
        ...


class MockChartClient(BaseChartClient):
    """Mock 图表理解客户端。"""

    def understand(self, image: Image.Image) -> ChartUnderstanding:
        return ChartUnderstanding(
            chart_type="bar",
            description="这是一个柱状图 (mock)，展示了各类别的数据对比。",
            data_summary={
                "categories": ["A", "B", "C"],
                "values": [100, 200, 150],
                "max": {"category": "B", "value": 200},
                "min": {"category": "A", "value": 100},
            },
            key_insights=[
                "B 类别数值最高 (200)",
                "A 类别数值最低 (100)",
                "三个类别的平均值为 150",
            ],
        )


class ChartUnderstandingEngine:
    """图表理解引擎。

    支持:
    - 柱状图 (bar)
    - 折线图 (line)
    - 饼图 (pie)
    - 散点图 (scatter)
    - 混合图表
    """

    SUPPORTED_TYPES = {"bar", "line", "pie", "scatter", "area", "mixed"}

    def __init__(self, client: Optional[BaseChartClient] = None):
        """初始化图表理解引擎。

        Args:
            client: 图表理解客户端。默认 MockChartClient。
        """
        self.client = client or MockChartClient()

    def understand(self, image: Image.Image) -> ChartUnderstanding:
        """理解图表内容。

        Args:
            image: PIL Image 对象。

        Returns:
            ChartUnderstanding: 理解结果。
        """
        return self.client.understand(image)

    def compare_charts(
        self, images: List[Image.Image]
    ) -> Dict[str, Any]:
        """对比多张图表。

        Args:
            images: 图片列表。

        Returns:
            dict: 对比结果。
        """
        results = []
        for i, img in enumerate(images):
            try:
                result = self.understand(img)
                results.append(result.model_dump())
            except Exception as e:
                logger.warning(f"第{i + 1}张图表理解失败: {e}")
                results.append({"error": str(e)})

        return {
            "chart_count": len(images),
            "results": results,
            "summary": f"共对比 {len(results)} 张图表 (mock 摘要)",
        }