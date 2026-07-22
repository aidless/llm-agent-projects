"""Vision 模块测试。"""

import pytest
from PIL import Image

from utils.image_utils import create_test_image
from vision.describer import ImageDescriber, MockVisionClient
from vision.chart_understanding import ChartUnderstandingEngine, MockChartClient
from vision.vqa import VQAEngine, MockVQAClient


class TestImageDescriber:
    """图片描述生成器测试。"""

    def test_describe_general(self):
        """测试通用描述。"""
        describer = ImageDescriber()
        img = create_test_image(200, 150)
        result = describer.describe(img, prompt_type="general")
        assert "mock" in result.description
        assert result.confidence > 0

    def test_describe_document(self):
        """测试文档描述。"""
        describer = ImageDescriber()
        img = create_test_image(800, 1000)
        result = describer.describe(img, prompt_type="document")
        assert isinstance(result.description, str)
        assert isinstance(result.labels, list)

    def test_describe_custom_prompt(self):
        """测试自定义提示。"""
        describer = ImageDescriber()
        img = create_test_image(100, 100)
        result = describer.describe(img, custom_prompt="描述这张图片的颜色")
        assert isinstance(result.description, str)

    def test_describe_multi_page(self):
        """测试多页描述。"""
        describer = ImageDescriber()
        images = [create_test_image(100, 100) for _ in range(3)]
        results = describer.describe_multi_page(images)
        assert len(results) == 3
        for r in results:
            assert isinstance(r.description, str)

    def test_default_prompts(self):
        """测试默认提示模板。"""
        assert "general" in ImageDescriber.DEFAULT_PROMPTS
        assert "document" in ImageDescriber.DEFAULT_PROMPTS
        assert "chart" in ImageDescriber.DEFAULT_PROMPTS
        assert "receipt" in ImageDescriber.DEFAULT_PROMPTS


class TestChartUnderstanding:
    """图表理解测试。"""

    def test_understand_chart(self):
        """测试图表理解。"""
        engine = ChartUnderstandingEngine()
        img = create_test_image(400, 300)
        result = engine.understand(img)
        assert result.chart_type == "bar"
        assert isinstance(result.description, str)
        assert isinstance(result.key_insights, list)

    def test_supported_types(self):
        """测试支持的图表类型。"""
        assert "bar" in ChartUnderstandingEngine.SUPPORTED_TYPES
        assert "line" in ChartUnderstandingEngine.SUPPORTED_TYPES
        assert "pie" in ChartUnderstandingEngine.SUPPORTED_TYPES

    def test_compare_charts(self):
        """测试图表对比。"""
        engine = ChartUnderstandingEngine()
        images = [create_test_image(300, 200) for _ in range(2)]
        result = engine.compare_charts(images)
        assert result["chart_count"] == 2
        assert len(result["results"]) == 2


class TestVQAEngine:
    """视觉问答测试。"""

    def test_single_qa(self):
        """测试单个问答。"""
        engine = VQAEngine()
        img = create_test_image(300, 200)
        result = engine.answer(img, "图片中有什么？")
        assert "mock" in result.answer
        assert result.question == "图片中有什么？"
        assert result.confidence > 0

    def test_batch_qa(self):
        """测试批量问答。"""
        engine = VQAEngine()
        images = [create_test_image(100, 100) for _ in range(3)]
        results = engine.batch_answer(images, "描述这张图片")
        assert len(results) == 3
        for r in results:
            assert isinstance(r.answer, str)

    def test_multi_turn_qa(self):
        """测试多轮问答。"""
        engine = VQAEngine()
        img = create_test_image(200, 200)
        questions = ["图片里有什么？", "主要颜色是什么？"]
        results = engine.multi_turn_qa(img, questions)
        assert len(results) == 2
        assert results[0].question == "图片里有什么？"
        assert results[1].question == "主要颜色是什么？"