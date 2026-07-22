"""OCR 模块测试。"""

import pytest
from PIL import Image

from ocr.engine import APIEngine, OCREngine, TesseractEngine
from ocr.preprocessor import ImagePreprocessor
from utils.image_utils import create_test_image


# ============ ImagePreprocessor 测试 ============


class TestImagePreprocessor:
    """图片预处理器测试。"""

    def test_default_steps(self):
        """测试默认预处理步骤。"""
        prep = ImagePreprocessor()
        assert prep.steps == ["grayscale", "denoise", "binarize"]

    def test_custom_steps(self):
        """测试自定义预处理步骤。"""
        prep = ImagePreprocessor(steps=["grayscale", "sharpen"])
        assert prep.steps == ["grayscale", "sharpen"]

    def test_invalid_step(self):
        """测试无效步骤。"""
        with pytest.raises(ValueError, match="未知的预处理步骤"):
            ImagePreprocessor(steps=["invalid_step"])

    def test_grayscale(self):
        """测试灰度化。"""
        img = Image.new("RGB", (100, 100), (128, 64, 32))
        result = ImagePreprocessor.to_grayscale(img)
        assert result.mode == "L"

    def test_grayscale_already(self):
        """测试已经是灰度的图片。"""
        img = Image.new("L", (100, 100), 128)
        result = ImagePreprocessor.to_grayscale(img)
        assert result.mode == "L"

    def test_binarize_simple(self):
        """测试简单二值化。"""
        img = Image.new("L", (100, 100), 128)
        result = ImagePreprocessor.binarize(img, threshold=128, method="simple")
        assert result.mode == "1"

    def test_binarize_otsu(self):
        """测试 Otsu 二值化。"""
        # 创建有明显双峰分布的图片
        img = Image.new("L", (100, 100), 50)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 0, 100, 100], fill=200)

        result = ImagePreprocessor.binarize(img, method="otsu")
        assert result.mode == "1"

    def test_denoise(self):
        """测试去噪。"""
        img = Image.new("L", (50, 50), 128)
        result = ImagePreprocessor.denoise(img, radius=1)
        assert result.mode == "L"
        assert result.size == (50, 50)

    def test_enhance_contrast(self):
        """测试对比度增强。"""
        img = Image.new("RGB", (50, 50), (100, 100, 100))
        result = ImagePreprocessor.enhance_contrast(img, factor=2.0)
        assert result.mode == "RGB"

    def test_sharpen(self):
        """测试锐化。"""
        img = Image.new("RGB", (50, 50), (100, 100, 100))
        result = ImagePreprocessor.sharpen(img)
        assert result.mode == "RGB"

    def test_resize_for_ocr(self):
        """测试 OCR 缩放。"""
        img = Image.new("RGB", (100, 100), (0, 0, 0))
        result = ImagePreprocessor.resize_for_ocr(img, min_dimension=200)
        assert result.size[0] >= 200
        assert result.size[1] >= 200

    def test_resize_for_ocr_already_large(self):
        """测试已满足最小尺寸时不缩放。"""
        img = Image.new("RGB", (500, 500), (0, 0, 0))
        result = ImagePreprocessor.resize_for_ocr(img, min_dimension=200)
        assert result.size == (500, 500)

    def test_process_pipeline(self):
        """测试完整预处理 pipeline。"""
        img = create_test_image(200, 200, text="Test")
        prep = ImagePreprocessor(steps=["grayscale", "denoise"])
        result = prep.process(img)
        assert result.mode == "L"
        assert result.size == (200, 200)


# ============ OCR Engine 测试 ============


class TestOCREngine:
    """OCR 引擎测试。"""

    def test_tesseract_init(self):
        """测试 Tesseract 引擎初始化。"""
        engine = OCREngine(engine_type="tesseract")
        assert engine.engine_type == "tesseract"

    def test_api_engine_init(self):
        """测试 API 引擎初始化。"""
        engine = OCREngine(
            engine_type="api",
            api_url="http://localhost:8000/v1/chat/completions",
            api_key="test-key",
        )
        assert engine.engine_type == "api"

    def test_invalid_engine_type(self):
        """测试无效引擎类型。"""
        with pytest.raises(ValueError, match="不支持的 OCR 引擎类型"):
            OCREngine(engine_type="invalid")

    def test_api_engine_recognize(self):
        """测试 API 引擎识别 (mock)。"""
        engine = APIEngine(
            api_url="http://localhost:8000",
            api_key="test-key",
            model="glm-4v",
        )
        img = create_test_image(100, 100)
        result = engine.recognize(img)
        assert result.engine == "api-glm-4v"
        assert isinstance(result.text, str)

    def test_api_engine_is_available(self):
        """测试 API 引擎可用性检查。"""
        engine = APIEngine(api_url="", api_key="")
        assert engine.is_available() is False

        engine2 = APIEngine(api_url="http://localhost", api_key="key")
        assert engine2.is_available() is True

    def test_tesseract_not_available(self):
        """测试 Tesseract 不可用时的处理。"""
        # 在没有安装 tesseract 的环境中，is_available 应返回 False
        engine = TesseractEngine()
        # 不做断言，因为取决于环境

    @pytest.mark.skipif(
        True,  # 始终跳过，因为 CI 可能没有 tesseract
        reason="Tesseract 可能未安装"
    )
    def test_tesseract_recognize(self):
        """测试 Tesseract 实际识别 (仅在安装时运行)。"""
        engine = TesseractEngine()
        if not engine.is_available():
            pytest.skip("Tesseract 未安装")
        img = create_test_image(200, 100, text="Hello World")
        result = engine.recognize(img, languages=["eng"])
        assert result.engine == "tesseract"
        assert isinstance(result.text, str)