"""OCR 图片预处理 pipeline。"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """图片预处理 pipeline。

    支持的预处理步骤:
    - 灰度化 (grayscale)
    - 二值化 (binarize)
    - 去噪 (denoise)
    - 对比度增强 (enhance_contrast)
    - 锐化 (sharpen)
    - 自适应缩放 (resize_for_ocr)
    """

    def __init__(self, steps: Optional[List[str]] = None):
        """初始化预处理器。

        Args:
            steps: 预处理步骤列表。默认: grayscale -> denoise -> binarize。
        """
        self.steps = steps or [
            "grayscale",
            "denoise",
            "binarize",
        ]
        self._validate_steps()

    def _validate_steps(self):
        valid = {
            "grayscale",
            "binarize",
            "denoise",
            "enhance_contrast",
            "sharpen",
            "resize_for_ocr",
        }
        for step in self.steps:
            if step not in valid:
                raise ValueError(f"未知的预处理步骤: {step}，可选: {valid}")

    def process(self, image: Image.Image) -> Image.Image:
        """按顺序执行预处理步骤。

        Args:
            image: 输入 PIL Image。

        Returns:
            PIL.Image.Image: 预处理后的图片。
        """
        result = image.copy()

        for step in self.steps:
            try:
                result = self._apply_step(result, step)
            except Exception as e:
                logger.warning(f"预处理步骤 '{step}' 失败: {e}")

        return result

    def _apply_step(self, image: Image.Image, step: str) -> Image.Image:
        """应用单个预处理步骤。"""
        if step == "grayscale":
            return self.to_grayscale(image)
        elif step == "binarize":
            return self.binarize(image)
        elif step == "denoise":
            return self.denoise(image)
        elif step == "enhance_contrast":
            return self.enhance_contrast(image)
        elif step == "sharpen":
            return self.sharpen(image)
        elif step == "resize_for_ocr":
            return self.resize_for_ocr(image)
        return image

    @staticmethod
    def to_grayscale(image: Image.Image) -> Image.Image:
        """灰度化。"""
        if image.mode != "L":
            return image.convert("L")
        return image

    @staticmethod
    def binarize(
        image: Image.Image,
        threshold: int = 128,
        method: str = "simple",
    ) -> Image.Image:
        """二值化。

        Args:
            image: 输入图片 (建议灰度图)。
            threshold: 二值化阈值 (simple 方法)。
            method: 二值化方法 (simple / otsu)。

        Returns:
            PIL.Image.Image: 二值化后的图片。
        """
        if image.mode != "L":
            image = image.convert("L")

        if method == "otsu":
            threshold = ImagePreprocessor._otsu_threshold(image)
        elif method != "simple":
            raise ValueError(f"未知的二值化方法: {method}")

        return image.point(lambda x: 255 if x > threshold else 0, mode="1")

    @staticmethod
    def _otsu_threshold(image: Image.Image) -> int:
        """计算 Otsu 最佳阈值。"""
        import numpy as np

        arr = np.array(image).flatten().astype(np.float32)
        hist, _ = np.histogram(arr, bins=256, range=(0, 256))
        total = arr.size

        sum_total = np.sum(np.arange(256) * hist)
        sum_bg = 0.0
        weight_bg = 0
        max_variance = 0.0
        best_threshold = 128

        for t in range(256):
            weight_bg += hist[t]
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            sum_bg += t * hist[t]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg
            variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if variance > max_variance:
                max_variance = variance
                best_threshold = t

        return best_threshold

    @staticmethod
    def denoise(image: Image.Image, radius: int = 1) -> Image.Image:
        """中值滤波去噪。"""
        return image.filter(ImageFilter.MedianFilter(size=radius * 2 + 1))

    @staticmethod
    def enhance_contrast(image: Image.Image, factor: float = 2.0) -> Image.Image:
        """对比度增强。"""
        from PIL import ImageEnhance

        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def sharpen(image: Image.Image, factor: float = 2.0) -> Image.Image:
        """锐化。"""
        from PIL import ImageEnhance

        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def resize_for_ocr(
        image: Image.Image,
        min_dimension: int = 1000,
    ) -> Image.Image:
        """为 OCR 调整图片大小，确保最小边不小于阈值。"""
        w, h = image.size
        if min(w, h) >= min_dimension:
            return image

        ratio = min_dimension / min(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)