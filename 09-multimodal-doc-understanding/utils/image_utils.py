"""图片工具模块 - 提供图片加载、格式转换、base64 编码等工具函数。"""

import base64
import io
from pathlib import Path
from typing import Optional, Tuple, Union

from PIL import Image


def load_image(
    source: Union[str, bytes, Path, Image.Image],
) -> Image.Image:
    """从文件路径、字节流或已有 PIL Image 加载图片。

    Args:
        source: 文件路径、字节流或 PIL Image 对象。

    Returns:
        PIL.Image.Image: 加载后的图片对象。

    Raises:
        ValueError: 无法识别的输入类型。
        FileNotFoundError: 文件不存在。
    """
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {path}")
        return Image.open(path)
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source))
    raise ValueError(f"不支持的图片源类型: {type(source)}")


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """将 PIL Image 转换为 base64 编码字符串。

    Args:
        image: PIL Image 对象。
        format: 图片格式，默认 PNG。

    Returns:
        str: base64 编码字符串。
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(b64_str: str) -> Image.Image:
    """将 base64 编码字符串转换为 PIL Image。

    Args:
        b64_str: base64 编码字符串。

    Returns:
        PIL.Image.Image: 解码后的图片对象。
    """
    image_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(image_bytes))


def get_image_info(image: Image.Image) -> dict:
    """获取图片基本信息。

    Args:
        image: PIL Image 对象。

    Returns:
        dict: 包含宽度、高度、模式、格式等信息的字典。
    """
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format or "unknown",
        "size_bytes": _estimate_image_size(image),
    }


def _estimate_image_size(image: Image.Image) -> int:
    """估算图片字节数。"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.tell()


def resize_image(
    image: Image.Image,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
) -> Image.Image:
    """等比例缩放图片。

    Args:
        image: PIL Image 对象。
        max_width: 最大宽度。
        max_height: 最大高度。

    Returns:
        PIL.Image.Image: 缩放后的图片。
    """
    if max_width is None and max_height is None:
        return image

    w, h = image.size
    ratio = 1.0

    if max_width and w > max_width:
        ratio = min(ratio, max_width / w)
    if max_height and h > max_height:
        ratio = min(ratio, max_height / h)

    if ratio < 1.0:
        new_size = (int(w * ratio), int(h * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def create_test_image(
    width: int = 800,
    height: int = 600,
    color: Tuple[int, int, int] = (255, 255, 255),
    text: str = "Test Document",
    mode: str = "RGB",
) -> Image.Image:
    """创建测试用图片。

    Args:
        width: 图片宽度。
        height: 图片高度。
        color: 背景色。
        text: 绘制文本。
        mode: 图片模式。

    Returns:
        PIL.Image.Image: 测试图片。
    """
    img = Image.new(mode, (width, height), color)
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.text((width // 10, height // 10), text, fill=(0, 0, 0))
    except Exception:
        pass
    return img


def save_temp_image(image: Image.Image, directory: str = "/tmp", prefix: str = "doc") -> str:
    """将图片保存为临时文件。

    Args:
        image: PIL Image 对象。
        directory: 临时目录。
        prefix: 文件名前缀。

    Returns:
        str: 临时文件路径。
    """
    from pathlib import Path
    import tempfile

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{prefix}_{id(image)}.png"
    image.save(str(path), "PNG")
    return str(path)