# ============================================
# 文件读写工具
# 支持读取和写入本地文件，带有安全路径检查
# ============================================

import os
import aiofiles
from pathlib import Path
from typing import Optional
from loguru import logger

try:
    from langchain_core.tools import tool
except ImportError:
    from langchain.tools import tool

# 允许操作的根目录（安全限制，防止路径穿越攻击）
ALLOWED_ROOT = os.getenv("WORKSPACE_DIR", "/workspace")


def _safe_path(file_path: str) -> str:
    """检查并规范化文件路径，确保不超出允许的根目录"""
    # 解析为绝对路径
    abs_path = os.path.abspath(file_path)
    root = os.path.abspath(ALLOWED_ROOT)

    # 检查路径是否在允许范围内
    if not abs_path.startswith(root):
        raise ValueError(f"文件路径 '{file_path}' 超出允许的操作范围 '{root}'")

    return abs_path


@tool
async def file_read_tool(file_path: str, encoding: str = "utf-8") -> str:
    """
    文件读取工具：读取指定路径的文件内容。

    Args:
        file_path: 文件路径（绝对路径或相对路径）
        encoding: 文件编码（默认 utf-8）

    Returns:
        str: 文件内容文本
    """
    abs_path = _safe_path(file_path)
    logger.info(f"[文件工具] 读取文件: {abs_path}")

    if not os.path.exists(abs_path):
        return f"错误：文件 '{abs_path}' 不存在"

    try:
        async with aiofiles.open(abs_path, mode="r", encoding=encoding) as f:
            content = await f.read()
        # 截断过长的内容
        if len(content) > 50000:
            content = content[:50000] + f"\n\n[... 内容过长，已截断。原始长度: {len(content)} 字符 ...]"
        logger.debug(f"[文件工具] 读取成功，长度: {len(content)} 字符")
        return content
    except Exception as e:
        error_msg = f"读取文件失败: {str(e)}"
        logger.error(f"[文件工具] {error_msg}")
        return f"错误：{error_msg}"


@tool
async def file_write_tool(file_path: str, content: str, encoding: str = "utf-8", mode: str = "w") -> str:
    """
    文件写入工具：将内容写入指定路径的文件。

    Args:
        file_path: 文件路径（绝对路径或相对路径）
        content: 要写入的内容
        encoding: 文件编码（默认 utf-8）
        mode: 写入模式（"w" 覆盖 / "a" 追加）

    Returns:
        str: 操作结果信息
    """
    abs_path = _safe_path(file_path)
    logger.info(f"[文件工具] 写入文件: {abs_path}, 模式: {mode}")

    # 确保父目录存在
    parent_dir = os.path.dirname(abs_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    try:
        async with aiofiles.open(abs_path, mode=mode, encoding=encoding) as f:
            await f.write(content)
        logger.debug(f"[文件工具] 写入成功，长度: {len(content)} 字符")
        return f"成功：已将 {len(content)} 字符写入 '{abs_path}'"
    except Exception as e:
        error_msg = f"写入文件失败: {str(e)}"
        logger.error(f"[文件工具] {error_msg}")
        return f"错误：{error_msg}"
