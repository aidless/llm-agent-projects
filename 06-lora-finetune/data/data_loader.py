"""
数据加载模块
支持从 JSONL、JSON、CSV 文件中加载训练数据
"""
import json
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    从 JSONL 文件中加载数据，每行一个 JSON 对象

    Args:
        file_path: JSONL 文件路径

    Returns:
        包含所有 JSON 对象的列表

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式错误
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"第 {line_num} 行解析失败: {e}")
                raise ValueError(f"JSONL 文件第 {line_num} 行格式错误: {e}")

    logger.info(f"从 {file_path} 加载了 {len(data)} 条数据")
    return data


def load_json(file_path: str, data_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    从 JSON 文件中加载数据

    Args:
        file_path: JSON 文件路径
        data_key: 如果 JSON 是嵌套结构，指定数据所在的 key

    Returns:
        包含所有数据条目的列表

    Raises:
        FileNotFoundError: 文件不存在
        KeyError: 指定的 data_key 不存在
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 如果指定了数据 key，提取对应的列表
    if data_key is not None:
        if data_key not in raw:
            raise KeyError(f"JSON 文件中不存在 key: '{data_key}'")
        data = raw[data_key]
    elif isinstance(raw, list):
        data = raw
    else:
        # 如果是字典但没有指定 key，尝试找到第一个列表类型的值
        for v in raw.values():
            if isinstance(v, list):
                data = v
                break
        else:
            raise ValueError("无法从 JSON 文件中识别数据列表，请指定 data_key")

    logger.info(f"从 {file_path} 加载了 {len(data)} 条数据")
    return data


def load_csv(
    file_path: str,
    instruction_col: str = "instruction",
    output_col: str = "output",
    input_col: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    从 CSV 文件中加载数据

    Args:
        file_path: CSV 文件路径
        instruction_col: 指令列名
        output_col: 输出列名
        input_col: 可选的输入列名（上下文）

    Returns:
        包含所有数据条目的列表
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "instruction": row.get(instruction_col, ""),
                "output": row.get(output_col, ""),
            }
            if input_col and input_col in row:
                item["input"] = row[input_col]
            data.append(item)

    logger.info(f"从 {file_path} 加载了 {len(data)} 条数据")
    return data
