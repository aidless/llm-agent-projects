"""
SFT 数据集构建模块
整合数据加载、格式化、分词的完整流水线，构建 HuggingFace Dataset 格式的 SFT 数据集
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class SFTDatasetBuilder:
    """
    SFT 数据集构建器

    完整的数据处理流水线：
    1. 从文件加载原始数据
    2. 格式化为指令-回答格式
    3. 构建用于 SFTTrainer 的数据集（或自定义 Dataset）

    Attributes:
        tokenizer_name_or_path: 分词器名称或路径
        max_seq_length: 最大序列长度
        template_name: 格式化模板名称
        val_split_ratio: 验证集划分比例
        seed: 随机种子
    """
    tokenizer_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    max_seq_length: int = 2048
    template_name: str = "alpaca"
    val_split_ratio: float = 0.05
    seed: int = 42

    def __post_init__(self):
        """初始化随机种子"""
        random.seed(self.seed)

    def build_from_file(
        self,
        data_path: str,
        data_format: str = "jsonl",
    ) -> Tuple[Any, Any]:
        """
        从文件构建 SFT 数据集（HuggingFace Dataset 格式）

        Args:
            data_path: 数据文件路径
            data_format: 数据格式，支持 "jsonl"、"json"、"csv"

        Returns:
            (train_dataset, val_dataset) 元组

        使用示例：
            from datasets import Dataset

            builder = SFTDatasetBuilder()
            train_ds, val_ds = builder.build_from_file("data/sample/train.jsonl")
        """
        # 第一步：加载数据
        data = self._load_data(data_path, data_format)
        logger.info(f"原始数据加载完成，共 {len(data)} 条")

        # 第二步：格式化
        formatted = self._format_data(data)
        logger.info(f"数据格式化完成，共 {len(formatted)} 条")

        # 第三步：划分训练集和验证集
        train_data, val_data = self._split_data(formatted)

        # 第四步：构建 HuggingFace Dataset
        train_dataset = self._build_hf_dataset(train_data)
        val_dataset = self._build_hf_dataset(val_data) if val_data else None

        logger.info(
            f"数据集构建完成: 训练集 {len(train_dataset)} 条, "
            f"验证集 {len(val_dataset) if val_dataset else 0} 条"
        )

        return train_dataset, val_dataset

    def build_instruction_dataset(
        self,
        data_path: str,
        data_format: str = "jsonl",
    ) -> Tuple[Any, Any]:
        """
        构建用于 SFTTrainer 的指令数据集

        与 build_from_file 不同，此方法返回包含 "text" 字段的 Dataset，
        适用于 HuggingFace TRL 的 SFTTrainer。

        Args:
            data_path: 数据文件路径
            data_format: 数据格式

        Returns:
            (train_dataset, val_dataset) 元组
        """
        # 加载数据
        data = self._load_data(data_path, data_format)

        # 使用格式化器获取格式化的文本
        from .formatter import InstructionFormatter
        formatter = InstructionFormatter(template_name=self.template_name)

        formatted_texts = []
        for item in data:
            try:
                text = formatter.format_single(item)
                formatted_texts.append({"text": text})
            except Exception as e:
                logger.warning(f"格式化失败: {e}")
                continue

        # 划分数据
        train_data, val_data = self._split_data(formatted_texts)

        # 构建 Dataset
        train_dataset = self._build_hf_dataset(train_data)
        val_dataset = self._build_hf_dataset(val_data) if val_data else None

        return train_dataset, val_dataset

    def _load_data(self, data_path: str, data_format: str) -> List[Dict[str, Any]]:
        """根据格式加载数据"""
        if data_format == "jsonl":
            from .data_loader import load_jsonl
            return load_jsonl(data_path)
        elif data_format == "json":
            from .data_loader import load_json
            return load_json(data_path)
        elif data_format == "csv":
            from .data_loader import load_csv
            return load_csv(data_path)
        else:
            raise ValueError(f"不支持的数据格式: {data_format}，支持: jsonl, json, csv")

    def _format_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化数据"""
        from .formatter import InstructionFormatter
        formatter = InstructionFormatter(template_name=self.template_name)

        formatted = []
        for item in data:
            try:
                text = formatter.format_single(item)
                prompt = formatter.get_prompt_only(item)
                output = item.get("output", "")

                formatted.append({
                    "text": text,
                    "prompt": prompt,
                    "output": output,
                    "instruction": item.get("instruction", ""),
                })
            except Exception as e:
                logger.warning(f"格式化失败: {e}")
                continue

        return formatted

    def _split_data(
        self, data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """划分训练集和验证集"""
        if not data:
            return [], []

        # 随机打乱
        indices = list(range(len(data)))
        random.shuffle(indices)

        val_size = max(1, int(len(data) * self.val_split_ratio))
        val_indices = set(indices[:val_size])
        train_indices = set(indices[val_size:])

        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]

        return train_data, val_data

    def _build_hf_dataset(self, data: List[Dict[str, Any]]) -> Any:
        """构建 HuggingFace Dataset"""
        from datasets import Dataset

        # 将所有字典的 key 统一
        if not data:
            return Dataset.from_dict({"text": [], "prompt": [], "output": [], "instruction": []})

        keys = data[0].keys()
        dataset_dict = {key: [item.get(key, "") for item in data] for key in keys}

        return Dataset.from_dict(dataset_dict)

    def save_processed_data(
        self,
        data: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        """
        保存处理后的数据到 JSONL 文件

        Args:
            data: 处理后的数据列表
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"处理后的数据已保存到: {output_path}")
