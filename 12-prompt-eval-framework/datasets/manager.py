"""数据集管理器，支持 JSON/JSONL/CSV 导入。"""

import csv
import json
import os
import random
from typing import Any, Dict, List, Optional


class DatasetManager:
    """数据集管理器。"""

    def __init__(self):
        self.datasets: Dict[str, Dict[str, Any]] = {}

    def create_dataset(
        self,
        name: str,
        input_key: str = "input",
        output_key: str = "output",
        data: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """创建新数据集。"""
        dataset = {
            "name": name,
            "input_key": input_key,
            "output_key": output_key,
            "data": data or [],
            "size": len(data) if data else 0,
        }
        self.datasets[name] = dataset
        return self._dataset_summary(dataset)

    def import_json(self, file_path: str, name: Optional[str] = None) -> Dict[str, Any]:
        """从 JSON 文件导入数据集。

        支持两种格式：
        1. 数组格式: [{"input": "...", "output": "..."}, ...]
        2. 对象格式: {"data": [...], "input_key": "input", "output_key": "output"}
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict) and "data" in raw:
            data = raw["data"]
            input_key = raw.get("input_key", "input")
            output_key = raw.get("output_key", "output")
        elif isinstance(raw, list):
            data = raw
            # 自动检测 key
            if data:
                keys = list(data[0].keys())
                input_key = "input" if "input" in keys else keys[0] if len(keys) > 0 else "input"
                output_key = "output" if "output" in keys else keys[1] if len(keys) > 1 else "output"
            else:
                input_key = "input"
                output_key = "output"
        else:
            raise ValueError("JSON 格式不支持，需要数组或包含 'data' 键的对象")

        dataset_name = name or os.path.splitext(os.path.basename(file_path))[0]
        return self.create_dataset(dataset_name, input_key, output_key, data)

    def import_jsonl(self, file_path: str, name: Optional[str] = None) -> Dict[str, Any]:
        """从 JSONL 文件导入数据集。"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        if not data:
            raise ValueError("JSONL 文件为空")

        keys = list(data[0].keys())
        input_key = "input" if "input" in keys else keys[0] if keys else "input"
        output_key = "output" if "output" in keys else keys[1] if len(keys) > 1 else "output"

        dataset_name = name or os.path.splitext(os.path.basename(file_path))[0]
        return self.create_dataset(dataset_name, input_key, output_key, data)

    def import_csv(self, file_path: str, name: Optional[str] = None, delimiter: str = ",") -> Dict[str, Any]:
        """从 CSV 文件导入数据集。"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                data.append(dict(row))

        if not data:
            raise ValueError("CSV 文件为空")

        keys = list(data[0].keys())
        input_key = "input" if "input" in keys else keys[0] if keys else "input"
        output_key = "output" if "output" in keys else keys[1] if len(keys) > 1 else "output"

        dataset_name = name or os.path.splitext(os.path.basename(file_path))[0]
        return self.create_dataset(dataset_name, input_key, output_key, data)

    def get_dataset(self, name: str) -> Dict[str, Any]:
        """获取数据集。"""
        if name not in self.datasets:
            raise ValueError(f"数据集 {name} 不存在")
        return self.datasets[name]

    def list_datasets(self) -> List[Dict[str, Any]]:
        """列出所有数据集。"""
        return [self._dataset_summary(ds) for ds in self.datasets.values()]

    def delete_dataset(self, name: str):
        """删除数据集。"""
        if name not in self.datasets:
            raise ValueError(f"数据集 {name} 不存在")
        del self.datasets[name]

    def sample(
        self,
        name: str,
        n: int,
        strategy: str = "random",
        seed: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """从数据集中采样。

        Args:
            name: 数据集名称
            n: 采样数量
            strategy: 采样策略 (random/first/last)
            seed: 随机种子
        """
        dataset = self.get_dataset(name)
        data = dataset["data"]

        if n >= len(data):
            return data

        if strategy == "random":
            rng = random.Random(seed)
            return rng.sample(data, n)
        elif strategy == "first":
            return data[:n]
        elif strategy == "last":
            return data[-n:]
        else:
            raise ValueError(f"未知采样策略: {strategy}")

    def get_inputs(self, name: str) -> List[str]:
        """获取数据集的所有输入。"""
        dataset = self.get_dataset(name)
        key = dataset["input_key"]
        return [item.get(key, "") for item in dataset["data"]]

    def get_outputs(self, name: str) -> List[str]:
        """获取数据集的所有预期输出。"""
        dataset = self.get_dataset(name)
        key = dataset["output_key"]
        return [item.get(key, "") for item in dataset["data"]]

    def _dataset_summary(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """生成数据集摘要。"""
        return {
            "name": dataset["name"],
            "input_key": dataset["input_key"],
            "output_key": dataset["output_key"],
            "size": dataset["size"],
        }