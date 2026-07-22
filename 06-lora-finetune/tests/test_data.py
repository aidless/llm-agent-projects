"""
数据处理模块测试
使用 mock 测试数据加载、格式化和数据集构建
"""
import json
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# 项目路径
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestDataLoader:
    """测试数据加载器"""

    def _create_jsonl_file(self, content):
        """创建临时 JSONL 文件"""
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        f.write(content)
        f.close()
        return f.name

    def _create_json_file(self, content):
        """创建临时 JSON 文件"""
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(content, f, ensure_ascii=False)
        f.close()
        return f.name

    def _create_csv_file(self, content):
        """创建临时 CSV 文件"""
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        f.write(content)
        f.close()
        return f.name

    def test_load_jsonl(self):
        """测试 JSONL 文件加载"""
        from data.data_loader import load_jsonl

        data_str = '\n'.join([
            json.dumps({"instruction": "测试问题1", "output": "测试答案1"}, ensure_ascii=False),
            json.dumps({"instruction": "测试问题2", "output": "测试答案2"}, ensure_ascii=False),
            json.dumps({"instruction": "测试问题3", "output": "测试答案3"}, ensure_ascii=False),
        ])

        filepath = self._create_jsonl_file(data_str)
        try:
            data = load_jsonl(filepath)
            assert len(data) == 3
            assert data[0]["instruction"] == "测试问题1"
            assert data[1]["output"] == "测试答案2"
        finally:
            os.unlink(filepath)

    def test_load_jsonl_empty_lines(self):
        """测试 JSONL 文件加载（含空行）"""
        from data.data_loader import load_jsonl

        data_str = '\n'.join([
            json.dumps({"instruction": "q1", "output": "a1"}),
            "",
            json.dumps({"instruction": "q2", "output": "a2"}),
            "  ",
            json.dumps({"instruction": "q3", "output": "a3"}),
        ])

        filepath = self._create_jsonl_file(data_str)
        try:
            data = load_jsonl(filepath)
            assert len(data) == 3
        finally:
            os.unlink(filepath)

    def test_load_jsonl_invalid_json(self):
        """测试 JSONL 文件加载（无效 JSON）"""
        from data.data_loader import load_jsonl

        filepath = self._create_jsonl_file('{"instruction": "valid"}\ninvalid json\n')
        try:
            with pytest.raises(ValueError):
                load_jsonl(filepath)
        finally:
            os.unlink(filepath)

    def test_load_jsonl_file_not_found(self):
        """测试文件不存在"""
        from data.data_loader import load_jsonl

        with pytest.raises(FileNotFoundError):
            load_jsonl("/nonexistent/path/data.jsonl")

    def test_load_json_list(self):
        """测试 JSON 文件加载（列表格式）"""
        from data.data_loader import load_json

        data = [
            {"instruction": "问题1", "output": "答案1"},
            {"instruction": "问题2", "output": "答案2"},
        ]

        filepath = self._create_json_file(data)
        try:
            loaded = load_json(filepath)
            assert len(loaded) == 2
            assert loaded[0]["instruction"] == "问题1"
        finally:
            os.unlink(filepath)

    def test_load_json_with_key(self):
        """测试 JSON 文件加载（嵌套格式，指定 key）"""
        from data.data_loader import load_json

        data = {
            "metadata": {"version": "1.0"},
            "examples": [
                {"instruction": "问题1", "output": "答案1"},
            ],
        }

        filepath = self._create_json_file(data)
        try:
            loaded = load_json(filepath, data_key="examples")
            assert len(loaded) == 1
            assert loaded[0]["instruction"] == "问题1"
        finally:
            os.unlink(filepath)

    def test_load_csv(self):
        """测试 CSV 文件加载"""
        from data.data_loader import load_csv

        csv_content = "instruction,output\n问题1,答案1\n问题2,答案2\n问题3,答案3\n"

        filepath = self._create_csv_file(csv_content)
        try:
            data = load_csv(filepath)
            assert len(data) == 3
            assert data[0]["instruction"] == "问题1"
            assert data[1]["output"] == "答案2"
        finally:
            os.unlink(filepath)


class TestFormatter:
    """测试数据格式化器"""

    def test_alpaca_format_with_input(self):
        """测试 Alpaca 格式（含 input）"""
        from data.formatter import InstructionFormatter

        formatter = InstructionFormatter(template_name="alpaca")
        item = {
            "instruction": "翻译以下句子",
            "input": "Hello, world!",
            "output": "你好，世界！",
        }

        result = formatter.format_single(item)
        assert "翻译以下句子" in result
        assert "Hello, world!" in result
        assert "你好，世界！" in result
        assert "### 指令:" in result
        assert "### 输入:" in result
        assert "### 回答:" in result

    def test_alpaca_format_no_input(self):
        """测试 Alpaca 格式（不含 input）"""
        from data.formatter import InstructionFormatter

        formatter = InstructionFormatter(template_name="alpaca")
        item = {
            "instruction": "什么是机器学习？",
            "output": "机器学习是AI的一个分支...",
        }

        result = formatter.format_single(item)
        assert "什么是机器学习？" in result
        assert "机器学习是AI的一个分支..." in result
        assert "### 输入:" not in result

    def test_chatml_format(self):
        """测试 ChatML 格式"""
        from data.formatter import InstructionFormatter

        formatter = InstructionFormatter(template_name="chatml")
        item = {
            "instruction": "你好",
            "output": "你好！有什么可以帮助你的？",
        }

        result = formatter.format_single(item)
        assert "<|im_start|>system" in result
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result
        assert "你好" in result

    def test_batch_format(self):
        """测试批量格式化"""
        from data.formatter import InstructionFormatter

        formatter = InstructionFormatter(template_name="alpaca")
        items = [
            {"instruction": f"问题{i}", "output": f"答案{i}"}
            for i in range(5)
        ]

        results = formatter.format_batch(items)
        assert len(results) == 5
        for i, r in enumerate(results):
            assert f"问题{i}" in r

    def test_prompt_only(self):
        """测试获取仅 prompt 部分"""
        from data.formatter import InstructionFormatter

        formatter = InstructionFormatter(template_name="alpaca")
        item = {"instruction": "什么是AI？"}

        prompt = formatter.get_prompt_only(item)
        assert "什么是AI？" in prompt
        assert "### 回答:" in prompt

    def test_unknown_template(self):
        """测试未知模板名称"""
        from data.formatter import InstructionFormatter

        with pytest.raises(ValueError, match="未知的模板名称"):
            InstructionFormatter(template_name="nonexistent_template")

    def test_simple_format(self):
        """测试 simple 格式"""
        from data.formatter import InstructionFormatter

        formatter = InstructionFormatter(template_name="simple")
        item = {"instruction": "问题", "output": "答案"}

        result = formatter.format_single(item)
        assert result == "问题\n答案"


class TestTokenizerWrapper:
    """测试分词器封装（使用 mock）"""

    def test_encode_text_without_load(self):
        """测试未加载分词器时编码"""
        from data.tokenizer_utils import TokenizerWrapper

        wrapper = TokenizerWrapper()
        with pytest.raises(RuntimeError, match="分词器未加载"):
            wrapper.encode_text("测试文本")

    def test_decode_without_load(self):
        """测试未加载分词器时解码"""
        from data.tokenizer_utils import TokenizerWrapper

        wrapper = TokenizerWrapper()
        with pytest.raises(RuntimeError, match="分词器未加载"):
            wrapper.decode([1, 2, 3])


class TestDatasetBuilder:
    """测试数据集构建器（使用 mock）"""

    def test_builder_init(self):
        """测试构建器初始化"""
        from data.dataset_builder import SFTDatasetBuilder

        builder = SFTDatasetBuilder(
            tokenizer_name_or_path="test-model",
            max_seq_length=1024,
            template_name="alpaca",
        )
        assert builder.max_seq_length == 1024
        assert builder.template_name == "alpaca"

    def test_split_data(self):
        """测试数据划分"""
        from data.dataset_builder import SFTDatasetBuilder

        builder = SFTDatasetBuilder(seed=42, val_split_ratio=0.2)

        data = [{"id": i} for i in range(100)]
        train, val = builder._split_data(data)

        assert len(train) == 80
        assert len(val) == 20

        # 确保没有重叠
        train_ids = {d["id"] for d in train}
        val_ids = {d["id"] for d in val}
        assert len(train_ids & val_ids) == 0

    def test_build_hf_dataset(self):
        """测试 HuggingFace Dataset 构建"""
        from data.dataset_builder import SFTDatasetBuilder

        builder = SFTDatasetBuilder()

        data = [
            {"text": f"文本{i}", "prompt": f"提示{i}"}
            for i in range(10)
        ]

        dataset = builder._build_hf_dataset(data)
        assert len(dataset) == 10
        assert "text" in dataset.column_names
        assert "prompt" in dataset.column_names

    def test_build_empty_dataset(self):
        """测试空数据集构建"""
        from data.dataset_builder import SFTDatasetBuilder

        builder = SFTDatasetBuilder()
        dataset = builder._build_hf_dataset([])
        assert len(dataset) == 0
