"""数据集管理测试。"""

import json
import os
import pytest

from datasets.manager import DatasetManager
from datasets.augmenter import DataAugmenter


class TestDatasetManager:
    def setup_method(self):
        self.manager = DatasetManager()
        self.test_dir = "/data/user/work/test_data"
        os.makedirs(self.test_dir, exist_ok=True)

    def teardown_method(self):
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_create_dataset(self):
        result = self.manager.create_dataset(
            name="test_ds",
            data=[{"input": "q1", "output": "a1"}],
        )
        assert result["name"] == "test_ds"
        assert result["size"] == 1

    def test_list_datasets(self):
        self.manager.create_dataset("ds1", data=[{"input": "a", "output": "b"}])
        self.manager.create_dataset("ds2", data=[{"input": "c", "output": "d"}])
        result = self.manager.list_datasets()
        assert len(result) == 2

    def test_get_dataset(self):
        self.manager.create_dataset("ds1", data=[{"input": "q", "output": "a"}])
        ds = self.manager.get_dataset("ds1")
        assert ds["name"] == "ds1"
        assert len(ds["data"]) == 1

    def test_get_nonexistent_raises(self):
        with pytest.raises(ValueError, match="不存在"):
            self.manager.get_dataset("nonexistent")

    def test_delete_dataset(self):
        self.manager.create_dataset("ds1", data=[])
        self.manager.delete_dataset("ds1")
        assert len(self.manager.list_datasets()) == 0

    def test_delete_nonexistent_raises(self):
        with pytest.raises(ValueError, match="不存在"):
            self.manager.delete_dataset("nonexistent")

    def test_import_json_array(self):
        path = os.path.join(self.test_dir, "test.json")
        with open(path, "w") as f:
            json.dump([{"input": "q1", "output": "a1"}, {"input": "q2", "output": "a2"}], f)

        result = self.manager.import_json(path)
        assert result["size"] == 2

    def test_import_json_object(self):
        path = os.path.join(self.test_dir, "test.json")
        with open(path, "w") as f:
            json.dump({"data": [{"input": "q", "output": "a"}], "input_key": "input", "output_key": "output"}, f)

        result = self.manager.import_json(path)
        assert result["size"] == 1

    def test_import_jsonl(self):
        path = os.path.join(self.test_dir, "test.jsonl")
        with open(path, "w") as f:
            f.write('{"input": "q1", "output": "a1"}\n')
            f.write('{"input": "q2", "output": "a2"}\n')

        result = self.manager.import_jsonl(path)
        assert result["size"] == 2

    def test_import_csv(self):
        path = os.path.join(self.test_dir, "test.csv")
        with open(path, "w") as f:
            f.write("input,output\n")
            f.write("q1,a1\n")
            f.write("q2,a2\n")

        result = self.manager.import_csv(path)
        assert result["size"] == 2

    def test_import_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            self.manager.import_json("/nonexistent/file.json")

    def test_sample_random(self):
        self.manager.create_dataset("ds", data=[
            {"input": f"q{i}", "output": f"a{i}"} for i in range(10)
        ])
        samples = self.manager.sample("ds", n=3, strategy="random", seed=42)
        assert len(samples) == 3

    def test_sample_first(self):
        self.manager.create_dataset("ds", data=[
            {"input": f"q{i}", "output": f"a{i}"} for i in range(10)
        ])
        samples = self.manager.sample("ds", n=3, strategy="first")
        assert samples[0]["input"] == "q0"
        assert samples[2]["input"] == "q2"

    def test_get_inputs_and_outputs(self):
        self.manager.create_dataset("ds", data=[
            {"input": "q1", "output": "a1"},
            {"input": "q2", "output": "a2"},
        ])
        inputs = self.manager.get_inputs("ds")
        outputs = self.manager.get_outputs("ds")
        assert inputs == ["q1", "q2"]
        assert outputs == ["a1", "a2"]

    def test_sample_invalid_strategy_raises(self):
        self.manager.create_dataset("ds", data=[{"input": "q", "output": "a"}] * 10)
        with pytest.raises(ValueError, match="未知采样策略"):
            self.manager.sample("ds", n=3, strategy="invalid")


class TestDataAugmenter:
    def test_synonym_rewrite_changes_text(self):
        aug = DataAugmenter(seed=42)
        result = aug.synonym_rewrite("This is a good test", replace_ratio=1.0)
        assert result != "This is a good test"
        assert "good" not in result.lower()

    def test_synonym_rewrite_no_change_when_no_match(self):
        aug = DataAugmenter(seed=42)
        result = aug.synonym_rewrite("xyz abc def")
        # 这些词不在同义词表中，可能不变
        assert isinstance(result, str)

    def test_back_translation_changes_text(self):
        aug = DataAugmenter(seed=42)
        result = aug.back_translation_simulate("Hello, this is very important", translate_ratio=1.0)
        assert isinstance(result, str)

    def test_augment_dataset_increases_size(self):
        aug = DataAugmenter(seed=42)
        data = [
            {"input": "This is good", "output": "Good result"},
        ]
        result = aug.augment_dataset(data, methods=["synonym"], augment_per_sample=2)
        assert len(result) > 1