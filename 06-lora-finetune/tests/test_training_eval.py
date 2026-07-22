"""
训练和评估模块测试
使用 mock 测试训练配置、实验追踪和评估器
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestTrainingConfig:
    """测试训练配置"""

    def test_default_config(self):
        """测试默认配置"""
        from training.training_config import TrainingConfig

        config = TrainingConfig()
        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 4
        assert config.gradient_accumulation_steps == 4
        assert config.learning_rate == 2e-4
        assert config.bf16 is True
        assert config.fp16 is False

    def test_to_dict(self):
        """测试转换为字典"""
        from training.training_config import TrainingConfig

        config = TrainingConfig(num_train_epochs=5, learning_rate=1e-4)
        d = config.to_dict()
        assert d["num_train_epochs"] == 5
        assert d["learning_rate"] == 1e-4

    def test_from_dict(self):
        """测试从字典创建"""
        from training.training_config import TrainingConfig

        d = {
            "num_train_epochs": 10,
            "per_device_train_batch_size": 8,
            "learning_rate": 5e-5,
        }
        config = TrainingConfig.from_dict(d)
        assert config.num_train_epochs == 10
        assert config.per_device_train_batch_size == 8
        assert config.learning_rate == 5e-5

    def test_to_transformers_args(self):
        """测试转换为 TrainingArguments"""
        from training.training_config import TrainingConfig

        config = TrainingConfig(
            num_train_epochs=1,
            output_dir="./test_output",
            fp16=False,
            bf16=False,
        )
        args = config.to_transformers_args()
        assert args.num_train_epochs == 1
        assert args.output_dir == "./test_output"


class TestExperimentTracker:
    """测试实验追踪器"""

    def test_create_tracker(self):
        """测试创建追踪器"""
        from training.experiment_tracker import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                experiment_name="test_exp",
                log_dir=tmpdir,
            )
            assert tracker.experiment_name == "test_exp"
            assert tracker.log_file.parent == Path(tmpdir)

    def test_log_config(self):
        """测试记录配置"""
        from training.experiment_tracker import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(experiment_name="test", log_dir=tmpdir)
            tracker.log_config({"lr": 0.001, "batch_size": 8})

            with open(tracker.log_file, "r") as f:
                data = json.load(f)
            assert data["config"]["lr"] == 0.001
            assert data["config"]["batch_size"] == 8

    def test_log_metric(self):
        """测试记录训练指标"""
        from training.experiment_tracker import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(experiment_name="test", log_dir=tmpdir)
            tracker.log_metric(step=10, loss=2.5, learning_rate=1e-4)
            tracker.log_metric(step=20, loss=1.8, learning_rate=9e-5)

            losses = tracker.get_metrics_history("loss")
            assert len(losses) == 2
            assert losses[0] == 2.5
            assert losses[1] == 1.8

    def test_log_eval(self):
        """测试记录评估结果"""
        from training.experiment_tracker import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(experiment_name="test", log_dir=tmpdir)
            tracker.log_eval(step=50, eval_loss=1.2, eval_metrics={"accuracy": 0.85})

            with open(tracker.log_file, "r") as f:
                data = json.load(f)
            assert len(data["eval_results"]) == 1
            assert data["eval_results"][0]["eval_loss"] == 1.2
            assert data["eval_results"][0]["metrics"]["accuracy"] == 0.85

    def test_get_summary(self):
        """测试获取摘要"""
        from training.experiment_tracker import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(experiment_name="test", log_dir=tmpdir)
            tracker.log_metric(step=10, loss=3.0)
            tracker.log_metric(step=20, loss=2.0)
            tracker.log_metric(step=30, loss=1.5)

            summary = tracker.get_summary()
            assert summary["total_steps"] == 3
            assert summary["min_loss"] == 1.5
            assert summary["max_loss"] == 3.0
            assert summary["final_loss"] == 1.5

    def test_log_final_results(self):
        """测试记录最终结果"""
        from training.experiment_tracker import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(experiment_name="test", log_dir=tmpdir)
            tracker.log_final_results({
                "total_steps": 100,
                "best_loss": 0.5,
            })

            with open(tracker.log_file, "r") as f:
                data = json.load(f)
            assert data["final_results"]["total_steps"] == 100
            assert "end_time" in data


class TestTaskEvaluator:
    """测试任务评估器"""

    def test_accuracy(self):
        """测试准确率计算"""
        from evaluation.task_evaluator import TaskEvaluator

        evaluator = TaskEvaluator(metrics=["accuracy"])
        predictions = ["北京", "上海", "深圳", "广州"]
        references = ["北京", "上海", "深圳", "杭州"]

        results = evaluator.evaluate(predictions, references)
        assert results["accuracy"] == 0.75

    def test_f1(self):
        """测试 F1 分数计算"""
        from evaluation.task_evaluator import TaskEvaluator

        evaluator = TaskEvaluator(metrics=["f1"])
        predictions = ["中国的首都是北京"]
        references = ["北京是中国的首都"]

        results = evaluator.evaluate(predictions, references)
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert 0 < results["f1"] <= 1.0

    def test_empty_predictions(self):
        """测试空预测"""
        from evaluation.task_evaluator import TaskEvaluator

        evaluator = TaskEvaluator(metrics=["accuracy", "f1"])
        results = evaluator.evaluate([], [])
        assert results["accuracy"] == 0.0
        assert results["f1"] == 0.0

    def test_mismatched_lengths(self):
        """测试长度不匹配"""
        from evaluation.task_evaluator import TaskEvaluator

        evaluator = TaskEvaluator(metrics=["accuracy"])
        with pytest.raises(ValueError, match="不匹配"):
            evaluator.evaluate(["a", "b"], ["c"])

    def test_custom_match_fn(self):
        """测试自定义匹配函数"""
        from evaluation.task_evaluator import TaskEvaluator

        evaluator = TaskEvaluator(
            metrics=["accuracy"],
            exact_match_fn=lambda pred, ref: pred.strip() in ref.strip(),
        )
        predictions = ["北京"]
        references = ["北京是中国的首都"]

        results = evaluator.evaluate(predictions, references)
        assert results["accuracy"] == 1.0

    def test_evaluate_generation(self):
        """测试生成评估"""
        from evaluation.task_evaluator import TaskEvaluator

        evaluator = TaskEvaluator(metrics=["accuracy"])
        prompts = ["问题1", "问题2"]
        predictions = ["答案是北京", "答案是上海"]
        references = ["北京", "上海"]

        results = evaluator.evaluate_generation(
            prompts=prompts,
            predictions=predictions,
            references=references,
            extract_fn=lambda x: x.replace("答案是", "").strip(),
        )

        assert "metrics" in results
        assert "details" in results
        assert results["total_samples"] == 2

    def test_tokenize(self):
        """测试分词"""
        from evaluation.task_evaluator import TaskEvaluator

        tokens = TaskEvaluator._tokenize("Hello World 你好世界")
        assert "hello" in tokens
        assert "world" in tokens
        assert "你" in tokens
        assert "好" in tokens


class TestLoggingCallback:
    """测试训练回调"""

    def test_on_log(self):
        """测试日志回调"""
        from training.sft_trainer import LoggingCallback

        with tempfile.TemporaryDirectory() as tmpdir:
            from training.experiment_tracker import ExperimentTracker
            tracker = ExperimentTracker(experiment_name="test", log_dir=tmpdir)

            callback = LoggingCallback(tracker=tracker, log_interval=1)

            # mock 参数
            mock_state = MagicMock()
            mock_state.global_step = 10

            callback.on_log(
                args=None,
                state=mock_state,
                control=None,
                logs={"loss": 1.5, "learning_rate": 1e-4},
            )

            losses = tracker.get_metrics_history("loss")
            assert len(losses) == 1
            assert losses[0] == 1.5

    def test_on_evaluate(self):
        """测试评估回调"""
        from training.sft_trainer import LoggingCallback

        with tempfile.TemporaryDirectory() as tmpdir:
            from training.experiment_tracker import ExperimentTracker
            tracker = ExperimentTracker(experiment_name="test", log_dir=tmpdir)

            callback = LoggingCallback(tracker=tracker, log_interval=1)

            mock_state = MagicMock()
            mock_state.global_step = 50

            callback.on_evaluate(
                args=None,
                state=mock_state,
                control=None,
                metrics={"eval_loss": 1.2, "eval_accuracy": 0.85},
            )

            with open(tracker.log_file, "r") as f:
                data = json.load(f)
            assert len(data["eval_results"]) == 1
