"""
SFT 训练器封装模块
封装 HuggingFace TRL 的 SFTTrainer，提供统一的训练接口和回调集成
"""
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class LoggingCallback:
    """
    训练日志回调

    在训练过程中记录 loss、learning_rate 等指标到 ExperimentTracker。
    兼容 HuggingFace Trainer 的回调接口。
    """

    def __init__(self, tracker: Any, log_interval: int = 10):
        """
        Args:
            tracker: ExperimentTracker 实例
            log_interval: 日志记录间隔
        """
        self.tracker = tracker
        self.log_interval = log_interval

    def on_log(self, args, state, control, logs=None, **kwargs):
        """训练日志回调"""
        if logs is None:
            return

        step = state.global_step
        if step % self.log_interval == 0 or step <= 1:
            loss = logs.get("loss", None)
            learning_rate = logs.get("learning_rate", None)
            grad_norm = logs.get("grad_norm", None)

            if loss is not None:
                self.tracker.log_metric(
                    step=step,
                    loss=loss,
                    learning_rate=learning_rate,
                    grad_norm=grad_norm,
                )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """评估回调"""
        if metrics is None:
            return

        step = state.global_step
        eval_loss = metrics.get("eval_loss", None)
        eval_metrics = {k: v for k, v in metrics.items() if k != "eval_loss"}

        if eval_loss is not None or eval_metrics:
            self.tracker.log_eval(
                step=step,
                eval_loss=eval_loss,
                eval_metrics=eval_metrics if eval_metrics else None,
            )

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束回调"""
        self.tracker.log_final_results({
            "total_steps": state.global_step,
            "total_epochs": int(state.epoch or 0),
            "best_metric": state.best_metric if hasattr(state, "best_metric") else None,
            "best_model_checkpoint": getattr(state, "best_model_checkpoint", None),
        })


class SFTTrainerWrapper:
    """
    SFT 训练器封装

    整合 TRL SFTTrainer、TrainingArguments 和 ExperimentTracker，
    提供简洁的训练接口。

    使用示例:
        trainer = SFTTrainerWrapper(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_config=training_config,
        )
        trainer.train()
        results = trainer.get_results()
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        training_config: Any,
        val_dataset: Optional[Any] = None,
        max_seq_length: int = 2048,
        experiment_name: str = "sft_experiment",
        log_dir: str = "./logs",
        dataset_text_field: str = "text",
    ):
        """
        初始化 SFT 训练器

        Args:
            model: 已应用 LoRA 的 PEFT 模型
            tokenizer: 分词器
            train_dataset: 训练数据集（HuggingFace Dataset）
            training_config: TrainingConfig 实例
            val_dataset: 验证数据集
            max_seq_length: 最大序列长度
            experiment_name: 实验名称
            log_dir: 日志目录
            dataset_text_field: 数据集中文本字段名
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_config = training_config
        self.max_seq_length = max_seq_length
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.dataset_text_field = dataset_text_field

        # 初始化实验追踪器
        self.tracker = self._create_tracker()

        # 初始化训练器
        self.trainer = self._create_trainer()

    def _create_tracker(self):
        """创建实验追踪器"""
        from .experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(
            experiment_name=self.experiment_name,
            log_dir=self.log_dir,
        )

        # 记录配置
        config = {
            "training": self.training_config.to_dict(),
            "max_seq_length": self.max_seq_length,
        }
        tracker.log_config(config)

        return tracker

    def _create_trainer(self):
        """创建 SFTTrainer"""
        from trl import SFTTrainer

        training_args = self.training_config.to_transformers_args()

        # 创建日志回调
        callback = LoggingCallback(
            tracker=self.tracker,
            log_interval=self.training_config.logging_steps,
        )

        # 构建训练器参数
        trainer_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "train_dataset": self.train_dataset,
            "args": training_args,
            "dataset_text_field": self.dataset_text_field,
            "max_seq_length": self.max_seq_length,
            "callbacks": [callback],
        }

        # 如果有验证集，添加评估
        if self.val_dataset is not None:
            trainer_kwargs["eval_dataset"] = self.val_dataset

        trainer = SFTTrainer(**trainer_kwargs)

        logger.info(
            f"SFTTrainer 创建完成: "
            f"训练样本数={len(self.train_dataset)}, "
            f"验证样本数={len(self.val_dataset) if self.val_dataset else 0}"
        )

        return trainer

    def train(self) -> Dict[str, Any]:
        """
        执行训练

        Returns:
            训练结果字典
        """
        logger.info("开始训练...")
        result = self.trainer.train()

        # 记录最终结果
        self.tracker.log_final_results({
            "training_time": str(result.metrics.get("train_runtime", "N/A")),
            "train_loss": result.metrics.get("train_loss", None),
            "eval_loss": result.metrics.get("eval_loss", None),
        })

        logger.info("训练完成!")
        logger.info(f"训练耗时: {result.metrics.get('train_runtime', 'N/A')}s")
        logger.info(f"最终训练 loss: {result.metrics.get('train_loss', 'N/A')}")

        return result.metrics

    def get_results(self) -> Dict[str, Any]:
        """
        获取训练结果摘要

        Returns:
            包含训练结果和实验摘要的字典
        """
        return self.tracker.get_summary()

    def save_model(self, output_dir: str) -> None:
        """
        保存模型

        Args:
            output_dir: 输出目录
        """
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"模型已保存到: {output_dir}")

    def evaluate(self) -> Dict[str, float]:
        """
        在验证集上评估模型

        Returns:
            评估指标字典
        """
        if self.val_dataset is None:
            logger.warning("没有验证集，跳过评估")
            return {}

        result = self.trainer.evaluate()
        logger.info(f"评估结果: {result}")
        return result
