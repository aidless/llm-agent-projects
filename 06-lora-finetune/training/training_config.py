"""
训练配置模块
管理训练超参数和训练器配置
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    训练配置

    管理 SFT 微调的所有训练超参数。

    Attributes:
        num_train_epochs: 训练轮数
        per_device_train_batch_size: 每设备训练 batch size
        per_device_eval_batch_size: 每设备评估 batch size
        gradient_accumulation_steps: 梯度累积步数（等效 batch size = batch_size * accum_steps）
        learning_rate: 学习率
        lr_scheduler_type: 学习率调度器类型
        warmup_ratio: 预热比例（占总步数的比例）
        weight_decay: 权重衰减
        max_grad_norm: 梯度裁剪最大范数
        fp16: 是否使用 fp16 混合精度
        bf16: 是否使用 bf16 混合精度
        logging_steps: 日志记录间隔
        save_strategy: 保存策略: "steps", "epoch", "no"
        save_steps: 保存间隔步数
        eval_strategy: 评估策略: "steps", "epoch", "no"
        eval_steps: 评估间隔步数
        save_total_limit: 最多保存的 checkpoint 数量
        seed: 随机种子
        output_dir: 输出目录
    """
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 100
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_total_limit: int = 3
    seed: int = 42
    output_dir: str = "./output"

    def to_transformers_args(self):
        """
        转换为 HuggingFace TrainingArguments

        Returns:
            transformers.TrainingArguments 实例
        """
        from transformers import TrainingArguments

        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            fp16=self.fp16,
            bf16=self.bf16,
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            eval_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            save_total_limit=self.save_total_limit,
            seed=self.seed,
            dataloader_pin_memory=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",  # 不使用 W&B
            ddp_find_unused_parameters=False,
        )

        logger.info(
            f"训练配置: epochs={self.num_train_epochs}, "
            f"batch_size={self.per_device_train_batch_size}, "
            f"grad_accum={self.gradient_accumulation_steps}, "
            f"lr={self.learning_rate}, "
            f"effective_bs={self.per_device_train_batch_size * self.gradient_accumulation_steps}"
        )

        return args

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "logging_steps": self.logging_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "save_total_limit": self.save_total_limit,
            "seed": self.seed,
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """从字典创建配置"""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)
