"""
实验追踪模块
基于 JSON 文件的简单实验日志记录，不依赖 W&B
"""
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    实验追踪器

    使用 JSON 文件记录训练过程中的指标，包括 loss、learning_rate、eval_metrics 等。
    每次实验生成一个独立的日志文件。

    Attributes:
        experiment_name: 实验名称
        log_dir: 日志保存目录
        auto_flush: 是否每次记录后自动写入文件
    """

    def __init__(
        self,
        experiment_name: str = "default",
        log_dir: str = "./logs",
        auto_flush: bool = True,
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.auto_flush = auto_flush

        # 日志数据结构
        self.log_data: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "config": {},
            "metrics": [],
            "eval_results": [],
            "final_results": {},
        }

        # 日志文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.json"

        logger.info(f"实验追踪器初始化完成: {self.log_file}")

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        记录实验配置

        Args:
            config: 配置字典（可包含训练参数、LoRA 参数等）
        """
        self.log_data["config"] = config
        if self.auto_flush:
            self.flush()

    def log_metric(
        self,
        step: int,
        loss: float,
        learning_rate: Optional[float] = None,
        grad_norm: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录训练指标

        Args:
            step: 当前训练步数
            loss: 当前 loss 值
            learning_rate: 当前学习率
            grad_norm: 梯度范数
            extra: 额外的自定义指标
        """
        entry = {
            "step": step,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
        }
        if learning_rate is not None:
            entry["learning_rate"] = learning_rate
        if grad_norm is not None:
            entry["grad_norm"] = grad_norm
        if extra is not None:
            entry.update(extra)

        self.log_data["metrics"].append(entry)
        if self.auto_flush:
            self.flush()

    def log_eval(
        self,
        step: int,
        eval_loss: Optional[float] = None,
        eval_metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录评估结果

        Args:
            step: 当前训练步数
            eval_loss: 评估 loss
            eval_metrics: 自定义评估指标字典
            extra: 额外的评估信息
        """
        entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }
        if eval_loss is not None:
            entry["eval_loss"] = eval_loss
        if eval_metrics is not None:
            entry["metrics"] = eval_metrics
        if extra is not None:
            entry.update(extra)

        self.log_data["eval_results"].append(entry)
        if self.auto_flush:
            self.flush()

    def log_final_results(self, results: Dict[str, Any]) -> None:
        """
        记录最终实验结果

        Args:
            results: 最终结果字典
        """
        self.log_data["final_results"] = results
        self.log_data["end_time"] = datetime.now().isoformat()
        self.flush()

    def flush(self) -> None:
        """将日志数据写入文件"""
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """
        获取实验摘要

        Returns:
            包含关键指标摘要的字典
        """
        metrics = self.log_data["metrics"]
        eval_results = self.log_data["eval_results"]

        summary = {
            "experiment_name": self.experiment_name,
            "total_steps": len(metrics),
            "log_file": str(self.log_file),
        }

        if metrics:
            losses = [m["loss"] for m in metrics]
            summary["min_loss"] = min(losses)
            summary["max_loss"] = max(losses)
            summary["final_loss"] = losses[-1]
            summary["avg_loss_last_10"] = sum(losses[-10:]) / len(losses[-10:])

        if eval_results:
            last_eval = eval_results[-1]
            summary["last_eval"] = last_eval

        return summary

    def get_metrics_history(self, key: str = "loss") -> List[float]:
        """
        获取指标历史

        Args:
            key: 指标名称

        Returns:
            指标值列表
        """
        return [m[key] for m in self.log_data["metrics"] if key in m]
