"""
LoRA 配置模块
提供 LoRA 和 QLoRA 的可配置参数管理
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """
    LoRA 微调配置

    管理 LoRA 微调的所有超参数，包括目标模块、rank、alpha、dropout 等。
    支持从字典或环境变量加载配置。

    Attributes:
        use_qlora: 是否使用 QLoRA（4-bit 量化 + LoRA）
        r: LoRA rank（秩），控制低秩矩阵的大小
        lora_alpha: LoRA 缩放因子，实际缩放为 alpha/r
        lora_dropout: LoRA 层的 dropout 比率
        target_modules: 需要应用 LoRA 的目标模块列表
        bias: bias 处理方式: "none"（默认）、"all"、"lora_only"
        task_type: 任务类型，SFT 通常为 "CAUSAL_LM"
        quantization_bit: QLoRA 量化位数（仅 use_qlora=True 时有效）
        double_quantization: 是否启用双重量化（QLoRA）
        quant_type: 量化类型: "nf4"（默认）或 "fp4"
    """
    use_qlora: bool = False  # 是否使用 QLoRA
    r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA alpha（缩放因子）
    lora_dropout: float = 0.05  # LoRA dropout
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj"
    ])
    bias: str = "none"  # bias 类型: "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"  # 任务类型
    quantization_bit: int = 4  # 量化位数
    double_quantization: bool = True  # 双重量化
    quant_type: str = "nf4"  # 量化类型

    def to_peft_config(self):
        """
        转换为 PEFT 库的 LoraConfig 对象

        Returns:
            peft.LoraConfig 实例
        """
        from peft import LoraConfig, TaskType

        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_CLS": TaskType.SEQ_CLS,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        }
        task_type = task_type_map.get(self.task_type, TaskType.CAUSAL_LM)

        config = LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=task_type,
        )

        logger.info(
            f"LoRA 配置: r={self.r}, alpha={self.lora_alpha}, "
            f"dropout={self.lora_dropout}, targets={self.target_modules}"
        )

        return config

    def get_bnb_config(self):
        """
        获取 QLoRA 的 BitsAndBytes 配置

        Returns:
            transformers.BitsAndBytesConfig 实例，如果非 QLoRA 模式则返回 None
        """
        if not self.use_qlora:
            return None

        from transformers import BitsAndBytesConfig

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=self.double_quantization,
            bnb_4bit_quant_type=self.quant_type,
            bnb_4bit_compute_dtype="bfloat16",
        )

        logger.info(
            f"QLoRA 配置: {self.quantization_bit}bit, "
            f"double_quant={self.double_quantization}, type={self.quant_type}"
        )

        return config

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "use_qlora": self.use_qlora,
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
            "quantization_bit": self.quantization_bit,
            "double_quantization": self.double_quantization,
            "quant_type": self.quant_type,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoRAConfig":
        """从字典创建配置"""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)


def get_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    use_qlora: bool = False,
    target_modules: Optional[List[str]] = None,
    **kwargs,
) -> LoRAConfig:
    """
    快捷方法：创建 LoRA 配置

    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha
        use_qlora: 是否使用 QLoRA
        target_modules: 目标模块列表
        **kwargs: 其他配置参数

    Returns:
        LoRAConfig 实例
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    return LoRAConfig(
        r=r,
        lora_alpha=lora_alpha,
        use_qlora=use_qlora,
        target_modules=target_modules,
        **kwargs,
    )


# 预定义的常用配置
PRESET_CONFIGS = {
    "small": LoRAConfig(r=4, lora_alpha=8, lora_dropout=0.05),
    "medium": LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.1),
    "large": LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.1),
    "qlora_default": LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.05, use_qlora=True),
    "qlora_large": LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.1, use_qlora=True),
}
