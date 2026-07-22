"""
模型加载模块
支持加载预训练模型并应用 LoRA/QLoRA 适配器
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelLoader:
    """
    模型加载器

    负责加载预训练语言模型，并可选地应用 LoRA/QLoRA 适配器。
    支持全精度、半精度（fp16/bf16）和量化加载。

    Attributes:
        model_name_or_path: 模型名称或本地路径
        use_gradient_checkpointing: 是否启用梯度检查点以节省显存
        torch_dtype: 模型权重的数据类型
        attn_implementation: 注意力实现方式
    """
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    use_gradient_checkpointing: bool = True
    torch_dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    attn_implementation: str = "eager"  # "eager", "sdpa", "flash_attention_2"

    def load_base_model(
        self,
        lora_config: Optional[Any] = None,
        device_map: str = "auto",
    ) -> Any:
        """
        加载基础模型（可选应用 LoRA）

        Args:
            lora_config: LoRAConfig 实例，如果提供则应用 LoRA
            device_map: 设备映射策略

        Returns:
            加载后的模型（可能是 PEFT 模型）
        """
        from transformers import AutoModelForCausalLM

        # 构建加载参数
        load_kwargs = {
            "pretrained_model_name_or_path": self.model_name_or_path,
            "trust_remote_code": True,
            "device_map": device_map,
        }

        # 设置数据类型
        if self.torch_dtype != "auto":
            import torch
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            load_kwargs["torch_dtype"] = dtype_map.get(self.torch_dtype)

        # 如果使用 QLoRA，添加量化配置
        if lora_config is not None and hasattr(lora_config, "get_bnb_config"):
            bnb_config = lora_config.get_bnb_config()
            if bnb_config is not None:
                load_kwargs["quantization_config"] = bnb_config

        # 注意力实现
        if self.attn_implementation != "eager":
            load_kwargs["attn_implementation"] = self.attn_implementation

        logger.info(f"正在加载模型: {self.model_name_or_path} ...")
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        # 启用梯度检查点
        if self.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            logger.info("已启用梯度检查点")

        logger.info(f"模型加载完成，参数量: {model.num_parameters():,}")

        # 应用 LoRA
        if lora_config is not None:
            model = self._apply_lora(model, lora_config)

        return model

    def load_tokenizer(self) -> Any:
        """
        加载与模型配套的分词器

        Returns:
            HuggingFace PreTrainedTokenizer 实例
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )

        # 确保有 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("已设置 pad_token = eos_token")

        return tokenizer

    def _apply_lora(self, model: Any, lora_config: Any) -> Any:
        """
        对模型应用 LoRA 适配器

        Args:
            model: 基础模型
            lora_config: LoRAConfig 实例

        Returns:
            应用 LoRA 后的 PEFT 模型
        """
        from peft import get_peft_model, prepare_model_for_kbit_training

        # 如果是 QLoRA，需要准备模型
        if lora_config.use_qlora:
            model = prepare_model_for_kbit_training(model)
            logger.info("已准备 QLoRA 模型")

        peft_config = lora_config.to_peft_config()
        model = get_peft_model(model, peft_config)

        # 打印可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"LoRA 适配器已应用: 可训练参数 {trainable_params:,} / "
            f"总参数 {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )

        # 打印 LoRA 层信息
        model.print_trainable_parameters()

        return model
