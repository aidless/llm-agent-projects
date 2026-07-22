"""
模型导出模块
提供合并 LoRA 权重并导出完整模型的功能
"""
from typing import Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelExporter:
    """
    模型导出器

    支持将 LoRA 适配器权重合并到基础模型中，
    并导出完整的模型到指定路径。
    """

    @staticmethod
    def merge_and_save(
        model: Any,
        output_dir: str,
        save_tokenizer: bool = True,
        tokenizer: Optional[Any] = None,
        safe_serialization: bool = True,
    ) -> str:
        """
        合并 LoRA 权重并保存完整模型

        Args:
            model: PEFT 模型（带有 LoRA 适配器）
            output_dir: 输出目录
            save_tokenizer: 是否同时保存分词器
            tokenizer: 分词器实例（如果 save_tokenizer=True 则需要）
            safe_serialization: 是否使用 safetensors 格式保存

        Returns:
            模型保存路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 检查是否是 PEFT 模型
        if hasattr(model, "merge_and_unload"):
            logger.info("正在合并 LoRA 权重到基础模型...")
            merged_model = model.merge_and_unload()
            logger.info("LoRA 权重合并完成")
        else:
            logger.info("模型不是 PEFT 模型，直接保存")
            merged_model = model

        # 保存模型
        logger.info(f"正在保存模型到: {output_path}")
        merged_model.save_pretrained(
            str(output_path),
            safe_serialization=safe_serialization,
        )

        # 保存分词器
        if save_tokenizer:
            if tokenizer is None:
                logger.warning("未提供分词器，跳过分词器保存")
            else:
                tokenizer.save_pretrained(str(output_path))
                logger.info(f"分词器已保存到: {output_path}")

        logger.info(f"模型导出完成: {output_path}")
        return str(output_path)

    @staticmethod
    def save_lora_adapter(
        model: Any,
        output_dir: str,
    ) -> str:
        """
        仅保存 LoRA 适配器权重（不合并）

        Args:
            model: PEFT 模型
            output_dir: 输出目录

        Returns:
            适配器保存路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(output_path))
            logger.info(f"LoRA 适配器已保存到: {output_path}")
        else:
            raise ValueError("模型不是 PEFT 模型，无法仅保存适配器")

        return str(output_path)

    @staticmethod
    def load_lora_adapter(
        base_model: Any,
        adapter_path: str,
    ) -> Any:
        """
        将保存的 LoRA 适配器加载到基础模型上

        Args:
            base_model: 基础模型
            adapter_path: 适配器路径

        Returns:
            加载了适配器的模型
        """
        from peft import PeftModel

        logger.info(f"正在从 {adapter_path} 加载 LoRA 适配器...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        logger.info("LoRA 适配器加载完成")

        return model
