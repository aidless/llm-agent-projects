#!/usr/bin/env python3
"""
模型合并和导出脚本

将 LoRA 适配器权重合并到基础模型，导出完整模型。

使用方式:
    python scripts/export_model.py \\
        --base_model meta-llama/Llama-2-7b-hf \\
        --adapter_path ./output/lora_adapter \\
        --output_dir ./output/merged_model
"""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 权重并导出完整模型")
    parser.add_argument("--base_model", type=str, required=True, help="基础模型路径")
    parser.add_argument("--adapter_path", type=str, required=True, help="LoRA 适配器路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--safe_serialization", action="store_true", default=True, help="使用 safetensors")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("模型合并和导出")
    logger.info(f"基础模型: {args.base_model}")
    logger.info(f"适配器: {args.adapter_path}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from models.model_exporter import ModelExporter

    # 加载基础模型
    logger.info("加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    # 加载适配器
    model = ModelExporter.load_lora_adapter(model, args.adapter_path)

    # 合并并保存
    logger.info("合并 LoRA 权重并保存...")
    ModelExporter.merge_and_save(
        model=model,
        output_dir=args.output_dir,
        save_tokenizer=True,
        tokenizer=tokenizer,
        safe_serialization=args.safe_serialization,
    )

    logger.info("=" * 60)
    logger.info(f"合并后的模型已保存到: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
