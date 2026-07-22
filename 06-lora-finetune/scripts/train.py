#!/usr/bin/env python3
"""
LoRA 微调训练脚本

使用方式:
    python scripts/train.py --config config/train_config.yaml

    或使用默认配置:
    python scripts/train.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

# 将项目根目录加入路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LoRA 微调训练脚本")
    parser.add_argument(
        "--data_path", type=str, default="data/sample/train.jsonl",
        help="训练数据路径"
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="模型名称或路径（优先使用环境变量 MODEL_NAME）"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="输出目录"
    )
    parser.add_argument(
        "--lora_r", type=int, default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=16,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--use_qlora", action="store_true",
        help="使用 QLoRA（4-bit 量化）"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="每设备 batch size"
    )
    parser.add_argument(
        "--grad_accum", type=int, default=4,
        help="梯度累积步数"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="学习率"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=2048,
        help="最大序列长度"
    )
    parser.add_argument(
        "--template", type=str, default="alpaca",
        help="提示模板名称"
    )
    parser.add_argument(
        "--experiment_name", type=str, default="lora_finetune",
        help="实验名称"
    )
    return parser.parse_args()


def main():
    """主训练流程"""
    args = parse_args()

    # 从环境变量读取模型名称
    model_name = args.model_name or os.environ.get("MODEL_NAME", "meta-llama/Llama-2-7b-hf")

    logger.info("=" * 60)
    logger.info("LoRA 微调训练")
    logger.info(f"模型: {model_name}")
    logger.info(f"数据: {args.data_path}")
    logger.info(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, qlora={args.use_qlora}")
    logger.info("=" * 60)

    # ===== 第一步：准备数据 =====
    logger.info("[1/5] 准备数据...")
    from data.dataset_builder import SFTDatasetBuilder

    builder = SFTDatasetBuilder(
        tokenizer_name_or_path=model_name,
        max_seq_length=args.max_seq_length,
        template_name=args.template,
        val_split_ratio=0.05,
    )

    train_dataset, val_dataset = builder.build_instruction_dataset(
        data_path=args.data_path,
        data_format="jsonl",
    )

    # ===== 第二步：配置 LoRA =====
    logger.info("[2/5] 配置 LoRA...")
    from models.lora_config import LoRAConfig

    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_qlora=args.use_qlora,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    # ===== 第三步：加载模型 =====
    logger.info("[3/5] 加载模型...")
    from models.model_loader import ModelLoader

    loader = ModelLoader(
        model_name_or_path=model_name,
        use_gradient_checkpointing=True,
        torch_dtype="auto",
    )

    model = loader.load_base_model(lora_config=lora_config)
    tokenizer = loader.load_tokenizer()

    # ===== 第四步：配置训练 =====
    logger.info("[4/5] 配置训练器...")
    from training.training_config import TrainingConfig
    from training.sft_trainer import SFTTrainerWrapper

    training_config = TrainingConfig(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        output_dir=args.output_dir,
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
    )

    trainer = SFTTrainerWrapper(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_config=training_config,
        max_seq_length=args.max_seq_length,
        experiment_name=args.experiment_name,
        log_dir=os.path.join(args.output_dir, "logs"),
    )

    # ===== 第五步：开始训练 =====
    logger.info("[5/5] 开始训练...")
    results = trainer.train()

    # 保存模型
    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    trainer.save_model(adapter_dir)

    # 输出结果
    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info(f"结果: {results}")
    logger.info(f"LoRA 适配器已保存: {adapter_dir}")
    logger.info("=" * 60)

    # 返回摘要
    summary = trainer.get_results()
    logger.info(f"实验摘要: {json.dumps(summary, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
