#!/usr/bin/env python3
"""
推理脚本

使用微调后的模型进行推理。

使用方式:
    # 使用合并后的模型
    python scripts/inference.py --model_path ./output/merged_model

    # 使用 LoRA 适配器
    python scripts/inference.py --model_path meta-llama/Llama-2-7b-hf --adapter_path ./output/lora_adapter

    # 交互模式
    python scripts/inference.py --model_path ./output/merged_model --interactive
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
    parser = argparse.ArgumentParser(description="LoRA 微调模型推理")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--adapter_path", type=str, default=None, help="LoRA 适配器路径")
    parser.add_argument("--prompt", type=str, default=None, help="推理提示文本")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    args = parser.parse_args()

    logger.info(f"加载模型: {args.model_path}")
    if args.adapter_path:
        logger.info(f"LoRA 适配器: {args.adapter_path}")

    from inference import InferenceEngine, GenerationConfig

    # 初始化推理引擎
    engine = InferenceEngine(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
    )

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=True,
        top_p=0.9,
    )

    if args.interactive:
        # 交互模式
        print("=" * 60)
        print("交互推理模式（输入 'quit' 退出）")
        print("=" * 60)
        while True:
            try:
                prompt = input("\n用户: ").strip()
                if prompt.lower() == "quit":
                    break
                if not prompt:
                    continue

                response = engine.generate(prompt, gen_config)
                print(f"\n助手: {response}")
            except KeyboardInterrupt:
                break
    elif args.prompt:
        # 单次推理
        response = engine.generate(args.prompt, gen_config)
        print(f"提示: {args.prompt}")
        print(f"回答: {response}")
    else:
        print("请提供 --prompt 或使用 --interactive 模式")
        sys.exit(1)


if __name__ == "__main__":
    main()
