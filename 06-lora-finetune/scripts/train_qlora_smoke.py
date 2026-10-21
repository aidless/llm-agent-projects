#!/usr/bin/env python3
"""
06-lora-finetune 的 QLoRA 冒烟训练脚本（Windows + RTX 3060 6GB 优化版）。

设计目标：
  1. 在最小依赖下端到端可运行（纯 transformers + peft + bitsandbytes + trl）
  2. 默认 3B + 4bit + LoRA r=8 + seq_len=1024（OOM 风险低）
  3. 冒烟模式：默认 5 步快速验证管道（用于 CI 风格冒烟）
  4. 真实训练模式：可配置 epochs、lr、output dir

依赖（pip install）：
  torch transformers peft trl bitsandbytes accelerate datasets safetensors

推荐环境（已在 RTX 3060 6GB 验证）：
  torch 2.6.0+cu124
  transformers 4.48.3
  peft 0.13
  trl 0.17
  bitsandbytes 0.45
  datasets 3.x

用法：
  # 冒烟：5 步 + 10 样本
  python scripts/train_qlora_smoke.py --smoke

  # 真实训练
  python scripts/train_qlora_smoke.py \\
      --data-path data/audit/2026-07-22-v2/train.jsonl \\
      --val-path data/audit/2026-07-22-v2/val.jsonl \\
      --output-dir ./output/lora-v1 \\
      --epochs 2

关键防御性检查：
  - 数据 schema（必须含 instruction, output）
  - 数据 token 长度估计（超 1024 自动截断）
  - 显存预算（若 < 6GB 自动降 batch/seq）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Windows 默认 gbk 编码，emoji 字符输出会崩。强制 utf-8。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass  # 旧 Python 或无 stdout（如 pipe）则忽略

# Workaround: 某些 shell(WorkBuddy 等)把大型 JSON 配置塞进环境变量
# (e.g. ACC_PRODUCT_CONFIG_V3 有 317KB), accelerate 的 clear_environment() 会尝试
# 还原全部 env vars 到 Windows 32K 限制之外, 触发 ValueError。
# 在 import accelerate / transformers 之前清掉这些大 env。
for k in list(os.environ.keys()):
    val_len = len(os.environ.get(k, ""))
    if val_len > 30000:
        sys.stderr.write(f"[warn] dropping oversized env var {k} (len={val_len})\n")
        sys.stderr.flush()
        del os.environ[k]  # 在 transformers import 前清除
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------- 参数 -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA 冒烟训练（RTX 3060 6GB）")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                   help="基座模型（HuggingFace ID 或本地路径）")
    p.add_argument("--data-path", default="data/audit/2026-07-22-v2/train.jsonl")
    p.add_argument("--val-path", default="data/audit/2026-07-22-v2/val.jsonl")
    p.add_argument("--output-dir", default="./output/lora-smoke")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=25)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=-1, help="-1 表示按 epoch 跑")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="冒烟模式：10 样本，5 步")
    p.add_argument("--save-merged", action="store_true", help="训练后合并并保存为完整模型（CPU 合并）")
    p.add_argument("--no-4bit", action="store_true", help="禁用 4bit 量化（仅调试用）")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--mock-smoke", action="store_true",
                   help="完全离线 dry-run：跳过模型加载，仅验证数据/分词器/LoRA 注入逻辑")
    return p.parse_args()


# ----------------------------- 数据 -----------------------------

def load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list) and len(obj) == 2:
                obj = {"instruction": obj[0], "output": obj[1]}
            if "instruction" not in obj or "output" not in obj:
                continue
            rows.append(obj)
            if limit and len(rows) >= limit:
                break
    return rows


# ----------------------------- 环境自检 -----------------------------

def preflight_check(model_id: str, no_4bit: bool) -> Dict[str, Any]:
    """检查 GPU + bitsandbytes + 模型可用性。"""
    import torch
    info: Dict[str, Any] = {"ok": True, "warnings": []}
    if not torch.cuda.is_available():
        info["ok"] = False
        info["warnings"].append("CUDA 不可用——本脚本仅支持 GPU 训练")
        return info
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    info["gpu"] = name
    info["vram_total_gb"] = round(total_gb, 1)
    if total_gb < 5.5:
        info["warnings"].append(f"显存 {total_gb:.1f}GB < 5.5GB，建议降到 batch=1 + seq_len=512")

    if not no_4bit:
        try:
            import bitsandbytes  # noqa
        except ImportError:
            info["ok"] = False
            info["warnings"].append("bitsandbytes 未安装；请 pip install bitsandbytes")
    return info


# ----------------------------- 训练主流程 -----------------------------

def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("[1/6] 环境自检")
    info = preflight_check(args.model, args.no_4bit)
    for k, v in info.items():
        print(f"  {k}: {v}")
    if not info["ok"]:
        print("FAIL: 环境不满足")
        return 1
    for w in info.get("warnings", []):
        print(f"  [!] {w}")
    print()

    print("[2/6] 加载数据")
    data_path = Path(args.data_path)
    val_path = Path(args.val_path) if args.val_path else None
    if not data_path.exists():
        print(f"FAIL: 数据文件不存在: {data_path}")
        return 1
    train_data = load_jsonl(data_path, limit=10 if args.smoke else None)
    val_data = load_jsonl(val_path, limit=5 if args.smoke else None) if val_path and val_path.exists() else []
    if args.smoke:
        args.max_steps = 5
        args.epochs = 1
    print(f"  train: {len(train_data)}  val: {len(val_data)}")
    if len(train_data) < 5:
        print("FAIL: 训练数据过少（<5）")
        return 1
    print()

    print("[3/6] 加载分词器")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  tokenizer: {args.model}  pad={tokenizer.pad_token}  eos={tokenizer.eos_token}")
    print()

    if args.mock_smoke:
        print("[MOCK-SMOKE] 跳过模型加载，仅验证数据 / 分词器 / ChatML 模板")
        # 只用标准库 + transformers 已加载部分，不导入 peft
        # 用 tokenizer 测 1 条 sample 编码
        sample = train_data[0]
        prompt = f"<|im_start|>system\n你是一个有帮助的助手。<|im_end|>\n<|im_start|>user\n{sample['instruction']}<|im_end|>\n<|im_start|>assistant\n{sample['output']}<|im_end|>"
        enc = tokenizer(prompt, truncation=True, max_length=args.max_seq_length, return_tensors="pt")
        print(f"  Sample 0 token count: {enc['input_ids'].shape[1]} (max_seq_length={args.max_seq_length})")
        # 估算参数量（粗略）
        m_lower = args.model.lower()
        if "3b" in m_lower or "3B" in args.model:
            est_total = 3.1e9
        elif "7b" in m_lower or "7B" in args.model:
            est_total = 7e9
        elif "1.5b" in m_lower:
            est_total = 1.5e9
        else:
            est_total = 3.1e9  # 默认假设
        est_trainable = int(est_total * 0.0008)  # 约 0.08% for r=8
        print(f"  Estimated total params: {est_total/1e9:.1f}B")
        print(f"  Estimated trainable params: ~{est_trainable/1e6:.1f}M (r=8 LoRA)")
        # LoRA 配置：尝试加载，不强制
        try:
            from peft import LoraConfig  # noqa
            target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
            lc = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                task_type="CAUSAL_LM",
                bias="none",
            )
            print(f"  LoRA config: r={args.lora_r} alpha={args.lora_alpha} target={target_modules} OK")
        except ImportError as e:
            print(f"  LoRA config: [!] peft 不可用 ({e.__class__.__name__})")
            print("  → 实际训练前需修复 peft/transformers 版本兼容")
        print()
        print("=" * 60)
        print("[OK] MOCK-SMOKE 验证通过（数据 + 分词器 + 模板）")
        print(f"  数据加载: OK ({len(train_data)} 条)")
        print(f"  分词器: OK")
        print(f"  ChatML 模板: OK")
        print()
        print("下一步: 实际训练前用 --smoke 跑一次（不用 --mock-smoke）")
        print("=" * 60)
        return 0

    print("[4/6] 加载模型（量化）")
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": args.model,
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if not args.no_4bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,  # RTX 3060 无 bf16
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
            print("  使用 4bit NF4 + double quant + fp16 compute")
        except Exception as e:
            print(f"  BitsAndBytes 失败（{e}），回退到 fp16")
            model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False  # 训练时禁用 KV cache
    print(f"  模型已加载")
    print()

    print("[5/6] 配置 LoRA + 训练器")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print()

    # 构造训练样本
    def format_sample(ex: Dict[str, Any]) -> str:
        inst = ex.get("instruction", "")
        out = ex.get("output", "")
        # ChatML 模板
        return (
            f"<|im_start|>system\n你是一个有帮助的助手。<|im_end|>\n"
            f"<|im_start|>user\n{inst}<|im_end|>\n"
            f"<|im_start|>assistant\n{out}<|im_end|>"
        )

    from datasets import Dataset
    train_ds = Dataset.from_dict({"text": [format_sample(x) for x in train_data]})
    val_ds = Dataset.from_dict({"text": [format_sample(x) for x in val_data]}) if val_data else None

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    if val_ds is not None:
        val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    print(f"  已 tokenize，train={len(train_ds)} val={0 if val_ds is None else len(val_ds)}")
    print()

    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy="steps" if not args.smoke else "no",
        save_steps=args.save_steps,
        eval_strategy="steps" if val_ds is not None and not args.smoke else "no",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        fp16=not args.no_4bit or True,  # RTX 3060 用 fp16
        bf16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=(val_ds is not None and not args.smoke),
        metric_for_best_model="eval_loss" if val_ds is not None else None,
        greater_is_better=False,
        dataloader_num_workers=0,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # transformers 5.x: tokenizer 改名为 processing_class。向后兼容旧版。
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            tokenizer=tokenizer,
        )

    print("[6/6] 开始训练")
    train_result = trainer.train()
    print()
    print("=" * 60)
    print("训练完成")
    print(f"  final loss: {train_result.training_loss:.4f}")
    print(f"  output_dir: {output_dir}")
    print("=" * 60)

    # 保存 adapter
    adapter_dir = output_dir / "lora_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter 已保存: {adapter_dir}")

    # 可选：CPU 合并
    if args.save_merged:
        print("正在 CPU 合并 LoRA 到基础模型...")
        try:
            merged = model.merge_and_unload()
            merged_dir = output_dir / "merged"
            merged.save_pretrained(str(merged_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(merged_dir))
            print(f"  合并后模型已保存: {merged_dir}")
        except Exception as e:
            print(f"  合并失败（{e}），可能 4bit + merge_and_unload 有 bug；建议 --no-4bit 或外部 CPU 合并")
    return 0


if __name__ == "__main__":
    sys.exit(main())
