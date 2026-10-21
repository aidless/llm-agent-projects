#!/usr/bin/env python3
"""
06-lora-finetune 全流程编排脚本。

阶段（可单跑或连跑）：
  1. base-eval  —— 评估基座模型在 test 集上的 baseline（**必须先做**，否则训练后无对照）
  2. train      —— QLoRA 训练
  3. eval       —— Base vs LoRA 对比评估 + LLM-judge（可选）

默认值假定 RTX 3060 6GB + Qwen2.5-3B-Instruct。

用法：
  # 跑完整流程
  python scripts/run_full_pipeline.py \\
      --stages base-eval,train,eval \\
      --base-model Qwen/Qwen2.5-3B-Instruct \\
      --data-dir data/audit/2026-07-22-v3 \\
      --output-dir output/v1

  # 仅跑 base 评估
  python scripts/run_full_pipeline.py \\
      --stages base-eval

  # 跳过 base eval 直接训练（不推荐）
  python scripts/run_full_pipeline.py \\
      --stages train --skip-base-eval
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_qlora_smoke.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval_base_vs_lora.py"


def run_step(cmd: List[str], cwd: Optional[Path] = None) -> int:
    """运行子步骤，实时打印输出。"""
    print()
    print("=" * 70)
    print("RUN: " + " ".join(cmd))
    print("=" * 70)
    proc = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT)
    if proc.returncode != 0:
        print(f"[X] 步骤失败: {cmd[0]}")
        sys.exit(proc.returncode)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stages", default="base-eval,train,eval",
                   help="逗号分隔的阶段：base-eval | train | eval")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--data-dir", default="data/audit/2026-07-22-v3",
                   help="含 train.jsonl/val.jsonl/test.jsonl 的目录")
    p.add_argument("--output-dir", default="output/v1",
                   help="基座/LoRA 评估输出目录")
    p.add_argument("--adapter-dir", default=None,
                   help="LoRA adapter 路径（eval 阶段用）；默认 <output-dir>/lora_adapter")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-merged", action="store_true",
                   help="训练后合并 LoRA 到基座（CPU 推荐）")
    p.add_argument("--judge", default="off",
                   help="off | openai（eval 阶段是否启用 LLM-judge）")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--skip-base-eval", action="store_true",
                   help="显式确认不跑 base eval 直接训练")
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    valid = {"base-eval", "train", "eval"}
    bad = [s for s in stages if s not in valid]
    if bad:
        print(f"[X] 未知阶段: {bad}; 仅支持 {valid}")
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir / "eval"
    adapter = args.adapter_dir or str(output_dir / "lora_adapter")
    train_data = Path(args.data_dir) / "train.jsonl"
    val_data = Path(args.data_dir) / "val.jsonl"
    test_data = Path(args.data_dir) / "test.jsonl"

    # 至少需要这些文件
    for p_path in [train_data, val_data, test_data]:
        if not p_path.exists():
            print(f"[X] 数据缺失: {p_path}")
            return 1

    # ----- 阶段 1: base eval -----
    baseline_path = eval_dir / "baseline_score.json"
    run_base = "base-eval" in stages
    if run_base:
        print("\n[阶段 1/3] 评估基座模型（baseline）")
        run_step([
            sys.executable, str(EVAL_SCRIPT),
            "--base-model", args.base_model,
            "--test-data", str(test_data),
            "--out-dir", str(eval_dir),
            "--max-new-tokens", str(args.max_new_tokens),
            "--baseline-only",
            "--judge", args.judge,
            "--judge-model", args.judge_model,
        ])
        if not baseline_path.exists():
            print(f"[X] baseline_score.json 没生成：{baseline_path}")
            return 1
        print(f"[v] Baseline 已保存: {baseline_path}")

    # ----- 阶段 2: 训练 -----
    if "train" in stages:
        # 防呆：如果 base eval 未跑且用户未明示跳过 → 强制要求
        if not baseline_path.exists() and not args.skip_base_eval:
            print()
            print("[!]️  **警告**: 检测到尚未运行 base eval。")
            print("   没有 baseline 就直接训练，eval 时无法对比 Win/Tie/Loss。")
            print()
            print("   建议先跑：python scripts/run_full_pipeline.py --stages base-eval")
            print("   显式跳过：--skip-base-eval")
            print()
            resp = input("仍要继续训练？[y/N] ").strip().lower()
            if resp != "y":
                print("已取消。")
                return 0
            args.skip_base_eval = True

        print("\n[阶段 2/3] QLoRA 训练")
        extra_save = ["--save-merged"] if args.save_merged else []
        run_step([
            sys.executable, str(TRAIN_SCRIPT),
            "--base-model", args.base_model,  # train 脚本叫 --model
            "--data-path", str(train_data),
            "--val-path",  str(val_data),
            "--output-dir", str(output_dir / "train"),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
        ] + extra_save)
        # train_qlora_smoke.py 的 output_dir 是其 internal 概念
        # LoRA adapter 会保存在 output_dir/lora_adapter/
        train_adapter_dir = output_dir / "train" / "lora_adapter"
        train_merged_dir = output_dir / "train" / "merged"
        if train_adapter_dir.exists():
            # 软链到 output_dir/lora_adapter
            target = Path(adapter)
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(train_adapter_dir.resolve())
            print(f"[v] LoRA adapter symlinked: {target} → {train_adapter_dir}")
            adapter = str(target)
        elif train_merged_dir.exists():
            # 退回使用 merged 的同等目录
            target = Path(adapter)
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(train_merged_dir.resolve())
            adapter = str(target)
            print(f"[!] 未发现 lora_adapter，用 merged 作 fallback: {adapter}")
        else:
            print(f"[X] 训练未产出 adapter 或 merged：{train_adapter_dir} / {train_merged_dir}")
            return 1

    # ----- 阶段 3: eval（base vs lora） -----
    if "eval" in stages:
        # 必须有 baseline + adapter
        if not baseline_path.exists():
            print(f"[X] eval 阶段需要 baseline_score.json：{baseline_path}")
            return 1
        if not Path(adapter).exists():
            print(f"[X] eval 阶段需要 LoRA adapter：{adapter}")
            return 1

        print("\n[阶段 3/3] Base vs LoRA 对比评估")
        # 跑一次 eval，但 out-dir 不能是 eval_dir（否则覆盖 baseline）
        compare_dir = output_dir / "compare"
        run_step([
            sys.executable, str(EVAL_SCRIPT),
            "--base-model", args.base_model,
            "--adapter", adapter,
            "--test-data", str(test_data),
            "--out-dir", str(compare_dir),
            "--max-new-tokens", str(args.max_new_tokens),
            "--judge", args.judge,
            "--judge-model", args.judge_model,
        ])
        # 对照 baseline
        compare_score = compare_dir / "baseline_score.json"
        if compare_score.exists():
            b = json.loads(baseline_path.read_text(encoding="utf-8"))
            c = json.loads(compare_score.read_text(encoding="utf-8"))
            print()
            print("=" * 70)
            print("基座 vs LoRA composite 对比（train 后 vs train 前）")
            print("=" * 70)
            key = "composite_mean"
            before = b.get("metrics", {}).get(key, "-")
            after_train = c.get("metrics", {}).get(key, "-")  # 这次跑的是 LoRA 的 composite
            print(f"  baseline.composite_mean (训练前)        = {before}")
            print(f"  LoRA.composite_mean (训练后)             = {after_train}")

    print()
    print("=" * 70)
    print(f"[OK] 全部完成 stages={stages}")
    print(f"   output_dir: {output_dir}")
    print(f"   eval_dir:   {eval_dir}")
    print(f"   adapter:    {adapter}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
