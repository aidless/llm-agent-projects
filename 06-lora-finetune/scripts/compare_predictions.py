#!/usr/bin/env python3
"""
预测对比辅助脚本：从 base_answers.jsonl + lora_answers.jsonl 抽取若干样本并排打印，
方便人工对比 base vs LoRA 表现。无需 GPU，只需两个 JSONL 文件。

用法：
  python scripts/compare_predictions.py \\
      --base output/v1/eval/base_answers.jsonl \\
      --lora output/v1/compare/lora_answers.jsonl \\
      --output output/v1/compare/compare_sample.md \\
      --n 5

或只对比 base（看 base 模型自身表现）：
  python scripts/compare_predictions.py \\
      --base output/v1/eval/base_answers.jsonl \\
      --n 5

或只对比 LoRA：
  python scripts/compare_predictions.py \\
      --base output/v1/eval/base_answers.jsonl \\
      --lora output/v1/compare/lora_answers.jsonl \\
      --base-only   # 不显示 base，只看 lora + gold
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_answers(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"\n... [truncated, total {len(s)} chars]"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="base 答案 JSONL")
    p.add_argument("--lora", default=None, help="LoRA 答案 JSONL（可选）")
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=5, help="抽几个样本")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", type=int, default=0, help="从第几条开始")
    args = p.parse_args()

    base_rows = load_answers(Path(args.base))
    lora_rows = load_answers(Path(args.lora)) if args.lora else None

    if lora_rows and len(lora_rows) != len(base_rows):
        print(f"[!] base 和 lora 行数不一致: {len(base_rows)} vs {len(lora_rows)}")

    # 选 sample
    import random
    rng = random.Random(args.seed)
    if args.n >= len(base_rows):
        sample_idx = list(range(len(base_rows)))
    else:
        sample_idx = rng.sample(range(len(base_rows)), args.n)
    sample_idx.sort()

    lines: List[str] = []
    lines.append("# Base vs LoRA 预测对比样本")
    lines.append("")
    lines.append(f"- base: `{args.base}` (n={len(base_rows)})")
    if lora_rows:
        lines.append(f"- lora: `{args.lora}` (n={len(lora_rows)})")
    lines.append(f"- 抽样: {len(sample_idx)} 条 (seed={args.seed})")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i in sample_idx:
        if i >= len(base_rows):
            continue
        b = base_rows[i]
        gold = b.get("gold", "")
        lines.append(f"## Sample {i}")
        lines.append("")
        lines.append(f"### Q: {b['instruction']}")
        lines.append("")
        lines.append(f"### Gold (参考答案):")
        lines.append("```")
        lines.append(truncate(gold, 1200))
        lines.append("```")
        lines.append("")
        if lora_rows and i < len(lora_rows):
            l = lora_rows[i]
            lines.append(f"### Base answer (前 800 字):")
            lines.append("```")
            lines.append(truncate(b.get("answer", ""), 800))
            lines.append("```")
            lines.append("")
            lines.append(f"### LoRA answer (前 800 字):")
            lines.append("```")
            lines.append(truncate(l.get("answer", ""), 800))
            lines.append("```")
            lines.append("")
            # 简单 diff
            ba = b.get("answer", "")
            la = l.get("answer", "")
            lines.append(f"**长度**: Base={len(ba)} chars, LoRA={len(la)} chars, Gold={len(gold)} chars")
            lines.append("")
        else:
            lines.append(f"### Answer (前 800 字):")
            lines.append("```")
            lines.append(truncate(b.get("answer", ""), 800))
            lines.append("```")
            lines.append("")
            lines.append(f"**长度**: {len(b.get('answer',''))} chars (Gold: {len(gold)} chars)")
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## 人工评分建议")
    lines.append("")
    lines.append("逐条对比 Base vs LoRA 在以下维度：")
    lines.append("")
    lines.append("1. **事实正确性** —— 哪个答案更准？")
    lines.append("2. **相关性** —— 是否答非所问？")
    lines.append("3. **完整性** —— 是否覆盖关键点？")
    lines.append("4. **简洁性** —— 是否太啰嗦？")
    lines.append("5. **格式** —— 是否含可读代码 / 列表？")
    lines.append("")
    lines.append("对每条给一个 verdict：`base_wins` / `tie` / `lora_wins`，汇总到末尾。")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Samples: {len(sample_idx)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
