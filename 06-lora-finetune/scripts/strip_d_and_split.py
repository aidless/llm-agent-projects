#!/usr/bin/env python3
"""
从 cleaned.jsonl 剔除 D 级样本（占位符），重新做 train/val/test 划分。

输入：cleaned.jsonl
输出：
  train.jsonl / val.jsonl / test.jsonl（固定 seed=42 可复现）
  splits.json
  quarantine_d.jsonl（被剔除的 D 级样本供 review）

复用 quality_sample.grade_sample 但只保留 A/B/C。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from quality_sample import grade_sample


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--exclude-grades", default="D", help="逗号分隔的排除等级")
    args = p.parse_args()

    exclude = set(g.strip().upper() for g in args.exclude_grades.split(","))
    rows: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list) and len(obj) == 2:
                obj = {"instruction": obj[0], "output": obj[1]}
            rows.append(obj)

    print(f"Input: {len(rows)} rows")
    keep, drop = [], []
    for r in rows:
        g = grade_sample(r)["grade"]
        if g in exclude:
            drop.append(r)
        else:
            keep.append(r)
    print(f"Kept: {len(keep)}, Dropped ({','.join(exclude)}): {len(drop)}")

    rng = random.Random(args.seed)
    idx = list(range(len(keep)))
    rng.shuffle(idx)
    n_train = int(round(len(keep) * args.train_ratio))
    n_val = int(round(len(keep) * args.val_ratio))
    n_test = len(keep) - n_train - n_val
    train = [keep[i] for i in idx[:n_train]]
    val = [keep[i] for i in idx[n_train:n_train + n_val]]
    test = [keep[i] for i in idx[n_train + n_val:]]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in [("train.jsonl", train), ("val.jsonl", val), ("test.jsonl", test), ("quarantine_d.jsonl", drop)]:
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "splits.json").write_text(json.dumps({
        "seed": args.seed,
        "input_size": len(rows),
        "kept_size": len(keep),
        "dropped_size": len(drop),
        "excluded_grades": sorted(exclude),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print(f"Wrote to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
