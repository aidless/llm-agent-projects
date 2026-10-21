#!/usr/bin/env python3
"""
数据集审计、清洗与划分工具（不依赖第三方包）。

输入:
  --input  数据 JSONL 路径
  --out-dir  输出目录
  --seed  随机种子（默认 42）
  --train-ratio / --val-ratio / --test-ratio  划分比例（默认 0.8/0.1/0.1）
  --max-inst-chars  指令超长阈值
  --max-out-chars  回答超长阈值
  --min-inst-chars  指令过短阈值
  --min-out-chars  回答过短阈值
  --exact-window  精确重复窗口（行内连续 n-gram）
  --ngram-size  用于近重复检测的 n-gram 大小
  --code-min-len  视为代码块的最小长度

输出:
  <out-dir>/dataset_audit.json     审计统计
  <out-dir>/dataset_audit.md       审计报告（人类可读）
  <out-dir>/cleaned.jsonl          通过清洗的数据
  <out-dir>/quarantine.jsonl       被剔除/标记为可疑的样本
  <out-dir>/train.jsonl
  <out-dir>/val.jsonl
  <out-dir>/test.jsonl
  <out-dir>/splits.json            划分元信息（用于复现）

设计原则:
  1. 完全确定性（只用 Python 标准库 + hash）
  2. 不调用任何外部 LLM，不联网
  3. 只做明显可观测的问题；细节由人工 review
  4. 代码块按 markdown ``` ``` 提取并尝试 ast.parse
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------- 数据读取 -----------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                # 单行 JSON 解析失败 → 计入 quarantine，不让整个审计崩溃
                items.append({"_bad_json": True, "_line": ln, "_err": str(e), "_raw": raw[:200]})
                continue
            if not isinstance(obj, dict):
                # 兼容 [instruction, output] 这种 2 元素列表写法
                if isinstance(obj, list) and len(obj) == 2 and all(isinstance(x, str) for x in obj):
                    items.append({"instruction": obj[0], "output": obj[1]})
                    continue
                items.append({"_bad_json": True, "_line": ln, "_err": "top-level not dict", "_raw": raw[:200]})
                continue
            items.append(obj)
    return items


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------- 启发式检查 -----------------------------

CODE_FENCE = re.compile(r"```([a-zA-Z0-9_+\-]*)\n(.*?)```", re.DOTALL)
INLINE_BACKTICK = re.compile(r"`([^`\n]+)`")

# 看似精确却无法核验的“硬指标”：
# 例如"召回率从 0.72 提升至 0.89"、"提升 15-25%"、"性能提升 10-100 倍"等。
HARD_METRIC_PATTERNS = [
    re.compile(r"(提升|提高|增长|降低|减少|提升到|增加到)\s*\d{1,3}\s*[%％xX倍]"),
    re.compile(r"\b\d+(\.\d+)?\s*%"),
    re.compile(r"\bfrom\s+\d+(\.\d+)?\s+to\s+\d+(\.\d+)?\b"),
    re.compile(r"\b\d+(\.\d+)?\s*-\s*\d+(\.\d+)?\s*(?:%|percent|个百分点|倍)"),
    re.compile(r"(提升|提高|增长|降低|减少|节约|节省)\s*\d+(\.\d+)?\s*(?:倍|个百分点|%)?"),
]

# 高风险句式（绝对化、未限定条件、可能未经实证）
ABSOLUTIST_PATTERNS = [
    re.compile(r"永远\s*(不会|是|是最好|是唯一)"),
    re.compile(r"(绝对|100%|百分百|完全|彻底)\s*(正确|无误|安全|可靠|可信赖)"),
    re.compile(r"没有任何\s*(风险|问题|副作用|缺陷|漏洞)"),
    re.compile(r"最佳实践(?!之一|是)"),
]

# 模板化信号
TEMPLATE_PREFIXES = [
    "**",
    "首先",
    "接下来",
    "1.",
    "1）",
    "1、",
]

URL_PATTERN = re.compile(r"https?://[^\s)）\]】]+")
URL_PATTERN = re.compile(r"https?://[^\s)）\]】]+")


def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """返回 [(language, code), ...]。"""
    return [(m.group(1) or "", m.group(2)) for m in CODE_FENCE.finditer(text)]


def try_parse_code(lang: str, code: str) -> Optional[str]:
    """返回错误信息；成功返回 None。仅对 Python/SQL 试解析。"""
    if lang.lower() in ("python", "py", "python3"):
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return f"python SyntaxError: {e.msg} (line {e.lineno})"
        except ValueError as e:
            return f"python ValueError: {e}"
    return None  # 其他语言跳过


def count_chinese_chars(s: str) -> int:
    return sum(1 for c in s if "\u4e00" <= c <= "\u9fff")


def normalize_for_dup(s: str) -> str:
    """用于近重复：去掉空白和标点，降低表面差异。"""
    s = re.sub(r"\s+", "", s)
    # 全角+半角标点统一去除
    s = re.sub(r"[\s,，.。!?？;；:：、\"'“”‘’()()（）【】\[\]<>\《》`~@#￥$%^&*+=/\\|_\-—–]+", "", s)
    return s


def ngrams(s: str, n: int) -> set:
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


# ----------------------------- 审计主流程 -----------------------------

def audit(
    rows: List[Dict[str, Any]],
    *,
    max_inst_chars: int,
    max_out_chars: int,
    min_inst_chars: int,
    min_out_chars: int,
    ngram_size: int,
) -> Dict[str, Any]:
    """对每行做四类检查：
      1) schema / length 硬校验（hard quarantine）
      2) 代码块可解析性（hard quarantine）
      3) 精确 / 近重复（hard quarantine）
      4) 硬指标 / 绝对化 / URL（soft warn，不剔除）

    返回值包含两组计数：hard_quarantine 与 soft_warn。
    """
    stats: Dict[str, Any] = {
        "total": len(rows),
        "schema_ok": 0,
        "schema_fail": 0,
        "too_long": 0,
        "too_short": 0,
        "code_block_count": 0,
        "code_parse_fail": 0,
        "hard_metric_count": 0,
        "absolutist_count": 0,
        "url_count": 0,
        "duplicates_exact": 0,
        "duplicates_near": 0,
        "issues": Counter(),
        "fields_missing": Counter(),
        "char_len_inst": [],
        "char_len_out": [],
        "lang_distribution": Counter(),
    }
    seen_exact: Dict[str, int] = {}
    seen_norm: Dict[str, Tuple[str, set]] = {}  # idx -> (norm_text, grams_set)
    quarantine_reasons: List[Tuple[int, List[str]]] = []  # hard quarantine
    soft_warn: List[Tuple[int, List[str]]] = []  # soft warn（保留）
    keep: List[int] = []
    near_pairs: List[Tuple[int, int]] = []

    for i, row in enumerate(rows):
        hard: List[str] = []
        soft: List[str] = []

        if row.get("_bad_json"):
            stats["schema_fail"] += 1
            quarantine_reasons.append((i, [f"bad_json:{row.get('_err','')}"]))
            continue

        if "instruction" not in row or "output" not in row:
            stats["schema_fail"] += 1
            for k in ("instruction", "output", "input"):
                if k not in row:
                    stats["fields_missing"][k] += 1
            quarantine_reasons.append((i, ["schema_missing"]))
            continue
        stats["schema_ok"] += 1

        inst = row.get("instruction") or ""
        out = row.get("output") or ""
        if not isinstance(inst, str) or not isinstance(out, str):
            quarantine_reasons.append((i, ["non_string_fields"]))
            continue

        inst_len = len(inst)
        out_len = len(out)
        stats["char_len_inst"].append(inst_len)
        stats["char_len_out"].append(out_len)

        if inst_len > max_inst_chars or out_len > max_out_chars:
            stats["too_long"] += 1
            hard.append("too_long")
        if inst_len < min_inst_chars or out_len < min_out_chars:
            stats["too_short"] += 1
            hard.append("too_short")

        codes = extract_code_blocks(out) + extract_code_blocks(inst)
        if codes:
            stats["code_block_count"] += len(codes)
            for lang, code in codes:
                stats["lang_distribution"][lang or "unspecified"] += 1
                err = try_parse_code(lang, code)
                if err:
                    stats["code_parse_fail"] += 1
                    hard.append(f"code_parse_fail:{err[:80]}")

        # 硬指标 → soft warn
        hit_hard_metric = False
        for pat in HARD_METRIC_PATTERNS:
            if pat.search(out + "\n" + inst):
                hit_hard_metric = True
                break
        if hit_hard_metric:
            stats["hard_metric_count"] += 1
            stats["issues"]["hard_metric"] += 1
            soft.append("hard_metric")

        # 绝对化 → soft warn
        hit_abs = False
        for pat in ABSOLUTIST_PATTERNS:
            if pat.search(out + "\n" + inst):
                hit_abs = True
                break
        if hit_abs:
            stats["absolutist_count"] += 1
            stats["issues"]["absolutist"] += 1
            soft.append("absolutist")

        # URL → soft warn
        if URL_PATTERN.search(out + "\n" + inst):
            stats["url_count"] += 1
            stats["issues"]["url"] += 1
            soft.append("url")

        # 重复 → hard quarantine
        h = hashlib.sha256((inst + "\n<<>>\n" + out).encode("utf-8")).hexdigest()
        if h in seen_exact:
            stats["duplicates_exact"] += 1
            stats["issues"]["duplicate_exact"] += 1
            hard.append("duplicate_exact")
        else:
            seen_exact[h] = i
            norm = normalize_for_dup(inst + " " + out)
            grams = ngrams(norm, ngram_size)
            match_idx = None
            for prev_idx, (prev_text, prev_grams) in seen_norm.items():
                if not prev_text or not norm:
                    continue
                inter = len(grams & prev_grams)
                union = len(grams | prev_grams) or 1
                if inter * 1.0 / union >= 0.85:
                    match_idx = prev_idx
                    break
            if match_idx is not None:
                stats["duplicates_near"] += 1
                stats["issues"]["duplicate_near"] += 1
                hard.append("duplicate_near")
                near_pairs.append((match_idx, i))
            else:
                seen_norm[i] = (norm, grams)

        if hard:
            quarantine_reasons.append((i, hard))
        else:
            keep.append(i)
            if soft:
                soft_warn.append((i, soft))

    # summary stats
    if stats["char_len_inst"]:
        stats["char_len_inst"] = {
            "min": min(stats["char_len_inst"]),
            "max": max(stats["char_len_inst"]),
            "mean": sum(stats["char_len_inst"]) / len(stats["char_len_inst"]),
        }
    if stats["char_len_out"]:
        stats["char_len_out"] = {
            "min": min(stats["char_len_out"]),
            "max": max(stats["char_len_out"]),
            "mean": sum(stats["char_len_out"]) / len(stats["char_len_out"]),
        }
    stats["lang_distribution"] = dict(stats["lang_distribution"])
    stats["issues"] = dict(stats["issues"])
    stats["fields_missing"] = dict(stats["fields_missing"])
    stats["keep"] = len(keep)
    stats["quarantine"] = len(quarantine_reasons)
    stats["quarantine_reasons"] = [
        {"index": i, "reasons": r} for (i, r) in quarantine_reasons
    ]
    stats["soft_warn_count"] = len(soft_warn)
    stats["soft_warn"] = [
        {"index": i, "reasons": r} for (i, r) in soft_warn
    ]
    stats["near_pairs_sample"] = near_pairs[:20]
    return stats


# ----------------------------- 划分 -----------------------------

def split_indices(
    n: int,
    *,
    train: float,
    val: float,
    test: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    # remainder goes to test
    n_test = n - n_train - n_val
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


# ----------------------------- 报告 -----------------------------

def render_markdown(stats: Dict[str, Any], out_dir: Path) -> str:
    lines: List[str] = []
    lines.append("# 06-lora-finetune 数据集审计报告")
    lines.append("")
    lines.append(f"- 总样本数: **{stats['total']}**")
    lines.append(f"- Schema 通过: {stats['schema_ok']}")
    lines.append(f"- Schema 失败: {stats['schema_fail']}")
    lines.append(f"- 保留: **{stats['keep']}**")
    lines.append(f"- 进入 hard quarantine: **{stats['quarantine']}**")
    lines.append(f"- Soft 警告（保留，仅标记）: **{stats.get('soft_warn_count', 0)}**")
    lines.append("")
    lines.append("## 长度统计（字符数）")
    inst = stats.get("char_len_inst", {}) or {}
    out_ = stats.get("char_len_out", {}) or {}
    lines.append("| 字段 | min | mean | max |")
    lines.append("|---|---|---|---|")
    lines.append(f"| instruction | {inst.get('min','-')} | {inst.get('mean','-'):.1f} | {inst.get('max','-')} |")
    lines.append(f"| output | {out_.get('min','-')} | {out_.get('mean','-'):.1f} | {out_.get('max','-')} |")
    lines.append("")
    lines.append("## 风险计数")
    lines.append(f"- 超长样本 (hard): {stats['too_long']}")
    lines.append(f"- 过短样本 (hard): {stats['too_short']}")
    lines.append(f"- 代码块: {stats['code_block_count']}（解析失败: {stats['code_parse_fail']}）")
    lines.append(f"- 含硬指标 (soft): {stats['hard_metric_count']}")
    lines.append(f"- 绝对化措辞 (soft): {stats['absolutist_count']}")
    lines.append(f"- 含 URL (soft): {stats['url_count']}")
    lines.append(f"- 精确重复 (hard): {stats['duplicates_exact']}")
    lines.append(f"- 近重复 (n-gram Jaccard >= 0.85, hard): {stats['duplicates_near']}")
    lines.append("")
    if stats.get("lang_distribution"):
        lines.append("## 代码块语言分布")
        for k, v in sorted(stats["lang_distribution"].items(), key=lambda x: -x[1]):
            lines.append(f"- `{k or 'unspecified'}`: {v}")
        lines.append("")
    if stats.get("quarantine_reasons"):
        lines.append("## Hard Quarantine 样本（前 30 条）")
        for item in stats["quarantine_reasons"][:30]:
            lines.append(f"- idx={item['index']}: {item['reasons']}")
        lines.append("")
    if stats.get("soft_warn"):
        lines.append("## Soft 警告分布（前 20 条）")
        for item in stats["soft_warn"][:20]:
            lines.append(f"- idx={item['index']}: {item['reasons']}")
        lines.append("")
    if stats.get("near_pairs_sample"):
        lines.append("## 近重复样本对（样例 20 组）")
        for a, b in stats["near_pairs_sample"]:
            lines.append(f"- {a} ↔ {b}")
        lines.append("")
    lines.append("## 训练门槛（在本机 RTX 3060 6GB 下）")
    lines.append("- 不建议直接训练 7B（OOM 风险高）。优先 3B + QLoRA + seq_len 1024 + batch 1 + grad_accum 8。")
    lines.append("- 当前数据量（≈300）建议 epochs=2，配合早期停止；val 集用于 loss 监控，不参与早停阈值。")
    lines.append("- 必须保持 test 集完全隔离，训练 / 调参 / prompt 调整都不能查看 test。")
    lines.append("- 即使训练后报告 loss 下降，也必须人工抽 20 条对比原始 vs 微调模型。")
    lines.append("")
    lines.append("## 后续动作")
    lines.append("1. 复核 hard quarantine 列表（代码解析失败、精确/近重复、超长）。")
    lines.append("2. 抽 30 条 soft warn 样本人工复核硬指标与绝对化措辞，必要时剔除。")
    lines.append("3. 确认 train/val/test 划分不重复（用 `splits.json` 复现）。")
    md = "\n".join(lines)
    (out_dir / "dataset_audit.md").write_text(md, encoding="utf-8")
    return md


# ----------------------------- CLI -----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="LoRA 数据集审计 + 划分")
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--max-inst-chars", type=int, default=4000)
    p.add_argument("--max-out-chars", type=int, default=8000)
    p.add_argument("--min-inst-chars", type=int, default=8)
    p.add_argument("--min-out-chars", type=int, default=20)
    p.add_argument("--ngram-size", type=int, default=12)
    p.add_argument("--code-min-len", type=int, default=12)
    args = p.parse_args(argv)

    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(inp)
    stats = audit(
        rows,
        max_inst_chars=args.max_inst_chars,
        max_out_chars=args.max_out_chars,
        min_inst_chars=args.min_inst_chars,
        min_out_chars=args.min_out_chars,
        ngram_size=args.ngram_size,
    )
    # 写 audit.json
    (out_dir / "dataset_audit.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # 写报告
    render_markdown(stats, out_dir)

    # 写 cleaned / quarantine
    keep_idx = set(
        i for i in range(len(rows)) if i not in {q["index"] for q in stats["quarantine_reasons"]}
    )
    cleaned = [r for i, r in enumerate(rows) if i in keep_idx]
    quarantined = [r for i, r in enumerate(rows) if i not in keep_idx]
    write_jsonl(out_dir / "cleaned.jsonl", cleaned)
    write_jsonl(out_dir / "quarantine.jsonl", quarantined)

    # 划分 cleaned 集
    tr, va, te = split_indices(
        len(cleaned),
        train=args.train_ratio,
        val=args.val_ratio,
        test=args.test_ratio,
        seed=args.seed,
    )
    train_rows = [cleaned[i] for i in tr]
    val_rows = [cleaned[i] for i in va]
    test_rows = [cleaned[i] for i in te]
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)
    (out_dir / "splits.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "train_size": len(train_rows),
                "val_size": len(val_rows),
                "test_size": len(test_rows),
                "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({
        "total": stats["total"],
        "keep": stats["keep"],
        "quarantine": stats["quarantine"],
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "duplicates_exact": stats["duplicates_exact"],
        "duplicates_near": stats["duplicates_near"],
        "code_parse_fail": stats["code_parse_fail"],
        "hard_metric": stats["hard_metric_count"],
        "absolutist": stats["absolutist_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
