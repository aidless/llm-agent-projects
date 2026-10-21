#!/usr/bin/env python3
"""
从 cleaned.jsonl 抽 N 条做客观质量评级。

评级完全基于客观特征（不靠 LLM-judge）：
  A — 含代码示例 + 长度合适（200-1500 字）+ 无硬指标
  B — 长度合适 + 含 1-2 项特征（短代码/技术细节）
  C — 含硬指标或"百科式"长答案（>2000 字）但仍可用
  D — 含 "TODO"/"...略"/"省略"占位符 / 模板化开头

输出：data/audit/2026-07-22-v2/quality_sample.md
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


CODE_FENCE = re.compile(r"```[a-zA-Z0-9_+\-]*\n(.*?)```", re.DOTALL)
HARD_METRIC = re.compile(r"(提升|提高|增长|降低|减少|提升到|增加到)\s*\d{1,3}\s*[%％xX倍]|\b\d+(\.\d+)?\s*%")
ABSOLUTIST = re.compile(r"永远\s*(不会|是|最好|唯一)|(100%|百分百|绝对|彻底)\s*(正确|可靠|无误|安全)|没有任何\s*(风险|问题|缺陷)")
PLACEHOLDER = re.compile(r"(TODO|FIXME|placeholder|占位|此处略|代码略|实际代码略|同上略|XX:|YY:|实现见|FIXME)")
TEMPLATE_PREFIX = re.compile(r"^(\*\*|首先|接下来|1\.|2\.|3\.|1\)|1、)")


def grade_sample(row: Dict[str, Any]) -> Dict[str, Any]:
    out = row.get("output", "")
    inst = row.get("instruction", "")
    n_chars = len(out)
    has_code = bool(CODE_FENCE.search(out))
    n_code_blocks = len(CODE_FENCE.findall(out))
    has_metric = bool(HARD_METRIC.search(out + "\n" + inst))
    has_absolutist = bool(ABSOLUTIST.search(out + "\n" + inst))
    has_placeholder = bool(PLACEHOLDER.search(out))
    template_start = bool(TEMPLATE_PREFIX.match(out.lstrip()))

    # 评级：A/B/C/D
    if has_placeholder:
        grade = "D"
        reason = "含占位符（TODO/...略/省略）"
    elif n_chars > 5000:
        grade = "C"
        reason = f"过长（{n_chars} 字符），可能百科式而非针对性回答"
    elif n_chars > 2500 and not has_code:
        grade = "C"
        reason = f"较长（{n_chars} 字符）但无代码示例，疑似百科式"
    elif has_code and n_code_blocks >= 1 and 200 <= n_chars <= 1500 and not has_metric:
        grade = "A"
        reason = "含代码 + 长度合适 + 无硬指标"
    elif has_code and 200 <= n_chars <= 2500 and not has_metric:
        grade = "B"
        reason = "含代码 + 长度合适"
    elif 100 <= n_chars <= 1500:
        grade = "B"
        reason = "长度合适"
    else:
        grade = "C"
        reason = f"其他 (chars={n_chars}, code={has_code})"

    if has_absolutist:
        reason += "；含绝对化措辞（需人工确认）"
    if has_metric:
        reason += "；含硬指标（需人工确认）"
    if template_start and n_chars > 800:
        reason += "；模板化开头 + 较长"

    return {
        "n_chars": n_chars,
        "n_code_blocks": n_code_blocks,
        "has_metric": has_metric,
        "has_absolutist": has_absolutist,
        "has_placeholder": has_placeholder,
        "template_start": template_start,
        "grade": grade,
        "reason": reason,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

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

    rng = random.Random(args.seed)
    sample_idx = rng.sample(range(len(rows)), min(args.n, len(rows)))

    lines: List[str] = []
    lines.append(f"# Cleaned 数据集质量抽检报告")
    lines.append("")
    lines.append(f"- 数据源: `{args.input}`")
    lines.append(f"- 总条数: {len(rows)}")
    lines.append(f"- 抽样: {len(sample_idx)} 条 (seed={args.seed})")
    lines.append("")

    # 整体统计
    grade_dist: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
    has_metric_count = 0
    has_absolutist_count = 0
    has_placeholder_count = 0
    has_code_count = 0
    for r in rows:
        g = grade_sample(r)
        grade_dist[g["grade"]] += 1
        if g["has_metric"]:
            has_metric_count += 1
        if g["has_absolutist"]:
            has_absolutist_count += 1
        if g["has_placeholder"]:
            has_placeholder_count += 1
        if g["n_code_blocks"] >= 1:
            has_code_count += 1
    lines.append("## 全数据集客观特征")
    lines.append("")
    lines.append("| 特征 | 数量 | 占比 |")
    lines.append("|---|---|---|")
    n = len(rows)
    lines.append(f"| 含代码块 | {has_code_count} | {has_code_count/n*100:.1f}% |")
    lines.append(f"| 含硬指标 | {has_metric_count} | {has_metric_count/n*100:.1f}% |")
    lines.append(f"| 含绝对化措辞 | {has_absolutist_count} | {has_absolutist_count/n*100:.1f}% |")
    lines.append(f"| 含占位符 | {has_placeholder_count} | {has_placeholder_count/n*100:.1f}% |")
    lines.append("")
    lines.append("## 全数据集评级分布（基于启发式）")
    lines.append("")
    lines.append("| 评级 | 数量 | 占比 | 含义 |")
    lines.append("|---|---|---|---|")
    grade_meaning = {
        "A": "含代码 + 长度合适 + 无硬指标——可直接用于训练",
        "B": "长度合适，可用于训练",
        "C": "较长或疑似百科式——可训练但可能含噪声",
        "D": "含占位符/代码省略——必须人工 review 或剔除",
    }
    for g in ["A", "B", "C", "D"]:
        c = grade_dist[g]
        lines.append(f"| {g} | {c} | {c/n*100:.1f}% | {grade_meaning[g]} |")
    lines.append("")
    lines.append("## 抽样样本")
    lines.append("")
    for i in sample_idx:
        r = rows[i]
        g = grade_sample(r)
        lines.append(f"### idx={i} | 评级: **{g['grade']}**")
        lines.append(f"- 字符数: {g['n_chars']}, 代码块: {g['n_code_blocks']}")
        lines.append(f"- 含硬指标: {g['has_metric']}, 含绝对化: {g['has_absolutist']}, 含占位符: {g['has_placeholder']}, 模板开头: {g['template_start']}")
        lines.append(f"- 评级理由: {g['reason']}")
        lines.append(f"")
        lines.append(f"**Q**: {r['instruction'][:120]}{'...' if len(r['instruction'])>120 else ''}")
        lines.append(f"")
        lines.append(f"**A (前 400 字)**: {r['output'][:400]}{'...' if len(r['output'])>400 else ''}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## 结论与建议")
    lines.append("")
    d_pct = grade_dist["D"] / n * 100
    c_pct = grade_dist["C"] / n * 100
    ab_pct = (grade_dist["A"] + grade_dist["B"]) / n * 100
    lines.append(f"- A+B 占比 **{ab_pct:.1f}%**：可直接训练")
    lines.append(f"- C 占比 **{c_pct:.1f}%**：建议训练后看 loss 是否被这些噪声影响")
    lines.append(f"- D 占比 **{d_pct:.1f}%**：必须剔除（占位符会让模型学会说'省略'）")
    lines.append("")
    if d_pct >= 5.0:
        lines.append("> **强建议**：先执行修复脚本移除 D 评级样本再训练，否则 LoRA 会学会用 'TODO'/'...' 回答问题")
    else:
        lines.append("> D 样本已 < 5%，可以继续训练但建议抽样复核")

    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Total: {n}")
    print(f"Grade distribution: A={grade_dist['A']} B={grade_dist['B']} C={grade_dist['C']} D={grade_dist['D']}")
    print(f"D 占比: {d_pct:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
