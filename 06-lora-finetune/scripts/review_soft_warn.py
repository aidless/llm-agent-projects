#!/usr/bin/env python3
"""
对 soft_warn 样本做启发式分类，给人工复核一个有序清单。

输入：dataset_audit.json 中的 soft_warn 列表
输出：data/audit/2026-07-22-v2/soft_warn_review.md（人类可读复核清单）
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def categorize(reasons: List[str], text: str) -> str:
    """给一个 soft warn 样本打一个复核优先级。"""
    text_low = text.lower()
    # 优先级 1（必须复核）：硬指标 + 绝对化同时出现
    if "absolutist" in reasons and "hard_metric" in reasons:
        return "P1_review"
    # 优先级 2：含百分比且明显"提升"等措辞
    if "hard_metric" in reasons:
        if any(p in text for p in ("提升", "提高", "增长", "降低", "减少", "增加到", "提升到")):
            return "P2_metric"
        return "P3_metric_mild"
    # 优先级 3：绝对化措辞
    if "absolutist" in reasons:
        if any(p in text for p in ("永远", "100%", "绝对", "彻底", "没有任何")):
            return "P2_absolutist_strong"
        return "P3_absolutist_mild"
    # URL
    if "url" in reasons:
        return "P3_url"
    return "P4_other"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--audit-dir", required=True)
    p.add_argument("--source", required=True, help="原始数据 JSONL，用于查看样本内容")
    args = p.parse_args()

    audit = json.loads(Path(args.audit_dir, "dataset_audit.json").read_text(encoding="utf-8"))
    src_rows: List[Dict[str, Any]] = []
    with open(args.source, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list) and len(obj) == 2:
                obj = {"instruction": obj[0], "output": obj[1]}
            src_rows.append(obj)

    soft = audit.get("soft_warn", [])
    by_priority: Dict[str, List[Dict[str, Any]]] = {}
    for w in soft:
        idx = w["index"]
        if idx >= len(src_rows):
            continue
        text = src_rows[idx].get("output", "") + "\n" + src_rows[idx].get("instruction", "")
        pri = categorize(w["reasons"], text)
        rec = {
            "idx": idx,
            "reasons": w["reasons"],
            "priority": pri,
            "instruction": src_rows[idx].get("instruction", "")[:80],
            "output_head": src_rows[idx].get("output", "")[:200],
        }
        by_priority.setdefault(pri, []).append(rec)

    # 输出 markdown
    lines = ["# Soft Warn 复核清单", ""]
    lines.append(f"**总计: {len(soft)} 条**")
    lines.append("")
    lines.append("| 优先级 | 数量 | 含义 |")
    lines.append("|---|---|---|")
    lines.append("| P1_review | %d | absolutist + hard_metric 同时出现 |" % len(by_priority.get("P1_review", [])))
    lines.append("| P2_metric | %d | 含百分比且有强提升措辞 |" % len(by_priority.get("P2_metric", [])))
    lines.append("| P2_absolutist_strong | %d | 含永远/100/绝对/彻底 |" % len(by_priority.get("P2_absolutist_strong", [])))
    lines.append("| P3_metric_mild | %d | 仅含百分比，无强措辞 |" % len(by_priority.get("P3_metric_mild", [])))
    lines.append("| P3_absolutist_mild | %d | 较弱绝对化措辞 |" % len(by_priority.get("P3_absolutist_mild", [])))
    lines.append("| P3_url | %d | 含外部 URL |" % len(by_priority.get("P3_url", [])))
    lines.append("")

    priority_order = ["P1_review", "P2_metric", "P2_absolutist_strong", "P3_metric_mild", "P3_absolutist_mild", "P3_url"]
    for pri in priority_order:
        items = by_priority.get(pri, [])
        if not items:
            continue
        lines.append(f"## {pri} ({len(items)} 条)")
        lines.append("")
        for it in items[:50]:
            lines.append(f"### idx={it['idx']}  reasons={it['reasons']}")
            lines.append(f"**问**: {it['instruction']}")
            lines.append("")
            lines.append(f"**答（前 200 字）**: {it['output_head']}")
            lines.append("")
            lines.append("**决策建议**: ")
            if pri == "P1_review":
                lines.append("- 必查。如为合理描述，保留；如为夸大表述，剔除。")
            elif pri == "P2_metric":
                lines.append("- 查具体数字是否有可证实出处。无则建议删。")
            elif pri == "P2_absolutist_strong":
                lines.append("- 查是否限定了条件。限定则可保留；无限定则改写或删。")
            elif pri.startswith("P3"):
                lines.append("- 可选。批量扫一眼，无明显问题可全部保留。")
            lines.append("")
        if len(items) > 50:
            lines.append(f"_（仅展示前 50 条，共 {len(items)} 条）_")
        lines.append("")

    out_path = Path(args.audit_dir, "soft_warn_review.md")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Total: {len(soft)} | "
          f"P1={len(by_priority.get('P1_review',[]))} "
          f"P2={len(by_priority.get('P2_metric',[])) + len(by_priority.get('P2_absolutist_strong',[]))} "
          f"P3={len(by_priority.get('P3_metric_mild',[])) + len(by_priority.get('P3_absolutist_mild',[])) + len(by_priority.get('P3_url',[]))}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
