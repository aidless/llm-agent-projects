#!/usr/bin/env python3
"""
Base vs LoRA 对比评估脚本。

输入：
  --base-model  HuggingFace 模型 ID 或本地路径（基座）
  --adapter     可选：LoRA adapter 路径（若提供则启用对比）
  --test-data   JSONL，含 instruction/output 字段
  --out-dir     输出目录

输出：
  out_dir/base_answers.jsonl      基座模型对 test 的回答
  out_dir/lora_answers.jsonl      LoRA 模型对 test 的回答（若提供 adapter）
  out_dir/eval_report.md          评分报告

评分维度（启发式，不依赖 LLM-judge）：
  1. 长度比：answer / gold 长度（<0.3 过短，>3.0 过长）
  2. 关键词覆盖：gold 答案中的核心词在 answer 中出现比例
  3. 中文占比（应 >= 0.5）
  4. 代码块语法（Python 块是否能 ast.parse）
  5. 是否含明显硬指标
  6. win rate（如果提供了 --adapter）

LLM-judge 模式（可选，需 OPENAI_API_KEY）：
  --judge openai
  加 G-Eval 风格的五维评分（准确性/完整性/代码正确/相关性/简洁性）
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------- 启发式评分 -----------------------------

CODE_RE = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL)
CODE_FALLBACK = re.compile(r"```([a-zA-Z0-9_+\-]*)\n(.*?)```", re.DOTALL)
HARD_METRIC_RE = re.compile(r"(提升|提高|增长|降低|减少|提升到|增加到)\s*\d{1,3}\s*[%％xX倍]|\b\d+(\.\d+)?\s*%")

STOPWORDS = set("的了是就在和与或但而等一二三四五六七八九十百千万这那一个一种一些")  # 中文停用词


def chinese_ratio(s: str) -> float:
    if not s:
        return 0.0
    ch = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    return ch / max(1, len(s))


def extract_keywords(text: str, k: int = 12) -> List[str]:
    """从 gold 答案中抽取前 k 个关键词（基于字符 n-gram 频率）。"""
    if not text:
        return []
    # 1) 简单分词（按非中文字符切分）
    tokens = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
    # 2) 过滤停用词
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 2]
    # 3) 频次
    cnt = Counter(tokens)
    return [w for w, _ in cnt.most_common(k)]


def keyword_coverage(gold: str, ans: str) -> float:
    kws = extract_keywords(gold, 12)
    if not kws:
        return 0.0
    hit = sum(1 for k in kws if k in ans)
    return hit / len(kws)


def length_ratio(gold: str, ans: str) -> float:
    if not gold:
        return 1.0
    return len(ans) / max(1, len(gold))


def code_parseable(ans: str) -> Tuple[bool, List[str]]:
    """检测 answer 中所有 Python 代码块是否能 ast.parse。"""
    errs: List[str] = []
    blocks = CODE_RE.findall(ans)
    if not blocks:
        blocks = [m.group(2) for m in CODE_FALLBACK.finditer(ans)
                  if m.group(1).lower() in ("python", "py", "python3", "")]
    ok = True
    for b in blocks:
        try:
            ast.parse(b)
        except SyntaxError as e:
            ok = False
            errs.append(f"{e.msg} L{e.lineno}")
    return ok, errs


def has_hard_metric(text: str) -> bool:
    return bool(HARD_METRIC_RE.search(text))


def score_answer(gold: str, ans: str) -> Dict[str, Any]:
    cr = chinese_ratio(ans)
    lr = length_ratio(gold, ans)
    kc = keyword_coverage(gold, ans)
    cp, cerrs = code_parseable(ans)
    hm = has_hard_metric(ans)
    # 复合分（无真相上限，仅参考）
    length_penalty = 0.0
    if lr < 0.3:
        length_penalty = 0.3
    elif lr > 3.0:
        length_penalty = 0.2
    composite = 0.4 * kc + 0.2 * (1.0 if cr >= 0.5 else 0.0) + 0.2 * (1.0 if cp else 0.0) + 0.2 * (1.0 - length_penalty)
    return {
        "chinese_ratio": round(cr, 3),
        "length_ratio": round(lr, 3),
        "keyword_coverage": round(kc, 3),
        "code_parseable": cp,
        "code_errors": cerrs[:3],
        "has_hard_metric": hm,
        "composite": round(composite, 3),
    }


# ----------------------------- 模型调用 -----------------------------

def load_model_and_tokenizer(model_path: str, adapter_path: Optional[str] = None,
                              use_4bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": model_path,
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if use_4bit:
        try:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception as e:
            print(f"  4bit 失败（{e}），回退 fp16")
            kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"  Loaded LoRA adapter: {adapter_path}")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, instruction: str, max_new_tokens: int = 512,
             temperature: float = 0.1) -> str:
    import torch
    prompt = (
        f"<|im_start|>system\n你是一个有帮助的助手。<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.01,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # 截断到下一个 <|im_end|>
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
    return text.strip()


# ----------------------------- 主流程 -----------------------------

# ----------------------------- LLM-judge -----------------------------

JUDGE_SYSTEM = """你是严格的 LLM 输出评分员。按 JSON 格式输出 4 个 1-5 分的字段：
  accuracy   —— 信息是否事实正确
  relevance  —— 是否切题、回答了用户问题
  completeness —— 是否覆盖关键点（不含硬指标也可；只看是否答非所问/漏关键点）
  conciseness —— 是否简洁（如果回答质量高但太长，给 3 分；答非所问则 1）
只返回 JSON，不要其他文字。"""

JUDGE_USER_TPL = """[Question]
{instruction}
[Reference Answer]
{gold}
[Model Answer]
{answer}
[Output]"""


def call_llm_judge(judge_client, model_name: str, instruction: str,
                   gold: str, answer: str) -> Dict[str, Any]:
    """通过 OpenAI 兼容接口调用 LLM-judge。无 client 时返回 None。"""
    if judge_client is None:
        return None
    try:
        resp = judge_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_USER_TPL.format(
                    instruction=instruction, gold=gold, answer=answer)},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        # JSON 解析（容忍 markdown 包裹）
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n", "", text)
            text = re.sub(r"\n```$", "", text)
        data = json.loads(text)
        return {k: int(data.get(k, 3)) for k in ["accuracy", "relevance", "completeness", "conciseness"]}
    except Exception as e:
        return {"error": f"{e.__class__.__name__}: {str(e)[:80]}",
                "accuracy": 0, "relevance": 0, "completeness": 0, "conciseness": 0}


def get_judge_client(args) -> Optional[Any]:
    """构造 OpenAI 兼容 judge 客户端。无 key 时返回 None。"""
    try:
        import openai
    except ImportError:
        print("  [!] openai 包未安装，--judge=openai 已禁用（启发式评）")
        return None
    if args.judge == "off" or not args.judge:
        return None
    if args.judge == "openai":
        api_key = args.judge_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("  [!] OPENAI_API_KEY 未设置，--judge=openai 已禁用")
            return None
        client = openai.OpenAI(
            api_key=api_key,
            base_url=args.judge_base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        return client
    return None


# ----------------------------- 主流程 -----------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", default=None, help="LoRA adapter 路径（启用对比）")
    p.add_argument("--test-data", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=0, help="0 表示全部")
    p.add_argument("--no-4bit", action="store_true")
    # 新增
    p.add_argument("--baseline-only", action="store_true",
                   help="仅评估 base 模型（不加载 LoRA）")
    p.add_argument("--judge", default="off",
                   help="off | openai（默认 off，仅启发式；需 OPENAI_API_KEY）")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--judge-api-key", default=None)
    p.add_argument("--judge-base-url", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_data: List[Dict[str, Any]] = []
    with open(args.test_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list) and len(obj) == 2:
                obj = {"instruction": obj[0], "output": obj[1]}
            test_data.append(obj)
    if args.limit:
        test_data = test_data[:args.limit]
    print(f"test 样本: {len(test_data)}")

    # 基座生成
    print("=== 加载基座模型 ===")
    base_model, base_tok = load_model_and_tokenizer(args.base_model, adapter_path=None,
                                                     use_4bit=not args.no_4bit)
    base_results = []
    for i, ex in enumerate(test_data):
        ans = generate(base_model, base_tok, ex["instruction"], max_new_tokens=args.max_new_tokens)
        base_results.append({"idx": i, "instruction": ex["instruction"], "gold": ex["output"], "answer": ans})
        if (i + 1) % 5 == 0:
            print(f"  base {i+1}/{len(test_data)}")
    del base_model
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()
    base_path = out_dir / "base_answers.jsonl"
    with base_path.open("w", encoding="utf-8") as f:
        for r in base_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  saved: {base_path}")

    # LoRA 生成（如提供）
    lora_results = None
    if args.adapter and not args.baseline_only:
        print("=== 加载 LoRA ===")
        lora_model, lora_tok = load_model_and_tokenizer(args.base_model, adapter_path=args.adapter,
                                                         use_4bit=not args.no_4bit)
        lora_results = []
        for i, ex in enumerate(test_data):
            ans = generate(lora_model, lora_tok, ex["instruction"], max_new_tokens=args.max_new_tokens)
            lora_results.append({"idx": i, "instruction": ex["instruction"], "gold": ex["output"], "answer": ans})
            if (i + 1) % 5 == 0:
                print(f"  lora {i+1}/{len(test_data)}")
        del lora_model
        gc.collect(); torch.cuda.empty_cache()
        lora_path = out_dir / "lora_answers.jsonl"
        with lora_path.open("w", encoding="utf-8") as f:
            for r in lora_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  saved: {lora_path}")

    # 评分
    print("=== 评分 ===")
    base_scores = [score_answer(r["gold"], r["answer"]) for r in base_results]
    lora_scores = [score_answer(r["gold"], r["answer"]) for r in lora_results] if lora_results else None

    # LLM-judge（如启用）
    judge_client = get_judge_client(args)
    base_judge = []
    lora_judge = []
    if judge_client:
        print(f"=== LLM-judge ({args.judge_model}) ===")
        for r in base_results:
            j = call_llm_judge(judge_client, args.judge_model,
                               r["instruction"], r["gold"], r["answer"])
            base_judge.append(j)
        for r in (lora_results or []):
            j = call_llm_judge(judge_client, args.judge_model,
                               r["instruction"], r["gold"], r["answer"])
            lora_judge.append(j)
    else:
        print("=== LLM-judge 跳过（未配置 judge 或无 API key） ===")

    def summarize(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not scores:
            return {}
        return {
            "n": len(scores),
            "composite_mean": round(sum(s["composite"] for s in scores) / len(scores), 3),
            "keyword_coverage_mean": round(sum(s["keyword_coverage"] for s in scores) / len(scores), 3),
            "chinese_ratio_mean": round(sum(s["chinese_ratio"] for s in scores) / len(scores), 3),
            "code_parseable_rate": round(sum(1 for s in scores if s["code_parseable"]) / len(scores), 3),
            "has_hard_metric_rate": round(sum(1 for s in scores if s["has_hard_metric"]) / len(scores), 3),
            "length_ratio_mean": round(sum(s["length_ratio"] for s in scores) / len(scores), 3),
        }

    report_lines: List[str] = []
    report_lines.append("# Base vs LoRA 评估报告")
    report_lines.append("")
    report_lines.append("## 评分（启发式，0-1 越大越好）")
    report_lines.append("")
    report_lines.append("| 指标 | Base | LoRA | Δ |")
    report_lines.append("|---|---|---|---|")
    base_s = summarize(base_scores)
    lora_s = summarize(lora_scores) if lora_scores else {}
    keys = ["composite_mean", "keyword_coverage_mean", "chinese_ratio_mean", "code_parseable_rate", "has_hard_metric_rate", "length_ratio_mean"]
    for k in keys:
        b = base_s.get(k, "-")
        l = lora_s.get(k, "-")
        if isinstance(b, (int, float)) and isinstance(l, (int, float)):
            delta = round(l - b, 3)
        else:
            delta = "-"
        report_lines.append(f"| {k} | {b} | {l} | {delta} |")
    report_lines.append("")

    if lora_scores:
        wins = sum(1 for b, l in zip(base_scores, lora_scores) if l["composite"] > b["composite"])
        losses = sum(1 for b, l in zip(base_scores, lora_scores) if l["composite"] < b["composite"])
        ties = len(base_scores) - wins - losses
        report_lines.append(f"## Win/Tie/Loss (composite)")
        report_lines.append(f"- LoRA wins: **{wins}**")
        report_lines.append(f"- LoRA ties: {ties}")
        report_lines.append(f"- LoRA losses: {losses}")
        report_lines.append("")
        # 显著性检验
        from math import sqrt
        diffs = [l["composite"] - b["composite"] for b, l in zip(base_scores, lora_scores)]
        if diffs:
            mean_d = sum(diffs) / len(diffs)
            var_d = sum((d - mean_d) ** 2 for d in diffs) / max(1, len(diffs) - 1)
            sd_d = sqrt(var_d) if var_d > 0 else 0
            t = mean_d / (sd_d / sqrt(len(diffs))) if sd_d > 0 else 0
            report_lines.append(f"### 配对 t 检验（不严谨版，仅参考）")
            report_lines.append(f"- mean(diff): {mean_d:.4f}")
            report_lines.append(f"- sd(diff):   {sd_d:.4f}")
            report_lines.append(f"- t = mean/(sd/sqrt(n)) = {t:.3f}")
            report_lines.append(f"- |t|>2 → 在 n={len(diffs)} 下粗略显著")
            report_lines.append("")
        if wins < losses:
            report_lines.append("> **结论**: LoRA 模型在 composite 指标上劣于基座。可能原因：训练过拟合 / 数据量不足 / 学习率不当。")
        elif wins > losses * 1.5:
            report_lines.append("> **结论**: LoRA 模型在 composite 指标上明显优于基座。")
        else:
            report_lines.append("> **结论**: 差异不显著，建议增加 test 集样本量或人工抽检。")

    # LLM-judge 报告（如果有）
    if base_judge:
        def judge_summary(scores: List[Dict[str, Any]]) -> Dict[str, float]:
            if not scores:
                return {}
            out = {}
            for k in ["accuracy", "relevance", "completeness", "conciseness"]:
                vals = [s.get(k, 0) for s in scores if k in s]
                if vals:
                    out[k] = round(sum(vals) / len(vals), 2)
            return out

        bj = judge_summary(base_judge)
        lj = judge_summary(lora_judge)
        report_lines.append("## LLM-judge 评分（1-5 分，模型越大）")
        report_lines.append("")
        report_lines.append("| 维度 | Base | LoRA | Δ |")
        report_lines.append("|---|---|---|---|")
        for k in ["accuracy", "relevance", "completeness", "conciseness"]:
            b = bj.get(k, "-")
            l = lj.get(k, "-")
            if isinstance(b, (int, float)) and isinstance(l, (int, float)):
                delta = round(l - b, 2)
            else:
                delta = "-"
            report_lines.append(f"| {k} | {b} | {l} | {delta} |")
        report_lines.append("")

    # 保存 baseline（用于跨次训练对比）
    baseline_path = out_dir / "baseline_score.json"
    baseline = {
        "model": args.base_model,
        "n": len(base_scores),
        "judge_used": bool(base_judge),
        "metrics": summarize(base_scores),
    }
    if base_judge:
        baseline["judge_mean"] = judge_summary(base_judge)
    baseline_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  saved: {baseline_path}")

    report_lines.append("## 局限")
    report_lines.append("- 启发式评分不等于人类判断；composite 仅作过滤维度，不替代 LLM-judge")
    report_lines.append("- keyword coverage 偏向 gold 与 answer 共享词的题目，对开放式回答不公平")
    report_lines.append("- composite 权重（0.4/0.2/0.2/0.2）是经验设置，可调整")
    report_lines.append("- test 集仅 27 条，Win/Tie/Loss 在此样本量下统计意义弱")
    report_lines.append("- 若启用了 LLM-judge，应**主要看 judge 评分**而非 composite")
    report_lines.append("- 强烈建议人工抽 10-20 条对比阅读")

    report_path = out_dir / "eval_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  saved: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
