#!/usr/bin/env python3
"""
端到端 LLM 应用流水线演示

串联以下项目展示完整的 LLM 应用流程：
  03 (Gateway) → 04 (Guardrails) → 01 (RAG) + 16 (KG Fusion) → 17 (Observability)

使用方式：
    # 默认（Mock 模式，无需 API Key）
    python demo_pipeline.py

    # 真实 LLM 模式（需要 OPENAI_API_KEY）
    export OPENAI_API_KEY=sk-xxx
    python demo_pipeline.py --real

    # 自定义查询
    python demo_pipeline.py --query "什么是向量数据库？"
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# 允许从项目根目录导入各子项目
ROOT = Path(__file__).resolve().parent
for project in ["03-multi-model-gateway", "01-rag-knowledge-base", "04-guardrails-chat",
                "17-llm-observability", "16-kg-rag-fusion"]:
    sys.path.insert(0, str(ROOT / project))


# ============================================================================
# 颜色与格式化
# ============================================================================

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


def cprint(text: str, color: str = "", bold: bool = False) -> None:
    """彩色打印。"""
    style = color + (Colors.BOLD if bold else "")
    print(f"{style}{text}{Colors.RESET}")


def header(text: str, level: int = 1) -> None:
    """打印分节标题。"""
    if level == 1:
        bar = "=" * 70
        print(f"\n{Colors.CYAN}{bar}{Colors.RESET}")
        cprint(f"  {text}", Colors.CYAN, bold=True)
        print(f"{Colors.CYAN}{bar}{Colors.RESET}\n")
    else:
        cprint(f"\n── {text} " + "─" * (60 - len(text)), Colors.YELLOW)


def step(num: int, name: str) -> None:
    """打印步骤标记。"""
    cprint(f"\n[Step {num}] {name}", Colors.BLUE, bold=True)


def info(label: str, value: Any, indent: int = 4) -> None:
    """打印键值对信息。"""
    pad = " " * indent
    label_str = cprint.__globals__  # noqa: F841  (avoid lint)
    print(f"{pad}{Colors.GRAY}{label}:{Colors.RESET} {value}")


def box(content: str, title: str = "") -> None:
    """在文本框中打印内容。"""
    width = 68
    print(f"{Colors.MAGENTA}┌{'─' * width}┐{Colors.RESET}")
    if title:
        title_padded = f" {title} "
        left_pad = (width - len(title_padded)) // 2
        right_pad = width - len(title_padded) - left_pad
        print(f"{Colors.MAGENTA}│{Colors.RESET}{' ' * left_pad}{Colors.BOLD}{title_padded}{Colors.RESET}{' ' * right_pad}{Colors.MAGENTA}│{Colors.RESET}")
        print(f"{Colors.MAGENTA}├{'─' * width}┤{Colors.RESET}")
    for line in content.split("\n"):
        # 自动换行处理
        while line:
            chunk = line[:width]
            line = line[width:]
            print(f"{Colors.MAGENTA}│{Colors.RESET} {chunk:<{width}} {Colors.MAGENTA}│{Colors.RESET}")
    print(f"{Colors.MAGENTA}└{'─' * width}┘{Colors.RESET}")


# ============================================================================
# 流水线组件（每个项目的核心逻辑封装）
# ============================================================================

class PipelineContext:
    """流水线共享上下文。

    优先使用 17-llm-observability 的真实 Tracer（如果可用），否则回退到 Mock 实现。
    """

    def __init__(self):
        # 尝试使用 17-llm-observability 的真实 tracer
        try:
            obs_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "17-llm-observability",
                "integrations",
                "demo_pipeline_tracer.py",
            )
            if os.path.exists(obs_path):
                import importlib.util

                spec = importlib.util.spec_from_file_location("tracer_module", obs_path)
                tracer_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(tracer_module)
                self._tracer = tracer_module.get_tracer("demo-pipeline")
                self._use_real_tracer = True
            else:
                self._tracer = None
                self._use_real_tracer = False
        except Exception as e:
            print(f"[warn] Failed to load real tracer: {e}. Using Mock.", file=sys.stderr)
            self._tracer = None
            self._use_real_tracer = False

        # 标准属性
        self.trace_id: str = ""
        self.spans: List[Dict[str, Any]] = []
        self.start_time: float = time.time()
        self.tokens_input: int = 0
        self.tokens_output: int = 0
        self.cost_estimate: float = 0.0
        self.safety_checks: List[Dict[str, Any]] = []
        self.retrieved_docs: List[Dict[str, Any]] = []
        self.errors: List[str] = []

    def start_trace(self, name: str = "demo-pipeline") -> None:
        """启动 trace span（兼容 API）."""
        if self._use_real_tracer:
            self._tracer.start_root_span(name)
            self.trace_id = self._tracer.trace_id or ""
        else:
            self.trace_id = f"trace-{int(time.time() * 1000)}"

    @property
    def tracer_mode(self) -> str:
        """返回当前使用的 tracer 类型."""
        return "real" if self._use_real_tracer else "mock"

    def add_span(self, name: str, duration_ms: float, attrs: Dict[str, Any] = None) -> None:
        """记录一个 span。"""
        self.spans.append({
            "name": name,
            "duration_ms": round(duration_ms, 2),
            "attrs": attrs or {},
        })

    def estimate_cost(self, model: str = "gpt-4o-mini") -> None:
        """根据 token 数估算成本（gpt-4o-mini: $0.15/M input, $0.6/M output）。"""
        cost = (self.tokens_input / 1_000_000) * 0.15 + (self.tokens_output / 1_000_000) * 0.6
        self.cost_estimate = round(cost, 6)


class Step1_Gateway:
    """03 - 多模型 LLM 网关：路由查询到合适的模型。"""

    def __init__(self, use_real: bool = False):
        self.use_real = use_real
        # Mock 分类器
        self.task_keywords = {
            "code": ["代码", "code", "python", "javascript", "编程"],
            "math": ["数学", "math", "计算", "公式"],
            "rag": ["文档", "知识", "知识库", "什么是"],
            "general": [],
        }

    def route(self, query: str, ctx: PipelineContext) -> str:
        """根据查询类型选择模型。"""
        t0 = time.time()
        try:
            query_lower = query.lower()
            # 分类
            if any(kw in query_lower for kw in self.task_keywords["code"]):
                task_type = "code"
                model = "gpt-4o"
            elif any(kw in query_lower for kw in self.task_keywords["math"]):
                task_type = "math"
                model = "gpt-4o"
            elif any(kw in query_lower for kw in self.task_keywords["rag"]):
                task_type = "rag"
                model = "gpt-4o-mini"
            else:
                task_type = "general"
                model = "gpt-4o-mini"

            elapsed = (time.time() - t0) * 1000
            ctx.add_span("gateway.route", elapsed, {
                "task_type": task_type,
                "model": model,
                "query_length": len(query),
            })
            return model
        except Exception as e:
            ctx.errors.append(f"Gateway error: {e}")
            return "gpt-4o-mini"  # fallback


class Step2_InputGuardrails:
    """04 - 输入侧安全护栏：检测 Prompt Injection 和敏感内容。"""

    def __init__(self, use_real: bool = False):
        self.use_real = use_real
        self.injection_patterns = [
            "ignore previous", "ignore above", "you are now", "system:",
            "忽略之前的", "你现在是", "系统提示",
        ]
        self.sensitive_keywords = ["bomb", "weapon", "hack", "密码", "攻击", "漏洞"]

    def check(self, query: str, ctx: PipelineContext) -> Dict[str, Any]:
        """执行安全检查。"""
        t0 = time.time()
        query_lower = query.lower()

        # Prompt Injection 检测
        injection_score = sum(1 for p in self.injection_patterns if p in query_lower)
        # 敏感词检测
        sensitive_score = sum(1 for p in self.sensitive_keywords if p in query_lower)

        result = {
            "is_safe": injection_score == 0 and sensitive_score == 0,
            "injection_score": injection_score,
            "sensitive_score": sensitive_score,
            "risk_level": "high" if injection_score > 0 else ("medium" if sensitive_score > 0 else "low"),
        }

        elapsed = (time.time() - t0) * 1000
        ctx.add_span("guardrails.input", elapsed, result)
        ctx.safety_checks.append({"stage": "input", **result})
        return result


class Step3_Retrieval:
    """01 + 16 - RAG + KG 融合检索。"""

    def __init__(self, use_real: bool = False):
        self.use_real = use_real
        # Mock 知识库
        self.knowledge_base = [
            {
                "id": "doc-001",
                "title": "RAG 最佳实践指南",
                "content": "RAG（Retrieval-Augmented Generation）是结合检索和生成的方法...",
                "metadata": {"source": "internal_doc.pdf", "page": 1},
                "entities": ["RAG", "检索", "生成"],
            },
            {
                "id": "doc-002",
                "title": "向量数据库对比",
                "content": "ChromaDB 适合原型，Milvus 适合生产，Weaviate 适合多模态...",
                "metadata": {"source": "vector_db_review.md", "page": 3},
                "entities": ["ChromaDB", "Milvus", "Weaviate"],
            },
            {
                "id": "doc-003",
                "title": "知识图谱与 RAG 融合",
                "content": "KG-RAG Fusion 通过实体链接和多跳推理提升答案质量...",
                "metadata": {"source": "kg_rag_fusion.pdf", "page": 5},
                "entities": ["知识图谱", "RAG", "实体链接"],
            },
        ]

    def retrieve(self, query: str, ctx: PipelineContext, top_k: int = 3) -> List[Dict[str, Any]]:
        """执行检索 + KG 增强。"""
        t0 = time.time()
        try:
            # Mock 向量检索：基于关键词重叠
            results = []
            query_words = set(query.lower().split())
            for doc in self.knowledge_base:
                doc_words = set(doc["title"].lower().split() + doc["content"].lower().split())
                overlap = len(query_words & doc_words)
                if overlap > 0:
                    score = overlap / len(query_words | doc_words)
                    results.append({
                        "doc_id": doc["id"],
                        "title": doc["title"],
                        "content": doc["content"][:200],
                        "score": round(score, 3),
                        "kg_entities": doc["entities"],
                        "metadata": doc["metadata"],
                    })

            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]

            elapsed = (time.time() - t0) * 1000
            ctx.add_span("rag.retrieve", elapsed, {
                "query": query[:50],
                "num_results": len(results),
                "top_score": results[0]["score"] if results else 0.0,
            })
            ctx.retrieved_docs = results
            return results
        except Exception as e:
            ctx.errors.append(f"Retrieval error: {e}")
            return []


class Step4_LLMGenerate:
    """LLM 生成（Mock 或真实）。"""

    def __init__(self, model: str = "gpt-4o-mini", use_real: bool = False):
        self.model = model
        self.use_real = use_real

    def generate(self, query: str, context_docs: List[Dict[str, Any]], ctx: PipelineContext) -> str:
        """基于检索上下文生成回答。"""
        t0 = time.time()
        try:
            if self.use_real:
                # 真实 LLM 调用
                try:
                    import openai
                    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    context_str = "\n\n".join(
                        f"[{d['title']}] {d['content']}" for d in context_docs
                    )
                    prompt = (
                        f"基于以下文档回答问题：\n\n{context_str}\n\n"
                        f"问题：{query}\n\n回答："
                    )
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "你是一个专业的技术助手。"},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=512,
                    )
                    answer = response.choices[0].message.content
                    ctx.tokens_input = response.usage.prompt_tokens
                    ctx.tokens_output = response.usage.completion_tokens
                    ctx.estimate_cost()
                except ImportError:
                    answer = self._mock_generate(query, context_docs)
            else:
                answer = self._mock_generate(query, context_docs)
                ctx.tokens_input = len(query) + sum(len(d["content"]) for d in context_docs)
                ctx.tokens_output = len(answer)

            elapsed = (time.time() - t0) * 1000
            ctx.add_span("llm.generate", elapsed, {
                "model": self.model,
                "tokens_in": ctx.tokens_input,
                "tokens_out": ctx.tokens_output,
                "answer_length": len(answer),
            })
            return answer
        except Exception as e:
            ctx.errors.append(f"LLM error: {e}")
            return f"[生成失败] {e}"

    def _mock_generate(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Mock 回答生成。"""
        if not context_docs:
            return f"（未检索到相关文档）针对 '{query}'，这是一个模拟回答。请提供 RAG 知识库内容以获得更好的答案。"
        top = context_docs[0]
        return (
            f"基于检索到的文档《{top['title']}》，针对您的问题「{query}」的回答：\n\n"
            f"{top['content']}\n\n"
            f"相关实体：{', '.join(top['kg_entities'])}\n"
            f"来源：{top['metadata'].get('source', 'unknown')}"
        )


class Step5_OutputGuardrails:
    """04 - 输出侧安全护栏：幻觉检测。"""

    def __init__(self):
        self.hallucination_indicators = [
            "可能", "也许", "我不确定", "approximately", "maybe",
        ]

    def check(self, answer: str, source_docs: List[Dict[str, Any]], ctx: PipelineContext) -> Dict[str, Any]:
        """幻觉检测 + 输出格式检查。"""
        t0 = time.time()
        # 简单的支撑度评估：答案中的关键词是否出现在源文档中
        answer_words = set(answer.lower().split())
        source_text = " ".join(d["content"].lower() for d in source_docs).split()
        source_words = set(source_text)
        coverage = len(answer_words & source_words) / max(1, len(answer_words))

        # 幻觉风险指标
        has_uncertainty = any(p in answer.lower() for p in self.hallucination_indicators)

        result = {
            "is_supported": coverage > 0.3,
            "coverage_score": round(coverage, 3),
            "has_uncertainty_marker": has_uncertainty,
            "hallucination_risk": "low" if coverage > 0.5 else ("medium" if coverage > 0.3 else "high"),
        }

        elapsed = (time.time() - t0) * 1000
        ctx.add_span("guardrails.output", elapsed, result)
        ctx.safety_checks.append({"stage": "output", **result})
        return result


class Step6_Observability:
    """17 - 可观测性：汇总 trace 和指标。"""

    def report(self, ctx: PipelineContext) -> Dict[str, Any]:
        """生成可观测性报告。"""
        t0 = time.time()
        total_duration = (time.time() - ctx.start_time) * 1000
        # 各阶段耗时
        stage_durations = {s["name"]: s["duration_ms"] for s in ctx.spans}
        # 错误统计
        error_count = len(ctx.errors)

        report = {
            "trace_id": ctx.trace_id,
            "total_duration_ms": round(total_duration, 2),
            "stage_durations": stage_durations,
            "tokens": {
                "input": ctx.tokens_input,
                "output": ctx.tokens_output,
                "total": ctx.tokens_input + ctx.tokens_output,
            },
            "cost_estimate_usd": ctx.cost_estimate,
            "errors": ctx.errors,
            "safety_checks": ctx.safety_checks,
            "num_retrieved_docs": len(ctx.retrieved_docs),
        }

        elapsed = (time.time() - t0) * 1000
        ctx.add_span("observability.report", elapsed)
        return report


# ============================================================================
# 流水线编排
# ============================================================================

async def run_pipeline(query: str, use_real: bool = False) -> Dict[str, Any]:
    """执行完整的流水线。"""
    ctx = PipelineContext()
    ctx.start_trace("demo-pipeline")

    cprint(f"\n🔍 用户查询: {query}", Colors.MAGENTA, bold=True)
    print(f"   {Colors.GRAY}Trace ID: {ctx.trace_id}{Colors.RESET}")

    # Step 1: Gateway 路由
    step(1, "03 多模型网关 — 任务路由")
    gateway = Step1_Gateway(use_real=use_real)
    selected_model = gateway.route(query, ctx)
    info("分类结果", gateway.task_keywords if hasattr(gateway, 'task_keywords') else "general")
    info("选中模型", selected_model, indent=4)

    # Step 2: 输入安全检查
    step(2, "04 输入安全护栏 — Prompt Injection / 敏感内容检测")
    guard = Step2_InputGuardrails(use_real=use_real)
    input_check = guard.check(query, ctx)
    info("风险等级", input_check["risk_level"])
    info("安全通过", "✅ 是" if input_check["is_safe"] else "❌ 否")

    if not input_check["is_safe"]:
        cprint("⚠️  查询未通过安全检查，流水线终止", Colors.RED, bold=True)
        return {"query": query, "blocked": True, "reason": input_check, "context": ctx}

    # Step 3: RAG + KG 检索
    step(3, "01 + 16 RAG 检索 + KG 融合")
    retriever = Step3_Retrieval(use_real=use_real)
    docs = retriever.retrieve(query, ctx, top_k=3)
    info("检索文档数", len(docs))
    if docs:
        for i, d in enumerate(docs, 1):
            info(f"  [{i}] {d['title']}", f"score={d['score']}", indent=4)

    # Step 4: LLM 生成
    step(4, f"LLM 生成 (model={selected_model})")
    generator = Step4_LLMGenerate(model=selected_model, use_real=use_real)
    answer = generator.generate(query, docs, ctx)
    info("Token 用量", f"{ctx.tokens_input} in + {ctx.tokens_output} out")
    info("估算成本", f"${ctx.cost_estimate:.6f}")

    # Step 5: 输出安全检查
    step(5, "04 输出安全护栏 — 幻觉检测")
    out_guard = Step5_OutputGuardrails()
    output_check = out_guard.check(answer, docs, ctx)
    info("支撑度", output_check["coverage_score"])
    info("幻觉风险", output_check["hallucination_risk"])

    # Step 6: 可观测性
    step(6, "17 可观测性 — Trace 汇总")
    obs = Step6_Observability()
    trace = obs.report(ctx)
    info("总耗时", f"{trace['total_duration_ms']:.1f} ms")
    info("Span 数", len(ctx.spans))

    return {
        "query": query,
        "blocked": False,
        "answer": answer,
        "context": ctx,
        "trace": trace,
        "input_check": input_check,
        "output_check": output_check,
        "docs": docs,
    }


# ============================================================================
# 输出报告
# ============================================================================


def export_report(result: Dict[str, Any], path: str) -> None:
    """导出报告到文件。支持 .json 或 .md 后缀。"""
    if result.get("blocked"):
        # 被拦截的报告
        payload = {
            "query": result["query"],
            "blocked": True,
            "reason": result["reason"],
        }
    else:
        ctx: PipelineContext = result["context"]
        payload = {
            "query": result["query"],
            "blocked": False,
            "answer": result["answer"],
            "trace_id": ctx.trace_id,
            "retrieved_docs": [
                {
                    "doc_id": d["doc_id"],
                    "title": d["title"],
                    "score": d["score"],
                    "kg_entities": d["kg_entities"],
                    "metadata": d["metadata"],
                }
                for d in result["docs"]
            ],
            "input_safety": result.get("input_check", {}),
            "output_safety": result.get("output_check", {}),
            "performance": {
                "total_duration_ms": result["trace"]["total_duration_ms"],
                "stage_durations": result["trace"]["stage_durations"],
                "tokens": result["trace"]["tokens"],
                "cost_estimate_usd": result["trace"]["cost_estimate_usd"],
                "errors": result["trace"]["errors"],
            },
            "timestamp": int(time.time()),
            "mode": "real" if os.getenv("OPENAI_API_KEY") else "mock",
        }

    # 确保父目录存在
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    if path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    elif path.endswith(".md") or path.endswith(".markdown"):
        md = render_markdown_report(payload)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
    else:
        raise ValueError(f"不支持的格式：{path}（需 .json 或 .md）")


def render_markdown_report(p: Dict[str, Any]) -> str:
    """渲染 Markdown 格式报告。"""
    lines = []
    lines.append(f"# Pipeline Execution Report")
    lines.append("")
    lines.append(f"- **Query**: {p['query']}")
    lines.append(f"- **Mode**: {p.get('mode', 'unknown')}")
    lines.append(f"- **Timestamp**: {p.get('timestamp', 'N/A')}")
    if p.get("trace_id"):
        lines.append(f"- **Trace ID**: {p['trace_id']}")
    lines.append("")

    if p.get("blocked"):
        lines.append("## ⚠️ Blocked")
        lines.append("")
        lines.append(f"**Reason**: {json.dumps(p['reason'], ensure_ascii=False)}")
        return "\n".join(lines)

    lines.append("## Answer")
    lines.append("")
    lines.append(p["answer"])
    lines.append("")

    if p.get("retrieved_docs"):
        lines.append("## Retrieved Documents")
        lines.append("")
        for i, d in enumerate(p["retrieved_docs"], 1):
            lines.append(f"### [{i}] {d['title']} (score={d['score']})")
            lines.append(f"- **Entities**: {', '.join(d['kg_entities'])}")
            lines.append(f"- **Source**: {d['metadata'].get('source', 'unknown')}")
            lines.append("")

    lines.append("## Safety Checks")
    lines.append("")
    inp = p.get("input_safety", {})
    out = p.get("output_safety", {})
    lines.append("| Stage | Risk Level | Details |")
    lines.append("|---|---|---|")
    lines.append(f"| Input | {inp.get('risk_level', 'N/A')} | injection={inp.get('injection_score', 0)}, sensitive={inp.get('sensitive_score', 0)} |")
    lines.append(f"| Output | {out.get('hallucination_risk', 'N/A')} | coverage={out.get('coverage_score', 0)} |")
    lines.append("")

    perf = p.get("performance", {})
    lines.append("## Performance")
    lines.append("")
    lines.append(f"- **Total duration**: {perf.get('total_duration_ms', 0):.2f} ms")
    lines.append(f"- **Tokens**: {perf.get('tokens', {}).get('input', 0)} in + {perf.get('tokens', {}).get('output', 0)} out")
    lines.append(f"- **Cost estimate**: ${perf.get('cost_estimate_usd', 0):.6f}")
    if perf.get("stage_durations"):
        lines.append("")
        lines.append("### Stage Durations")
        lines.append("")
        lines.append("| Stage | Duration (ms) |")
        lines.append("|---|---|")
        for name, dur in perf["stage_durations"].items():
            lines.append(f"| {name} | {dur:.2f} |")
    if perf.get("errors"):
        lines.append("")
        lines.append("### Errors")
        lines.append("")
        for e in perf["errors"]:
            lines.append(f"- {e}")

    return "\n".join(lines)


def print_summary(result: Dict[str, Any]) -> None:
    """打印汇总报告。"""
    header("流水线执行报告", level=1)

    if result["blocked"]:
        cprint("\n❌ 查询被安全护栏拦截", Colors.RED, bold=True)
        box(json.dumps(result["reason"], indent=2, ensure_ascii=False), "拦截原因")
        return

    # 用户查询
    cprint("\n📝 用户查询", Colors.CYAN, bold=True)
    box(result["query"])

    # 检索结果
    cprint("\n📚 检索结果", Colors.CYAN, bold=True)
    docs_text = ""
    for i, d in enumerate(result["docs"], 1):
        docs_text += f"[{i}] 《{d['title']}》 (score={d['score']})\n"
        docs_text += f"    {d['content'][:150]}...\n"
        docs_text += f"    实体: {', '.join(d['kg_entities'])}\n\n"
    box(docs_text or "无")

    # 答案
    cprint("\n💡 生成答案", Colors.GREEN, bold=True)
    box(result["answer"])

    # 安全检查
    cprint("\n🛡️  安全检查", Colors.YELLOW, bold=True)
    safety_text = ""
    for check in result["context"].safety_checks:
        safety_text += f"[{check.get('stage', 'unknown')}] "
        if check.get("stage") == "input":
            safety_text += f"风险等级: {check.get('risk_level', 'unknown')}, "
            safety_text += f"injection={check.get('injection_score', 0)}, "
            safety_text += f"sensitive={check.get('sensitive_score', 0)}\n"
        else:
            safety_text += f"幻觉风险: {check.get('hallucination_risk', 'unknown')}, "
            safety_text += f"支撑度: {check.get('coverage_score', 0)}\n"
    box(safety_text)

    # Trace 详情
    cprint("\n📊 性能 Trace", Colors.BLUE, bold=True)
    trace = result["trace"]
    trace_text = f"Trace ID: {trace['trace_id']}\n"
    trace_text += f"总耗时: {trace['total_duration_ms']:.2f} ms\n\n"
    trace_text += "各阶段耗时：\n"
    for name, dur in trace["stage_durations"].items():
        bar = "█" * int(dur / 5)  # 简易柱状图
        trace_text += f"  {name:<25} {dur:>7.2f} ms  {bar}\n"
    trace_text += f"\nToken: {trace['tokens']['input']} in + {trace['tokens']['output']} out\n"
    trace_text += f"成本估算: ${trace['cost_estimate_usd']:.6f}"
    if trace["errors"]:
        trace_text += f"\n错误: {len(trace['errors'])}"
    box(trace_text)

    # 评估
    cprint("\n📈 总体评估", Colors.MAGENTA, bold=True)
    output_check = result["output_check"]
    evaluations = [
        ("检索结果", f"{len(result['docs'])} 个文档 (top score: {result['docs'][0]['score'] if result['docs'] else 'N/A'})"),
        ("输入安全", "✅ 通过" if result["input_check"]["is_safe"] else "❌ 拦截"),
        ("幻觉风险", {"low": "✅ 低", "medium": "⚠️ 中", "high": "❌ 高"}.get(output_check["hallucination_risk"], "❓ 未知")),
        ("支撑度", f"{output_check['coverage_score']:.1%}"),
        ("总耗时", f"{trace['total_duration_ms']:.1f} ms"),
        ("Token 成本", f"${trace['cost_estimate_usd']:.6f}"),
    ]
    eval_text = ""
    for label, value in evaluations:
        eval_text += f"  {label:<15} {value}\n"
    box(eval_text)


def main():
    parser = argparse.ArgumentParser(description="端到端 LLM 流水线演示")
    parser.add_argument("--query", "-q", type=str,
                        default="如何使用 RAG 和知识图谱融合提升企业文档搜索？",
                        help="要查询的问题")
    parser.add_argument("--real", action="store_true",
                        help="使用真实 LLM（需 OPENAI_API_KEY 环境变量）")
    parser.add_argument("--export-report", type=str, default=None, metavar="PATH",
                        help="导出报告到指定路径（支持 .json 或 .md 后缀）")
    args = parser.parse_args()

    cprint("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║          LLM Agent Projects — End-to-End Pipeline Demo           ║
║                                                                  ║
║   03 Gateway → 04 Guardrails → 01 RAG + 16 KG → 17 Observability ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""", Colors.CYAN, bold=True)

    # 检查 real 模式
    if args.real and not os.getenv("OPENAI_API_KEY"):
        cprint("❌ --real 模式需要 OPENAI_API_KEY 环境变量", Colors.RED, bold=True)
        cprint("回退到 Mock 模式\n", Colors.YELLOW)
        args.real = False

    if not args.real:
        cprint("📌 当前为 Mock 模式（无需 API Key）", Colors.GRAY)
        cprint("   使用 --real 启用真实 LLM 调用\n", Colors.GRAY)

    # 执行流水线
    try:
        result = asyncio.run(run_pipeline(args.query, use_real=args.real))
        print_summary(result)

        # 导出报告（如果指定）
        if args.export_report:
            export_report(result, args.export_report)
            cprint(f"\n📁 报告已导出到: {args.export_report}", Colors.GREEN, bold=True)
    except KeyboardInterrupt:
        cprint("\n\n⚠️  用户中断", Colors.YELLOW)
        sys.exit(1)
    except Exception as e:
        cprint(f"\n\n❌ 流水线执行失败: {e}", Colors.RED)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()