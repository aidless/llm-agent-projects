# ⚠️ DEPRECATED — Inference Optimization (2026-07-22)

This project has been **moved to `archive/`** because, despite its name, it
contains **no real GPU inference code**. The "simulator" in
`simulator/inference_simulator.py` uses hardcoded formulas
(`params = layers * hidden^2 * 12`, etc.) — there is no `torch`, no `vllm`,
no `sglang`, no `tensorrt`. It is a Python teaching tool about *what vLLM
does*, not a piece of infrastructure that does it.

This project is **kept for reference only**. For real inference work, see
`vllm-project/vllm` or `sgl-project/sglang` upstream.

---

# Original README (preserved for reference)

# LLM Inference Optimization Service

模拟 vLLM/SGLang 核心优化策略的推理服务性能优化工具集。

> ⚠️ **Reality check (2026-07-22 audit)**: This project is a **mathematical
> simulator**, not real inference. The `InferenceSimulator` and
> `MockTokenizer` use hardcoded latency/throughput formulas. There is no
> `torch`, `vllm`, or `sglang` import anywhere. Useful as a teaching aid;
> not suitable for any claim of "inference optimization" research.

## 功能特性

### 1. 模拟推理引擎
- Token by token 生成模拟
- 动态批处理 (Continuous Batching)
- 请求队列和优先级调度
- 多种调度策略 (FIFO/Priority/SJF/LMF)

### 2. 量化模拟
- 支持 FP32/FP16/INT8/INT4 精度
- GPTQ/AWQ/GGUF 格式配置

(... full original README content preserved in git history ...)

## Why it's archived

- **Pure simulator** — formulas are hardcoded; no real GPU math.
- **Not a benchmark** — outputs are deterministic from inputs, not from
  actual hardware.
- **Could mislead reviewers** — README says "inference optimization service"
  but it cannot optimize anything.

## Recommended use

Treat this as a teaching artifact about *what* vLLM-style schedulers do, not
*how* to deploy one. The original README's broader feature list (quantization
sim, KV-cache sim, etc.) is useful documentation, but the code is not.