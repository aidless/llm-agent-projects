# 06-lora-finetune 端到端运行手册

本手册给出从「已审计数据集」到「Base vs LoRA 对比报告」的完整可执行步骤。所有命令假设你已经在 `06-lora-finetune/` 目录下。

---

## 0. 前置条件

### 硬件
- 单卡 RTX 3060 6GB（已实测配置）—— **不要用 7B**，会 OOM
- 显存 ≥ 5.5GB，否则自动降级但 batch 必须保持 1

### 软件
- Python 3.10+（推荐 3.11）
- CUDA 12.4

### Python 依赖（一次性安装）
```bash
# 必须
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.48.3 peft==0.13 trl==0.17 bitsandbytes==0.45
pip install accelerate datasets safetensors

# 测试/审计
pip install pytest
```

> **注意**：不要 `pip install unsloth`（在 Windows 上有冲突链，已知会导致 `torchvision::nms does not exist`）。

### HF 模型下载路径
你的 C 盘已经满了。把 HF 缓存重定向到 F 盘：
```bash
# Linux/Git Bash
export HF_HOME=F:/hf_cache
export TRANSFORMERS_CACHE=F:/hf_cache

# PowerShell
$env:HF_HOME = "F:/hf_cache"
$env:TRANSFORMERS_CACHE = "F:/hf_cache"
```

**国内访问 HuggingFace 必加镜像**（不设会卡在 10 秒超时）：
```bash
# Linux/Git Bash
export HF_ENDPOINT=https://hf-mirror.com

# PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

**首次登录 HuggingFace**（如果模型是 gated）：
```bash
huggingface-cli login
```

---

## 1. 数据状态确认

### 1.1 检查数据已生成
```bash
ls data/audit/2026-07-22-v2/
# 期望看到：
#   cleaned.jsonl  dataset_audit.json  dataset_audit.md
#   quarantine.jsonl  soft_warn_review.md
#   splits.json  train.jsonl  val.jsonl  test.jsonl
```

### 1.2 查看审计摘要
```bash
cat data/audit/2026-07-22-v2/splits.json
cat data/audit/2026-07-22-v2/dataset_audit.md | head -25
```

**当前状态**：
- 309 条原始 → 275 条 cleaned（hard quarantine 34 条）
- train / val / test = 220 / 28 / 27
- Soft warn 106 条待人工（已生成清单，见 `soft_warn_review.md`）

---

## 2. 单元测试（CPU-only，先跑）

不需要 GPU，先验证所有评分函数、审计脚本、数据加载路径都对：

```bash
cd 06-lora-finetune
pytest tests/test_audit_split.py tests/test_eval_scorers.py -v
```

**期望**：48 passed。如果失败，可能是数据 regenerations 后路径不同，重跑一次审计即可。

---

## 3. 冒烟训练（5 步验证管道）

这是「验证训练代码不报错」的冒烟测试。耗时约 2-5 分钟（包括模型加载 + 数据传输）：

```bash
cd 06-lora-finetune
python scripts/train_qlora_smoke.py --smoke
```

**期望输出**：
```
[1/6] 环境自检
  gpu: NVIDIA GeForce RTX 3060 Laptop GPU
  vram_total_gb: 6.0
[2/6] 加载数据
  train: 10  val: 5
[3/6] 加载分词器
[4/6] 加载模型（量化）
  使用 4bit NF4 + double quant + fp16 compute
[5/6] 配置 LoRA + 训练器
  trainable params: 2.5M || all params: 3B || trainable%: 0.083
[6/6] 开始训练
{'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': 1.0}
============================================================
训练完成
  final loss: <2-3 之间是合理的>
============================================================
LoRA adapter 已保存: output/lora-smoke/lora_adapter
```

**如果失败**：
- `ImportError: bitsandbytes`：重装
- `CUDA out of memory`：把 `--max-seq-length 512`，把 `--batch-size 1` 写死
- `KeyError: 'q_proj' 等`：模型架构不同，调整 `--target-modules`

---

## 4. 真实训练（约 1-3 小时）

冒烟通过后，启动真实训练：

```bash
cd 06-lora-finetune
python scripts/train_qlora_smoke.py \
  --data-path data/audit/2026-07-22-v2/train.jsonl \
  --val-path  data/audit/2026-07-22-v2/val.jsonl \
  --output-dir ./output/lora-v1 \
  --epochs 2 \
  --lr 1e-4 \
  --save-merged
```

**耗时估计**（RTX 3060 6GB）：
- 220 训练样本 × 2 epochs = 440 步
- 步长约 5-10 秒（seq_len=1024, batch=1, grad_accum=8）
- 总计 ~1-2 小时

**输出**：
- `output/lora-v1/lora_adapter/` — LoRA 权重（关键文件）
- `output/lora-v1/merged/` — 合并后完整模型（可选）
- `output/lora-v1/checkpoint-*/` — 中间检查点

**监控**：
- Loss 应在前 20 步从 ~2.0 降到 ~1.5 以下
- 如果 loss 长时间不下降：调 `--lr 5e-5`，或换 `--lora-r 16`
- 如果 loss 降到 0.5 以下持续 5+ 轮：可能过拟合，加 `--early-stop`
- Val loss 与 train loss 差距 >0.5：过拟合迹象，加权 `--weight-decay 0.01`

---

## 5. Base vs LoRA 评估（约 30-60 分钟）

训练完成后，对比基座和微调模型在 test 集上的表现：

```bash
cd 06-lora-finetune
python scripts/eval_base_vs_lora.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --adapter  ./output/lora-v1/lora_adapter \
  --test-data data/audit/2026-07-22-v2/test.jsonl \
  --out-dir  ./output/eval-v1 \
  --max-new-tokens 512
```

**输出**：
- `output/eval-v1/base_answers.jsonl` — 基座对 27 条 test 的回答
- `output/eval-v1/lora_answers.jsonl` — LoRA 对 27 条 test 的回答
- `output/eval-v1/eval_report.md` — 评分报告（含 Win/Tie/Loss）

**判读 eval_report.md**：
- `composite_mean` 是 0-1 的综合分
- `Win` 数 > `Loss` 数 × 1.5：LoRA 明显有效
- `Win` ≈ `Loss`：差异不显著——可能数据量不足 / 学习率不当
- `Win` < `Loss`：训练破坏了基座能力，可能：
  - 过拟合（loss 已在 <0.5 持续）
  - lr 太大（loss NaN）
  - 数据被污染（人工抽几条看是否合理）

---

## 6. 故障排查

### 6.1 训练时 loss = nan
- 学习率太大 → 降到 `5e-5`
- 数据里有空 answer → 检查 `cleaned.jsonl`

### 6.2 eval 时 LoRA 输出乱码
- 可能没等 token cache 生效 → 加 `--temperature 0.1` 而非 default
- 可能 adapter 加载失败 → 检查 `lora_adapter/adapter_config.json` 是否存在

### 6.3 数据集问题
- 重新审计：`python scripts/audit_dataset.py --input data/sample/train.jsonl --out-dir data/audit/$(date +%Y-%m-%d)`
- 重新生成：参考 `scripts/gen_part2.py` 等的格式，在 `data/sample/train.jsonl` 后追加

### 6.4 Pytest 失败
- 重跑：`pytest tests/test_audit_split.py tests/test_eval_scorers.py -v`
- 如果 fixture 警告：用 `@classmethod` 替换 `@pytest.fixture(scope="class")`

### 6.5 模型下载太慢
- HF 模型推荐：`Qwen/Qwen2.5-3B-Instruct` (~6GB)
- 备选：`Qwen/Qwen2.5-1.5B-Instruct`（更快但效果差）
- 用 `--model` 指向本地路径也行（前提是 transformers 能加载）

---

## 7. 落地后的记录项

完成后请更新：
- [ ] 把最终 loss 写进 `data/audit/last_train_loss.txt`
- [ ] 把 eval_report.md 的 Win/Tie/Loss 写进 README
- [ ] 把训练曲线（如果有 tensorboard）截图保存
- [ ] 给 `06-lora-finetune/README.md` 加 `## v1 训练结果` 一节

---

## 8. 可选的下一步（如果你想做得更好）

1. **人工复核 soft warn**：打开 `data/audit/2026-07-22-v2/soft_warn_review.md`，按 P1 → P2 → P3 顺序 review
2. **修复 hard quarantine**：剩 24 个 Python 代码错误（缩进 / 配对）—— 见 `data/audit/2026-07-22-v2/quarantine.jsonl`
3. **多轮次训练**：用 eval 的失败 case 作为新训练样本，做 DPO/RLHF
4. **LLM-judge**（已集成）：见下文 §9
5. **跨模型对比**：训练多个基座（Qwen2.5-3B、Llama3-3B、Qwen2.5-1.5B），对比效果

---

## 9. LLM-judge 用法（可选）

`eval_base_vs_lora.py` 已支持 `--judge openai`，会自动调用 OpenAI 兼容接口给每个回答打分（accuracy / relevance / completeness / conciseness，1-5 分）。

```bash
# 需要：export OPENAI_API_KEY=sk-...
# 需要：pip install openai

# 在 base eval 阶段用 LLM-judge
python scripts/run_full_pipeline.py \
  --stages base-eval \
  --judge openai \
  --judge-model gpt-4o-mini

# 在 base vs lora 对比阶段用
python scripts/run_full_pipeline.py \
  --stages base-eval,train,eval \
  --judge openai
```

**注意**：
- `eval_report.md` 中会同时输出启发式 composite 和 LLM-judge 两套评分
- 若 OPENAI_API_KEY 未设置，自动回退到纯启发式
- 配对 t 检验输出在 composite 之后

---

## 10. 全流程编排器（推荐）

`scripts/run_full_pipeline.py` 是统一的编排器，按正确顺序跑所有阶段：

```bash
# 完整流程（base-eval → train → eval）
python scripts/run_full_pipeline.py \
  --stages base-eval,train,eval \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --data-dir data/audit/2026-07-22-v3 \
  --output-dir output/v1 \
  --epochs 2 \
  --judge openai

# 单独跑某个阶段
python scripts/run_full_pipeline.py --stages base-eval
python scripts/run_full_pipeline.py --stages train --skip-base-eval
python scripts/run_full_pipeline.py --stages eval --base-model Qwen/...
```

**关键特性**：
- **强约束**：不跑 base eval 直接训练 → 交互式确认（防止没 baseline 还硬训）
- **结果对比**：最后打印 baseline vs LoRA 的 composite 差值
- **adapter 自动 symlink**：训练产出的 lora_adapter 软链到 output_dir/lora_adapter
- **三处输出**：`output/v1/eval/baseline_score.json`（baseline）、`output/v1/train/lora_adapter/`（adapter）、`output/v1/compare/eval_report.md`（对比）

---

## 11. v3 数据集（推荐用于训练）

`data/audit/2026-07-22-v3/` 是剔除 D 级（占位符样本）后的 v3 划分：

| 阶段 | 数量 |
|---|---:|
| 总（cleaned） | 272 |
| Train | 218 |
| Val | 27 |
| Test | 27 |
| 剔除 D | 3 |

**用 v3 替代 v2** 是因为占位符会教模型说"...略"——剔除后训练更安全。

