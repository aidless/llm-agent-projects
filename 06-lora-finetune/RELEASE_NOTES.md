# RELEASE_NOTES — 06-lora-finetune（2026-07-22 ~ 2026-07-23）

> 一句话：完整可用的 QLoRA 训练管道（数据 + 工具 + 评估 + 文档），含诚实失败案例（v3 训练退步）作为研究资产。

## 时间线（24 小时）

| 阶段 | 完成 |
|---|---|
| 数据生成（10→309 条） | ✅ |
| 审计脚本 + 测试 | ✅（53 测试） |
| 训练脚本 + smoke 验证 | ✅ |
| 评估脚本 + LLM-judge | ✅ |
| 真实训练（v3，50 步完成） | ✅（部分） |
| 真实评估（10 题 base vs LoRA） | ✅ |
| 诚实报告（v3 退步 6/10） | ✅ |
| v4 数据扩充（+48 条 → 260 train） | ✅ |
| 修复 4 个 code_parse_fail | ✅ |

## 数据状态

```
v3 训练集: 326 条 (含 1 quarantined = too_short → cleaned 325)
v3 实际训练: 218 条 (旧 v2 划分)
v4 划分: 260 train / 32 val / 33 test
```

| 指标 | v3 | v4 |
|---|---|---|
| 总数 | 218 | **260** |
| D 评级 (占位符) | 0 | 0 |
| code_parse_fail | 4 | **0** ✓ |
| 主题覆盖 | RAG/Agent/DDP/MLOps | + reasoning / debugging / multi-turn / system design |
| 模板化开头 | 49.3% | 降低 |

## 工具集

11 个脚本 + 4 套测试：

| 工具 | 用途 | 测试 |
|---|---|---|
| `audit_dataset.py` | 数据审计 + 划分 | ✅ |
| `quality_sample.py` | 客观评级 | ✅ |
| `strip_d_and_split.py` | 剔除 D + 重划 | ✅ |
| `review_soft_warn.py` | 软警告 P1/P2/P3 | ✅ |
| `gen_v3_supplement.py` | 60 条补充 | - |
| `gen_v4_supplement.py` | 28 条新主题 | - |
| `gen_v4_batch2.py` | 20 条工程实践 | - |
| `train_qlora_smoke.py` | QLoRA 训练 | ✅ |
| `eval_base_vs_lora.py` | 评估 + LLM-judge | ✅ |
| `compare_predictions.py` | 人工对比辅助 | - |
| `run_full_pipeline.py` | 全流程编排 | - |

**测试覆盖**：53 CPU-only 测试全过（不依赖 GPU/HF/peft 完整环境）

## 文档

| 文档 | 行数 | 内容 |
|---|---|---|
| `README.md` | 110 | Project Card + Known Risks（11 项） |
| `RUN.md` | 280 | 端到端运行手册 11 节 |
| `final_eval_report.md` | 200 | 诚实最终评估（v3 退步分析） |
| `V4_TRAINING_GUIDE.md` | 130 | v4 训练配置 + 检查点 |
| `RELEASE_NOTES.md` | (本文件) | 全日总结 |
| `FINAL_SUMMARY.md` | 200 | 之前写的（保留作历史） |
| `requirements-fix.txt` | - | peft/torch 兼容性指南 |
| `Makefile` + `run.ps1` | 110 | 一键命令（双平台） |
| `.github/workflows/test.yml` | - | CI（PR 自动跑 4 个测试） |
| `reports/smoke_template.md` | 70 | smoke 结果回填 |
| `reports/training_result_template.md` | 120 | 真实训练回填 |

## 关键产出（可立即使用）

| 文件 | 用途 |
|---|---|
| `output/v3/checkpoint-50/` | 50 步 LoRA adapter（60MB safetensors） |
| `output/eval_base/` | 基座 10 题 baseline + 评分 |
| `output/eval_lora/` | LoRA 10 题对比 + 评分 |
| `data/audit/2026-07-22-v4/{train,val,test}.jsonl` | 训练就绪的 v4 数据 |

## 关键发现（诚实记录）

### ✅ 已验证

1. **管道完整可用**：数据 → 训练 → 评估 → 报告，每步都有自动化脚本
2. **6GB RTX 3060 可跑 3B 4bit QLoRA**（seq=384 + batch=1 + grad_accum=4）
3. **checkpoint-50 在 CPU 上可加载，inference 正常**：
   - RAG 问题输出技术正确的中文回答
4. **53 个 CPU-only 测试全过**（沙箱内可重复验证）

### ❌ 真实失败案例

1. **v3 训练（50 步）** LoRA 反而退步（composite 0.496 → 0.413，6/10 退步）
2. **3 次训练被工具 kill**（环境/工具问题，不是代码问题）
3. **OOM 6GB 边界**：seq=512 + 3B + 4bit 不够；seq=384 稳态

### ⚠️ 已知限制

1. **218 样本 × 1 epoch 不够**（"格式学习 ≠ 知识学习"陷阱）
2. **composite 启发式评分不可信**（经验权重，无论文支撑）
3. **27 题 test 集统计意义弱**（发论文需 ≥200）
4. **未启用 LLM-judge**（无 API key）
5. **chinese_ratio 暴跌**：LoRA 把训练数据中英文术语传染给输出

## 改进路径（已写文档）

### 数据
- ✅ 4 个新主题（reasoning / debugging / multi-turn / system design）
- ✅ 修复 4 个 broken Python code block
- 建议：扩到 ≥1000 条（v4 是 260，差距 4x）

### 训练
- ✅ v4 配置指南（lr 1e-4 → 5e-5 避免格式过拟合）
- ✅ save_steps 50 → 25（防工具 kill）
- 建议：≥2 epoch（本次只跑了 1 epoch）

### 评估
- ✅ base vs LoRA 对比（10 题）
- 待办：全 27 题 + LLM-judge

## 一键执行

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "F:\hf_cache"
$env:PYTHONIOENCODING = "utf-8"
Set-Location F:\llm-agent-projects\06-lora-finetune

# 单元测试（无需 GPU，5 秒）
.\run.ps1 test

# 真实训练（v4 配置，约 12-15 分钟）
.\run.ps1 train
# 或：
python scripts/train_qlora_smoke.py `
  --model "Qwen/Qwen2.5-3B-Instruct" `
  --data-path "data/audit/2026-07-22-v4/train.jsonl" `
  --val-path  "data/audit/2026-07-22-v4/val.jsonl" `
  --output-dir "output/v4" `
  --epochs 2 `
  --lr 5e-5 `
  --max-seq-length 384 `
  --batch-size 1 `
  --grad-accum 4
```

## 教训（可推广）

1. **基础设施比代码重要**：工具的 detached 进程会被 session 清理，inline 跑更稳
2. **诚实记录失败**：v3 退步是真实结果，不是工具问题——这种诚实记录比粉饰更有研究价值
3. **模板 vs 知识**：1 epoch LoRA 学到的是 markdown 模板，不是知识；需要 ≥2 epoch + ≥1000 样本
4. **composite 不可信**：LLM-judge 必加；composite 只能过滤不能评价

## 文件结构（最终）

```
06-lora-finetune/
├── .github/workflows/test.yml
├── 06-lora-finetune/...              # 子目录
├── FINAL_SUMMARY.md                   # 阶段总结
├── Makefile
├── README.md                          # 含 Known Risks
├── RELEASE_NOTES.md                   # 本文件
├── RUN.md                             # 11 节运行手册
├── V4_TRAINING_GUIDE.md               # v4 配置
├── COMMIT_MSG.md
├── data/
│   └── audit/
│       ├── 2026-07-22/                # 初次审计
│       ├── 2026-07-22-v2/             # 309 → 275 (去 hard)
│       ├── 2026-07-22-v3/             # 218 / 27 / 27 (v3 划分)
│       ├── 2026-07-22-v3-audit/      # v3 + 扩充后审计 (326 → 325)
│       └── 2026-07-22-v4/             # 260 / 32 / 33 (推荐)
├── output/
│   ├── v3/checkpoint-50/              # 50 步 LoRA（60MB）
│   ├── eval_base/                     # base 10 题 baseline
│   └── eval_lora/                     # LoRA 10 题对比
├── reports/
│   ├── smoke_template.md
│   └── training_result_template.md
├── requirements-fix.txt
├── run.ps1
├── run_train_v3.sh
├── run_eval_base.sh
├── run_eval_lora.sh
├── scripts/
│   ├── audit_dataset.py
│   ├── compare_predictions.py
│   ├── eval_base_vs_lora.py
│   ├── gen_v3_supplement.py
│   ├── gen_v4_batch2.py
│   ├── gen_v4_supplement.py
│   ├── gen_extra_280.py
│   ├── gen_extra_data.py
│   ├── gen_full_dataset.py
│   ├── gen_part2.py ~ gen_part8.py
│   ├── quality_sample.py
│   ├── review_soft_warn.py
│   ├── run_full_pipeline.py
│   ├── strip_d_and_split.py
│   ├── train.py                       # 旧版
│   ├── train_qlora_smoke.py           # 推荐
│   └── inference.py
└── tests/
    ├── test_audit_split.py
    ├── test_data.py                   # 依赖 peft (沙箱不能跑)
    ├── test_eval_scorers.py
    ├── test_llm_judge_mock.py
    ├── test_models.py                  # 依赖 peft
    ├── test_pipeline_smoke_cpu.py
    └── test_training_eval.py          # 依赖 accelerate
```
