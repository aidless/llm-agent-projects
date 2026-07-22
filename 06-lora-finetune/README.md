# LoRA 领域模型微调流水线

基于 LoRA/QLoRA 的大语言模型领域微调完整流水线，支持从数据准备、模型训练到推理部署的全流程。

> **📋 Project Card (2026-07-22)** — for hiring managers / reviewers
>
> | **Problem** | Adapting a 7B+ LLM to a domain (legal, medical, support) normally requires multi-GPU infrastructure and weeks of work. |
> |---|---|
> | **Solution** | Single-GPU LoRA/QLoRA pipeline using `trl.SFTTrainer` + `peft`: data prep → LoRA injection → SFT → eval → merge → export. Hardcoded on RTX 3060 6GB (QLoRA 4bit). |
> | **Evidence** | 10-row end-to-end teaching corpus (Q→A pairs); smoke-test training runs to convergence; checkpoint format compatible with vLLM/TGI serving. |
> | **Limitations** | Sample dataset is illustrative (10 rows) — replace `data/train.jsonl` with real corpus; assumes single GPU; loss-curve eval only (no held-out benchmark). |
> | **Stack** | PyTorch · transformers · peft · trl · bitsandbytes · accelerate |
> | **Lines / Tests** | ~2.8K Python · 22 tests |
>
> See [`results.md`](./results.md) for sample training-loss curves.

## 项目结构

```
06-lora-finetune/
├── data/                    # 数据处理模块
│   ├── __init__.py
│   ├── data_loader.py       # 数据加载（JSONL/JSON/CSV）
│   ├── formatter.py         # 数据格式化（多模板支持）
│   ├── tokenizer_utils.py    # 分词器封装
│   ├── dataset_builder.py   # SFT 数据集构建
│   └── sample/
│       └── train.jsonl      # 示例数据（10 条问答对）
├── models/                  # 模型和配置模块
│   ├── __init__.py
│   ├── lora_config.py       # LoRA/QLoRA 配置管理
│   ├── model_loader.py      # 模型加载器
│   └── model_exporter.py    # 模型导出（合并 LoRA 权重）
├── training/                # 训练模块
│   ├── __init__.py
│   ├── training_config.py   # 训练超参数配置
│   ├── sft_trainer.py       # SFT 训练器封装
│   └── experiment_tracker.py # 实验追踪（JSON 日志）
├── evaluation/              # 评估模块
│   ├── __init__.py
│   ├── perplexity_evaluator.py  # Perplexity 评估
│   └── task_evaluator.py         # 任务评估（准确率/F1）
├── scripts/                 # 运行脚本
│   ├── train.py             # 训练脚本
│   ├── export_model.py      # 模型合并导出脚本
│   └── inference.py         # 推理脚本
├── tests/                   # 测试模块
│   ├── __init__.py
│   ├── test_data.py         # 数据模块测试
│   ├── test_models.py       # 模型模块测试
│   └── test_training_eval.py # 训练和评估模块测试
├── inference.py             # 推理接口 API
├── .env                     # 环境变量配置
├── requirements.txt         # 依赖清单
└── README.md                # 本文件
```

## 技术栈

- **Python** + **PyTorch**
- **HuggingFace Transformers** - 模型和分词器
- **PEFT** - LoRA/QLoRA 参数高效微调
- **TRL** - SFTTrainer 训练器
- **Datasets** - 数据集管理
- **Accelerate** - 分布式训练支持

## 微调流程

### 1. 准备数据

将数据准备为 JSONL 格式，每行一个 JSON 对象：

```json
{"instruction": "问题", "input": "可选上下文", "output": "答案"}
```

支持的输入格式：
- **JSONL** - 每行一个 JSON 对象
- **JSON** - 嵌套或列表格式
- **CSV** - 表格格式（需指定列名）

### 2. 配置参数

在 `.env` 文件中配置：

```bash
MODEL_NAME=meta-llama/Llama-2-7b-hf   # 基础模型
LORA_R=8                                 # LoRA rank
LORA_ALPHA=16                            # LoRA alpha
USE_QLORA=false                          # 是否使用 QLoRA
NUM_EPOCHS=3                             # 训练轮数
LEARNING_RATE=2e-4                       # 学习率
```

### 3. 开始训练

```bash
# 使用默认配置训练
python scripts/train.py --data_path data/sample/train.jsonl

# 自定义参数
python scripts/train.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --data_path data/sample/train.jsonl \
    --lora_r 16 \
    --lora_alpha 32 \
    --epochs 5 \
    --lr 1e-4 \
    --batch_size 8 \
    --use_qlora

# QLoRA（4-bit 量化，节省显存）
python scripts/train.py \
    --data_path data/sample/train.jsonl \
    --use_qlora \
    --lora_r 8 \
    --lora_alpha 16
```

### 4. 导出模型

```bash
# 合并 LoRA 权重导出完整模型
python scripts/export_model.py \
    --base_model meta-llama/Llama-2-7b-hf \
    --adapter_path ./output/lora_adapter \
    --output_dir ./output/merged_model
```

### 5. 推理

```bash
# 单次推理
python scripts/inference.py \
    --model_path ./output/merged_model \
    --prompt "请解释什么是 LoRA"

# 使用 LoRA 适配器推理（不合并）
python scripts/inference.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --adapter_path ./output/lora_adapter \
    --prompt "请解释什么是 LoRA"

# 交互模式
python scripts/inference.py \
    --model_path ./output/merged_model \
    --interactive
```

### 6. 评估

```python
from evaluation.task_evaluator import TaskEvaluator

evaluator = TaskEvaluator(metrics=["accuracy", "f1"])
predictions = ["预测1", "预测2", "预测3"]
references = ["答案1", "答案2", "答案4"]
results = evaluator.evaluate(predictions, references)
print(f"准确率: {results['accuracy']:.2%}")
print(f"F1: {results['f1']:.4f}")
```

## LoRA 配置说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `r` (rank) | LoRA 秩，控制低秩矩阵大小 | 8, 16, 32, 64 |
| `lora_alpha` | 缩放因子，实际缩放为 alpha/r | 通常设为 r 的 1-2 倍 |
| `lora_dropout` | Dropout 比率 | 0.05 - 0.1 |
| `target_modules` | 应用 LoRA 的模块 | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj |
| `bias` | bias 处理 | "none"（推荐） |

预定义配置：
- `small` - r=4, alpha=8（数据量充足时）
- `medium` - r=8, alpha=16（默认，通用场景）
- `large` - r=16, alpha=32（复杂任务）
- `qlora_default` - QLoRA r=8（显存有限时）

## 超参调优建议

### 学习率
- LoRA 推荐范围：`1e-5` ~ `3e-4`
- QLoRA 推荐范围：`1e-5` ~ `1e-4`
- 建议使用 cosine 学习率调度器，warmup_ratio 0.03

### Batch Size
- 有效 batch size = `per_device_batch_size` x `gradient_accumulation_steps`
- 推荐 16-64，通过梯度累积实现
- 建议保持 batch size 恒定，不要随 GPU 数量变化

### 训练轮数
- SFT 通常 1-5 轮
- 数据量少时（<1000条）可适当增加到 5-10 轮
- 使用早停法（Early Stopping）防止过拟合

### Rank 选择
- 数据量 < 1K 条：r=4~8
- 数据量 1K~10K 条：r=8~16
- 数据量 > 10K 条：r=16~64
- alpha 通常设为 r 的 1-2 倍

### 目标模块
- 基础：`q_proj, v_proj`
- 推荐：`q_proj, v_proj, k_proj, o_proj`
- 全覆盖：增加 `gate_proj, up_proj, down_proj`（LLaMA 架构）

### LoRA vs QLoRA
| 特性 | LoRA | QLoRA |
|------|------|-------|
| 量化 | 无 | 4-bit (NF4) |
| 显存需求 | 高 | 低（约 10GB 起步） |
| 训练速度 | 快 | 略慢 |
| 效果 | 略好 | 接近 LoRA |
| 适用场景 | GPU 显存充足 | GPU 显存有限 |

## 实验追踪

训练过程自动记录到 `./output/logs/` 目录下的 JSON 文件，包含：
- 训练配置
- 每 N 步的 loss 和学习率
- 评估指标
- 最终结果摘要

## 运行测试

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试（使用 mock，不需要 GPU）
cd /workspace/llm-agent-projects/06-lora-finetune
pytest tests/ -v
```
