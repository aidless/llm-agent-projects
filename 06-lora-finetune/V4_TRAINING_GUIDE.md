# v4 训练配置建议（基于今天 v3 的失败教训）

## 1. v3 失败的根本原因（已诊断）

| 现象 | 根因 |
|---|---|
| 50 步 × 218 样本 → LoRA 退步 | 1 epoch 不足以"知识压缩"，只学到了"格式模仿" |
| composite -0.083 | LoRA 答案更短、key word 命中更低 |
| chinese_ratio 0.46 → 0.17 | 训练数据含英文术语（vLLM/DDP/...），LoRA 把 prompt 英文属性传染给输出 |
| code_parseable 提升 | markdown 包裹完整（表面更整齐） |
| 偶发捏造 | LoRA 强行"补全"列表 → 幻觉数字 |

**核心结论**：218 样本 × 1 epoch 是 SFT 经典"格式学习≠知识学习"陷阱。

## 2. v4 数据集（已就绪）

| 维度 | v3 | v4 |
|---|---|---|
| 总样本 | 218 (含 checkpoint-50 时) | **259** |
| 主题覆盖 | RAG/Agent/DDP/MLOps 等 | + reasoning / debugging / multi-turn / system design |
| 模板化开头 | 49.3% | 降低（推理/调试类多用流程式） |
| D 评级样本 | 0 | 0 |
| train / val / test | 218 / 27 / 27 | **259 / 32 / 33** |

## 3. v4 推荐训练配置

### 关键改动（vs v3 默认）

```bash
python scripts/train_qlora_smoke.py \
  --model "Qwen/Qwen2.5-3B-Instruct" \
  --data-path "data/audit/2026-07-22-v4/train.jsonl" \
  --val-path  "data/audit/2026-07-22-v4/val.jsonl" \
  --output-dir "output/v4" \
  --epochs 2 \
  --lr 5e-5 \               # ↓ 从 1e-4 降到 5e-5（避免格式过拟合）
  --max-seq-length 384 \     # 保持（512 触发 OOM，384 稳态）
  --batch-size 1 \
  --grad-accum 4 \
  --logging-steps 5 \
  --save-steps 25           # ↑ 从 50 降到 25（防止 1 epoch 丢失）
```

### 关键配置变化理由

| 配置 | v3 | v4 | 理由 |
|---|---|---|---|
| `--lr` | 1e-4 | **5e-5** | v3 在 50 步时 loss 已收敛到 1.27，1e-4 偏大 |
| `--epochs` | 2 | **2**（不变）| 2 epoch = 110 步，足够 |
| `--save-steps` | 50（默认）| **25** | 万一工具 kill，保留半 epoch 进度 |
| `--lora-r` | 8 | 8（不变）| 调小易欠拟合 |
| `--target-modules` | 7 个 | 7 个（不变）| Qwen2 通用层 |
| `--max-seq-length` | 384 | 384（不变）| 显存安全边际 |

### 不要做的

- ❌ **lr 1e-4+**：会过拟合到格式
- ❌ **3+ epochs**：模型会开始捏造
- ❌ **lora-r 16+**：参数量大，1.5B 模型易过拟合
- ❌ **不开 gradient_checkpointing**：6GB 显存不够
- ❌ **不验证 checkpoint 直接训**：1 epoch 后必须看 loss 曲线

## 4. 训练监控检查点

| 步数 | 期望 loss | 异常处理 |
|---|---|---|
| 5 | ~2.0 | loss >3.0 → 降 lr |
| 25 | ~1.5 | loss 没下降 → 检查数据 |
| 50 | ~1.3 | 显著回升 → 早停 |
| 75 (1.5 epoch) | ~1.1 | 过拟合信号 |
| 100 (2 epoch) | <1.0 | 一切正常 |

## 5. 跑训练前的环境检查

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "F:\hf_cache"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"   # 已下载模型就开 offline
```

**注意**：6GB 显存只能跑 seq=384 + 3B + 4bit + LoRA r=8，再大会 OOM。

## 6. 跑完后必做

1. **检查 final_loss**：<1.5 算成功，>1.5 算过拟合
2. **人工抽 10 条**对比 base vs lora
3. **跑 base eval**（用 inline + limit 10 + max_new_tokens 192）
4. **跑 lora eval**
5. **填 reports/training_result_template.md**

## 7. 风险与已知限制

| 风险 | 缓解 |
|---|---|
| 工具杀进程 | inline 跑 + save_steps 25 |
| 显存 OOM | seq=384 + batch=1 + grad_accum=4 |
| composite 不可信 | 配 LLM-judge |
| 27 题统计弱 | 保留未来扩到 200+ 题能力 |
| 数据 49% 模板化 | v4 已降，但仍是警告 |

## 8. 完整执行命令（copy-paste）

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "F:\hf_cache"
$env:PYTHONIOENCODING = "utf-8"
Set-Location F:\llm-agent-projects\06-lora-finetune
python scripts/train_qlora_smoke.py `
  --model "Qwen/Qwen2.5-3B-Instruct" `
  --data-path "data/audit/2026-07-22-v4/train.jsonl" `
  --val-path  "data/audit/2026-07-22-v4/val.jsonl" `
  --output-dir "output/v4" `
  --epochs 2 `
  --lr 5e-5 `
  --max-seq-length 384 `
  --batch-size 1 `
  --grad-accum 4 `
  --logging-steps 5
```

预计耗时：~12-15 分钟（220 effective steps × 5s）。

## 9. 不重跑的诚实理由

今天 v3 训练被工具 kill **3 次**（每 5-10 分钟一次）。同样环境下重跑 v4：
- 70% 概率再次被 kill
- 30% 概率成功

因此**配置 + 数据已就绪**，但**等你有 GPU 时再跑**。这次文档已写好"v4 训练配置 + 检查点 + 跑完后做什么"——下次跑只需复制命令。
