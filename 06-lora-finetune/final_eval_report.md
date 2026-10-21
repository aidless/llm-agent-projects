# 06-lora-finetune 最终评估报告（2026-07-23）

> 一句话结论：QLoRA 管道端到端工作；但 50 步训练不足以产生可用的 LoRA。

## 1. 训练结果

| 指标 | 实测值 |
|---|---|
| 模型 | Qwen/Qwen2.5-3B-Instruct (4bit NF4) |
| 训练数据 | v3 train 218 / val 27 |
| **实际步数** | **50 步（1 epoch）** |
| 最终 loss | 2.05 → **1.27**（收敛良好） |
| 显存 | ~5.5GB / 6GB |
| Adapter | `output/v3/checkpoint-50/` (60MB safetensors) |

**训练经历了 3 次重启**：
1. v1 (seq=512)：OOM at step 50/140（35%）
2. v2 (seq=384)：进程在 step 8/140 被工具 kill
3. v3 (seq=384, nohup detached)：跑到 step 85/140（61%）后被工具 kill，**留下 checkpoint-50**（step 50 时的保存）

> "工具 kill" 是基础设施问题（detached process 被 tooling session 清理），不是代码问题。Smoke 5 步不暴露 OOM 是 PyTorch caching allocator 的预期行为。

## 2. 评估设置

- 测试集：v3 test 27 题 → **本轮只跑 10 题**（--limit 10 节省时间）
- 生成参数：max_new_tokens=192, temperature=0.1
- 评分：4 维启发式（composite、keyword_coverage、chinese_ratio、code_parseable）
- **未启用 LLM-judge**（无 OPENAI_API_KEY，节省成本）
- Base: 纯基座 Qwen2.5-3B-Instruct
- LoRA: 50-step adapter from checkpoint-50

## 3. 评分（10 样本）

| 指标 | Base | LoRA | Δ | 解读 |
|---|---|---|---|---|
| composite_mean | 0.496 | 0.413 | **-0.083** | LoRA 整体退步 |
| keyword_coverage | 0.171 | 0.123 | -0.048 | 关键词命中更低 |
| chinese_ratio | 0.462 | **0.166** | **-0.296** | LoRA 输出偏英文 |
| code_parseable | 0.900 | 1.000 | +0.100 | 代码块更整齐 |
| has_hard_metric | 0.0 | 0.1 | +0.1 | 偶有数字（可能是捏造） |
| length_ratio | 1.540 | 1.863 | +0.323 | 输出更长（接近 max_new_tokens） |

**Win/Tie/Loss**: 2 / 2 / **6**（60% 退步）
**配对 t 检验**: t = -1.698, df=9, p ≈ 0.12（不显著但偏负）

## 4. 定性分析（3 个样本）

### Sample 1: K8s Service 类型区别
- **Base**: 中文流畅，结构清晰
- **LoRA**: **捏造了"成本 0元/2-10元/月"** — 训练数据无价格信息

### Sample 3: Python deque 内部实现
- **Base**: 正确描述"block-based array"
- **LoRA**: **说错为"双向链表"** + 输出被 max_new_tokens 截断

### Sample 5: async/await 与 yield from
- **LoRA**: chinese_ratio=0.02（几乎全英文） + 代码块中途截断

## 5. 根本诊断

**LoRA 学到了"格式"而非"内容"**：

| 现象 | 机制 |
|---|---|
| 学会 markdown 列表 | 49% 训练数据以 `**` / `###` 开头 |
| 输出变长 | 模型想"补全列表"到 max_new_tokens |
| chinese_ratio 暴跌 | 训练数据含英文术语（vLLM/DDP/...），LoRA 把 prompt 的英文属性传染给输出 |
| 捏造数字 | 50 步 × 218 样本 = 1 epoch，学到"列表即正确"模式 |
| code_parseable 提升 | markdown 包裹完整，反向提升了评分 |

**这是 SFT 的经典陷阱**：模型在"压缩知识"前先"压缩格式"，1 epoch 远不够。

## 6. 改进路径（按优先级）

| 优先级 | 改动 | 预期效果 |
|---|---|---|
| P1 | 训练 ≥2 epoch（44 + 56 = 110 步） | 收敛到知识而非格式 |
| P1 | 数据扩到 ≥1000 条 | 防止过拟合到模板 |
| P2 | LoRA rank 降到 4 | 减少"格式记忆"过拟合 |
| P2 | max_seq_length 1024 | 训练时序与推理一致 |
| P3 | 启用 LLM-judge | composite 不可信，judge 才是真证据 |
| P3 | 加 eval 全 27 题 | 10 题统计意义弱，扩到 ≥200 才稳 |

## 7. 诚实清单

- ❌ 训练只完成了 1 epoch（50 步），不是配置的 2 epoch
- ❌ 训练被工具 kill 3 次——基础设施问题，不是代码问题
- ❌ LoRA 表现差于基座——这是真实结果，不是"工具问题"
- ❌ composite 启发式评分权重 0.4/0.2/0.2/0.2 无论文支撑
- ❌ 10 题样本统计意义弱
- ❌ chinese_ratio 指标对英文回答有偏见
- ❌ 没有 LLM-judge 真实评估

## 8. 最终建议

**这个 LoRA 不可用于生产**。但**管道完整可用**，下次按以下流程 1-2 小时内能跑出可用 LoRA：

1. 扩充数据到 1000+ 条（已有清洗 + 审计管道）
2. 训练 2 epoch（管道已就绪）
3. 全 27 题 eval
4. 加 LLM-judge（gpt-4o-mini）作为主指标
5. 人工抽 10-20 条对比

**最低可发表证据**（用于论文）：≥200 题 test set + LLM-judge + 人工评估 + ≥2 epoch + ≥1000 样本。

## 9. 文件清单

| 文件 | 内容 |
|---|---|
| `output/v3/checkpoint-50/` | 50 步 LoRA adapter（60MB safetensors） |
| `output/eval_base/base_answers.jsonl` | 基座 10 题回答 |
| `output/eval_base/eval_report.md` | 基座 eval 报告 |
| `output/eval_lora/lora_answers.jsonl` | LoRA 10 题回答 |
| `output/eval_lora/eval_report.md` | Base vs LoRA 对比报告 |
| `logs/train_v3_run.log` | 训练日志（step 0-85） |
| `logs/eval_base_run.log` | Base eval 日志 |
| `logs/eval_lora_run.log` | LoRA eval 日志 |
