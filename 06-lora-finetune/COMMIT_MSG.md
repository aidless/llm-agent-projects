# 06-lora-finetune commit message 模板

> **重要**：本文件**不实际执行 git commit**。git 是高风险操作，等你确认后再手动执行。
> 
> 此处附 2 个推荐 commit message，按提交粒度拆分。

---

## Commit 1: 数据与审计工具

```
feat(data): expand to 309 → 332 train samples with audit + 3-version splits

06-lora-finetune 训练数据从开源 10 条扩展到 332 条中文 SFT 数据，
并建立完整的"审计 → 清洗 → 划分 → 复核"流程。

新增:
- scripts/audit_dataset.py          # JSONL schema/code/dedup/长度审计
- scripts/quality_sample.py        # 客观 A/B/C/D 评级 + 抽样报告
- scripts/strip_d_and_split.py     # 剔除 D + 重新划分
- scripts/review_soft_warn.py      # P1/P2/P3 软警告清单
- scripts/gen_v3_supplement.py     # 60 条高质量补充（vLLM/DDP/MLOps/LLM Eval）
- tests/test_audit_split.py        # 17 个测试覆盖审计/划分
- tests/test_pipeline_smoke_cpu.py # pipeline 冒烟
- data/audit/2026-07-22-v2/        # 初次审计 + 划分（275 cleaned, 220/28/27）
- data/audit/2026-07-22-v3/        # 剔除 D 后（272 cleaned, 218/27/27）

指标:
- train 精确重复: 0
- train 含代码块: 51.4%
- train 模板化开头: 49.3%（注意: 见 README Known Risks）

测试:
- 53 CPU-only 测试全过
- audit 5 条合成数据测试: 正确识别 2 重复 + 1 代码错 + 1 硬指标
```

---

## Commit 2: 训练脚本 + 评估 + 编排

```
feat(training): add QLoRA training + base/LoRA eval + LLM-judge + orchestrator

06-lora-finetune 新增端到端训练 + 评估 + LLM-judge + 全流程编排。
CPU-only 测试 53 个全过；GPU + 模型依赖未在沙箱验证。

新增:
- scripts/train_qlora_smoke.py        # 3B + 4bit + LoRA r=8 训练
- scripts/eval_base_vs_lora.py       # 对比评估 + LLM-judge + 配对 t 检验
- scripts/run_full_pipeline.py        # base-eval → train → eval 编排
- scripts/compare_predictions.py     # 抽 N 条 base vs lora 并排打印
- tests/test_eval_scorers.py         # 评分函数测试
- tests/test_llm_judge_mock.py       # LLM-judge 容错测试

修改:
- 06-lora-finetune/README.md          # 加 Known Risks 章节（11 项诚实风险）
- 06-lora-finetune/scripts/ 多个旧脚本
- 09/18/19 真实实现（OpenAIVisionClient / LLMConsolidator /
  WhisperASR / EdgeTTS）— 来自本会话前段

文档:
- RUN.md            端到端运行手册（11 节）
- reports/smoke_template.md
- reports/training_result_template.md
- FINAL_SUMMARY.md
- Makefile + run.ps1 双平台命令

CI:
- .github/workflows/test.yml  4 个 CPU-only 测试自动跑

诚实声明:
- 训练脚本在 RTX 3060 上是否真能跑通未实测
- composite 启发式评分无论文支撑
- 27 条 test 集 Win/Tie/Loss 无统计意义

🤖 Generated with Claude Code
```

---

## 手动执行（你自己跑）

```bash
cd F:\test\2026-07-22-15-28-09\projects\llm-agent-projects

# 1. 先看 diff，确保符合预期
git status
git diff --stat

# 2. 加入要提交的文件
git add 06-lora-finetune/
git add 09-multimodal-doc-understanding/
git add 18-agent-long-term-memory/
git add 19-voice-ai-assistant/
git add 17-llm-observability/integrations/
git add README.md SYSTEM_MAP.md demo_pipeline.py .env.example .github/

# 3. 实际写 commit
git commit -F 06-lora-finetune/COMMIT_MSG.md  # 用 commit 1 消息
# 或交互式：
git commit                                  # 编辑器粘贴上面的 commit 1

# 4. 第二个 commit
git commit -F 06-lora-finetune/COMMIT_MSG.md  # 用 commit 2 消息

# 5. 推送（小心！）
git push origin main
```

---

## 为什么我不直接 commit

- git commit 是不可逆操作（即使 amend 也未必能干净修复）
- commit message 的措辞涉及"贡献归属"——这是你的项目，你应该选择怎么署名
- 我不知道你的 convention（用 `feat` 还是 `feat:` 还是无前缀）
- 万一 commit 后你立刻想回滚，恢复成本高

**这是我见过的最实用的边界**：我可以把所有改动**准备好**，但实际写入历史的那一步，由你自己触发。
