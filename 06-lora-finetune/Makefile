# 06-lora-finetune Makefile
# 遵循"先 smoke 再 train 再 eval"的顺序

# ---- 路径 ----
PY ?= python3
DATA_SAMPLE ?= data/sample/train.jsonl
DATA_V3 ?= data/audit/2026-07-22-v3/train.jsonl
DATA_V3_TEST ?= data/audit/2026-07-22-v3/test.jsonl
AUDIT_DIR ?= data/audit/$(shell date +%Y-%m-%d)
BASE_MODEL ?= Qwen/Qwen2.5-3B-Instruct
OUTPUT_DIR ?= output/v1
LOG_DIR ?= logs

# ---- 帮助 ----
.PHONY: help
help:
	@echo "06-lora-finetune 快速命令"
	@echo ""
	@echo "  make test        跑 CPU-only 单元测试（无需 GPU）"
	@echo "  make audit       对 data/sample/train.jsonl 做一次审计"
	@echo "  make sample      抽检 + 评级 cleaned 数据"
	@echo "  make smoke       5 步冒烟训练（验证 GPU + 数据兼容性）"
	@echo "  make full        完整流程 base-eval → train → eval"
	@echo "  make base-eval   仅 base 模型评估（训练前必跑）"
	@echo "  make compare     从 base/lora 答案抽 N 条并排"
	@echo "  make clean       清理 __pycache__ / logs"
	@echo ""

# ---- 测试（无需 GPU）----
.PHONY: test
test:
	$(PY) -m pytest tests/test_audit_split.py \
		tests/test_eval_scorers.py \
		tests/test_pipeline_smoke_cpu.py \
		tests/test_llm_judge_mock.py \
		-v --tb=short

.PHONY: test-quick
test-quick:
	$(PY) -m pytest tests/test_audit_split.py tests/test_eval_scorers.py -q

# ---- 审计 ----
.PHONY: audit
audit:
	$(PY) scripts/audit_dataset.py --input $(DATA_SAMPLE) --out-dir $(AUDIT_DIR)

.PHONY: audit-v3
audit-v3:
	$(PY) scripts/audit_dataset.py --input $(DATA_V3) --out-dir $(AUDIT_DIR)-v3

.PHONY: strip-d
strip-d:
	$(PY) scripts/strip_d_and_split.py \
		--input $(AUDIT_DIR)/cleaned.jsonl \
		--out-dir $(AUDIT_DIR)

.PHONY: sample
sample:
	$(PY) scripts/quality_sample.py \
		--input $(AUDIT_DIR)/cleaned.jsonl \
		--output $(AUDIT_DIR)/quality_sample.md \
		--n 10

.PHONY: review-warn
review-warn:
	$(PY) scripts/review_soft_warn.py \
		--audit-dir $(AUDIT_DIR) \
		--source $(DATA_SAMPLE)

# ---- 训练（需要 GPU）----
.PHONY: smoke
smoke:
	@mkdir -p $(LOG_DIR)
	$(PY) scripts/train_qlora_smoke.py --smoke 2>&1 | tee $(LOG_DIR)/smoke_$(shell date +%Y%m%d_%H%M).log

.PHONY: train
train:
	@mkdir -p $(LOG_DIR)
	$(PY) scripts/train_qlora_smoke.py \
		--model $(BASE_MODEL) \
		--data-path $(DATA_V3)/train.jsonl \
		--val-path $(DATA_V3)/val.jsonl \
		--output-dir $(OUTPUT_DIR)/train \
		--epochs 2 \
		--lr 1e-4 \
		2>&1 | tee $(LOG_DIR)/train_$(shell date +%Y%m%d_%H%M).log

# ---- 评估 ----
.PHONY: base-eval
base-eval:
	@mkdir -p $(OUTPUT_DIR)/eval
	$(PY) scripts/eval_base_vs_lora.py \
		--base-model $(BASE_MODEL) \
		--test-data $(DATA_V3_TEST) \
		--out-dir $(OUTPUT_DIR)/eval \
		--baseline-only \
		--max-new-tokens 512

.PHONY: eval
eval: base-eval
	$(PY) scripts/eval_base_vs_lora.py \
		--base-model $(BASE_MODEL) \
		--adapter $(OUTPUT_DIR)/lora_adapter \
		--test-data $(DATA_V3_TEST) \
		--out-dir $(OUTPUT_DIR)/compare \
		--max-new-tokens 512

.PHONY: eval-judge
eval-judge:
	$(PY) scripts/run_full_pipeline.py \
		--stages base-eval,eval \
		--judge openai \
		--judge-model gpt-4o-mini \
		--data-dir data/audit/2026-07-22-v3 \
		--output-dir $(OUTPUT_DIR)

# ---- 全流程 ----
.PHONY: full
full:
	$(PY) scripts/run_full_pipeline.py \
		--stages base-eval,train,eval \
		--data-dir data/audit/2026-07-22-v3 \
		--output-dir $(OUTPUT_DIR) \
		--epochs 2 \
		--judge openai 2>&1 | tee $(LOG_DIR)/full_$(shell date +%Y%m%d_%H%M).log

# ---- 对比辅助 ----
.PHONY: compare
compare:
	$(PY) scripts/compare_predictions.py \
		--base $(OUTPUT_DIR)/eval/base_answers.jsonl \
		--lora $(OUTPUT_DIR)/compare/lora_answers.jsonl \
		--output $(OUTPUT_DIR)/compare_sample.md \
		--n 10

# ---- 清理 ----
.PHONY: clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/

.PHONY: clean-output
clean-output:
	rm -rf $(OUTPUT_DIR) $(LOG_DIR)/
	@echo "Cleaned $(OUTPUT_DIR) and $(LOG_DIR)/"
	@echo "WARNING: This deletes trained adapter, eval results, and logs"
	@echo "Re-run 'make full' to restore"

# ---- 调试用 ----
.PHONY: data-info
data-info:
	@echo "=== Data files ==="
	@wc -l $(DATA_V3)/*.jsonl 2>/dev/null || echo "v3 data not found"
	@echo ""
	@echo "=== Audit outputs ==="
	@ls -la data/audit/ 2>/dev/null | tail -10
	@echo ""
	@echo "=== Test count ==="
	@$(PY) -m pytest tests/test_audit_split.py tests/test_eval_scorers.py tests/test_pipeline_smoke_cpu.py tests/test_llm_judge_mock.py --collect-only -q 2>&1 | tail -3
