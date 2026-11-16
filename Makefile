# Root entry points.
# `make check` is the collection evidence: offline pytest for every active
# project, then a drift test that the README tier table matches the matrix.
# `make check-install` is the one networked step (pip + NLTK tokenizers).
PYTHON ?= python3
SUB    ?= 20-llm-eval-benchmark

.PHONY: eval eval-save eval-selftest test-20 check check-refresh check-install

# Install the shared offline environment. Needs network. Not part of `make check`.
check-install:
	$(PYTHON) -m pip install -r requirements-check.txt
	$(PYTHON) scripts/bootstrap_nltk.py

# Re-run offline tests. Exit 1 when counts, tiers, or the README table drift
# from results/status-matrix.json. Does not rewrite that file.
check:
	$(PYTHON) scripts/offline_check.py --compare
	$(PYTHON) scripts/drift_check.py

# Re-run offline tests and overwrite the committed matrix + README table.
check-refresh:
	$(PYTHON) scripts/offline_check.py --refresh
	$(PYTHON) scripts/drift_check.py

# 验收入口：跑 eval/cases.jsonl 的 20 条题目，20/20 才算通过（退出码原样透传）。
eval:
	$(PYTHON) eval/run_cases.py

# 重建 eval/run-report.json（逐条 id/axis/ok/detail）。
eval-save:
	$(PYTHON) eval/run_cases.py --save

# 反证：篡改 EV-01 期望值后必须变红（19/20）。
eval-selftest:
	$(PYTHON) eval/run_cases.py --selftest

# Project 20's own pytest. The collection gate is `make check`, not this target.
test-20:
	cd $(SUB) && $(PYTHON) -m pytest tests/ -q --tb=line
