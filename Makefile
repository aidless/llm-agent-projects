# 根级一键入口。只新增本文件，不改既有 CI（.github/workflows/*）。
PYTHON ?= python3
SUB    ?= 20-llm-eval-benchmark

.PHONY: eval eval-save eval-selftest test-20

# 验收入口：跑 eval/cases.jsonl 的 20 条题目，20/20 才算通过（退出码原样透传）。
eval:
	$(PYTHON) eval/run_cases.py

# 重建 eval/run-report.json（逐条 id/axis/ok/detail）。
eval-save:
	$(PYTHON) eval/run_cases.py --save

# 反证：篡改 EV-01 期望值后必须变红（19/20）。
eval-selftest:
	$(PYTHON) eval/run_cases.py --selftest

# 既有 pytest 套件是另一回事：当前状态见 results/baseline.txt，不在 eval 里。
test-20:
	cd $(SUB) && $(PYTHON) -m pytest tests/ -q --tb=line
