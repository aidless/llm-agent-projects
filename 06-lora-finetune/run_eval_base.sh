#!/usr/bin/env bash
# Base eval launcher
cd "F:/test/2026-07-22-15-28-09/projects/llm-agent-projects/06-lora-finetune"
mkdir -p output/eval_base logs
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="F:/hf_cache"
export HF_HUB_OFFLINE="1"
export PYTHONIOENCODING="utf-8"
nohup python -u scripts/eval_base_vs_lora.py --base-model "Qwen/Qwen2.5-3B-Instruct" --test-data "data/audit/2026-07-22-v3/test.jsonl" --out-dir "output/eval_base" --baseline-only --max-new-tokens 384 > logs/eval_base_run.log 2>&1 &
echo $! > /tmp/eval_base.pid
disown
