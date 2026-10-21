#!/usr/bin/env bash
# Detached training launcher v3
cd "F:/test/2026-07-22-15-28-09/projects/llm-agent-projects/06-lora-finetune"
mkdir -p output/v3 logs
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="F:/hf_cache"
export PYTHONIOENCODING="utf-8"
nohup python -u scripts/train_qlora_smoke.py --model "Qwen/Qwen2.5-3B-Instruct" --data-path "data/audit/2026-07-22-v3/train.jsonl" --val-path "data/audit/2026-07-22-v3/val.jsonl" --output-dir "output/v3" --epochs 2 --lr 1e-4 --max-seq-length 384 --batch-size 1 --grad-accum 4 --logging-steps 5 > logs/train_v3_run.log 2>&1 &
echo $! > /tmp/train_v3.pid
disown
