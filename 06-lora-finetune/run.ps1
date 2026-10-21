# run.ps1 - 06-lora-finetune 常用命令封装（Windows 友好）
# 等价于 Makefile，但用 PowerShell 写

param(
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"
$DataV3 = "data/audit/2026-07-22-v3"
$BaseModel = "Qwen/Qwen2.5-3B-Instruct"
$OutputDir = "output/v1"
$LogDir = "logs"

function Help {
    Write-Host "06-lora-finetune PowerShell 命令" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  .\run.ps1 test        跑 CPU-only 单元测试"
    Write-Host "  .\run.ps1 audit       对 data/sample/train.jsonl 做审计"
    Write-Host "  .\run.ps1 sample      抽检 + 评级"
    Write-Host "  .\run.ps1 smoke       5 步冒烟训练"
    Write-Host "  .\run.ps1 base-eval   仅 base 模型评估"
    Write-Host "  .\run.ps1 full        完整流程（base-eval → train → eval）"
    Write-Host "  .\run.ps1 compare     从 base/lora 答案抽 N 条对比"
    Write-Host "  .\run.ps1 info        数据/审计/测试 summary"
    Write-Host "  .\run.ps1 clean       清理 __pycache__ / .pytest_cache"
}

function Test {
    python -m pytest tests/test_audit_split.py tests/test_eval_scorers.py tests/test_pipeline_smoke_cpu.py tests/test_llm_judge_mock.py -v --tb=short
}

function Audit {
    $auditDir = "data/audit/$((Get-Date).ToString('yyyy-MM-dd'))"
    python scripts/audit_dataset.py --input data/sample/train.jsonl --out-dir $auditDir
}

function Sample {
    $auditDir = "data/audit/$((Get-Date).ToString('yyyy-MM-dd'))"
    python scripts/quality_sample.py --input $auditDir/cleaned.jsonl --output $auditDir/quality_sample.md --n 10
}

function Smoke {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $logFile = "$LogDir/smoke_$((Get-Date).ToString('yyyyMMdd_HHmm')).log"
    python scripts/train_qlora_smoke.py --smoke 2>&1 | Tee-Object -FilePath $logFile
}

function Train {
    Write-Host "⚠ Make sure base-eval is done before training!" -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $logFile = "$LogDir/train_$((Get-Date).ToString('yyyyMMdd_HHmm')).log"
    python scripts/train_qlora_smoke.py `
        --model $BaseModel `
        --data-path "$DataV3/train.jsonl" `
        --val-path "$DataV3/val.jsonl" `
        --output-dir "$OutputDir/train" `
        --epochs 2 `
        --lr 1e-4 `
        2>&1 | Tee-Object -FilePath $logFile
}

function BaseEval {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $logFile = "$LogDir/base-eval_$((Get-Date).ToString('yyyyMMdd_HHmm')).log"
    python scripts/eval_base_vs_lora.py `
        --base-model $BaseModel `
        --test-data "$DataV3/test.jsonl" `
        --out-dir "$OutputDir/eval" `
        --baseline-only `
        --max-new-tokens 512 `
        2>&1 | Tee-Object -FilePath $logFile
}

function Full {
    python scripts/run_full_pipeline.py `
        --stages base-eval,train,eval `
        --data-dir $DataV3 `
        --output-dir $OutputDir `
        --epochs 2 `
        --judge openai `
        2>&1 | Tee-Object -FilePath "$LogDir/full_$((Get-Date).ToString('yyyyMMdd_HHmm')).log"
}

function Compare {
    if (-not (Test-Path "$OutputDir/eval/base_answers.jsonl")) {
        Write-Host "❌ base_answers.jsonl 不存在，先跑 base-eval" -ForegroundColor Red
        return
    }
    if (Test-Path "$OutputDir/compare/lora_answers.jsonl") {
        python scripts/compare_predictions.py `
            --base "$OutputDir/eval/base_answers.jsonl" `
            --lora "$OutputDir/compare/lora_answers.jsonl" `
            --output "$OutputDir/compare_sample.md" `
            --n 10
    } else {
        python scripts/compare_predictions.py `
            --base "$OutputDir/eval/base_answers.jsonl" `
            --output "$OutputDir/compare_sample.md" `
            --n 10
    }
}

function Info {
    Write-Host "=== Data files ===" -ForegroundColor Cyan
    if (Test-Path $DataV3) {
        Get-ChildItem "$DataV3/*.jsonl" | ForEach-Object { Write-Host "  $($_.Name): $(($_ | Get-Content).Count) lines" }
    } else {
        Write-Host "  v3 data not found"
    }
    Write-Host ""
    Write-Host "=== Audit outputs ===" -ForegroundColor Cyan
    if (Test-Path "data/audit/") {
        Get-ChildItem "data/audit/" | Sort-Object LastWriteTime -Descending | Select-Object -First 5 | ForEach-Object {
            Write-Host "  $($_.Name) ($($_.LastWriteTime.ToString('yyyy-MM-dd HH:mm')))"
        }
    }
    Write-Host ""
    Write-Host "=== Test count ===" -ForegroundColor Cyan
    python -m pytest tests/test_audit_split.py tests/test_eval_scorers.py tests/test_pipeline_smoke_cpu.py tests/test_llm_judge_mock.py --collect-only -q 2>&1 | Select-Object -Last 3
}

function Clean {
    Get-ChildItem -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Recurse -Directory -Filter ".pytest_cache" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Recurse -Directory -Filter ".mypy_cache" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Recurse -Directory -Filter ".ruff_cache" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Cleaned" -ForegroundColor Green
}

# --- 路由 ---
switch ($Command) {
    "help"      { Help }
    "test"      { Test }
    "audit"     { Audit }
    "sample"    { Sample }
    "smoke"     { Smoke }
    "train"     { Train }
    "base-eval" { BaseEval }
    "full"      { Full }
    "compare"   { Compare }
    "info"      { Info }
    "clean"     { Clean }
    default     { Write-Host "Unknown command: $Command" -ForegroundColor Red; Help }
}
