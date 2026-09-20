param(
    [string]$InputDir = (Join-Path $PSScriptRoot "data\attacker_gc_curriculum_v18"),
    [string]$OutputDir = (Join-Path $PSScriptRoot ("data\attacker_gc_curriculum_v18_stage6_carry_route_fix_" + (Get-Date -Format "yyyyMMdd_HHmmss")))
)

$ErrorActionPreference = "Stop"

$bestBundlePath = Join-Path $InputDir "best_by_eval_bundle.json"
if (-not (Test-Path -LiteralPath $bestBundlePath)) {
    throw "best_by_eval_bundle.json was not found: $bestBundlePath"
}

$bestBundle = Get-Content -Raw -Encoding UTF8 -LiteralPath $bestBundlePath | ConvertFrom-Json
$episodeOffset = [int]$bestBundle.episode

$env:GC_TRAINING_MODE = "carry_route_priority"
$env:GC_FACING_WEIGHT = "0.01"
$env:GC_CARRY_SITE_ENTRY_BONUS = "12.0"
$env:GC_PLANT_PROGRESS_BONUS = "7.0"
$env:GC_SITE_HOLD_BONUS = "4.0"
$env:GC_CARRY_NO_ENTRY_PENALTY = "10.0"
$env:GC_TIMEOUT_PENALTY = "6.0"
$env:GC_SPIKE_DROP_PENALTY = "5.0"
$env:GC_ROUTE_STALL_PENALTY = "5.0"
$env:GC_TEAM_COLLISION_PENALTY = "2.5"
$env:GC_ESCORT_SUPPORT_BONUS = "0.75"
$env:GC_GUARD_SUPPRESSION_FACTOR = "0.5"

Write-Host "[GC v18 Stage 6] Structural carry route fix"
Write-Host "[GC v18 Stage 6] entry quality > facing > escort"
Write-Host "[GC v18 Stage 6] input episode: $episodeOffset"
Write-Host "[GC v18 Stage 6] output dir: $OutputDir"

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
if ((Get-ChildItem -LiteralPath $OutputDir -Force | Measure-Object).Count -gt 0) {
    throw "Output directory is not empty: $OutputDir. Use a new directory for each training run."
}

$trainer = Join-Path $PSScriptRoot "train_attacker_gc_real_curriculum.py"
if (-not (Test-Path -LiteralPath $trainer)) {
    throw "Missing trainer: $trainer"
}

& "D:\git\python\python.exe" -X utf8 $trainer `
    --init-carry (Join-Path $InputDir "dqn_attacker_carry_gc_best_by_eval.pt") `
    --init-escort (Join-Path $InputDir "dqn_attacker_escort_gc_best_by_eval.pt") `
    --init-guard (Join-Path $InputDir "dqn_attacker_guard_gc_best_by_eval.pt") `
    --output-dir $OutputDir --episode-offset $episodeOffset `
    --episodes 2200 --curriculum-episodes 1600 `
    --navigation-bootstrap-episodes 800 `
    --navigation-demo-weight 3.0 --navigation-retention-weight 0.7 `
    --ultimate-classification-weight 3.0 --facing-supervision-weight 0.01 `
    --n-step 5 --final-mix 0.25 `
    --start-stats 50 10 0 --final-stats 100 60 40 --seed 2026110200 `
    --eval-seeds 3026091700 5026091700 6026091700 4026091700 7026091700 8026091700 `
    --holdout-seeds 9026092000 10026092000 11026092000 `
    --eval-interval 250 --eval-episodes 30 --holdout-episodes 50 `
    --lr 0.00001 --max-updates 32 `
    --max-no-entry-rate 0.25 --max-timeout-rate 0.10 `
    --feature-only-phases guard `
    --action-only-phases guard

exit $LASTEXITCODE
