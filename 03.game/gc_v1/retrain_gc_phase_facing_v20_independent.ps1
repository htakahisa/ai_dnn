param(
    [string]$InputDir = (Join-Path $PSScriptRoot "data\attacker_gc_curriculum_v19_facing_head_only"),
    [string]$OutputDir = (Join-Path $PSScriptRoot "data\attacker_gc_curriculum_v20_independent_facing")
)

$ErrorActionPreference = "Stop"

$bestBundlePath = Join-Path $InputDir "best_by_eval_bundle.json"
if (-not (Test-Path -LiteralPath $bestBundlePath)) {
    throw "best_by_eval_bundle.json was not found: $bestBundlePath"
}
$bestBundle = Get-Content -Raw -Encoding UTF8 -LiteralPath $bestBundlePath | ConvertFrom-Json
$episodeOffset = [int]$bestBundle.episode

$trainer = Join-Path $PSScriptRoot "train_attacker_gc_real_curriculum.py"
& "D:\git\python\python.exe" -X utf8 $trainer `
    --init-carry (Join-Path $InputDir "dqn_attacker_carry_gc_best_by_eval.pt") `
    --init-escort (Join-Path $InputDir "dqn_attacker_escort_gc_best_by_eval.pt") `
    --init-guard (Join-Path $InputDir "dqn_attacker_guard_gc_best_by_eval.pt") `
    --output-dir $OutputDir --episode-offset $episodeOffset `
    --episodes 1000 --curriculum-episodes 750 `
    --navigation-bootstrap-episodes 0 `
    --navigation-demo-weight 0 --navigation-retention-weight 0 `
    --ultimate-classification-weight 0 --facing-supervision-weight 1.0 `
    --facing-regression-tolerance 0.05 `
    --n-step 5 --final-mix 0.25 `
    --start-stats 50 10 0 --final-stats 100 60 40 --seed 2026110400 `
    --eval-seeds 3026091700 5026091700 6026091700 4026091700 7026091700 8026091700 `
    --holdout-seeds 9026092000 10026092000 11026092000 `
    --eval-interval 250 --eval-episodes 30 --holdout-episodes 50 `
    --lr 0.0001 --max-updates 16 `
    --facing-only-phases carry escort `
    --freeze-phases guard

exit $LASTEXITCODE
