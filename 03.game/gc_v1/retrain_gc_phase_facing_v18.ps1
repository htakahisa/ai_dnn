param(
    [string]$InputDir = (Join-Path $PSScriptRoot "data\attacker_gc_curriculum_v17"),
    [string]$OutputDir = (Join-Path $PSScriptRoot "data\attacker_gc_curriculum_v18")
)

$gcBundlePath = Join-Path $InputDir "best_by_eval_bundle.json"
if (-not (Test-Path -LiteralPath $gcBundlePath)) {
    throw "best_by_eval_bundle.json was not found: $gcBundlePath"
}

$gcTrainer = Join-Path $PSScriptRoot "train_attacker_gc_real_curriculum.py"
& "D:\git\python\python.exe" -X utf8 $gcTrainer `
    --init-carry (Join-Path $InputDir "dqn_attacker_carry_gc_best_by_eval.pt") `
    --init-escort (Join-Path $InputDir "dqn_attacker_escort_gc_best_by_eval.pt") `
    --init-guard (Join-Path $InputDir "dqn_attacker_guard_gc_best_by_eval.pt") `
    --output-dir $OutputDir `
    --episodes 3000 --curriculum-episodes 1500 --navigation-bootstrap-episodes 1250 `
    --navigation-demo-weight 1.5 --navigation-retention-weight 0.35 `
    --ultimate-classification-weight 2.0 --facing-supervision-weight 0.5 `
    --n-step 5 --final-mix 0.25 `
    --start-stats 50 10 0 --final-stats 100 60 40 --seed 2026110100 `
    --eval-seeds 3026091700 5026091700 6026091700 4026091700 7026091700 8026091700 `
    --holdout-seeds 9026092000 10026092000 11026092000 `
    --eval-interval 250 --eval-episodes 30 --holdout-episodes 50 `
    --lr 0.00005 --max-updates 32 `
    --feature-only-phases carry escort guard `
    --action-only-phases carry escort guard

exit $LASTEXITCODE
