param(
    [string]$OutputDir = (Join-Path $PSScriptRoot "data\attacker_gc_curriculum_v12")
)

$gcInitDir = Join-Path $PSScriptRoot "data\attacker_gc_curriculum_v11"
$gcBundlePath = Join-Path $gcInitDir "best_by_eval_bundle.json"
if (-not (Test-Path -LiteralPath $gcBundlePath)) {
    throw "best_by_eval_bundle.json was not found: $gcBundlePath"
}
$gcBundle = Get-Content -Raw -Encoding UTF8 -LiteralPath $gcBundlePath | ConvertFrom-Json
if ([int]$gcBundle.episode -ne 0) {
    throw "Expected the preserved v11 episode 0 checkpoint, got episode $($gcBundle.episode)"
}

$gcTrainer = Join-Path $PSScriptRoot "train_attacker_gc_real_curriculum.py"
& "D:\git\python\python.exe" -X utf8 $gcTrainer `
    --init-carry (Join-Path $gcInitDir "dqn_attacker_carry_gc_best_by_eval.pt") `
    --init-escort (Join-Path $gcInitDir "dqn_attacker_escort_gc_best_by_eval.pt") `
    --init-guard (Join-Path $gcInitDir "dqn_attacker_guard_gc_best_by_eval.pt") `
    --output-dir $OutputDir `
    --episodes 2000 --curriculum-episodes 1000 --navigation-bootstrap-episodes 750 `
    --navigation-demo-weight 1.0 --navigation-retention-weight 0.20 --n-step 5 --final-mix 0.20 `
    --start-stats 50 10 0 --final-stats 100 60 40 --seed 2026098000 `
    --eval-seeds 3026091700 5026091700 6026091700 4026091700 7026091700 8026091700 `
    --holdout-seeds 9026092000 10026092000 11026092000 `
    --eval-interval 250 --eval-episodes 30 --holdout-episodes 50 `
    --lr 0.00002 --max-updates 32 --freeze-phases guard `
    --feature-only-phases carry escort

exit $LASTEXITCODE
