param(
    [string]$InputDir = (Join-Path $PSScriptRoot "data\attacker_gc_curriculum_v17"),
    [string]$OutputDir = ""
)

$latestBundlePath = Join-Path $InputDir "latest_bundle.json"
if (-not (Test-Path -LiteralPath $latestBundlePath)) {
    throw "latest_bundle.json was not found: $latestBundlePath"
}
$latestBundle = Get-Content -Raw -Encoding UTF8 -LiteralPath $latestBundlePath | ConvertFrom-Json
$episodeOffset = [int]$latestBundle.episode
$remainingEpisodes = 3000 - $episodeOffset
if ($remainingEpisodes -le 0) {
    throw "Training already reached episode $episodeOffset (target: 3000)."
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $PSScriptRoot "data\attacker_gc_curriculum_v17_resume_$episodeOffset"
}

$gcTrainer = Join-Path $PSScriptRoot "train_attacker_gc_real_curriculum.py"
& "D:\git\python\python.exe" -X utf8 $gcTrainer `
    --init-carry (Join-Path $InputDir "dqn_attacker_carry_gc_latest.pt") `
    --init-escort (Join-Path $InputDir "dqn_attacker_escort_gc_latest.pt") `
    --init-guard (Join-Path $InputDir "dqn_attacker_guard_gc_latest.pt") `
    --output-dir $OutputDir --episode-offset $episodeOffset `
    --episodes $remainingEpisodes --curriculum-episodes 1500 --navigation-bootstrap-episodes 1250 `
    --navigation-demo-weight 1.5 --navigation-retention-weight 0.35 `
    --ultimate-classification-weight 2.0 --n-step 5 --final-mix 0.25 `
    --start-stats 50 10 0 --final-stats 100 60 40 --seed 2026103000 `
    --eval-seeds 3026091700 5026091700 6026091700 4026091700 7026091700 8026091700 `
    --holdout-seeds 9026092000 10026092000 11026092000 `
    --eval-interval 250 --eval-episodes 30 --holdout-episodes 50 `
    --lr 0.00005 --max-updates 32 `
    --feature-only-phases carry escort guard `
    --action-only-phases carry escort guard

exit $LASTEXITCODE
