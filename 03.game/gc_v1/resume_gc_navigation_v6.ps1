param(
    [string]$StoppedDir = (Join-Path $PSScriptRoot "data\attacker_gc_curriculum_v6"),
    [string]$OutputDir = (Join-Path $PSScriptRoot "data\attacker_gc_curriculum_v6_resume_750")
)

$gcBundlePath = Join-Path $StoppedDir "latest_bundle.json"
if (-not (Test-Path -LiteralPath $gcBundlePath)) {
    throw "latest_bundle.json was not found: $gcBundlePath"
}
$gcBundle = Get-Content -Raw -Encoding UTF8 -LiteralPath $gcBundlePath | ConvertFrom-Json
$gcOffset = [int]$gcBundle.episode
$gcTotalEpisodes = 2000
$gcRemaining = $gcTotalEpisodes - $gcOffset
if ($gcRemaining -le 0) {
    throw "No remaining episodes: saved=$gcOffset target=$gcTotalEpisodes"
}

$gcTrainer = Join-Path $PSScriptRoot "train_attacker_gc_real_curriculum.py"
& "D:\git\python\python.exe" -X utf8 $gcTrainer `
    --init-carry (Join-Path $StoppedDir "dqn_attacker_carry_gc_latest.pt") `
    --init-escort (Join-Path $StoppedDir "dqn_attacker_escort_gc_latest.pt") `
    --init-guard (Join-Path $StoppedDir "dqn_attacker_guard_gc_latest.pt") `
    --output-dir $OutputDir `
    --episode-offset $gcOffset --episodes $gcRemaining --curriculum-episodes 1000 `
    --navigation-bootstrap-episodes 500 --navigation-demo-weight 1.0 `
    --navigation-retention-weight 0.15 --n-step 5 --final-mix 0.20 `
    --start-stats 50 10 0 --final-stats 100 60 40 `
    --eval-seeds 3026091700 5026091700 6026091700 4026091700 7026091700 8026091700 `
    --holdout-seeds 9026092000 10026092000 11026092000 `
    --eval-interval 250 --eval-episodes 30 --holdout-episodes 50 `
    --lr 0.00002 --max-updates 32

exit $LASTEXITCODE
