param(
    [string]$Python = "D:\git\python\python.exe",
    [int]$SearchEpisodes = 4000,
    [int]$RetakeEpisodes = 4000,
    [int]$EvalEvery = 250,
    [int]$EvalEpisodesPerSeed = 30
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable was not found: $Python"
}

$Workspace = Split-Path -Parent $PSScriptRoot
Push-Location $Workspace
try {
    Write-Host "=== Backing up current defender runtime models ==="
    & $Python -u -m gc_v1.promote_defender_gc --backup-only
    if ($LASTEXITCODE -ne 0) {
        throw "Defender model backup failed with exit code $LASTEXITCODE"
    }

    Write-Host "=== Training defender search (movement + facing + ability + ultimate) ==="
    & $Python -u -m gc_v1.train_defender_search_gc `
        --episodes $SearchEpisodes `
        --eval-every $EvalEvery `
        --eval-episodes-per-seed $EvalEpisodesPerSeed
    if ($LASTEXITCODE -ne 0) {
        throw "Defender search training failed with exit code $LASTEXITCODE"
    }

    Write-Host "=== Training defender retake (movement + facing + ability + ultimate) ==="
    & $Python -u -m gc_v1.train_defender_retake_gc `
        --episodes $RetakeEpisodes `
        --eval-every $EvalEvery `
        --eval-episodes-per-seed $EvalEpisodesPerSeed
    if ($LASTEXITCODE -ne 0) {
        throw "Defender retake training failed with exit code $LASTEXITCODE"
    }

    Write-Host "=== Defender training complete; best_by_eval models are active ==="
} finally {
    Pop-Location
}
