<#
Joint Carry/Escort movement training with protected tactical heads.
Repeated invocations overwrite the same current training/selection artifacts.
#>

$ErrorActionPreference = "Stop"
$pythonPath = "D:\git\python\python.exe"
$gcDir = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$legacySourceDir = Join-Path $gcDir "data\attacker_gc_curriculum_v22_escort_support_20260922_094149\training"
$runRoot = Join-Path $gcDir "data\attacker_gc_escort_support_current"
$trainingDir = Join-Path $runRoot "training"
$selectedDir = Join-Path $runRoot "selected"
$snapshotDir = Join-Path $runRoot "source"
$snapshotBundle = Join-Path $snapshotDir "source_bundle.json"
$sourceEpisode = 1250

# Only a previously packaged safe winner may seed the next run. Never copy an
# evaluation winner that the selector rejected for a safety regression.
$selectedBundle = Join-Path $selectedDir "best_escort_support_ab_bundle.json"
$selectedModels = @{
  carry = Join-Path $selectedDir "dqn_attacker_carry_gc_best_escort_support_ab.pt"
  escort = Join-Path $selectedDir "dqn_attacker_escort_gc_best_escort_support_ab.pt"
  guard = Join-Path $selectedDir "dqn_attacker_guard_gc_best_escort_support_ab.pt"
}
if ((Test-Path -LiteralPath $selectedBundle) -and
    -not (@($selectedModels.Values | Where-Object { -not (Test-Path -LiteralPath $_) }).Count)) {
  New-Item -ItemType Directory -Path $snapshotDir -Force | Out-Null
  foreach ($phase in @("carry", "escort", "guard")) {
    Copy-Item -LiteralPath $selectedModels[$phase] -Destination (Join-Path $snapshotDir "dqn_attacker_${phase}_gc_source.pt") -Force
  }
  Copy-Item -LiteralPath $selectedBundle -Destination $snapshotBundle -Force
}

$snapshotModels = @{
  carry = Join-Path $snapshotDir "dqn_attacker_carry_gc_source.pt"
  escort = Join-Path $snapshotDir "dqn_attacker_escort_gc_source.pt"
  guard = Join-Path $snapshotDir "dqn_attacker_guard_gc_source.pt"
}
if ((Test-Path -LiteralPath $snapshotBundle) -and
    -not (@($snapshotModels.Values | Where-Object { -not (Test-Path -LiteralPath $_) }).Count)) {
  $carryModel = $snapshotModels.carry
  $escortModel = $snapshotModels.escort
  $guardModel = $snapshotModels.guard
  $sourceMetadata = Get-Content -LiteralPath $snapshotBundle -Raw | ConvertFrom-Json
  if ($null -eq $sourceMetadata.episode) {
    throw "Source bundle has no episode: $snapshotBundle"
  }
  $sourceEpisode = [int]$sourceMetadata.episode
} else {
  $carryModel = Join-Path $legacySourceDir "dqn_attacker_carry_gc_best_by_eval.pt"
  $escortModel = Join-Path $legacySourceDir "dqn_attacker_escort_gc_best_facing.pt"
  $guardModel = Join-Path $legacySourceDir "dqn_attacker_guard_gc_best_by_eval.pt"
}
foreach ($path in @($pythonPath, $carryModel, $escortModel, $guardModel)) {
  if (-not (Test-Path -LiteralPath $path)) {
    throw "Required file not found: $path"
  }
}

# Continue learning from the last candidate, while keeping the previously
# selected safe bundle as the independent deployment baseline. Copy first:
# --overwrite-output replaces the training directory below.
$trainCarryModel = $carryModel
$trainEscortModel = $escortModel
$trainGuardModel = $guardModel
$trainSourceEpisode = $sourceEpisode
$latestBundle = Join-Path $trainingDir "latest_bundle.json"
$latestModels = @{
  carry = Join-Path $trainingDir "dqn_attacker_carry_gc_latest.pt"
  escort = Join-Path $trainingDir "dqn_attacker_escort_gc_latest.pt"
  guard = Join-Path $trainingDir "dqn_attacker_guard_gc_latest.pt"
}
if ((Test-Path -LiteralPath $latestBundle) -and
    -not (@($latestModels.Values | Where-Object { -not (Test-Path -LiteralPath $_) }).Count)) {
  $latestMetadata = Get-Content -LiteralPath $latestBundle -Raw | ConvertFrom-Json
  if ($null -ne $latestMetadata.episode -and [int]$latestMetadata.episode -gt $sourceEpisode) {
    $resumeDir = Join-Path $runRoot "resume"
    New-Item -ItemType Directory -Path $resumeDir -Force | Out-Null
    foreach ($phase in @("carry", "escort", "guard")) {
      Copy-Item -LiteralPath $latestModels[$phase] -Destination (Join-Path $resumeDir "dqn_attacker_${phase}_gc_resume.pt") -Force
    }
    $trainCarryModel = Join-Path $resumeDir "dqn_attacker_carry_gc_resume.pt"
    $trainEscortModel = Join-Path $resumeDir "dqn_attacker_escort_gc_resume.pt"
    $trainGuardModel = Join-Path $resumeDir "dqn_attacker_guard_gc_resume.pt"
    $trainSourceEpisode = [int]$latestMetadata.episode
  }
}

Write-Host "=== GC joint Carry/Escort movement training ===" -ForegroundColor Cyan
Write-Host "Run directory: $runRoot"
Write-Host "Safe baseline episode: $sourceEpisode; training starts from episode: $trainSourceEpisode"
Write-Host "Guard, ability and ultimate are frozen; Carry/Escort movement and facing train together."

$trainArgs = @(
  "-X", "utf8",
  (Join-Path $gcDir "train_attacker_gc_real_curriculum.py"),
  "--init-carry", $trainCarryModel,
  "--init-escort", $trainEscortModel,
  "--init-guard", $trainGuardModel,
  "--output-dir", $trainingDir,
  "--overwrite-output",
  "--episodes", "1000",
  "--episode-offset", "$trainSourceEpisode",
  "--curriculum-episodes", "350",
  "--navigation-bootstrap-episodes", "300",
  "--eval-interval", "250",
  "--eval-episodes", "100",
  "--holdout-episodes", "50",
  "--seed", "2026092700",
  "--eval-seeds", "3126092700", "5126092700", "7126092700",
  "--holdout-seeds", "9226092800", "10226092800", "11226092800",
  "--lr", "0.00001",
  "--lr-decay-start", "250",
  "--lr-decay-episodes", "500",
  "--lr-final-scale", "0.20",
  "--max-updates", "4",
  "--max-no-entry-rate", "0.30",
  "--max-timeout-rate", "0.10",
  "--max-carrier-death-increase", "0.05",
  "--max-plant-regression", "0.05",
  "--navigation-demo-weight", "1.0",
  "--orb-demo-weight", "4.0",
  "--orb-demo-updates", "4",
  "--orb-collection-updates", "4",
  "--orb-collection-lr-multiplier", "20",
  "--orb-teacher-final-probability", "0.05",
  "--navigation-retention-weight", "0.10",
  "--ultimate-classification-weight", "0.0",
  "--facing-supervision-weight", "1.0",
  "--freeze-phases", "guard",
  "--movement-only-phases", "carry", "escort",
  "--train-facing-with-movement",
  "--movement-td-updates", "1",
  "--movement-demo-updates", "2",
  "--relative-schedules",
  "--early-stop-patience", "3",
  "--movement-guardrail-patience", "2"
)
& $pythonPath $trainArgs
if ($LASTEXITCODE -ne 0) {
  throw "Carry/Escort movement training failed with exit code $LASTEXITCODE"
}

Write-Host "=== Packaging the evaluation-seed winner ===" -ForegroundColor Cyan
$selectArgs = @(
  "-X", "utf8",
  (Join-Path $gcDir "select_gc_escort_support_v22.py"),
  "--source-carry", $carryModel,
  "--source-escort", $escortModel,
  "--source-guard", $guardModel,
  "--training-dir", $trainingDir,
  "--output-dir", $selectedDir,
  "--overwrite-output",
  "--allow-data-revision-mismatch",
  "--allow-facing-changes",
  "--max-no-entry-rate", "0.30",
  "--max-timeout-rate", "0.10",
  "--max-quiet-stall-increase", "0.05",
  "--max-carrier-death-increase", "0.05",
  "--max-plant-regression", "0.05",
  "--holdout-seeds", "1326092900", "1426092900", "1526092900",
  "--holdout-episodes", "50"
)
& $pythonPath $selectArgs
if ($LASTEXITCODE -ne 0) {
  throw "Carry/Escort movement selection failed with exit code $LASTEXITCODE"
}

Write-Host "=== Deploying selected attacker models (with automatic backup) ===" -ForegroundColor Cyan
& $pythonPath -u -m gc_v1.promote_attacker_gc `
  --carry-source (Join-Path $selectedDir "dqn_attacker_carry_gc_best_escort_support_ab.pt") `
  --escort-source (Join-Path $selectedDir "dqn_attacker_escort_gc_best_escort_support_ab.pt") `
  --guard-source (Join-Path $selectedDir "dqn_attacker_guard_gc_best_escort_support_ab.pt")
if ($LASTEXITCODE -ne 0) {
  throw "Attacker model deployment failed with exit code $LASTEXITCODE"
}

Write-Host "=== GC Carry/Escort movement run completed ===" -ForegroundColor Green
Write-Host "Training: $trainingDir"
Write-Host "Selected: $selectedDir"
