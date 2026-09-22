<#
Isolated Escort movement training with a fake-wait bodyguard objective.
Repeated invocations overwrite the same current training/selection artifacts.
#>

$ErrorActionPreference = "Stop"
$pythonPath = "D:\git\python\python.exe"
$gcDir = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$sourceDir = Join-Path $gcDir "data\attacker_gc_curriculum_v22_escort_support_20260922_094149\training"
$runRoot = Join-Path $gcDir "data\attacker_gc_escort_support_current"
$trainingDir = Join-Path $runRoot "training"
$selectedDir = Join-Path $runRoot "selected"

$carryModel = Join-Path $sourceDir "dqn_attacker_carry_gc_best_by_eval.pt"
$escortModel = Join-Path $sourceDir "dqn_attacker_escort_gc_best_facing.pt"
$guardModel = Join-Path $sourceDir "dqn_attacker_guard_gc_best_by_eval.pt"
foreach ($path in @($pythonPath, $carryModel, $escortModel, $guardModel)) {
  if (-not (Test-Path -LiteralPath $path)) {
    throw "Required file not found: $path"
  }
}

Write-Host "=== GC v22 isolated Escort support training ===" -ForegroundColor Cyan
Write-Host "Run directory: $runRoot"
Write-Host "Carry, Guard, facing, ability and ultimate are frozen."

$trainArgs = @(
  "-X", "utf8",
  (Join-Path $gcDir "train_attacker_gc_real_curriculum.py"),
  "--init-carry", $carryModel,
  "--init-escort", $escortModel,
  "--init-guard", $guardModel,
  "--output-dir", $trainingDir,
  "--overwrite-output",
  "--episodes", "3000",
  "--episode-offset", "1250",
  "--curriculum-episodes", "500",
  "--navigation-bootstrap-episodes", "350",
  "--eval-interval", "250",
  "--eval-episodes", "100",
  "--holdout-episodes", "50",
  "--seed", "2026092700",
  "--eval-seeds", "3126092700", "5126092700", "7126092700",
  "--holdout-seeds", "9226092800", "10226092800", "11226092800",
  "--lr", "0.00002",
  "--max-updates", "4",
  "--max-no-entry-rate", "0.30",
  "--max-timeout-rate", "0.10",
  "--max-carrier-death-increase", "0.05",
  "--max-plant-regression", "0.05",
  "--navigation-demo-weight", "1.0",
  "--navigation-retention-weight", "0.15",
  "--ultimate-classification-weight", "0.0",
  "--facing-supervision-weight", "0.0",
  "--freeze-phases", "carry", "guard",
  "--movement-only-phases", "escort",
  "--movement-td-updates", "1",
  "--movement-demo-updates", "4",
  "--relative-schedules",
  "--early-stop-patience", "6",
  "--movement-guardrail-patience", "2"
)
& $pythonPath $trainArgs
if ($LASTEXITCODE -ne 0) {
  throw "Escort support training failed with exit code $LASTEXITCODE"
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
  "--max-no-entry-rate", "0.30",
  "--max-timeout-rate", "0.10",
  "--max-carrier-death-increase", "0.05",
  "--max-plant-regression", "0.05",
  "--holdout-seeds", "1326092900", "1426092900", "1526092900",
  "--holdout-episodes", "50"
)
& $pythonPath $selectArgs
if ($LASTEXITCODE -ne 0) {
  throw "Escort support selection failed with exit code $LASTEXITCODE"
}

Write-Host "=== GC v22 Escort support run completed ===" -ForegroundColor Green
Write-Host "Training: $trainingDir"
Write-Host "Selected: $selectedDir"
