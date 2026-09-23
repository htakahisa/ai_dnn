<#
GC v21 isolated Carry movement training.

Creates a timestamped run directory and never removes an earlier run.
Escort, Guard, facing, ability, plant, and ultimate behavior stay frozen.
#>

$ErrorActionPreference = "Stop"
$pythonPath = "D:\git\python\python.exe"
$gcDir = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$sourceDir = Join-Path $gcDir "data\attacker_gc_curriculum_v20_independent_facing"
$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runRoot = Join-Path $gcDir ("data\attacker_gc_curriculum_v21_carry_movement_" + $runStamp)
$warmDir = Join-Path $runRoot "warm"
$selectedDir = Join-Path $runRoot "selected"

$carryModel = Join-Path $sourceDir "dqn_attacker_carry_gc_best_phase_facing_ab.pt"
$escortModel = Join-Path $sourceDir "dqn_attacker_escort_gc_best_phase_facing_ab.pt"
$guardModel = Join-Path $sourceDir "dqn_attacker_guard_gc_best_phase_facing_ab.pt"

foreach ($path in @($pythonPath, $carryModel, $escortModel, $guardModel)) {
  if (-not (Test-Path -LiteralPath $path)) {
    throw "Required file not found: $path"
  }
}

Write-Host "=== GC v21 isolated Carry movement training ===" -ForegroundColor Cyan
Write-Host "Run directory: $runRoot"
Write-Host "Only Carry STAY/UP/DOWN/LEFT/RIGHT output rows are trainable."

$trainArgs = @(
  "-X", "utf8",
  (Join-Path $gcDir "train_attacker_gc_real_curriculum.py"),
  "--init-carry", $carryModel,
  "--init-escort", $escortModel,
  "--init-guard", $guardModel,
  "--output-dir", $warmDir,
  "--episodes", "10000",
  "--episode-offset", "1250",
  "--curriculum-episodes", "800",
  "--navigation-bootstrap-episodes", "500",
  "--eval-interval", "250",
  "--eval-episodes", "100",
  "--holdout-episodes", "50",
  "--seed", "2026092400",
  "--eval-seeds", "3126092400", "5126092400", "7126092400",
  "--holdout-seeds", "9226092500", "10226092500", "11226092500",
  "--lr", "0.00002",
  "--max-updates", "4",
  "--max-no-entry-rate", "0.30",
  "--max-timeout-rate", "0.10",
  "--navigation-demo-weight", "1.0",
  "--navigation-retention-weight", "0.15",
  "--ultimate-classification-weight", "0.0",
  "--facing-supervision-weight", "0.0",
  "--freeze-phases", "escort", "guard",
  "--movement-only-phases", "carry",
  "--movement-td-updates", "1",
  "--movement-demo-updates", "4",
  "--relative-schedules",
  "--early-stop-patience", "6",
  "--movement-guardrail-patience", "2",
  "--max-quiet-stall-increase", "0.05"
)

& $pythonPath $trainArgs
if ($LASTEXITCODE -ne 0) {
  throw "Carry movement training failed with exit code $LASTEXITCODE"
}

Write-Host "=== Packaging the evaluation-seed winner ===" -ForegroundColor Cyan
$selectArgs = @(
  "-X", "utf8",
  (Join-Path $gcDir "select_gc_carry_movement_v21.py"),
  "--source-dir", $sourceDir,
  "--warm-dir", $warmDir,
  "--output-dir", $selectedDir,
  "--max-no-entry-rate", "0.30",
  "--max-timeout-rate", "0.10",
  "--holdout-seeds", "1326092600", "1426092600", "1526092600",
  "--holdout-episodes", "50"
)

& $pythonPath $selectArgs
if ($LASTEXITCODE -ne 0) {
  throw "Carry movement selection failed with exit code $LASTEXITCODE"
}

Write-Host "=== Deploying selected attacker models (with automatic backup) ===" -ForegroundColor Cyan
& $pythonPath -u -m gc_v1.promote_attacker_gc `
  --carry-source (Join-Path $selectedDir "dqn_attacker_carry_gc_best_carry_movement_ab.pt") `
  --escort-source (Join-Path $selectedDir "dqn_attacker_escort_gc_best_carry_movement_ab.pt") `
  --guard-source (Join-Path $selectedDir "dqn_attacker_guard_gc_best_carry_movement_ab.pt")
if ($LASTEXITCODE -ne 0) {
  throw "Attacker model deployment failed with exit code $LASTEXITCODE"
}

Write-Host "=== GC v21 Carry movement run completed ===" -ForegroundColor Green
Write-Host "Training: $warmDir"
Write-Host "Selected: $selectedDir"
