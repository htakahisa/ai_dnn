<#
GC v20 phase-facing A/B selection script
実行方法: powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\run_phase_selection.ps1
#>

$sourceDir = ".\gc_v1\data\attacker_gc_curriculum_v20_independent_facing"
$pythonPath = "D:\git\python\python.exe"

# 引数を配列で渡すことでPowerShellの構文解析エラーを回避
$arguments = @(
  "-X", "utf8",
  ".\gc_v1\select_gc_phase_facing_ab.py",
  "--models-dir", $sourceDir,
  "--eval-seeds", "3026091700", "5026091700", "6026091700", "4026091700", "7026091700", "8026091700",
  "--holdout-seeds", "9026092000", "10026092000", "11026092000",
  "--allow-data-revision-mismatch"
)

Write-Host "Starting v20 phase-facing A/B selection..." -ForegroundColor Cyan
& $pythonPath $arguments

if ($LASTEXITCODE -eq 0) {
  Write-Host "`nPhase selection completed successfully! Best models saved to: $sourceDir" -ForegroundColor Green
} else {
  Write-Error "Phase selection failed with exit code $LASTEXITCODE"
  exit 1
}