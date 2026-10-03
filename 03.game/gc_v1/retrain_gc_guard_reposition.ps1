param(
  [int]$Episodes = 1000,
  [string]$OutputDir = "",
  [string]$PythonPath = "D:\git\python\python.exe"
)
$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if (-not $OutputDir) {
  $OutputDir = Join-Path $PSScriptRoot ("data\guard_reposition_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
}
Push-Location -LiteralPath $workspace
try {
  & $PythonPath -X utf8 (Join-Path $PSScriptRoot "train_guard_reposition_gc.py") `
    --episodes $Episodes --output-dir $OutputDir
  if ($LASTEXITCODE -ne 0) { throw "Guard training failed: exit code $LASTEXITCODE" }
} finally {
  Pop-Location
}
