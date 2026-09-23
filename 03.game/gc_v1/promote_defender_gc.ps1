param(
    [string]$Python = "D:\git\python\python.exe",
    [string]$SearchSource = "",
    [string]$RetakeSource = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Workspace = Split-Path -Parent $PSScriptRoot
Push-Location $Workspace
try {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Python executable was not found: $Python"
    }

    $arguments = @("-u", "-m", "gc_v1.promote_defender_gc")
    if ($SearchSource) { $arguments += @("--search-source", $SearchSource) }
    if ($RetakeSource) { $arguments += @("--retake-source", $RetakeSource) }
    if ($DryRun) { $arguments += "--dry-run" }

    & $Python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Defender model promotion failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
