param(
    [string]$Python = "D:\git\python\python.exe",
    [string]$CarrySource = "",
    [string]$EscortSource = "",
    [string]$RetrieveSource = "",
    [string]$GuardSource = "",
    [switch]$Selected,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Workspace = Split-Path -Parent $PSScriptRoot
Push-Location $Workspace
try {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Python executable was not found: $Python"
    }
    $arguments = @("-u", "-m", "gc_v1.promote_attacker_gc")
    foreach ($item in @(
        @("carry", $CarrySource), @("escort", $EscortSource),
        @("retrieve", $RetrieveSource), @("guard", $GuardSource)
    )) {
        if ($item[1]) { $arguments += @("--$($item[0])-source", $item[1]) }
    }
    if ($Selected) { $arguments += "--selected" }
    if ($DryRun) { $arguments += "--dry-run" }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Attacker model promotion failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
