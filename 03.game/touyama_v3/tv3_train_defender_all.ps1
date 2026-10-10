# Run from touyama_v3: .\tv3_train_defender_all.ps1
# Training settings are defined at the top of each Python script.
# Opponent AI keys:
# - gc_v1
# - concon_v1
# - omoko_v1
# - fnatic_v3
# - frc_v1
# - toru_ai_v4
# - ghost_champions_v2
# ghost_champions_v2 is not selectable yet.
# One AI: .\tv3_train_defender_all.ps1 -Opponents frc_v1
# Multiple AIs: .\tv3_train_defender_all.ps1 -Opponents gc_v1,frc_v1
# With powershell -File, use one quoted comma-separated argument:
# powershell -NoProfile -ExecutionPolicy Bypass -File .\tv3_train_defender_all.ps1 -Opponents "gc_v1,frc_v1"
param(
    [string[]]$Opponents = @()
)

$ErrorActionPreference = 'Stop'

$PythonCommand = 'py'
# Set persistent targets here; empty uses each Python script's target constants.
$DefaultOpponents = @()
$AvailableOpponents = @('gc_v1', 'concon_v1', 'omoko_v1', 'fnatic_v3', 'frc_v1', 'toru_ai_v4')
$SelectedOpponents = $DefaultOpponents
if ($PSBoundParameters.ContainsKey('Opponents')) {
    $SelectedOpponents = $Opponents
}
$SelectedOpponents = @($SelectedOpponents | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Select-Object -Unique)
foreach ($Opponent in $SelectedOpponents) {
    if ($AvailableOpponents -cnotcontains $Opponent) {
        throw "Unknown opponent AI: '$Opponent'. Available: $($AvailableOpponents -join ', ')"
    }
}
$PythonArguments = @()
if ($SelectedOpponents.Count -gt 0) {
    $PythonArguments = @('--opponents') + $SelectedOpponents
}
$TrainingScripts = @(
    'tv3_train_defender_analysis.py'
    'tv3_train_defender_search.py'
    'tv3_collect_defender_retake.py'
    'tv3_train_defender_retake.py'
)

Push-Location -LiteralPath $PSScriptRoot
try {
    foreach ($TrainingScript in $TrainingScripts) {
        Write-Host "Starting: $PythonCommand $TrainingScript $($PythonArguments -join ' ')"
        & $PythonCommand $TrainingScript @PythonArguments
        if ($LASTEXITCODE -ne 0) {
            throw "$TrainingScript failed (exit code: $LASTEXITCODE). Remaining steps were stopped."
        }
    }
    Write-Host 'All defender training steps completed.'
}
finally {
    Pop-Location
}
