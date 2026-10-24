# Run from touyama_v3: .\tv3_train_attacker_all.ps1
# Training settings are defined at the top of each Python script.
# Opponent AI keys:
# - gc_v1
# - concon_v1
# - omoko_v1
# - fnatic_v3
# - frc_v1
# - toru_ai_v4
# - ghost_champions_v2
# One AI: .\tv3_train_attacker_all.ps1 -Opponents frc_v1
# Multiple AIs: .\tv3_train_attacker_all.ps1 -Opponents gc_v1,frc_v1
# With powershell -File, use one quoted comma-separated argument:
# powershell -NoProfile -ExecutionPolicy Bypass -File .\tv3_train_attacker_all.ps1 -Opponents "gc_v1,frc_v1"
param(
    [string[]]$Opponents = @()
)

$ErrorActionPreference = 'Stop'

$PythonCommand = 'py'
$PlantTrainingProfile = 'common' # analysis再学習後は共通plantを再構築。右専用は単独実行。
# Set persistent targets here; empty uses each Python script's target constants.
$DefaultOpponents = @()
$AvailableOpponents = @('concon_v1', 'omoko_v1', 'fnatic_v3', 'frc_v1', 'toru_ai_v4', 'ghost_champions_v2')
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
    'tv3_train_attacker_analysis.py'
    'tv3_train_attacker_plant.py'
    'tv3_collect_attacker_guard.py'
    'tv3_train_attacker_guard.py'
)

Push-Location -LiteralPath $PSScriptRoot
try {
    foreach ($TrainingScript in $TrainingScripts) {
        $StepArguments = @($PythonArguments)
        if ($TrainingScript -eq 'tv3_train_attacker_plant.py') {
            $StepArguments += @('--training-profile', $PlantTrainingProfile)
        }
        Write-Host "Starting: $PythonCommand $TrainingScript $($StepArguments -join ' ')"
        & $PythonCommand $TrainingScript @StepArguments
        if ($LASTEXITCODE -ne 0) {
            throw "$TrainingScript failed (exit code: $LASTEXITCODE). Remaining steps were stopped."
        }
    }
    Write-Host 'All attacker training steps completed.'
}
finally {
    Pop-Location
}
