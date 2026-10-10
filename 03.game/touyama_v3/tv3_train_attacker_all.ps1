# Run from touyama_v3: .\tv3_train_attacker_all.ps1
# Training settings are defined at the top of each Python script.
$ErrorActionPreference = 'Stop'

$PythonCommand = 'py'
$TrainingScripts = @(
    'tv3_train_attacker_analysis.py'
    'tv3_train_attacker_plant.py'
    'tv3_collect_attacker_guard.py'
    'tv3_train_attacker_guard.py'
)

Push-Location -LiteralPath $PSScriptRoot
try {
    foreach ($TrainingScript in $TrainingScripts) {
        Write-Host "Starting: $PythonCommand $TrainingScript"
        & $PythonCommand $TrainingScript
        if ($LASTEXITCODE -ne 0) {
            throw "$TrainingScript failed (exit code: $LASTEXITCODE). Remaining steps were stopped."
        }
    }
    Write-Host 'All attacker training steps completed.'
}
finally {
    Pop-Location
}
