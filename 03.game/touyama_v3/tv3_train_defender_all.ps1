# Run from touyama_v3: .\tv3_train_defender_all.ps1
# Training settings are defined at the top of each Python script.
$ErrorActionPreference = 'Stop'

$PythonCommand = 'py'
$TrainingScripts = @(
    'tv3_train_defender_analysis.py'
    'tv3_train_defender_search.py'
    'tv3_collect_defender_retake.py'
    'tv3_train_defender_retake.py'
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
    Write-Host 'All defender training steps completed.'
}
finally {
    Pop-Location
}
