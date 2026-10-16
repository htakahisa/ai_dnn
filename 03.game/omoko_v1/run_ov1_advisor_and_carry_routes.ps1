$ErrorActionPreference = 'Stop'

$scripts = @(
    'ability_advisor_v1/ov1_collect_attacker_ability_advisor.py',
    'ability_advisor_v1/ov1_train_attacker_ability_advisor.py',
    'ability_advisor_v1/ov1_evaluate_attacker_ability_advisor.py',
    'ov1_evaluate_attacker_carry_routes.py',
    'ov1_summarize_attacker_carry_routes.py'
)

Push-Location $PSScriptRoot
try {
    foreach ($script in $scripts) {
        Write-Host ''
        Write-Host "Running: python $script"
        & python $script
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            throw "python $script failed with exit code $exitCode. Remaining scripts were not run."
        }
    }
}
finally {
    Pop-Location
}

Write-Host ''
Write-Host 'All advisor and carry-route scripts completed successfully.'
