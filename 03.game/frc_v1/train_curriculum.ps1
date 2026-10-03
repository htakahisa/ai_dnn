param(
    [string]$Python = 'D:\git\python\python.exe',
    [ValidateRange(1, 100000000)][int]$StepsPerStage = 10000,
    [ValidateRange(0, 1000000)][int]$TeacherSteps = 512,
    [ValidateRange(0, 1)][double]$ImitationWeight = 0.03,
    [int]$Seed = 1,
    [string]$Device = 'cpu',
    [string]$OutputDir = 'frc_v1/checkpoints'
)

$ErrorActionPreference = 'Stop'
$frcWorkspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Push-Location -LiteralPath $frcWorkspace
try {
    foreach ($side in @('A', 'D')) {
        $stages = @('threats', 'support', 'entry', $(if ($side -eq 'A') { 'attack' } else { 'defense' }), 'balemoon', 'match')
        $previousCheckpoint = $null
        foreach ($stage in $stages) {
            $checkpointName = if ($stage -eq 'match') { "${side}_policy.pt" } else { "${side}_${stage}.pt" }
            $outputPath = Join-Path $OutputDir $checkpointName
            $trainingArgs = @('-m', 'frc_v1.train', '--side', $side, '--stage', $stage,
                '--steps', "$StepsPerStage", '--seed', "$Seed", '--device', $Device, '--output', $outputPath)
            if ($previousCheckpoint) { $trainingArgs += @('--resume', $previousCheckpoint) }
            if ($stage -in @('support', 'entry', 'match') -and $TeacherSteps -gt 0) {
                $trainingArgs += @('--teacher-steps', "$TeacherSteps")
            }
            if ($stage -in @('attack', 'defense', 'match') -and $ImitationWeight -gt 0) {
                $trainingArgs += @('--imitation-weight', "$ImitationWeight")
            }
            & $Python @trainingArgs
            if ($LASTEXITCODE -ne 0) { throw "FRC training failed: side=$side stage=$stage" }
            $previousCheckpoint = $outputPath
        }
    }
} finally {
    Pop-Location
}
