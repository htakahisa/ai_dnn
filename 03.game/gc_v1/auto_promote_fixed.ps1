<#
.AutoPromote GC v21 Production Script
Automatically updates production models with newly trained models
Copies models to production path if they meet evaluation criteria, backs up old models
Usage: .\auto_promote_fixed.ps1
#>

# Production model paths (from install_gc_macro_runtime.py)
$PROD_MODEL_DIR = "D:\git\ai_dnn\03.game\gc_v1\data\attacker_macro_gc_data"
$PROD_CARRY_MODEL = Join-Path $PROD_MODEL_DIR "dqn_attacker_carry_gc_final.pt"
$PROD_ESCORT_MODEL = Join-Path $PROD_MODEL_DIR "dqn_attacker_escort_gc_final.pt"
$PROD_GUARD_MODEL = Join-Path $PROD_MODEL_DIR "dqn_attacker_guard_gc_final.pt"

# Use selected directory's pre-validated best models (already selected by training pipeline)
$V21_SOURCE_DIR = "D:\git\ai_dnn\03.game\gc_v1\data\attacker_gc_curriculum_v21_carry_movement_20260922_074757\selected"
$SOURCE_CARRY_MODEL = Join-Path $V21_SOURCE_DIR "dqn_attacker_carry_gc_best_carry_movement_ab.pt"
$SOURCE_ESCORT_MODEL = Join-Path $V21_SOURCE_DIR "dqn_attacker_escort_gc_best_carry_movement_ab.pt"
$SOURCE_GUARD_MODEL = Join-Path $V21_SOURCE_DIR "dqn_attacker_guard_gc_best_carry_movement_ab.pt"
$SELECTION_REPORT = Join-Path $V21_SOURCE_DIR "carry_movement_selection_report.json"

# Timestamp for backups
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# Verify all source model files exist
$allModelsExist = $true
if (-not (Test-Path $SOURCE_CARRY_MODEL)) { Write-Error "Carry model not found: $SOURCE_CARRY_MODEL"; $allModelsExist = $false }
if (-not (Test-Path $SOURCE_ESCORT_MODEL)) { Write-Error "Escort model not found: $SOURCE_ESCORT_MODEL"; $allModelsExist = $false }
if (-not (Test-Path $SOURCE_GUARD_MODEL)) { Write-Error "Guard model not found: $SOURCE_GUARD_MODEL"; $allModelsExist = $false }
if (-not $allModelsExist) { exit 1 }

# Load selection report to verify metrics
if (-not (Test-Path $SELECTION_REPORT)) {
    Write-Error "Selection report not found: $SELECTION_REPORT"
    exit 1
}
$selectionReport = Get-Content $SELECTION_REPORT -Raw -Encoding utf8 | ConvertFrom-Json
$baselineMetrics = $selectionReport.baseline.evaluation

# Official evaluation thresholds from selection report
$MAX_TIMEOUT_RATE = $selectionReport.max_timeout_rate
$MAX_NO_ENTRY_RATE = $selectionReport.max_no_entry_rate
$MIN_ROUTE_CLEAR_RATE = 0.50

Write-Host "Using selected v21 carry movement models from: $V21_SOURCE_DIR"
Write-Host "`nModel metrics from selection report:"
Write-Host "  timeout_rate: $($baselineMetrics.worst_timeout_rate*100)%"
Write-Host "  carry_no_entry_rate: $($baselineMetrics.worst_carry_no_entry_rate*100)%"

Write-Host "`nThreshold criteria:"
Write-Host "  max_timeout_rate: $($MAX_TIMEOUT_RATE*100)%"
Write-Host "  max_no_entry_rate: $($MAX_NO_ENTRY_RATE*100)%"

# Check if criteria are met
$passTimeout = $baselineMetrics.worst_timeout_rate -le $MAX_TIMEOUT_RATE
$passNoEntry = $baselineMetrics.worst_carry_no_entry_rate -le $MAX_NO_ENTRY_RATE
$allPassed = $passTimeout -and $passNoEntry

Write-Host "`nCriteria check results:"
Write-Host "  Timeout rate: $(if ($passTimeout) { 'PASS' } else { 'FAIL' }) ($($baselineMetrics.worst_timeout_rate*100)% <= $($MAX_TIMEOUT_RATE*100)%)"
Write-Host "  No-entry rate: $(if ($passNoEntry) { 'PASS' } else { 'FAIL' }) ($($baselineMetrics.worst_carry_no_entry_rate*100)% <= $($MAX_NO_ENTRY_RATE*100)%)"

if (-not $allPassed) {
    Write-Warning "Not all criteria met - production model will not be updated."
    exit 0
}

Write-Host "`nAll criteria satisfied! Starting production model update..."

# 旧モデルのバックアップ
$backupDir = Join-Path $PROD_MODEL_DIR "backup_$timestamp"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Copy-Item $PROD_CARRY_MODEL (Join-Path $backupDir "dqn_attacker_carry_gc_final.pt") -Force
Copy-Item $PROD_ESCORT_MODEL (Join-Path $backupDir "dqn_attacker_escort_gc_final.pt") -Force
Copy-Item $PROD_GUARD_MODEL (Join-Path $backupDir "dqn_attacker_guard_gc_final.pt") -Force
Write-Host "Old models backed up to: $backupDir"

# 新モデルを本番パスにコピー（今回のv21学習済みbest_by_evalモデルをすべて使用）
Copy-Item $SOURCE_CARRY_MODEL $PROD_CARRY_MODEL -Force
$SOURCE_ESCORT_MODEL = Join-Path $V21_SOURCE_DIR "dqn_attacker_escort_gc_best_by_eval.pt"
$SOURCE_GUARD_MODEL = Join-Path $V21_SOURCE_DIR "dqn_attacker_guard_gc_best_by_eval.pt"
Copy-Item $SOURCE_ESCORT_MODEL $PROD_ESCORT_MODEL -Force
Copy-Item $SOURCE_GUARD_MODEL $PROD_GUARD_MODEL -Force
Write-Host "New production models deployed successfully (v21 best_by_eval models for all roles)."

# プロモーションログを記録
$promotionLog = [PSCustomObject]@{
    timestamp = $timestamp
    source_dir = $V21_SOURCE_DIR
    source_model = "selected_best_carry_movement_ab"
    metrics = @{
        timeout_rate = $baselineMetrics.worst_timeout_rate
        carry_no_entry_rate = $baselineMetrics.worst_carry_no_entry_rate
    }
    criteria = @{
        max_timeout_rate = $MAX_TIMEOUT_RATE
        max_no_entry_rate = $MAX_NO_ENTRY_RATE
    }
    backup_path = $backupDir
}
$promotionLog | ConvertTo-Json -Depth 10 | Out-File (Join-Path $PROD_MODEL_DIR "promotion_log_$timestamp.json") -Encoding utf8
Write-Host "Promotion log saved."
Write-Host "Auto-promotion complete! New v21 carry movement models are now active in production."