<#
.AutoPromote GC v21 Production Script
本番環境で使用されるモデルを自動的に更新するスクリプト
学習済みモデルが評価基準を満たした場合、自動的に本番パスにコピーし、旧モデルはバックアップ
使用方法: .\auto_promote_gc_v21_prod.ps1 [--episode <エピソード番号>]
例: .\auto_promote_gc_v21_prod.ps1 --episode 1250
#>

# 引数解析
$targetEpisode = $null
for ($i = 0; $i -lt $args.Count; $i++) {
    if ($args[$i] -eq "--episode" -and $i + 1 -lt $args.Count) {
        $targetEpisode = [int]$args[$i + 1]
        $i++
    }
}

# 本番環境のモデルパス（install_gc_macro_runtime.py から抽出した正規パス）
$PROD_MODEL_DIR = "D:\git\ai_dnn\03.game\gc_v1\data\attacker_macro_gc_data"
$PROD_CARRY_MODEL = Join-Path $PROD_MODEL_DIR "dqn_attacker_carry_gc_final.pt"
$PROD_ESCORT_MODEL = Join-Path $PROD_MODEL_DIR "dqn_attacker_escort_gc_final.pt"
$PROD_GUARD_MODEL = Join-Path $PROD_MODEL_DIR "dqn_attacker_guard_gc_final.pt"

# 学習済みv21モデルのソースパス（最新の学習ディレクトリに修正）
$V21_SOURCE_DIR = "D:\git\ai_dnn\03.game\gc_v1\data\attacker_gc_curriculum_v22_escort_support_20260922_094149\training"
$SOURCE_CARRY_MODEL = Join-Path $V21_SOURCE_DIR "dqn_attacker_carry_gc_best_by_eval.pt"
$SOURCE_EVAL_HISTORY = Join-Path $V21_SOURCE_DIR "evaluation_history.json"

# コードベースの公式評価基準（select_gc_carry_movement_v21.py から引用）
$MAX_TIMEOUT_RATE = 0.05    # タイムアウト率5%以下
$MAX_NO_ENTRY_RATE = 0.35   # エントリー不実行率35%以下
$MIN_ROUTE_CLEAR_RATE = 0.50 # ルートクリア率50%以上

# タイムスタンプ
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# 評価履歴を読み込み
if (-not (Test-Path $SOURCE_EVAL_HISTORY)) {
    Write-Error "Evaluation history not found: $SOURCE_EVAL_HISTORY"
    exit 1
}

$evalHistory = Get-Content $SOURCE_EVAL_HISTORY -Raw -Encoding utf8 | ConvertFrom-Json
$bestEntry = $null
$bestScoreViolation = [float]::MaxValue

# エピソード指定がある場合は指定エピソードを使用
if ($targetEpisode -ne $null) {
    Write-Host "手動指定されたエピソードを使用します: $targetEpisode"
    $bestEntry = $evalHistory | Where-Object { $_.episode -eq $targetEpisode }
    if (-not $bestEntry) {
        Write-Error "指定されたエピソード $targetEpisode が評価履歴に見つかりません"
        exit 1
    }
}

# エピソードが指定されていない場合は自動で最良を選択
if ($targetEpisode -eq $null) {
    Write-Host "エピソードが指定されていません - 履歴から最良モデルを自動選択します..."
    # コードベースと同じ選択ロジックで最良エントリを選ぶ（存在するPTファイルだけ対象）
    foreach ($entry in $evalHistory) {
        $candidateModel = Join-Path $V21_SOURCE_DIR "dqn_attacker_carry_gc_ep$($entry.episode).pt"
        if (-not (Test-Path $candidateModel)) {
            continue # ファイルが存在しないエピソードはスキップ
        }
        $noEntry = $entry.carry_no_entry_rate
        $timeout = $entry.timeout_rate
        $violation = [Math]::Max(0.0, $noEntry - $MAX_NO_ENTRY_RATE) + [Math]::Max(0.0, $timeout - $MAX_TIMEOUT_RATE)
        
        if ($violation -lt $bestScoreViolation) {
            $bestScoreViolation = $violation
            $bestEntry = $entry
        }
    }
    if (-not $bestEntry) {
        # ファイルが存在しなくても評価履歴から最良のエピソードを選択する
        Write-Warning "評価履歴に有効なモデルファイルが見つかりませんでした - 履歴から最良のエピソードを選択してbest_by_eval.ptを使用します"
        foreach ($entry in $evalHistory) {
            $noEntry = $entry.carry_no_entry_rate
            $timeout = $entry.timeout_rate
            $violation = [Math]::Max(0.0, $noEntry - $MAX_NO_ENTRY_RATE) + [Math]::Max(0.0, $timeout - $MAX_TIMEOUT_RATE)
            
            if ($violation -lt $bestScoreViolation) {
                $bestScoreViolation = $violation
                $bestEntry = $entry
            }
        }
    }
}

# 選択されたエピソードに対応するPTファイルを設定
if (Test-Path (Join-Path $V21_SOURCE_DIR "dqn_attacker_carry_gc_ep$($bestEntry.episode).pt")) {
    $SOURCE_CARRY_MODEL = Join-Path $V21_SOURCE_DIR "dqn_attacker_carry_gc_ep$($bestEntry.episode).pt"
} else {
    # ファイルが存在しない場合はbest_by_eval.ptを使用
    Write-Warning "EP$($bestEntry.episode)のモデルファイルが見つかりません - best_by_eval.ptを使用します"
    $SOURCE_CARRY_MODEL = Join-Path $V21_SOURCE_DIR "dqn_attacker_carry_gc_best_by_eval.pt"
}

# モデルファイルの存在確認
if (-not (Test-Path $SOURCE_CARRY_MODEL)) {
    Write-Error "Model file not found: $SOURCE_CARRY_MODEL"
    exit 1
}

Write-Host "Using verified evaluation result (EP$($bestEntry.episode)):"
Write-Host "  timeout_rate: $([math]::Round($bestEntry.timeout_rate*100,2))%"
Write-Host "  carry_no_entry_rate: $([math]::Round($bestEntry.carry_no_entry_rate*100,2))%"
Write-Host "  designated_route_clear_rate: $([math]::Round($bestEntry.designated_route_clear_rate*100,2))%"

Write-Host "`nThreshold criteria (max limits):"
Write-Host "  max_timeout_rate: $($MAX_TIMEOUT_RATE*100)%"
Write-Host "  max_no_entry_rate: $($MAX_NO_ENTRY_RATE*100)%"
Write-Host "  min_route_clear_rate: $($MIN_ROUTE_CLEAR_RATE*100)%"

# 基準達成チェック
$passTimeout = $bestEntry.timeout_rate -le $MAX_TIMEOUT_RATE
$passNoEntry = $bestEntry.carry_no_entry_rate -le $MAX_NO_ENTRY_RATE
$passRouteClear = $bestEntry.designated_route_clear_rate -ge $MIN_ROUTE_CLEAR_RATE
$allPassed = $passTimeout -and $passNoEntry -and $passRouteClear

Write-Host "`nCriteria check results:"
Write-Host "  Timeout rate: $(if ($passTimeout) { 'PASS' } else { 'FAIL' }) (value: $([math]::Round($bestEntry.timeout_rate*100,2))% <= $($MAX_TIMEOUT_RATE*100)%)"
Write-Host "  No-entry rate: $(if ($passNoEntry) { 'PASS' } else { 'FAIL' }) (value: $([math]::Round($bestEntry.carry_no_entry_rate*100,2))% <= $($MAX_NO_ENTRY_RATE*100)%)"
Write-Host "  Route clear rate: $(if ($passRouteClear) { 'PASS' } else { 'FAIL' }) (value: $([math]::Round($bestEntry.designated_route_clear_rate*100,2))% >= $($MIN_ROUTE_CLEAR_RATE*100)%)"

# プロモーションログを記録（基準未達成でも常に保存）
$promotionLog = [PSCustomObject]@{
    timestamp = $timestamp
    source_episode = $bestEntry.episode
    source_dir = $V21_SOURCE_DIR
    source_model = "auto_selected_best"
    selection_violation = $bestScoreViolation
    criteria_met = $allPassed
    metrics = @{
        timeout_rate = $bestEntry.timeout_rate
        carry_no_entry_rate = $bestEntry.carry_no_entry_rate
        designated_route_clear_rate = $bestEntry.designated_route_clear_rate
    }
    criteria = @{
        max_timeout_rate = $MAX_TIMEOUT_RATE
        max_no_entry_rate = $MAX_NO_ENTRY_RATE
        min_route_clear_rate = $MIN_ROUTE_CLEAR_RATE
    }
    gaps_to_criteria = @{
                timeout_gap = [Math]::Max(0, $bestEntry.timeout_rate - $MAX_TIMEOUT_RATE)
                no_entry_gap = [Math]::Max(0, $bestEntry.carry_no_entry_rate - $MAX_NO_ENTRY_RATE)
                route_clear_gap = [Math]::Max(0, $MIN_ROUTE_CLEAR_RATE - $bestEntry.designated_route_clear_rate)
            }
    backup_path = $null
}

if (-not $allPassed) {
    Write-Warning "`nNot all criteria met - production model will not be updated."
    Write-Host "`nRemaining gaps to meet criteria:"
    Write-Host "  Timeout rate remaining: $([math]::Round(($bestEntry.timeout_rate - $MAX_TIMEOUT_RATE)*100,2))%"
            Write-Host "  No-entry rate remaining: $([math]::Max(0, [math]::Round(($bestEntry.carry_no_entry_rate - $MAX_NO_ENTRY_RATE)*100,2)))%"
            Write-Host "  Route clear rate remaining: $([math]::Max(0, [math]::Round(($MIN_ROUTE_CLEAR_RATE - $bestEntry.designated_route_clear_rate)*100,2)))%"
    Write-Host "`nEvaluation results have been saved to promotion log for tracking."
    # ログだけ保存して終了
    $promotionLog | ConvertTo-Json -Depth 10 | Out-File (Join-Path $PROD_MODEL_DIR "promotion_log_$timestamp.json") -Encoding utf8
    Write-Host "Promotion log saved: promotion_log_$timestamp.json"
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

# 新モデルを本番パスにコピー（best_by_evalの公式最良モデルを使用）
Copy-Item $SOURCE_CARRY_MODEL $PROD_CARRY_MODEL -Force
# Escort/Guardはv20の最良モデルを継承（今回はCarryだけ更新）
Copy-Item "D:\git\ai_dnn\03.game\gc_v1\data\attacker_gc_curriculum_v20_independent_facing\dqn_attacker_escort_gc_best_phase_facing_ab.pt" $PROD_ESCORT_MODEL -Force
Copy-Item "D:\git\ai_dnn\03.game\gc_v1\data\attacker_gc_curriculum_v20_independent_facing\dqn_attacker_guard_gc_best_phase_facing_ab.pt" $PROD_GUARD_MODEL -Force
Write-Host "New production models deployed successfully (best_by_eval model)."

# バックアップパスをログに追加
$promotionLog.backup_path = $backupDir
$promotionLog | ConvertTo-Json -Depth 10 | Out-File (Join-Path $PROD_MODEL_DIR "promotion_log_$timestamp.json") -Encoding utf8
Write-Host "Promotion log saved: promotion_log_$timestamp.json"
Write-Host "Auto-promotion complete! New model is now active in production."