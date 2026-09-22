<#
.AutoPromote GC v21 Production Script
譛ｬ逡ｪ迺ｰ蠅・〒菴ｿ逕ｨ縺輔ｌ繧九Δ繝・Ν繧定・蜍慕噪縺ｫ譖ｴ譁ｰ縺吶ｋ繧ｹ繧ｯ繝ｪ繝励ヨ
蟄ｦ鄙呈ｸ医∩繝｢繝・Ν縺瑚ｩ穂ｾ｡蝓ｺ貅悶ｒ貅縺溘＠縺溷ｴ蜷医∬・蜍慕噪縺ｫ譛ｬ逡ｪ繝代せ縺ｫ繧ｳ繝斐・縺励∵立繝｢繝・Ν縺ｯ繝舌ャ繧ｯ繧｢繝・・
菴ｿ逕ｨ譁ｹ豕・ .\auto_promote_gc_v21_prod.ps1 [--episode <繧ｨ繝斐た繝ｼ繝臥分蜿ｷ>]
萓・ .\auto_promote_gc_v21_prod.ps1 --episode 1250
#>

# 蠑墓焚隗｣譫・
$targetEpisode = $null
for ($i = 0; $i -lt $args.Count; $i++) {
    if ($args[$i] -eq "--episode" -and $i + 1 -lt $args.Count) {
        $targetEpisode = [int]$args[$i + 1]
        $i++
    }
}

# 譛ｬ逡ｪ迺ｰ蠅・・繝｢繝・Ν繝代せ・・nstall_gc_macro_runtime.py 縺九ｉ謚ｽ蜃ｺ縺励◆豁｣隕上ヱ繧ｹ・・
$PROD_MODEL_DIR = "D:\git\ai_dnn\03.game\gc_v1\data\attacker_macro_gc_data"
$PROD_CARRY_MODEL = Join-Path $PROD_MODEL_DIR "dqn_attacker_carry_gc_final.pt"
$PROD_ESCORT_MODEL = Join-Path $PROD_MODEL_DIR "dqn_attacker_escort_gc_final.pt"
$PROD_GUARD_MODEL = Join-Path $PROD_MODEL_DIR "dqn_attacker_guard_gc_final.pt"

# 蟄ｦ鄙呈ｸ医∩v21繝｢繝・Ν縺ｮ繧ｽ繝ｼ繧ｹ繝代せ
$V21_SOURCE_DIR = "D:\git\ai_dnn\03.game\gc_v1\data\attacker_gc_curriculum_v21_carry_movement_warm_tuned"
$SOURCE_CARRY_MODEL = Join-Path $V21_SOURCE_DIR "dqn_attacker_carry_gc_best_by_eval.pt"
$SOURCE_EVAL_HISTORY = Join-Path $V21_SOURCE_DIR "evaluation_history.json"

# 繧ｳ繝ｼ繝峨・繝ｼ繧ｹ縺ｮ蜈ｬ蠑剰ｩ穂ｾ｡蝓ｺ貅厄ｼ・elect_gc_carry_movement_v21.py 縺九ｉ蠑慕畑・・
$MAX_TIMEOUT_RATE = 0.05    # 繧ｿ繧､繝繧｢繧ｦ繝育紫5%莉･荳・
$MAX_NO_ENTRY_RATE = 0.35   # 繧ｨ繝ｳ繝医Μ繝ｼ荳榊ｮ溯｡檎紫35%莉･荳・
$MIN_ROUTE_CLEAR_RATE = 0.50 # 繝ｫ繝ｼ繝医け繝ｪ繧｢邇・0%莉･荳・

# 繧ｿ繧､繝繧ｹ繧ｿ繝ｳ繝・
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# 隧穂ｾ｡螻･豁ｴ繧定ｪｭ縺ｿ霎ｼ縺ｿ
if (-not (Test-Path $SOURCE_EVAL_HISTORY)) {
    Write-Error "Evaluation history not found: $SOURCE_EVAL_HISTORY"
    exit 1
}

$evalHistory = Get-Content $SOURCE_EVAL_HISTORY -Raw -Encoding utf8 | ConvertFrom-Json
$bestEntry = $null
$bestScoreViolation = [float]::MaxValue

# 繧ｨ繝斐た繝ｼ繝画欠螳壹′縺ゅｋ蝣ｴ蜷医・謖・ｮ壹お繝斐た繝ｼ繝峨ｒ菴ｿ逕ｨ
if ($targetEpisode -ne $null) {
    Write-Host "謇句虚謖・ｮ壹＆繧後◆繧ｨ繝斐た繝ｼ繝峨ｒ菴ｿ逕ｨ縺励∪縺・ $targetEpisode"
    $bestEntry = $evalHistory | Where-Object { $_.episode -eq $targetEpisode }
    if (-not $bestEntry) {
        Write-Error "謖・ｮ壹＆繧後◆繧ｨ繝斐た繝ｼ繝・$targetEpisode 縺瑚ｩ穂ｾ｡螻･豁ｴ縺ｫ隕九▽縺九ｊ縺ｾ縺帙ｓ"
        exit 1
    }
}

# 繧ｨ繝斐た繝ｼ繝峨′謖・ｮ壹＆繧後※縺・↑縺・ｴ蜷医・閾ｪ蜍輔〒譛濶ｯ繧帝∈謚・
if ($targetEpisode -eq $null) {
    Write-Host "繧ｨ繝斐た繝ｼ繝峨′謖・ｮ壹＆繧後※縺・∪縺帙ｓ - 螻･豁ｴ縺九ｉ譛濶ｯ繝｢繝・Ν繧定・蜍暮∈謚槭＠縺ｾ縺・.."
    # 繧ｳ繝ｼ繝峨・繝ｼ繧ｹ縺ｨ蜷後§驕ｸ謚槭Ο繧ｸ繝・け縺ｧ譛濶ｯ繧ｨ繝ｳ繝医Μ繧帝∈縺ｶ・亥ｭ伜惠縺吶ｋPT繝輔ぃ繧､繝ｫ縺縺大ｯｾ雎｡・・
    foreach ($entry in $evalHistory) {
        $candidateModel = Join-Path $V21_SOURCE_DIR "dqn_attacker_carry_gc_ep$($entry.episode).pt"
        if (-not (Test-Path $candidateModel)) {
            continue # 繝輔ぃ繧､繝ｫ縺悟ｭ伜惠縺励↑縺・お繝斐た繝ｼ繝峨・繧ｹ繧ｭ繝・・
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
        Write-Error "隧穂ｾ｡螻･豁ｴ縺ｫ譛牙柑縺ｪ繝｢繝・Ν繝輔ぃ繧､繝ｫ縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ縺ｧ縺励◆ - best_by_eval.pt縺ｫ繝輔か繝ｼ繝ｫ繝舌ャ繧ｯ縺励∪縺・
        $bestEntry = $evalHistory | Select-Object -Last 1
    }
}

# 驕ｸ謚槭＆繧後◆繧ｨ繝斐た繝ｼ繝峨↓蟇ｾ蠢懊☆繧輝T繝輔ぃ繧､繝ｫ繧定ｨｭ螳・
if (Test-Path (Join-Path $V21_SOURCE_DIR "dqn_attacker_carry_gc_ep$($bestEntry.episode).pt")) {
    $SOURCE_CARRY_MODEL = Join-Path $V21_SOURCE_DIR "dqn_attacker_carry_gc_ep$($bestEntry.episode).pt"
} else {
    # 繝輔ぃ繧､繝ｫ縺悟ｭ伜惠縺励↑縺・ｴ蜷医・best_by_eval.pt繧剃ｽｿ逕ｨ
    Write-Warning "EP$($bestEntry.episode)縺ｮ繝｢繝・Ν繝輔ぃ繧､繝ｫ縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ - best_by_eval.pt繧剃ｽｿ逕ｨ縺励∪縺・
    $SOURCE_CARRY_MODEL = Join-Path $V21_SOURCE_DIR "dqn_attacker_carry_gc_best_by_eval.pt"
}

# 繝｢繝・Ν繝輔ぃ繧､繝ｫ縺ｮ蟄伜惠遒ｺ隱・
if (-not (Test-Path $SOURCE_CARRY_MODEL)) {
    Write-Error "Model file not found: $SOURCE_CARRY_MODEL"
    exit 1
}

Write-Host "Using verified evaluation result (EP$($bestEntry.episode)):"
Write-Host "  timeout_rate: $($bestEntry.timeout_rate*100)%"
Write-Host "  carry_no_entry_rate: $($bestEntry.carry_no_entry_rate*100)%"
Write-Host "  designated_route_clear_rate: $($bestEntry.designated_route_clear_rate*100)%"

Write-Host "`nThreshold criteria (max limits):"
Write-Host "  max_timeout_rate: $($MAX_TIMEOUT_RATE*100)%"
Write-Host "  max_no_entry_rate: $($MAX_NO_ENTRY_RATE*100)%"
Write-Host "  min_route_clear_rate: $($MIN_ROUTE_CLEAR_RATE*100)%"

# 蝓ｺ貅夜＃謌舌メ繧ｧ繝・け
$passTimeout = $bestEntry.timeout_rate -le $MAX_TIMEOUT_RATE
$passNoEntry = $bestEntry.carry_no_entry_rate -le $MAX_NO_ENTRY_RATE
$passRouteClear = $bestEntry.designated_route_clear_rate -ge $MIN_ROUTE_CLEAR_RATE
$allPassed = $passTimeout -and $passNoEntry -and $passRouteClear

Write-Host "`nCriteria check results:"
Write-Host "  Timeout rate: $(if ($passTimeout) { 'PASS' } else { 'FAIL' }) (value: $($bestEntry.timeout_rate*100)% <= $($MAX_TIMEOUT_RATE*100)%)"
Write-Host "  No-entry rate: $(if ($passNoEntry) { 'PASS' } else { 'FAIL' }) (value: $($bestEntry.carry_no_entry_rate*100)% <= $($MAX_NO_ENTRY_RATE*100)%)"
Write-Host "  Route clear rate: $(if ($passRouteClear) { 'PASS' } else { 'FAIL' }) (value: $($bestEntry.designated_route_clear_rate*100)% >= $($MIN_ROUTE_CLEAR_RATE*100)%)"

if (-not $allPassed) {
    Write-Warning "Not all criteria met - production model will not be updated."
    exit 0
}

Write-Host "`nAll criteria satisfied! Starting production model update..."

# 譌ｧ繝｢繝・Ν縺ｮ繝舌ャ繧ｯ繧｢繝・・
$backupDir = Join-Path $PROD_MODEL_DIR "backup_$timestamp"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Copy-Item $PROD_CARRY_MODEL (Join-Path $backupDir "dqn_attacker_carry_gc_final.pt") -Force
Copy-Item $PROD_ESCORT_MODEL (Join-Path $backupDir "dqn_attacker_escort_gc_final.pt") -Force
Copy-Item $PROD_GUARD_MODEL (Join-Path $backupDir "dqn_attacker_guard_gc_final.pt") -Force
Write-Host "Old models backed up to: $backupDir"

# 譁ｰ繝｢繝・Ν繧呈悽逡ｪ繝代せ縺ｫ繧ｳ繝斐・・・est_by_eval縺ｮ蜈ｬ蠑乗怙濶ｯ繝｢繝・Ν繧剃ｽｿ逕ｨ・・
Copy-Item $SOURCE_CARRY_MODEL $PROD_CARRY_MODEL -Force
# Escort/Guard縺ｯv20縺ｮ譛濶ｯ繝｢繝・Ν繧堤ｶ呎価・井ｻ雁屓縺ｯCarry縺縺第峩譁ｰ・・
Copy-Item "D:\git\ai_dnn\03.game\gc_v1\data\attacker_gc_curriculum_v20_independent_facing\dqn_attacker_escort_gc_best_phase_facing_ab.pt" $PROD_ESCORT_MODEL -Force
Copy-Item "D:\git\ai_dnn\03.game\gc_v1\data\attacker_gc_curriculum_v20_independent_facing\dqn_attacker_guard_gc_best_phase_facing_ab.pt" $PROD_GUARD_MODEL -Force
Write-Host "New production models deployed successfully (best_by_eval model)."

# 繝励Ο繝｢繝ｼ繧ｷ繝ｧ繝ｳ繝ｭ繧ｰ繧定ｨ倬鹸
$promotionLog = [PSCustomObject]@{
    timestamp = $timestamp
    source_episode = $bestEntry.episode
    source_dir = $V21_SOURCE_DIR
    source_model = "auto_selected_best"
    selection_violation = $bestScoreViolation
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
    backup_path = $backupDir
}
$promotionLog | ConvertTo-Json -Depth 10 | Out-File (Join-Path $PROD_MODEL_DIR "promotion_log.json") -Encoding utf8
Write-Host "Promotion log saved."
Write-Host "Auto-promotion complete! New model is now active in production."
}
