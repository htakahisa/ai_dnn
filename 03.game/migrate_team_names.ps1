# One-time mechanical migration of exact team-name strings in JSON.
# Keep source filenames and an untouched backup of each modified document.
$ErrorActionPreference = 'Stop'
$pairs = ConvertFrom-Json -InputObject '[["\u3068\u3046\u3084\u307e\u30b2\u30fc\u30df\u30f3\u30b0","Touyama Gaming"],["\u30c9\u30e9\u30b4\u30f3\u30c6\u30a4\u30eb","Dragon Tail"],["\u30d6\u30e9\u30c3\u30c9\u30e0\u30a6\u30f3","Blood Moon"],["Leo\u8ef8","Leo And Friends"],["\u30d5\u30ea\u30fc\u30ca\u30af\u30e9\u30b7\u30c3\u30af","Furina Classic"],["\u30d5\u30ea\u30fc\u30ca\u30bf\u30eb\u30bf\u30ea\u30e4","Furina Tartaglia"],["\u65e5\u672c\u4ee3\u8868","Japan All-Stars"],["\u30af\u30a4\u30fc\u30f3\u30ba\u30d5\u30e9\u30ef\u30fc\u30ae\u30e3\u30f3\u30d3\u30c3\u30c8","Queen''s Flower Gambit"],["\u30a2\u30a4\u30cd\u30af\u30e9\u30a4\u30cd","Eine Kleine"],["\u500b\u4eba\u80fd\u529b\u30d1","Team Elites"],["\u30d6\u30e9\u30c3\u30c9\u30e0\u30fc\u30f3","Blood Moon"]]'
$workspacePath = [IO.Path]::GetFullPath($PSScriptRoot)
$backupPath = Join-Path $workspacePath 'team_name_migration_backup'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$changed = 0
$files = & rg --files --no-ignore $workspacePath -g '*.json' -g '!**/.git/**' -g '!**/team_name_migration_backup/**'
foreach ($file in $files) {
    $targetPath = [IO.Path]::GetFullPath($file)
    if (-not $targetPath.StartsWith($workspacePath + [IO.Path]::DirectorySeparatorChar)) {
        throw "Outside workspace: $targetPath"
    }
    $original = [IO.File]::ReadAllText($targetPath, $utf8)
    $updated = $original
    foreach ($pair in $pairs) {
        $oldValue = '"' + $pair[0] + '"'
        $newValue = '"' + $pair[1] + '"'
        $updated = $updated.Replace($oldValue, $newValue)
        $escaped = '"' + (-join ($pair[0].ToCharArray() | ForEach-Object {
            if ([int]$_ -gt 127) { '\u{0:x4}' -f [int]$_ } else { [string]$_ }
        })) + '"'
        $updated = $updated.Replace($escaped, $newValue)
    }
    if ($updated -cne $original) {
        $relativePath = $targetPath.Substring($workspacePath.Length + 1)
        $savedPath = Join-Path $backupPath $relativePath
        if (-not (Test-Path -LiteralPath $savedPath)) {
            New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($savedPath)) -Force | Out-Null
            Copy-Item -LiteralPath $targetPath -Destination $savedPath
        }
        [IO.File]::WriteAllText($targetPath, $updated, $utf8)
        $changed++
        Write-Output "Migrated: $relativePath"
    }
}
Write-Output "Updated JSON files: $changed"

