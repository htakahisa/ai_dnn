# 警戒ポイント設定 v1

`watch_points_map.py` の `NEW_MAZE_STR` にある `4` が座標の正本である。JSONを直接編集せず、
次の変換コマンドで `watch_points.json` を再生成する。

```powershell
python -m coach_v1.convert_watch_points_map --write
```

変換時は現在のJSONと同じ座標のmetadataを保持し、新しい座標には決定的な初期metadataを付ける。
マップ変更時は `map` の寸法とSHA-256を更新し、全ポイントを見直して再学習する。設定hashはJSONを
canonical化して算出し、checkpoint metadataの `watch_points_hash` に保存する。

## point形式

| field | 形式 | 意味 |
|---|---|---|
| `id` | snake_case文字列 | 設定内で一意なID |
| `position` | `[row, column]` | 壁以外のマス。座標の重複は禁止 |
| `importance` | 1～5の整数 | 学習・分析用の重要度 |
| `facing` | `N/NE/E/SE/S/SW/W/NW` | 推奨警戒方向 |
| `sides` | 配列 | `attacker`、`defender` の対象陣営 |
| `situations` | 配列 | `carry/retrieve/guard/search/retake` の対象状況 |
| `random_radius` | 1～3の整数 | 周辺配置候補の最大半径 |
| `tags` | 配列 | 地点の戦術的分類。直接的なability命令は保存しない |

検証ではroot・map・pointの未知fieldも拒否する。陣営と状況の不整合、範囲外、壁上、重複、
無効な方向・重要度・半径・タグに加え、推奨方向の直近マスが壁またはマップ外の設定も拒否する。

## 検証と表示

```powershell
python -m coach_v1.validate_watch_points
python -m coach_v1.validate_watch_points --side attacker --situation guard
python -m coach_v1.validate_watch_points --no-map
```

表示上の `↑ ↗ → ↘ ↓ ↙ ← ↖` が推奨方向を示す。このデータは学習分布、事前情報、履歴、
分析に使うためのものであり、推論時の強制移動や自動経路探索には使用しない。
