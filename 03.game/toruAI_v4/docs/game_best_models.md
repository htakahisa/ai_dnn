# 通常ゲームのbestモデル

`03.game/` を作業ディレクトリとして `python run_game.py` を実行し、AI選択で「Toru AI v4」を選ぶ。攻撃側・守備側とも、実際の相手AI名から `toruAI_v4/data/best/<相手AI>/` のモデルを選ぶ。攻守交替後も相手側のAIに合わせて読み込む。

| 側 | フェーズ | ファイル名 |
| --- | --- | --- |
| 攻撃 | 解析 | attacker_analysis_best.pt |
| 攻撃 | 設置まで | attacker_plant_best.pt |
| 攻撃 | 設置後 | attacker_guard_best.pt |
| 守備 | 解析 | defender_analysis_best.pt |
| 守備 | 設置前 | search_best.pt |
| 守備 | 左サイトの設置後 | retake_L_best.pt |
| 守備 | 右サイトの設置後 | retake_R_best.pt |

相手AIは `gc_v1`、`touyama_v2`、`omoko_v1`、`fnatic_v3`、`frc_v1`、`toru_ai_v3`。analysisは各フェーズの入力として使い、独立した行動フェーズにはしない。学習途中のlatestは読み込まない。

モデルの入力schema、相手、サイト、学習元モデルのhashを検証する。plantが参照したanalysis、guardが参照したanalysis・plant、searchが参照したanalysis、retakeが参照したanalysis・searchの一致が必要。モデルを単独で再学習して依存hashが変わった場合は、後続モデルも更新する必要がある。

未対応の相手や学習時と異なるマップでは汎用controllerに切り替える。必須のanalysis・plantまたはanalysis・searchを読めない場合も汎用controllerになる。guardや片側retakeのみ読めない場合は、該当する設置後フェーズだけ汎用controllerを使う。読み込み結果と切替理由は起動時のコンソールに表示する。

## 学習せずに読み込みを確認する

`tv4_check_best_runtime.py` 冒頭の `CHECK_OPPONENTS`、`LIVE_ROUND_OPPONENTS`、`OWN_PRESETS`、`SEED`、`MAX_ROUND_STEPS`、`OUTPUT_PATH` が検証設定。全6相手について通常ゲームのfactoryを使って攻守両側を初期化し、全bestが利用可能であることを検証する。`LIVE_ROUND_OPPONENTS` の相手には1ラウンドを実行し、設置後の切替状況を記録する。勝率評価用のスクリプトではない。

```powershell
Set-Location toruAI_v4
python tv4_check_best_runtime.py
```

結果は `toruAI_v4/logs/best_runtime_check.json` に保存する。学習とモデルの更新は行わない。
