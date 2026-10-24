# competitionの試合再現

作業ディレクトリは `D:\git\ai_dnn\03.game`。

```powershell
Set-Location D:\git\ai_dnn\03.game
python run_competition_manager.py
```

通常の実行にコマンドラインオプションは不要です。既存の画面でチーム・AI・Seed・シリーズ形式・シールド設定を選択します。同じ開始条件とSeedなら、描画ON/OFFや再生速度、一時停止、スキップによらず同じ試合になります。再現用の推論条件は `simulation_runtime.py` 冒頭の `COMPETITION_INFERENCE_DEVICE = "cpu"` に固定しています。GPUとの数値差を避けるため、この値はCPUのまま使用してください。

Python・NumPy・PyTorchの乱数と、AI内部の独立した乱数を試合単位で管理します。画面の別の処理が乱数を消費しても試合の乱数列は変わりません。試合終了・例外時には呼び出し元の乱数状態と推論設定を復元します。

## 保存したマップを再現する

更新後の試合結果には、各マップの開始状態 `reproduction` が保存されます。チームの選手順、IGL、スパイク担当、使用AI、開始時のシリーズ勝敗、疲労、戦術記憶、シールド設定とSeedを含みます。2マップ目以降の状態も独立して復元できます。

画面の **保存試合を再現** を押し、`series_data/` 内の `*_original.json`、または `competition_results/` のシリーズ結果JSONを選びます。複数マップある場合はマップ番号を指定します。観戦画面を最後まで実行すると、保存した全tickの状態・戦闘統計・試合結果との一致を検証します。GC3の履歴ログのファイル名は実行ごとに別名で保存し、一致検証から除外します。再現操作は新しい大会結果やレーティングを登録しません。

試合の再計算には、元の試合と同じゲームコード、選手能力、コンボ、AIモデル、Python・NumPy・PyTorchの環境を使用します。変更があって結果が違えば、不一致を表示します。旧形式のJSONには開始状態がないため、この機能では再計算できません。ユーザー操作を含む試合や、モデル変更後の過去の展開は、既存の保存済みリプレイで確認してください。

## 検証

```powershell
Set-Location D:\git\ai_dnn\03.game
python -m unittest test.test_game_match_reproduction -v
```

検証設定は `test/test_game_match_reproduction.py` 冒頭の `TEST_SEED` と `TEST_TEAMS` です。同一条件のフルマッチ、別プロセス・別のPythonハッシュSeed、描画中の別処理による乱数消費、JSON保存後のシリーズ途中マップの再現、独立したAI乱数を検証します。
