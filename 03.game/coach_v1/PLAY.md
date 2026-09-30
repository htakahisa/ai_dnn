# coach_v1 を試合で使う

`run_game.py` と `run_competition_manager.py` の controller 欄で **Coach v1** を選びます。coach_v1 は固定 5 人編成で学習されているため、担当チームには **Gorigons** プリセット（ごりまる、ごんごん、ごんた、くんた、くりまる）を使ってください。攻守交代時も同じチームの攻撃・防衛モデルに自動で切り替わります。

## モデルの切り替え

[`game_controller.py`](game_controller.py) の `MODEL_PROFILES` にモデル一式の checkpoint を登録し、`ACTIVE_MODEL` をその名前に変えます。パスはすべて `coach_v1/checkpoints/` からの相対パスです。攻撃 coach、防衛 coach、5 人の character、ごんごんの防衛専用モデルをそれぞれ指定してください。

ファイルを編集せず一回の起動だけ切り替える場合は、環境変数 `COACH_V1_MODEL` に登録済みの名前を設定できます。PowerShell の例:

```powershell
$env:COACH_V1_MODEL = "current"
python run_game.py
```

`COACH_V1_MODEL` は `ACTIVE_MODEL` より優先されます。未登録の名前や存在しないファイルは試合開始時にエラーになります。checkpoint の観測仕様や警戒ポイント設定が現在の coach_v1 と合わない場合も、既存の loader が読み込みを拒否します。
