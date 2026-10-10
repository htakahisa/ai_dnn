# Windowsのチェックポイント保存失敗と再開

2026-10-10、FRCの52セット目のlatest保存でファイル置換がWinError 5により拒否された。旧latestは読み取り専用ではなく、51セット・plant version 9として正常に読み込めた。拒否したプロセスは特定できていない。

`tv3_checkpoint.py` の置換再試行を10回・0.2秒間隔から61回・0.5秒間隔へ変更した。待機は最大30秒。それでも置換できない場合、旧latestを保持し、torch.saveが完了した新しい一時ファイルも残す。例外メッセージには復旧用のファイルパスを表示する。書き込み自体が失敗した不完全な一時ファイルは削除する。

今回の旧処理ではfinallyで一時ファイルを削除しており、52セット目の一時ファイルは残っていない。FRCは51セットから再開する。保存処理と再開回数の設定だけの変更なので、新規学習やモデルの互換性変更は不要。

並列学習は相手ごとに完了数が違う。`tv3_train_attacker_plant.py` 冒頭を次に設定すると、各相手のlatestから合計100セットまで進められる。

```python
TARGET_OPPONENTS = ("gc_v1", "concon_v1", "omoko_v1", "fnatic_v3", "frc_v1", "toru_ai_v4")
RESUME_TRAINING = True
EVALUATION_ONLY = False
TRAINING_SETS = 100
TOTAL_TRAINING_SET_LIMIT = 100
```

`TRAINING_SETS` は追加セット数の上限、`TOTAL_TRAINING_SET_LIMIT` は通算の上限。例えば51セット完了なら49セット追加、61セット完了なら39セット追加する。通算上限を設けず追加セットを学習するときは `TOTAL_TRAINING_SET_LIMIT = None` に戻す。

作業ディレクトリは `03.game/touyama_v3/`。学習はユーザーが実行する。

```powershell
Set-Location touyama_v3  # 03.gameから移動
python tv3_train_attacker_plant.py
```
