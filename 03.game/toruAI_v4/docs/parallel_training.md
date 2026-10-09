# 相手AIごとの並列学習

各スクリプト冒頭の `MAX_PARALLEL_WORKERS = 3` が最大同時実行数。3なら最大3相手を同時学習し、完了した枠で次の相手を開始する。1なら従来の順次実行。相手が1種類なら同一プロセスで実行する。通常は定数を変更してオプションなしで起動する。

Pythonのスレッドではなく、相手AIごとの別プロセスを使用する。ゲーム側の作業ディレクトリ変更、乱数seed、stdout/stderrの切替、PyTorch・敵AIの状態を分離するため。`TORCH_THREADS`が指定する一つのプロセス内の計算スレッド数とは別で、既定は1のまま。

## 対象

| 学習スクリプト | 並列単位 | ログ |
|---|---|---|
| `tv4_train_attacker_analysis.py` | 相手AI | `logs/attacker_analysis/<相手AI>/training.log` |
| `tv4_train_attacker_plant.py` | 相手AI | `logs/attacker_plant/<相手AI>/training.log` |
| `tv4_train_attacker_guard.py` | 相手AI | `logs/attacker_guard/<相手AI>/training.log` |
| `tv4_train_defender_analysis.py` | 相手AI | `logs/defender_analysis/<相手AI>/training.log` |
| `tv4_train_defender_search.py` | 相手AI | `logs/defender/search/<相手AI>/training.log`（従来どおり） |
| `tv4_train_retake.py` | 相手AI。L/Rは同じworker | `logs/defender/<相手AI>/retake/training.log` |

表のログは既定の保存先で並列起動した場合。ログ定数を変更した場合もその下の相手AI別ディレクトリへ保存する。最大1の順次実行では従来の保存先を使用する。既存のrootのtraining.logは並列起動では更新されない。

## 実行

作業ディレクトリは `03.game/toruAI_v4/`。plantの場合、`tv4_train_attacker_plant.py` 冒頭の `MAX_PARALLEL_WORKERS` を確認し、通常は次のコマンドだけで実行する。

```powershell
Set-Location C:\Users\ronet\MyProject\git\AI_dnn\03.game\toruAI_v4
python tv4_train_attacker_plant.py
```

他の学習も表のスクリプト名へ置き換えてオプションなしで起動する。一時的に順次実行へ上書きする場合だけ `python tv4_train_attacker_plant.py --max-workers 1` を使う。

モデル・latest・bestの保存先、学習回数、評価頻度、評価seed・編成、採用基準を変更しない。並列数の変更だけで新規学習を要求しない。retakeのデータ識別hashは選択する相手AIの部分集合に依存させず、同じケースデータで相手を分けても同じ識別値を使う。学習中の乱数列が順次実行と完全に一致することは保証しない。bestは各相手の共通独立評価結果で選ぶ。

子プロセスの出力は同じコンソールに表示する。開始・完了・失敗と相手AI名を親が表示し、学習ログの行も相手AI名で区別する。失敗またはCtrl+Cでは実行中workerを停止し、次の相手を開始しない。再開は従来どおり完了セットのlatestを使用する。

前段と後段は従来の順番で実行する。analysis完了後にplant、plant完了後にguard収集、収集完了後にguard学習へ進む。収集スクリプトは共通のmanifest、ケース一覧、進捗を書くため、今回の並列化対象にしない。

最大3で必ず3倍速になるわけではない。CPUとメモリを各workerが使用するため、実際の速度は環境による。起動済みの学習には今回のコード変更は反映されず、次回起動から有効になる。

検証は実対戦や本学習を開始せず、軽量な子プロセスで最大同時数、実際の重なり、作業ディレクトリ・ログの分離、引数引継ぎ、失敗・中断時の停止を確認する。
## 推論デバイス

toru AI v4 の学習・評価・ケース収集は、相手AIの生成に `device="cpu"` を明示指定する。
保存ケースも `load_case(..., device="cpu")` で読み込み、保存時のGPU情報をCPUへ置き換える。
v4自身の学習モデルもCPUで動作する。設定用の追加コマンドライン引数は不要。

推論APIでは `_build_team_ai(key, device="cpu")`、または各AIのコントローラーに
`device="cpu"` / `device="cuda"` を渡して変更できる。
touyama_v2・GC・omokoの推論は、各サブモデルまで指定を引き継ぐ。
未指定時は従来の自動選択・CPU既定値を維持し、CPU推論コンテキスト内ではCPUを選ぶ。

通常の実行は `03.game/toruAI_v4/` を作業ディレクトリにして、
対象スクリプト冒頭の学習回数・再開設定などの名前付き定数を確認し、
`python tv4_train_retake.py` または `python tv4_train_attacker_guard.py` をオプションなしで実行する。
実行中のプロセスにはソース変更が反映されないため、次回起動からCPU指定が適用される。

