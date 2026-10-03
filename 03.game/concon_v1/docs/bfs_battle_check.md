# BFSによる戦闘・移動確認

プロジェクトルートで実行します。学習やattackerモデルの読み込みは行いません。
既存の経由点・同期・道譲りを守り、BFS距離が小さくなる移動を選択します。
交戦時の停止・射撃、IQ知覚、スパイク回収、対戦相手は実ゲームの処理です。

```powershell
python check_co1_bfs_battle.py -map A3 --rounds 3 --opponents gc_v1
```

標準では100tick、プラント完了またはラウンド終了までを確認します。
プラント後の防御・起爆までの試合は対象に含みません。
相手を複数指定すると、それぞれに対して指定回数実行します。

```powershell
python check_co1_bfs_battle.py -map A2 --rounds 10 --opponents gc_v1 omoko_v1 --seed 0
```

コンソールに終了理由、射撃数・命中数、両チームの死亡tickと位置、attackerの死亡時の経由点、停滞情報を表示します。
座標は `(row, col)`、左上が `(0, 0)` です。
`STOP` は同じ経路状態が5tick続いた場合に表示します。
`controller_hold` は交戦や敵情報による待機も含むため、必ずしも移動不能ではありません。
射撃ログとリプレイを合わせて判断してください。サイト到着後の護衛待機は停滞集計から除外します。

結果は `concon_v1/data/diagnostics/bfs_battle_A3.json` と `.html` に保存します。
HTMLをブラウザで開くと、各tickのマップ上の位置、HP、死亡、射撃線、停滞情報を確認できます。
外部ライブラリやサーバーは不要です。同じマップの再実行では結果を上書きします。
保存先を変更する場合は `--output concon_v1/data/diagnostics/example.json` を指定します。

`--stuck-ticks 3` で停滞検出を早められます。
`--max-ticks 300` で時間制限を延長できますが、通常の100tickでの到達可否とは区別してください。
