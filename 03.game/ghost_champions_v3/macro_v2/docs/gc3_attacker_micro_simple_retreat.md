# 簡易マクロの撤退練習（v4 run）

**この文書は完了済みv4 runの説明です。現在は[近距離から戻る撤退練習のv5 run](gc3_attacker_micro_near_retreat.md)が通常設定です。**

マクロはテスト用の簡単なロジックのまま、進軍・待機に撤退を追加します。A/Bと設置予定マス、ユーザーの5の目的地候補はラウンド中固定します。下位モデルは一つで、指示入力層だけを追加学習します。マクロ自体や設置後ガードは学習しません。

- 戦闘開始後、最低12tick進軍します。セットアップ中のtickはこの時間に数えません。
- 実際の自分のスポーンから歩行距離6マス以上離れたプレイヤーを、一人ずつ撤退させます。各プレイヤーは1ラウンド1回までです。
- 撤退の目標は本人のラウンド開始時のスポーン座標です。到達、または最大12tickで撤退練習を終えます。
- 撤退後は2tick待機し、同じ設置計画への進軍を再開します。全員同時には撤退させません。
- 設置できるスパイク保持者、設置中・オーブ回収中のプレイヤーは対象から外します。撤退中にこれらの条件になった場合は練習を終了し、通常の指示へ戻します。設置後は新たな撤退練習を始めません。

撤退は指示であり、実際の移動はミクロの確率分布から選びます。練習の時間切れをスポーン到達成功とは扱いません。`retreat_practice_events` に開始時刻と終了理由、既存の評価には撤退の接近率・到達回数を記録します。`instruction_ready` と `scenario_instruction_ready` は今回、進軍・待機・撤退の3指示すべてで測定対象があり、各75%以上の場合だけTrueになります。

## 設定と実行

設定場所は `gc3_train_attacker_micro_directives.py` 冒頭です。

| 定数 | 通常設定 |
| --- | --- |
| `RUN_NAME` | `attacker_micro_directives_v4` |
| `DIRECTIVE_CHECKPOINT` | 完了したv3の `latest.pt`、ST154991 |
| `ADDITIONAL_TRAINING_STEPS` | 20,000判断、合計ST174991まで |
| `SIMPLE_RETREAT_AFTER_TICKS` | 12 |
| `SIMPLE_RETREAT_MIN_DISTANCE` | 6 |
| `SIMPLE_RETREAT_TICKS` | 12 |
| `SIMPLE_RETREAT_WAIT_TICKS` | 2 |
| `RESUME` | 初回False、同じrunの途中再開はTrue |

作業ディレクトリは `D:\git\ai_dnn\03.game\ghost_champions_v3\macro_v2` です。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3\macro_v2
python gc3_train_attacker_micro_directives.py
```

新モデルは `data/attacker_micro_directives_v4/`、新ログは `logs/attacker_micro_directives_v4/` に保存します。元のモデルとログは保持します。BCや教師データ収集はやり直しません。

グラフは別のPowerShellで同じ作業ディレクトリから実行します。既定はv4です。すでに開いている旧グラフは再起動して新しい対象を読み込んでください。

6つの枠に、平均報酬、return、エントロピー、設置率・勝率、待機・進軍・撤退の対応率、撤退練習の開始・スポーン到達・時間切れ・中断件数を表示します。5秒ごとに自動更新し、横軸は今回のrunの開始stepから表示します。撤退の時間切れを到達成功に含めません。

```powershell
python tools/gc3_watch_attacker_directive_metrics.py
```

実装・テスト・更新なしの試合確認では、この追加学習を開始しません。通常の実戦モデルは自動で切り替えません。
