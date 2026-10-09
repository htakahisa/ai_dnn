# リテイクのデータ収集と学習

## 2026-10-09：戦闘と解除期限の修正

retakeの目的はdefenderのラウンド勝利。設置後に敵を倒すだけ、生存するだけでは成功に数えない。bestは引き続き教師・探索なしの独立評価のリテイク勝率を最優先する。

attacker plantの知見を `tv4_retake_combat.py` に取り込んだ。公開視認・最終目撃・壁・スモーク・味方の身体から各味方の射撃相手を選び、停止射撃・低HP/失明時の退避・援護確認を教師例にする。FLASH/RECONは着弾と効果範囲、SMOKEは敵の射線を確認して教師が使用を選ぶ。通常アビリティとULTの合法候補は維持し、推論時の行動は学習したQ値で選ぶ。

解除担当は公開の距離・解除進捗・HPで決め、教師は接触と援護を確認する。味方が解除している間は援護を教える。残り時間から移動・解除に必要な時間を引き、期限が迫れば合流待ちや交戦停止を続けず解除へ進む。安全確認によって必ず勝てるという意味ではない。合流待ちは2人以上が準備できた場合、最大6tickで突入を開始する。

勝利加点は+8、敗北は-8。先に死亡したキャラの過去行動へ後の勝利加点を付けない。停止射撃命中・実際に発生したアビリティ効果を補助評価する。投擲の遅延効果は投げた元のtransitionに加点・減点し、未公開の答え合わせ情報を入力・教師判断に戻さない。解除担当が非交戦時に解除も進行もせず待機する場合は小さく減点する。教師例をreplayに保存し、教師模倣lossも追加する。

教師確率は95%から60セットかけて10%まで減らす。設定は `tv4_train_retake.py` 冒頭の `TEACHER_START_PROBABILITY`、`TEACHER_MIN_PROBABILITY`、`TEACHER_DECAY_SETS`、`DEMONSTRATION_WEIGHT`。実力評価・通常推論では教師と探索を0にする。セット数や対象AIなど既存のユーザー設定は維持した。

最初の戦闘修正ではretakeを519要素、方式を `rally_entry_public_combat_deadline_v2` とした。爆発負けの調査後は解除可能範囲・担当・期限の入力を追加して528要素・v3とした。相手別の調査で死亡した解除担当による合法マスクの閉塞を修正し、現在は528要素のまま `living_defuser_opponent_learning_v4`。searchの入力505要素・保存契約・学習目的は維持する。新しい列をゼロで初期化し、searchの既存列と重みを引き継ぐ。旧retakeのreplay・optimizer・best評価は新条件として再開しない。

解除担当は継続して割り当て、担当の目標を入口やスパイク中心ではなく、エンジンが受け付ける周囲8マスを含む解除可能範囲にする。担当の死亡・味方の解除開始・時間内に担当が到達できなくなった場合に引き継ぐ。教師は開始済みの解除を単なる新規接触で中断しない。推論では合法な候補からQ値で選び、解除を強制しない。

担当の解除範囲への前進を補助加点し、交戦や投擲を含めて進行していない時間を減点する。解除進捗の報酬は増分・減少の両方を扱い、開始→中断の反復で加点を取り直せないようにする。`tv4_retake_combat.py` 冒頭の定数で調整する。学習ミニバッチは `tv4_train_retake.py` の `DEFUSER_SAMPLE_FRACTION = .50` により、半分を解除担当の例、残り半分を全員の例から抽出し、担当の行動を援護・待機例に埋もれさせない。

v4では、`frc_v1/actions.py` の解除中判定から死亡者を除く。死亡時にゲームが解除ロックを解放してもキャラの進捗は残るため、公開マスクで後任のDEFUSEを禁止していた不具合を直した。通常推論では引き続き学習したQ値で行動を選ぶ。

待機が多いモデルには、向きと別に前進・待機・アビリティ・解除の種類を学ぶlossを追加した。`DEMONSTRATION_KIND_WEIGHT`、`MODEL_LEARNING_OVERRIDES`、`DEFUSER_SAMPLE_FRACTION` を学習スクリプト冒頭で確認できる。frc左は種類模倣1.0・担当抽出75%、fnatic左右・omoko左右・frc右・touyama右は0.75・65%、その他は0.50・50%。これは相手・左右別モデルの学習設定であり、特定編成や座標への推論ルールではない。

目標は相手・左右別に `TARGET_RETAKE_WIN_RATE = .50`。前回の独立評価で目標未満だったモデルだけ、次の更新を `BELOW_TARGET_UPDATE_MULTIPLIER = 3` 倍にする。既定の基準更新回数100なら300回。セット数は変更しない。評価seedは6へ増やし、各回同じ計画で通常setup/searchから評価する。左右の設置数を人工的に均等化しない。12設置未満の評価は目標判定を「評価数不足」とし、少数の50%を達成と報告しない。

相手別の調査・同じ重みでの不具合修正比較は [retake_opponent_review.md](retake_opponent_review.md) を参照する。学習方法を変えた後のモデル性能は、新規学習後の教師・探索なしの評価で確認する。

`defuse_diagnostics` には各tickの解除担当・目標・距離・合法性・選択行動・教師行動・進捗・残り時間・到達可能性を記録する。解除を開始していない爆発、解除範囲に入っていない爆発、解除中断の原因を独立評価とbestにも集計する。詳細は [retake_deadline_review.md](retake_deadline_review.md)。

searchと収集時のhashが一致する既存の設置ケースは再利用できる。今回のコード修正だけを理由にsearch再学習・ケース再収集は必要ない。現在起動中の学習プロセスには修正が反映されない。現在の処理の終了または停止後に、`TRAINING_MODE = "fresh"` を確認してretakeを新規学習する。通常保存先は変更しない。修正作業ではモデルを書き換えていない。

```powershell
Set-Location C:\Users\ronet\MyProject\git\AI_dnn\03.game\toruAI_v4
python tv4_train_retake.py
```

`COMBAT_ROUND_LOG_ENABLED = True` では、各workerのretakeログディレクトリに `combat_rounds.jsonl` を保存する。通常学習の `log_context` と、教師・探索なしの評価の `log_context` を区別し、死亡前4tickの公開情報・実射撃・投擲効果・終了理由・被害HPを残す。学習ログにも戦闘診断の集計を表示する。各起動で今回のログを上書きする。

実ゲームの比較結果と限界は [retake_combat_review.md](retake_combat_review.md) を参照する。

学習は `MAX_PARALLEL_WORKERS = 3` で相手AIごとに並列実行する。L/Rは同じworker。収集は順次実行。学習ログの保存先と設定は [並列学習](parallel_training.md) を参照する。

作業ディレクトリは `03.game/toruAI_v4/` です。

search の `search_best.pt` を更新したら、`python tv4_collect_retake.py`、`python tv4_train_retake.py` の順に実行します。通常は保存先の変更や日付の指定は不要です。

収集は毎回同じ `data/retake_cases/` の収集ファイルと共有テンソルを置き換えます。学習は既定の `TRAINING_MODE = "fresh"` で新規に行い、latest と条件が変わった best も同じ保存先で更新します。収集完了を確認してから学習してください。

```powershell
Set-Location toruAI_v4
python tv4_collect_retake.py
```

学習済みの `data/best/<相手AI>/search_best.pt` とサイト分析モデルを固定し、通常の setup/search から設置された状態を収集します。既定は6種類の相手AIそれぞれ Left 50件、Right 50件、合計600件です。ここでいう1件は1ラウンドの設置状態であり、search 学習の12ラウンド単位の「1セット」とは異なります。

味方編成は search の学習用プリセットから、相手とキャラクターが重ならないものを順番に切り替えます。設置前に決着したラウンド、設置時に defender が全滅したラウンド、収集上限に達したサイドの状態は保存しません。設置位置や生存人数は変更しません。設置完了 tick の残りの defender 行動を待機にして、その tick の終了時点を保存します。収集ではモデルを更新しません。

保存先は `data/retake_cases/` です。

- `collection.json`: 収集条件、search/analysis のハッシュ。
- `cases.jsonl`: 相手、サイド、味方編成、設置位置、残り時間、全キャラクターのHP・位置・アビリティ残量。
- `blocks.jsonl`: 再開に必要な実行済み12ラウンドブロックの記録。
- `*.case.gz`: 全キャラクター、相手コントローラー、IQ知覚、ゲーム内効果、乱数状態を含むスナップショット。
- `tensors/`: 共通のモデル重みを内容ごとに一度だけ保存。

収集の通常設定も `tv4_collect_retake.py` 冒頭の定数を編集します。再開・追加する場合は `DEFAULT_RESUME = True` に設定し、同じ条件で実行します。試行上限は `DEFAULT_MAX_BLOCKS`（相手ごとの累計ブロック数）です。

```powershell
python tv4_collect_retake.py
```

相手が一方のサイドに偏る場合は、指定件数に達しないことがあります。スクリプトは不足件数を表示して終了コード2を返します。同じ条件で収集を続ける場合は `DEFAULT_RESUME = True` に設定します。条件を変えて再収集する場合は `DEFAULT_RESUME = False` で同じ保存先に収集し直してください。任意の別保存先は `DEFAULT_OUTPUT` または一時的な `--output-dir` で指定できます。

保存状態からの学習:

```powershell
python tv4_train_retake.py
```

通常の設定は `tv4_train_retake.py` 冒頭の定数を編集します。コマンドラインオプションは必要な場合だけ一時的な上書きに使用します。

| 定数 | 既定値・意味 |
| --- | --- |
| `TRAINING_SETS` | スクリプト冒頭の設定値（現在10セット） |
| `TRAINING_OPPONENTS` | 全6種類の相手AI |
| `CASES_DIRECTORY` | `data/retake_cases/` |
| `TRAINING_MODE` | `"fresh"`（新規学習）。再開は `"resume"`、評価のみは `"eval"` |
| `EVALUATION_INTERVAL` | 1。毎セット、同じ編成・seedで評価し、各AI・左右別に最良モデルを保存 |
| `EVALUATION_SEED_COUNT` | 評価seed数6 |
| `UPDATES_PER_SET` / `BATCH_SIZE` | 更新回数100 / バッチサイズ64 |
| `RANDOM_SEED` | 42 |
| `TRAINING_PRESETS` / `EVALUATION_PRESETS` | 学習・評価用の味方編成 |
| `SEARCH_DIRECTORY` / `DATA_DIRECTORY` / `BEST_DIRECTORY` / `LOG_DIRECTORY` | 入力search・学習保存先・採用モデル保存先・ログ保存先 |

この経路の1セットは、各相手AIの全保存ケースをランダム順で1回ずつ学習する単位です。左右それぞれ50件なら、各モデルは1セットで50ラウンドを学習します。`TRAINING_SETS` に学習するセット数を設定します。学習時は保存された search の知覚履歴と相手AIの進行状態を維持し、defender のリテイクモデルだけを学習対象に差し替えます。双方のサイドに少なくとも1件必要です。収集データを変更した後は、同じ学習保存先で新規学習してください。再開時は同じデータセットが必要です。

評価は収集ケースの再生ではなく、学習で使用しない味方編成と固定seedで、通常の search から12ラウンドを進めます。既定の評価編成は `Eine Kleine`、`SUPES`、`BBL` です。

各セットの学習後に評価し、リテイク勝率→解除回数→平均報酬の順で既存bestと比較します。初回はそのセットのモデルを保存し、以降は改善したときだけ更新します。最終セットの成績が悪化してもbestは上書きしません。latestは直近のセットの再開用モデルです。全6AI・左右各50件、評価6seedでは、1セットあたり学習600ラウンド、評価432ラウンドです。

`CASES_DIRECTORY = None` に変更すると、毎回 search から12ラウンドを進める経路になります。`tv4_train_defender_search.py --phase retake` も、同じリテイク用定数を使用します。

## 合流とアビリティ

concon v1 の合流方式を参考に、マップの小文字 `a/b/c` に継続的な担当地点を割り当て、到達可能な味方が揃うと突入します。残り時間が不足する場合や単独生存の場合は待機を解除します。大文字 `A/B/C` は対応する入口として使い、入口を通過した後は設置地点へ向かいます。移動、交戦、アビリティの選択は学習モデルが行います。

`F/R/S` は FLASH/RECON/SMOKE の投下先候補です。FLASH/RECON は concon v1 と同じ投射経路計算を使い、マーク地点に着弾する照準を求めます。定点のないスキルも、公開された敵位置、設置地点、周辺のマップ上の地点、自分の地点を候補として学習します。ASH、RAMP、DANCE と各キャラクターの発動可能なアルティメットも合法マスクの範囲で選択できます。DANCE は味方を対象にします。HUNT はゲーム側の自動効果、SERENADE は死亡時の自動発動として扱われます。定点は使用候補であり、使用を強制するものではありません。

既定の保存先:

- 学習再開用: `data/defender/retake/<相手AI>/L_latest.pt`、`R_latest.pt`
- 採用モデル: `data/best/<相手AI>/retake_L_best.pt`、`retake_R_best.pt`
- 学習ログ: `logs/defender/retake/training.log`

学習済み search、分析、ケースの内容、合流・定点マップ、リテイク方式が変わった状態では再開できません。検証を別ディレクトリで行う場合は、`--search-dir data/best` で入力searchを指定し、`--best-dir` と `--data-dir` と `--log-dir` を検証用の場所にしてください。

少数での収集確認:

```powershell
python tv4_collect_retake.py --cases-per-site 1 --opponents gc_v1 --output-dir data/retake_cases_small
python tv4_train_retake.py --cases-dir data/retake_cases_small --opponents gc_v1 --sets 1 --updates 1 --eval-seeds 1 --search-dir data/best --data-dir data/retake_small --best-dir data/retake_small_best --log-dir logs/retake_small
```

ケースは本プロジェクトのPythonオブジェクトを復元する形式です。自分で収集したものを、同じコード・環境で使用してください。
