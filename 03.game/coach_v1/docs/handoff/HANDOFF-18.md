# HANDOFF-18: Issue 001 の再現と未採用候補

## 結論

**P0-A と P0-D は未解決。実験候補を正式 checkpoint へ昇格していない。** 先に両 P0 を解決してから、P1 と Task 17 の ablation 残課題へ進む。現行の正式評価基準は Task 16 の attacker/defender checkpoint のまま。

## 固定条件での再現

`python -X utf8 -m coach_v1.diagnose_issue001 --seed 1701 --opponent omoko_v1 --starts-as defender --output coach_v1/reports/issue001_baseline_1701.json` で、試合終了後の replay/referee と公開観測に基づく coach 決定・行動ログを保存した。actor には replay/referee の実座標を渡していない。seed 1700、attacker 開始も `issue001_baseline.json` に保存した。

| 条件 | score | coach attacker の plant | coach defender が受けた plant | defuse 試行 | attacker 時間切れ |
|---|---:|---:|---:|---:|---:|
| 現行 seed 1700、attacker 開始 | 0–13 | 0/12 | 1 | 0 | 5 |
| 現行 seed 1701、defender 開始 | 1–13 | 0/2 | 10（左6、右4） | 0 | 2 |

seed 1701 の第1ラウンドでは、左 `(9,3)` への設置を公開観測と coach 決定ログの双方が保持していた。設置後の最初の coach 指示は5人全員 `MOVE_E`、実行ログも右移動だった。観測欠落や5人分指示の取り違えではなく、**現 checkpoint の方策出力が設置位置と矛盾**していた。seed 1700 ではキャリアーが生存したまま `(17,28)` 付近で停滞して時間切れになる例があった。詳細なラウンド、行動、時刻は JSON を参照。

追加したサイト経路距離・2マス往復の計測では、seed 1700 の attacker 時間切れ5件中3件でキャリアーが同じ2マスをラウンド末尾から82～84 tick 往復した。第2ラウンドは `(17,28)` と `(16,28)` を交互に要求し、サイトまで21マスを残して終了した。移動阻止はこのラウンドで0件。移動不能ではなく **coach が往復を出力**した。残る2件と全滅ラウンドは別原因として扱う。

現行 checkpoint SHA-256: attacker `3ae623afda2a8a96e042d1b12560abea77c014ce68028a5d2ea2def0e25b0ab7`、defender `a175cfdaf08a3c6c3c0bb01b06ef85ce8b225d8ef0adc783897e2aa896c70cdc`。

## 実装と実験

- `coordinator.py` の `DecisionAudit` に、公開 spike 座標・公開キャリアー slot・5人分の coach 指示を追加。試合後に観測→指示→実行を追える。行動制御は変更していない。
- `diagnose_issue001.py` でラウンド別の設置地点、キャリアーの位置、要求行動、移動阻止、defuse 試行、checkpoint SHA を保存する。再現試合の終了後だけ replay/referee を読む。
- 学習局面の plantable tile 数は左7、右14だった。従来の tile 均等サンプリングは物理サイトを左1/3、右2/3にしていたため、左右サイトを交互に選ぶ `_balanced_plant_cell` を追加した。
- 学習中、敵の先行行動で最後の味方が死亡した場合に coach を待って例外になる学習環境の終了処理を修正した。
- 実験候補用に、左右サイト・設置・落下位置への相対座標と slot 位置の局所マップ特徴を加える設定を `CoachActorModel` に追加。旧 checkpoint の既定設定・重み読み込みは維持し、候補だけ別ディレクトリに保存した。
- defender の左右設置を同一最適化バッチへ入れる `paired_sites` 収集を追加。実験候補で使用した。

候補 `checkpoints/experiments/issue001_local/coach/{attacker,defender}/latest.pt` は `train_issue001.py` で元の Task 16 重みから初期化し、attacker を30サイクル、defender を左右ペアの retake 局面30サイクル学習した。候補 SHA-256: attacker `f9e0fd18f1871edd92e32279eb1b8317f490516f8774408f3b5d3d71bbd2948c`、defender `97ba199c8e18657b3f17d58f76ed76065fcb983b45ea966053005bbdd6f69ff5`。

| 条件 | score | coach attacker の plant | coach defender が受けた plant | defuse 試行 |
|---|---:|---:|---:|---:|
| 候補 seed 1700、attacker 開始 | 0–13 | 0/12 | 1 | 0 |
| 候補 seed 1701、defender 開始 | 0–13 | 0/1 | 12 | 0 |

候補は P0 の改善を示さず、seed 1701 の match score は悪化した。結果は `reports/issue001_local_1700.json` と `reports/issue001_local_1701.json` に保存した。前段階の `issue001_position` と `issue001_geometry` 候補も不採用で、結果を `reports/issue001_candidate_1701.json`、`reports/issue001_geometry_*.json`、`reports/issue001_paired_1701.json` に残した。これらは候補選択用の再現 seed のため、独立 holdout の改善証拠ではない。改善候補ができてから独立 seed・3相手で比較する。

### 長距離局面と重点損失の追加検証

実戦で右側にいた defender から左 `(9,3)` までの経路距離は40～55マスで、従来の group-up 学習局面の最大35マスを超えていた。右端 `(19,42)` からは55マスで、爆発まで55 tick、解除に6 tick が必要なため、設置後に最短で動いても間に合わない。設置前の左右配置を含めた改善が必須。attacker の rally 局面も実際の初期距離を含むよう20～50マスに広げ、defender group-up を8～55、ability-retake を5～45へ広げた。

`issue001_distance` 候補では左右ペアの retake と初期配置、full-round を混ぜ、さらにキャリアー移動と設置後移動の教師損失を重点化した。最終 checkpoint SHA-256 は attacker `aa91f7b928050d0fc5c3d73fa94a8c83832f3ed22d281e653f57a612ed1e088a`、defender `e501bdf2e868d0fa98f1a8365f1c359bdf2329fdfbd4a812235cd7db60d962b9`。再現 seed 1701 では左設置5例中1例で6 tick 連続 DEFUSE に成功したが、右設置7例は0回成功。seed 1700 の attacker plant は0/12のまま。

再現する場合は side ごとに新しい空ディレクトリを `--directory` で指定し、`python -X utf8 -m coach_v1.train_issue001 --side attacker --cycles 30` と defender の同コマンドを実行する。その後、同じディレクトリで attacker に `--cycles 20 --carrier-move-weight 4`、defender に `--cycles 20 --retake-move-weight 3` をそれぞれ追加実行する。学習先は元の Task 16 checkpoint と別である。

候補選択に使っていない seed 1800/1801 を同一相手 omoko_v1・同一開始陣営で比較した。`reports/issue001_holdout_{baseline,candidate}_{1800,1801}.json` に checkpoint hash とラウンド別の記録を保存。

| 未使用 seed | 条件 | score | attacker plant | defender が受けた plant | defuse 成功 |
|---|---|---:|---:|---:|---:|
| 1800、attacker 開始 | 現行 | 0–13 | 1/12 | 左1 | 0 |
| 1800、attacker 開始 | 候補 | 0–13 | 0/12 | 左1 | 0 |
| 1801、defender 開始 | 現行 | 0–13 | 0/1 | 左9、右2 | 0 |
| 1801、defender 開始 | 候補 | 0–13 | 0/1 | 左4、右8 | 0 |

候補は独立 seed で attacker plant を悪化させ、defender retake の改善を再現できなかった。**正式 checkpoint は更新しない。** 2試合だけでは改善可能性を否定できないが、合格条件を満たさない候補に他相手の大量評価を重ねていない。

seed 1800 の候補は12攻撃ラウンドすべて全滅し、キャリアーが最後に確認された時点でサイトまで22～26マス残していた。2マス往復は解消したが、到達・生存は改善していない。時間切れの解消だけを plant 改善とみなさない。

## 次に必要なこと

1. attacker のキャリアーが停止・衝突・死亡する tick を診断ログで分類する。特に現候補は時間切れが減っても死亡が増え、plant は増えていない。coach の教師軌跡と on-policy 軌跡の位置別行動一致率を計測し、サイト到達までの長距離経路と生存を同時に改善する。
2. defender は、左右設置後の同じ開始位置に対して coach の5人分指示と教師を比較し、設置前に片サイトへ全員移動する偏りも合わせて修正する。左右の到達・DEFUSE・成功を別々に測る。
3. 既知の再現 seed では候補を選別するだけとし、合格見込みの候補を用意してから未使用 seed で omoko_v1・touyama_v2・gc_v1、両開始陣営を同条件比較する。左右別・キャリアー別・設置前後別の件数を残す。

Task 15 では既に opponent pool 上の DAgger を反復しても教師損失の低下と実戦性能が一致しないと確認されている（`HANDOFF-15.md`）。今回の固定 default 相手の模倣を単純に増やすだけで実戦改善を期待しない。次の候補はキャリアーの往復・全滅、設置前配置、左右別 defuse の実戦指標を選択条件に含める。

## 検証

- `python -X utf8 -m unittest discover -s coach_v1 -p 'test_coach_v1_*.py' -q`: 209件成功。
- `python -X utf8 -m unittest coach_v1.test_coach_v1_issue001 -q`: 4件成功。左右交互サンプル、左右同一バッチ、位置入力差、長距離 retake 局面を確認。
- `python -X utf8 -m unittest coach_v1.test_coach_v1_task12_attacker coach_v1.test_coach_v1_task13_defender coach_v1.test_coach_v1_issue001 -q`: 20件成功（重点損失追加後）。
- `python -X utf8 -m compileall -q coach_v1` と `git diff --check`: 成功。
- core ファイル、正式 checkpoint、既存のユーザー文書移動は変更していない。

## プロジェクト継続判断のための教師方策評価

ユーザーの依頼で、これ以上の再学習を行わず、既存の attacker/defender 教師を診断専用 coach としてフルマッチへ接続した。`diagnose_issue001.py --teacher` は通常の公開 `CoachObservation` と合法 action mask だけを教師へ渡し、キャラクターモデル・マップ・相手・seed は現行評価と同じ。教師の経路探索は本番推論に追加していない。omoko_v1、touyama_v2、gc_v1 に対し seed 1700 の attacker 開始と1701 の defender 開始、計6試合を実行した。詳細は `reports/issue001_teacher_{omoko,touyama,gc}_{1700,1701}.json`。

| 相手 | coach round 勝利 | attacker plant | defender が受けた plant | defuse 成功 | match 勝利 |
|---|---:|---:|---:|---:|---:|
| omoko_v1 | 5 | 0/17 | 7（左7） | 2（左2） | 0/2 |
| touyama_v2 | 9 | 12/17 | 18（左14、右4） | 5（左5） | 0/2 |
| gc_v1 | 7 | 10/20 | 11（左6、右5） | 5（左1、右4） | 0/2 |
| 合計 | 21/99 round | 22/54 | 36 | 12/36 | 0/6 |

omoko_v1 への攻撃は12ラウンド連続で時間切れになった試合があり、キャリアーは全てサイトまで2マスで停止した。第1ラウンド終了時、味方はサイト `(9,3)`、入口 `(10,3)`、キャリアー `(11,3)`、後続 `(12,3)` と一列に並び、キャリアーは合法移動を失って `STAY` を出し続けた。現在の attacker 教師が**全員を同じ最短地点へ送る**ため、味方自身が設置を妨げる。これは未視認情報・ゲーム core の問題ではなく、教師の役割割当と経路の問題である。

defender 教師では左右とも実際の解除成功を確認した。一方、全相手で match 勝利0/6のため、教師をそのまま模倣しても競争力達成の根拠にはならない。既存の学習済み coach はこの教師の有効局面も十分再現できておらず、教師自体にも明確な欠陥がある。**現行の教師・模倣学習を続けるだけでは改善しない**という判断を支持する。役割分担を持つ教師／訓練目標と、相手別の実戦検証を組み直すなら改善余地はあるが、それは局所修正ではなく coach_v1 の再設計になる。
