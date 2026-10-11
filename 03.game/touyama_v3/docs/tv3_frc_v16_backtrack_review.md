# 良かった重みへ戻す比較（2026-10-12）

保存bestは引き続きv16 set10。hashはcd9079b8813a14e05edd4751bef61751e48e3aea0a9db688aecab401d369d453。モデルファイルの上書き、schemaの読み替え、追加学習は行っていない。

## 重みと実行処理を分けた対照

seed804000142、start_round=1、対象8ラウンド。前7ラウンドは両方とも保存best v16で実行し、対象開始時にのみ比較するモデルを切り替えた。前ラウンド結果、対象開始時のPython/NumPy/Torch乱数hash、対象の実行schema hashは一致。

| 対象ラウンドの重み / 実行処理 | 結果 |
| --- | --- |
| 保存v16 / 元のv16 | 時間切れ、5v2、HP460 |
| 保存v16の重みをメモリ内で初期化 / v19 | 69tick設置、5v2、HP460 |
| 学習済みv19 set5の重み / 同じv19 | 59tick設置、4v2、HP320 |

v16の重みをv19へコピーした対照は、新規v19の学習済みモデルではない。元の保存v16のschemaやファイルは変更せず、対象の実行版を明示する診断だけ。

最初に行動が異なるのは16tickのRECON役slot3。同じ観測配列・最終mask hashで、旧重みはABILITY、新重みはN。旧重みは16tickにRECON、30tickにFLASHを使い、FLASHは敵4へ作用。新重みは17tickに異なる目標へRECONを使い、この対象ラウンドではFLASH使用なし。以後の差は行動・射撃・乱数消費の変化も含む。

新v19だけで前7ラウンドも動かした実力評価では、同じseedの8ラウンド目は90tickの2v3設置だった。今回の4v2は前7ラウンドを旧bestに固定した別条件の対照であり、実力評価値の訂正ではない。特定の疲労だけに原因を帰す根拠でもない。

この一例では、最新の目標修正・終盤候補処理と旧重みの組み合わせが、元の時間切れを解消しつつ人数をより多く残した。コードを丸ごと戻すことと、良かった重みに戻すことは分けて判断する必要がある。ただし一例で全面採用や72ラウンドの性能改善とは扱わない。現行best v16を維持。

ログは`logs/attacker_route_risk_review/v16_weights_v19_execution_r8_control.json`、`v19_weights_v19_execution_r8_matched_prefix.json`、要約は`v16_backtrack_r8_review.json`。

## 次回の学習で、途中へ戻って調べられるようにする

これまではlatestが各セットで上書きされ、評価checkpointはset5など評価した時点だけだった。今回のv19学習のset1～4の重みは後から復元できない。既存のreplay・optimizerから過去の重みを作ったことにしない。

`tv3_train_attacker_plant.py`冒頭にSAVE_REVIEW_WEIGHT_CHECKPOINTS=Trueを追加。今後ユーザーが学習を実行したとき、学習前と各セット終了時の重みだけを保存する。

保存先は`data/attacker_plant/<相手>/review_checkpoints/v<版>_<実行ごとのUUID>/set_000000.pt`、`set_000001.pt`など。実行ごとにディレクトリを変えるため、前の実行のset5も消えない。再開の場合の開始checkpointはその完了set番号になる。

schema・analysis hash・warm_start hash・セット番号・開始セットを記録し、evaluation_status=not_evaluated、checkpoint_kind=diagnostic_weights_onlyを明示。optimizer/replayや架空の評価結果は保存しない。学習前set0の重みには旧モデル由来の知識があるので、warm_start hashを必ず残す。診断スクリプトのSOURCE_PLANT_CHECKPOINTで明示指定して読み、未評価の重みをbestに昇格させない。

実際の学習はこの作業では行っていない。通常の5セット上限・評価頻度・報酬・学習損失は変更していない。次回は途中checkpointを使って、悪化が始まった段階を評価し、良かった段階まで戻って比較できる。新しい学習の実行依頼は現時点で行わない。

重み保存テスト1件とplant報酬・選定・回帰24件、計25件通過。別実行・別セットの保存が以前の重みを壊さないこと、由来が残ること、未評価の明示とoptimizer/replayの不保存を確認。診断スクリプトの構文確認・差分空白チェックも通過。
