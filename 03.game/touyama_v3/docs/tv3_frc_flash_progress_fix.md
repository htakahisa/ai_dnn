# 確認済みフラッシュ後の前進と集合待ち（plant v16）

## 停止原因と修正

seed804000042の第4ラウンドを、直前まで保存best set90のv13で動かしてから比較した。56・57tickで所有フラッシュの公開爆発を確認でき、効果時間の入力も正だったが、保持者のEはrecent_edge_limitで禁止され、STAYしか選べなかった。

v16では、選択経路上で目標距離を縮める一歩について、既知の入射線がすべて確認済みの所有フラッシュで覆われる場合だけ往復制限を解除する。エンジンの合法mask、味方の移動先予約、経路の通路制限は保持する。フラッシュを投げただけ・爆発前・効果期限後・覆われない射線がある場合は解除しない。前進行動は強制しない。

教師も同じ一歩の条件で助言を合わせる。移動を続けると12,40で集合待ちが優先されることも確認したため、保持者が集合地点付近にいて、選択入口の既知の射線が確認済みフラッシュで覆われ、到達距離＋設置所要tick＋1tickの余裕が残る場合だけ集合目標を解放する。効果期限はexclusiveなので余裕tickを入れる。これも目標の待ち解除であり、モデルの前進・設置行動を強制しない。

設置地点で同じ時間条件を満たす場合、教師はその窓を設置に使う例を出す。未知の敵の位置・実際のblind値は入力に読まない。覆われるのは既知の射線であり、全敵が無力化された保証ではない。

## 限定比較

教師だけの診断。重み更新0回、対象ラウンドより前の結果・対象開始時の乱数hashを照合。

| 条件 | 第4ラウンド | 第6ラウンド |
| --- | --- | --- |
| v16、今回の修正前 | 時間切れ、3v4 | 88tickで設置、3v3 |
| 往復制限の限定解除 | 93tickで設置、3v3 | 74tickで設置、4v3 |
| 教師の前進だけを追加 | 時間切れ、2v4 | 79tickで設置、4v3 |
| 集合待ち・設置時間条件まで整合 | 93tickで設置、3v3 | 64tickで設置、4v4 |

第4ラウンドが最初のフラッシュ中に即座に設置できた、という結果ではない。途中に残る待機や、その後の交戦も含めて93tickで設置した。早い設置と人数優位も別の指標であり、64tick・4v4が74tick・4v3より必ず勝ちやすいとは主張しない。

## 12ラウンドと未検証事項

同一seed・同一保存重みの教師比較では設置8→9、人数有利での設置5→7、人数同数以上での設置8→8、時間切れ3→3、全滅1→0。教師を入れない旧重み比較は設置9→8、人数有利7→5、人数同数以上9→7、時間切れ2→3。新しい入力・教師を学習したモデルの性能は未検証。1seedの教師比較を実力改善と扱わない。

記録は `logs/attacker_route_risk_review/set90_r4_flash_window_baseline.json`、`set90_r4_flash_edge_probe.json`、`set90_r4_flash_edge_integrated.json`、`set90_r6_flash_edge_integrated.json`、`set90_r4_flash_teacher_integrated.json`、`set90_r6_flash_teacher_integrated.json`、`set90_r4_flash_rally_integrated.json`、`set90_r6_flash_rally_integrated.json`、`set90_flash_rally_teacher_12round_final.json`、`set90_flash_rally_model_12round_final.json`。

228件の非学習テスト通過。確認前・期限後・覆われない射線・自分のblind・予約・エンジン禁止・後退方向・設置完了の時間余裕を検証。射線エンジン・シナリオ・analysis入力は変更していない。既存bestの内容と保存版は維持し、旧v13は元の実行経路で読む。

## 次の検証学習（ユーザーが実行）

ここからはplantの再学習が必要。通常の新規plant版を16にし、`tv3_train_attacker_plant.py` 冒頭を次の短い検証設定にした。

- `TARGET_OPPONENTS=("frc_v1",)`
- `TRAINING_SETS=20`、`TOTAL_TRAINING_SET_LIMIT=20`
- `EVALUATION_INTERVAL=5`
- `RESUME_TRAINING=False`、`WARM_START_FROM_BEST=True`

同じ入力構造・盤面・analysis依存のv13 best重みだけを初期値に使い、新しいoptimizer/replay・0からのセット数で始める。元bestのバイト列とhashはwarm_start_sourcesへ退避する。保存済みbestは元の版で再評価し、選定指標を上回った場合だけ更新する。旧90セットをv16として読み替えない。検証した全対象リストは`ALL_TARGET_OPPONENTS`に保持し、Ghost Champions v2を削除していない。

作業ディレクトリは `03.game/touyama_v3/`。通常実行：

```powershell
python tv3_train_attacker_plant.py
```

まずset5/10の `[実力評価][教師・探索なし]` とサイト別評価を確認する。教師付き学習成功率を実力と扱わず、右時間切れ・人数有利設置・能力使用を比較する。20セットで不足か、改善しないかも含めて判断する。analysisはこの比較では固定し、更新しない。guardの再収集・学習は新plantを採用した後に依存hashをそろえて行う。まだ全対象一括学習を実行する段階ではない。

こちらでは学習を実行していない。以前の「再学習開始は保留」は教師停滞の候補を検証するまでの判断であり、現在は上記の短いplant検証学習へ進む段階。
