# HANDOFF-05: coach観測エンコーダー

## 2026-09-24 Task 05 完了記録

### 事前確認と範囲

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、`03.DESIGN.md`、先行HANDOFF、
  Task 03のチーム知覚、Task 04のbelief、固定マップ、警戒ポイント設定、
  checkpoint metadataを確認した。
- Task 05はcoach actorの観測だけを実装した。キャラクター固有観測、
  critic観測、学習シナリオ、モデル、実行コーディネーターは対象外。
- 作業開始前から変更されていた `coach_v1/10.task_template.md` は変更していない。

### 実装

- `CoachObservationEncoder` を追加。入力は `TeamPerceptionSnapshot` と
  `BeliefSnapshot`、および呼び出し側が明示するside適合のsituationに限定した。
- 出力は固定shapeの `float32` 配列: CNN grid `[27, 26, 44]`、
  非グリッドvector `[84]`。固定map、警戒ポイント、共有視野、クリア履歴、
  現在の合法な敵目撃、最終目撃履歴、公開spike、smoke、味方5 slotを符号化する。
- channel／vectorの正確な順序と正規化は `observation/README.md` および
  `COACH_GRID_CHANNELS`、`COACH_VECTOR_FIELDS` に固定した。
- clear／last-seen／警戒ポイント確認の経過tickは256でclipして0～1へ変換する。
  presence channelを別に設け、未確認と「確認済みage 0」を区別する。
  256 tick以上を安全／危険と判定するロジックは入れていない。
- 警戒ポイントは現在sideとsituationに適合するものだけをchannel化する。
  importanceと推奨facingを含むが、移動先や経路には変換しない。
- 味方slot順は固定ロスター順。死亡slotはgridとvector内でゼロpaddingし、
  後続slotを詰めない。
- encoder生成時に固定マップのshapeとセル、警戒ポイント設定とmap hashを検証。
  `validate_checkpoint` で観測version、map hash、警戒ポイントhash、roster、coach sideを照合する。
- 入力のtick、side、roster、grid shape、watch point集合、age、座標等を検証し、
  非有限値の出力を拒否する。配列はコピーして読み取り専用で返す。

### 未視認敵情報の監査

- encoderは生ゲーム、`Character`、`game_state["chars"]`、critic情報を引数に持たない。
  敵の位置channelはTask 03が合法に報告した現在目撃とTask 04が保持する
  過去の合法な最終目撃だけから作る。敵の実位置、HP、未確認のspike所持者はない。
- 固定マップ上の二つの実ゲーム状態で、未視認敵の実位置だけを変え、
  `TeamPerceptionBuilder` → `BeliefMemory` → `CoachObservationEncoder` を実行した。
  gridとvectorの全要素が完全一致した。
- 合法な目撃後に敵が未視認となるテストでは、現在目撃channelが消え、
  最終目撃地点と経過tickだけが残ることを確認した。

### core変更判断

coreファイルの変更は不要で、変更していない。既存の安全なTask 03／04 DTOと
固定map設定だけでTask 05のactor観測を構築できた。

### 変更ファイル

- `coach_v1/observation/__init__.py`（新規: 公開API）
- `coach_v1/observation/coach_encoder.py`（新規: encoderと検証）
- `coach_v1/observation/README.md`（新規: v1観測仕様）
- `coach_v1/test_coach_v1_task05_coach_encoder.py`（新規: 自動テスト）
- `coach_v1/HANDOFF-05.md`（新規: 本記録）

### テスト結果

- `python -m unittest -v coach_v1.test_coach_v1_task05_coach_encoder
  coach_v1.test_coach_v1_task04_belief_memory
  coach_v1.test_coach_v1_task03_team_perception
  coach_v1.test_coach_v1_task02_watch_points
  coach_v1.test_coach_v1_task01_foundation
  coach_v1.test_coach_v1_task00_design`: 62件成功（Task 05は9件）。
- `python -m compileall -q coach_v1`: 成功。
- `git diff --check`: errorなし。

### 残課題と次の推奨タスク

- round timerとdetonation timerはTask 03のDTOに存在せず、現v1観測には含めていない。
  後続で合法な公開timerの導入が必要と判断した場合は、Task 03境界と
  観測versionを同時に見直す。
- Task 06では警戒ポイント中心の70/20/10配置と合法ランダム配置を作り、
  actor観測に敵の実位置が漏れないことを継続検証する。
- 学習／推論への接続と1tickに1回のcacheは後続Task 10以降。
- 次の推奨タスク: **Task 06 警戒ポイント中心のシナリオ生成**。
