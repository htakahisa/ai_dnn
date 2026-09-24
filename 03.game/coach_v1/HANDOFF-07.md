# HANDOFF-07: キャラクター固有モデルの環境・行動仕様

## 2026-09-24 Task 07 完了記録

### 事前確認と範囲

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、`03.DESIGN.md`、Task 00～06 の実装・引き継ぎ、ゲームの既存戻り値・アビリティ成立条件を確認した。
- Task 07 はキャラクター観測、行動マスク、coach 指示を外部入力とする行動変換、初期 1v1・2v1 カリキュラムの契約に限定した。モデル、trainer、報酬、ゲーム状態生成、coordinator は実装していない。
- 作業前から変更されていた `coach_v1/10.task_template.md` は触れていない。

### 実装

- `CharacterObservationEncoder` を追加。既存の安全な coach 観測 27 grid channel・84 vector field を再利用し、自分の位置と coach 指示を加えた `float32 [29,26,44]` grid、`float32 [106]` vector を返す。固定マップ・警戒ポイント hash とキャラクター checkpoint metadata の照合を持つ。
- 行動は 8方向 facing、通常アビリティ使用有無、対象マスのみ。移動 action は持たない。行動マスクは生死、残数、キャラクター種別、固定マップの床・壁、投射物の初手の壁判定、coach 目的行動を反映する。HUNT の通常 active ability は常に不可。
- `CharacterEnvironment.prepare` は coach 指示と知覚 DTO を受け取り、任意の `1v1`・`2v1` 段階の生存人数を検査する。`resolve` はマスク外の action を拒否し、ゲームの既存 controller tuple に変換する。通常アビリティを使う場合は現在地を返して移動を止める。PLANT・DEFUSE を優先する。
- 観測の shape、field 順、mask 意味、ゲームとの戻り値を `observation/CHARACTER.md` に文書化した。

### 未視認敵情報の監査

- character encoder と環境の入力は `TeamPerceptionSnapshot`、`BeliefSnapshot`、coach 指示のみ。Scenario の敵実座標、生ゲーム、Character、critic を受け取らない。
- 固定マップに異なる未視認敵実位置を生成し、`TeamPerceptionBuilder` → `BeliefMemory` → character encoder を通した。両局面で目撃は空、grid・vector・target mask は完全一致した。
- 味方5人の合法な目撃、clear age は既存の team snapshot と belief をそのまま使う。未確認マスを永久に安全とは判定しない。

### core変更判断と残課題

- core変更は不要。既存 controller は `(座標, {"facing": ...})`、`(現在地, {"ability": ..., "target": ...})`、`(現在地, "PLANT"/"DEFUSE")` を受け取る。アビリティ中の移動停止も既存ゲームで実施される。
- 既存 `battle_logic.py` の ABILITY 分岐は facing 適用前に戻るため、ability payload の facing 指定は現状適用されない。Task 10 のゲーム接続時に、coordinator 側で安全に facing を適用できるか確認する。代替案は core の ability 分岐で facing を処理する仕様変更だが、既存挙動への影響を検証してから別途判断する。facing のみ／通常移動中の facing 指定は機能する。
- `1v1`・`2v1` は生存人数の入力契約まで。敵配置・局面生成と報酬・学習ループは後続の trainer が担当する。

### 変更ファイル

- `coach_v1/observation/character_encoder.py`（新規）
- `coach_v1/observation/__init__.py`（公開 API 追加）
- `coach_v1/observation/CHARACTER.md`（新規）
- `coach_v1/training/character_environment.py`（新規）
- `coach_v1/training/__init__.py`（公開 API 追加）
- `coach_v1/test_coach_v1_task07_character_environment.py`（新規、自動テスト7件）
- `coach_v1/HANDOFF-07.md`（新規、本記録）

### テスト結果

- `python -m unittest -q coach_v1.test_coach_v1_task07_character_environment coach_v1.test_coach_v1_task06_scenario_generator coach_v1.test_coach_v1_task05_coach_encoder coach_v1.test_coach_v1_task04_belief_memory coach_v1.test_coach_v1_task03_team_perception coach_v1.test_coach_v1_task02_watch_points coach_v1.test_coach_v1_task01_foundation coach_v1.test_coach_v1_task00_design`: 75件成功（Task 07 は7件）。
- Task 07 テストでゲームの投射経路とマスクの判定も比較した。

### 次の推奨タスク

**Task 08: キャラクター固有モデルと trainer の実装**。この観測・mask・行動契約を使って、1キャラクターの学習パイプラインと検証指標を作る。
