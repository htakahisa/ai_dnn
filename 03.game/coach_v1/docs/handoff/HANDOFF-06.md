# HANDOFF-06: 警戒ポイント中心のシナリオ生成

## 2026-09-24 Task 06 完了記録

### 事前確認と範囲

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、`03.DESIGN.md`、Task 00～05 の HANDOFF、
  固定マップ、警戒ポイント設定、Task 03～05 の知覚・belief・coach観測を確認した。
- Task 06 は学習用の敵配置生成、ログ、分布レポートに限定した。ゲーム状態への適用、
  character/coach環境、critic、モデル、推論コーディネーターは実装していない。
- 作業前から変更されていた `coach_v1/10.task_template.md` は触れていない。

### 実装

- `ScenarioGenerator(seed)` を追加。actor side と situation に合う警戒ポイントから、
  敵を 70% point、20% jitter、10% legal random のカテゴリ抽選で配置する。
- point は `importance` で重み付け。jitter は警戒ポイントから 1～`random_radius`
  歩行ステップ以内のセルを使い、他のpointセルは除外する。
- 敵陣営の5つの初期配置セルからの最短歩行距離を事前計算し、`elapsed_ticks` 以内に
  到達できるセルだけに配置する。複数敵の場合は、各敵を別々の初期配置セルに
  割り当てられることも検査する。壁、範囲外、現在の合法な味方視野、指定された
  味方占有セル、同じシナリオ内の敵重複を除外する。
- 初期tickなどで候補のないカテゴリは抽選から除き、残りの重みを正規化する。
  候補が全くなければ `ScenarioGenerationError` を返す。カテゴリ実績は
  `EnemyPlacement.category` と分布レポートに残す。
- シナリオには seed、連番、map hash、警戒ポイントhash、配置カテゴリ、
  point ID、スポーンからの距離を記録する。JSONL ログ保存関数と割合レポート関数を追加。

### 未視認敵情報の監査

- `Scenario` と `EnemyPlacement` は学習側専用の実配置データ。actor encoder に渡すAPIは
  追加していない。生成器は既存の敵実位置を入力として受け取らない。
- 生成した異なる2つの未視認敵配置を実ゲーム相当の状態へ置き、
  `TeamPerceptionBuilder` → `BeliefMemory` → `CoachObservationEncoder` を実行した。
  目撃はどちらもなく、actor の grid/vector は全要素が完全一致した。
- 学習側の呼び出し元は、そのtickの合法な `currently_visible` を必須入力として渡す。
  実際の敵位置は知覚境界を経ずに actor に渡してはならない。

### core変更判断

coreファイルの変更は不要だった。固定マップと設定を読み取り、Task 03 の合法な
視野マスクを呼び出し元から受け取るだけで必要な制約を実現できた。

### 変更ファイル

- `coach_v1/training/__init__.py`（新規、公開API）
- `coach_v1/training/scenario_generator.py`（新規、生成・ログ・レポート）
- `coach_v1/training/README.md`（新規、学習専用APIの使い方）
- `coach_v1/test_coach_v1_task06_scenario_generator.py`（新規、自動テスト6件）
- `coach_v1/reports/task06_distribution.json`（新規、固定seedサンプルの分布）
- `coach_v1/logs/task06_sample.jsonl`（新規、固定seedサンプルの配置ログ。
  repositoryの `logs/` ignore対象なので作業環境に保存）
- `coach_v1/HANDOFF-06.md`（新規、本記録）

### テスト結果

- `python -m unittest -v coach_v1.test_coach_v1_task06_scenario_generator
  coach_v1.test_coach_v1_task05_coach_encoder
  coach_v1.test_coach_v1_task04_belief_memory
  coach_v1.test_coach_v1_task03_team_perception
  coach_v1.test_coach_v1_task02_watch_points
  coach_v1.test_coach_v1_task01_foundation
  coach_v1.test_coach_v1_task00_design`: 68件成功（Task 06 は6件）。
- `python -m compileall -q coach_v1`: 成功。
- 固定seed `20260924`、attacker/carry、経過100tick、1000シナリオ×敵5人で、
  point 3481/5000 = 69.62%、jitter 1003/5000 = 20.06%、
  random 516/5000 = 10.32%。
- `git diff --check`: errorなし。

### 残課題と次の推奨タスク

- シナリオを実ゲームへ適用する環境とcritic入力は後続タスクの責任。
  Task 06 のログは学習側の敵実情報なので actor 入力へ接続しない。
- 次は **Task 07: キャラクター固有モデルの環境・行動仕様**。coach移動指示を
  外部入力として受け、facing と通常アビリティだけを学習する境界を実装する。
