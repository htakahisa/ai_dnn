# HANDOFF-10: チーム実行コーディネーター

## 2026-09-24 Task 10 完了記録

### 事前確認と範囲

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、`03.DESIGN.md`、Task 03～09 の知覚・belief・観測・character 実装、`team_ai.py`、ゲーム側の controller 戻り値と setup/live 呼び出しを確認した。
- 作業開始前から変更されていた `coach_v1/10.task_template.md` と `omoko_v1` 関連ファイルには触れていない。
- Task 11 の coach 学習・checkpoint、戦術規則、経路探索、core 変更、UI のチーム選択肢追加は実装していない。

### 実装

- `TeamExecutionCoordinator` は一方の side の固定5 slot を担当する。ゲームの最初の `decide_move` で安全なチーム snapshot、belief、coach 観測を作り、外部から渡された coach actor を一度だけ呼ぶ。5人分の `CoachInstruction` と snapshot を `(side, round, phase, tick)` に相当する key で保持する。
- 残りのキャラクターには同じ snapshot と各 slot の coach 指示から character 観測を作り、各固有 actor の facing・通常 ability 判断を実行する。既存 `CharacterEnvironment.resolve` の優先順位で目的行動、ability、coach 移動をゲームの controller 戻り値に変換する。
- ABILITY と PLANT/DEFUSE はゲーム側の facing 適用箇所より先に return するため、coordinator が検証後の facing をゲームの character に適用する。`facing_forced_this_tick` は尊重する。ability 要求は現在地を返すため同 tick に移動しない。
- setup/live・tick・round の変化でキャッシュを再生成し、round 変更時と `reset_round` 時に belief を消去する。死亡 slot は character actor を呼ばない。要求行動の `ActionLog` を記録する。固定マップ以外は `set_game` で拒否する。
- `team_ai.py` に汎用の `handles_team_perception` 判定を2行追加した。これにより独自の安全なチーム知覚を持つ controller は既存の一人用 IQ wrapper を通らず、ゲームの `DualRoleTeamAI` に接続できる。core ファイルは変更していない。

### テストと情報境界

- 新規自動テスト9件。coach の同 tick 一回実行、最初に呼ぶキャラクター順と後続移動による入力不変、ability 時の停止と facing、coach 移動の保持、setup/live/round 切替、死亡者、wrapper 接続、固定マップ検査を確認した。
- 歩行可能な2つの未視認敵実位置を入れ替えても、coach と character の grid/vector が完全一致した。合法な目撃は5人に共通に届き、次 tick に視界を失うと last-seen/clear age が1に増えることも確認した。actor に生ゲームや敵の実座標を渡す経路はない。
- `python -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q`: **90件成功**。`python -m compileall -q coach_v1`、`git diff --check -- team_ai.py coach_v1` 成功。

### 残課題と次の推奨タスク

- coach actor は Task 11 の学習モデルが未実装のため注入方式。ゲームの通常メニューへ coach_v1 を常設するのは、attacker/defender の checkpoint が揃ってから行う。
- `ActionLog` は controller が要求した行動を記録する。当初未実施だった実ゲーム評価は下記追補で実施した。ゲーム全体の勝率や coach 学習の評価は引き続き対象外。
- 次は **Task 11: coach 学習環境とモデル**。この coordinator の actor 契約に5人分の移動・意図を返すモデルを接続し、side 別 checkpoint を作る。

### 変更ファイル

- 新規: `coach_v1/coordinator.py`、`coach_v1/test_coach_v1_task10_coordinator.py`、`coach_v1/HANDOFF-10.md`
- 更新: `team_ai.py`（汎用 wrapper bypass 2行）

## 2026-09-24 追補: Task 09 から持ち越した実ゲーム評価

HANDOFF-09 の「Task 10 接続後の実戦評価を経て判断する」に対し、初回の Task 10 完了時にはダミー actor のテストしかなく、評価が欠けていた。本追補では Task 09 の学習済み character checkpoint 5人を実際の headless ゲームと Task 10 coordinator に接続した。Task 11 はまだ実装していない。

### 方法

- `evaluate_task10_rollout.py` で attacker/defender を別々に評価した。coach は固定 `STAY/HOLD` のダミー、相手は既存 `default`。1 tick につき coach が一度だけ呼ばれる実ゲームループで各モデルを推論した。
- 自然配置で seed 7・各80 live tick を実行すると、LOS が通る敵の機会は attacker 8件、defender 0件で、ability 発動は双方0件だった。この条件では facing の実戦評価に足りない。
- 接敵テストは seed 0～9 の10回、各最大20 live tick。defender setup 終了後、**評価環境だけ**が相手5人を合法な近距離床へ配置した。actor には敵の実位置を渡さず、既存 `TeamPerceptionBuilder` の合法な目撃と belief だけを使った。
- facing 指標は、ゲームによる被弾後の強制 facing を除き、チームに合法的に共有された敵の報告位置のいずれかへ **45度以内**に向いた行動の割合。これは実ゲーム中の視線方向の診断値であり、教師ラベルに対する「正解率」や勝率ではない。ゲーム側の出力適用と、通常 ability のチャージ消費・flash/recon の敵への効果イベントも記録した。

### 結果

| checkpoint | side | 共有目撃への45度以内 | 非強制時の facing 適用 | ability 発動成功 |
|---|---|---:|---:|---:|
| 正式 `best.pt` | attacker | 186/394 = 47.2% | 527/527 | 40/40 |
| 正式 `best.pt` | defender | 191/485 = 39.4% | 603/603 | 40/40 |
| 実験 `facing_best.pt` | attacker | 161/359 = 44.8% | 577/577 | 40/40 |
| 実験 `facing_best.pt` | defender | 111/493 = 22.5% | 607/607 | 40/40 |

正式checkpointの slot 別45度以内率（attacker / defender）は、ごりまる33/44・3/42、ごんごん67/91・28/144、ごんた15/109・119/160、くんた62/82・2/79、くりまる9/68・39/60。特定の side とキャラクターで弱さが目立つ。ただし、共有された敵を見ることが常に最善とは限らず、この値だけで戦術的な正解率は断定できない。

正式checkpointでは、attacker と defender のそれぞれで smoke 10回、flash 20回、recon 10回がゲームに受理された。相手への効果イベントは attacker で flash 21件・recon 40件、defender で flash 37件・recon 50件。これは影響を受けた敵の延べ件数であり、スモークの位置の良さ、kill への寄与、勝率を表さない。

### 判断と限界

- 学習済みモデルの facing はゲーム側で適用されている。通常時は1130/1130行動でモデル指定と一致した。被弾後の強制 facing はゲーム仕様として別に扱った。
- `facing_best.pt` への一律差し替えは、同じ接敵テストで双方の side とも指標が下がるため採用しない。
- 正式checkpointにも side/slot ごとの大きな偏りがある。特に attacker のごんた・くりまる、defender のごりまる・ごんごん・くんたは、近距離接敵と共有目撃を含むデータを追加して再学習・再評価する候補。ゼロからのモデル設計し直しや全員一律再学習の必要性までは、この固定 `STAY` coach と人工接敵条件からは判断できない。学習した coach を使った自然な移動・勝率評価は Task 11 以降に残る。

### 追加ファイルと再現

- 新規: `coach_v1/evaluate_task10_rollout.py`、`coach_v1/reports/task10_rollout.json`、`coach_v1/reports/task10_near_rollout.json`、`coach_v1/reports/task10_near_facing_best_rollout.json`
- 実行例: `python -m coach_v1.evaluate_task10_rollout --near --ticks 20 --seed 0 --seeds 10 --output coach_v1/reports/task10_near_rollout.json`
- 実験checkpoint比較: 上記に `--variant facing_best` を追加する。
