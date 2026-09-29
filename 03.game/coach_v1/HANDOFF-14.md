# HANDOFF-14: フルマッチ統合

## 2026-09-29 対応範囲

`01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、`03.DESIGN.md`、Task 10～13 の
HANDOFF、`TeamExecutionCoordinator`、side swap／round reset／headless／replay の
既存ゲームライフサイクルを確認してから実装した。Task 15 の self-play、opponent pool、
追加学習、推論時の戦術規則・経路探索は実装していない。

core ファイル（`abilities_los.py`、`battle_logic.py`、`game_core.py`、`map_data.py`、
`map_data_defender_setup.py`、`run_game.py`）は変更していない。既存の
`DualRoleTeamAI` と `VisualFPSBattle` の公開接続口だけで統合できたため、core 変更案は
採用していない。作業開始前から変更されていた `10.task_template.md` には触れていない。

## 実装

- `team_ai.py` に正式 runtime factory `build_coach_v1_team` を追加した。attacker は
  `learning_coach_attacker` の Task 12 checkpoint、defender は
  `learning_coach_defender` の Task 13 checkpointを読み、side ごとに独立した
  coordinator、belief memory、recurrent state、5キャラ policy を持つ。
- `DualRoleTeamAI` が通常戦13R開始時とOTの side swapで対応controllerを選ぶ。固定
  Gorigons slot順と side 別coach checkpointを混ぜず、キャラクターmodelは既存の
  正式checkpointを使う。ごんごんの既存side別選択も保持した。
- `full_match.py` に本番headless match実行、seed固定、score対応、round reason集計、
  replay構造検査、side checkpoint切替、round先頭memory resetの監査、JSON保存を追加した。
- `run_full_match.py` にCLIを追加した。相手AI／preset、coach開始side、seed、device、
  report／任意replay出力を指定できる。出力は `coach_v1` 配下に置く。
- coordinatorへ `DecisionAudit` を追加した。1 team-wide coach計算につき1件だけ、
  round／phase／tick／side／belief memory tickを記録する。観測配列、生ゲーム、敵座標は
  保持しない。既存の1 tick 1回cacheと行動決定は変更していない。

## 自動テストと情報境界

新規 `test_coach_v1_task14_full_match.py` の4件で以下を確認した。

- attacker／defender checkpoint loaderが別々に呼ばれ、5つのcharacter checkpointが
  固定slotへ対応する。
- 未視認敵だけを合法な2地点で入れ替えた2ゲームについて、factoryから作った
  coordinatorのcoach grid/vectorとcharacter grid/vectorが完全一致する。actorへ
  未視認敵の実位置は混入しない。
- 実 `VisualFPSBattle` の短縮headless matchがsetup、複数round、side swap、時間切れ、
  match終了、replay／report保存まで完走する。
- side変更後は正しいcoordinatorを使い、各roundで最初のbelief memory tickが0になる。

全テスト結果:

- `python -X utf8 -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q`:
  **168件成功**。
- `python -X utf8 -m compileall -q coach_v1`: 成功。
- `git diff --check`: 成功。

## 正式checkpointによる本番フルマッチ

実行:

```text
python -X utf8 -m coach_v1.run_full_match --opponent-ai default \
  --opponent-preset "Ghost Champions" --coach-start-side attacker \
  --seed 14 --report coach_v1/reports/task14_full_match.json
```

固定map、通常round timer、通常setup、通常13本先取、side swap有効。coach_v1は
attacker Task 12／defender Task 13／正式5キャラcheckpoint、相手は既存default AI。

| 項目 | 結果 |
|---|---:|
| スコア | coach_v1 8 - 13 Ghost Champions |
| round数 | 21 |
| replay frame | 2,398 |
| plant発生round | 18 |
| defuse決着 | 3 |
| detonation決着 | 9 |
| defender全滅決着 | 6 |
| 時間切れ | 3 |
| attacker側coach一括決定 | 1,017 |
| defender側coach一括決定 | 1,120 |

例外なく完走し、全21roundでsetup／live replayを確認した。
`memory_reset_ok`、`side_checkpoint_switch_ok`、`replay_ok` はすべて `true`。
詳細なround recordは `reports/task14_full_match.json` に保存した。replayは全frameを
メモリ上で構造・可視性メタデータまで検査し、ファイル肥大化を避けるため今回の既定実行では
replay本体JSONを保存していない。CLIの `--replay` 指定で保存できる。

## 変更ファイル

- 更新: `coordinator.py`
- 新規: `team_ai.py`、`full_match.py`、`run_full_match.py`
- 新規テスト: `test_coach_v1_task14_full_match.py`
- 新規結果: `reports/task14_full_match.json`
- 新規引継ぎ: `HANDOFF-14.md`

## 残課題と次の推奨タスク

- 1 seed／既存default AIとの統合確認であり、勝率や汎化性能の評価ではない。今回の8-13を
  checkpoint退行判定には使わない。
- 本番試合はplant／defuse／detonation／全滅／時間切れをすべて通った。attacker wipeは
  このseedでは発生しなかったが、ゲーム共通終端処理を変更しておらず、既存回帰テストも成功した。
- replay本体を恒常的に保存すると容量が大きいため任意出力とした。比較評価で必要な試合だけ
  `--replay` を指定する。
- 次は **Task 15: self-playとopponent pool**。omoko_v1、touyama_v2、gc_v1、
  coach_v1過去checkpointを混ぜ、複数seedの昇格／regression基準を作る。Task 13で残った
  過剰rotationも推論規則ではなく、opponent poolと学習／validation基準で改善する。
