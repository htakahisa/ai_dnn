# HANDOFF-13: defender coach の学習

## 2026-09-29 実装結果

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、`03.DESIGN.md`、`HANDOFF-12.md` と既存の trainer／観測／coordinator を確認してから実装した。
- defender 専用の8段階カリキュラム（初期配置、警戒地点保持、敵目撃への反応、ローテーション、複数人集合、アビリティretake、defuse担当と護衛、1ラウンド全体）を実ゲーム環境へ追加した。初期配置は既存 defender setup を20 tick実行し、残りの段階も実際の移動・戦闘・plant／defuse遷移を使用する。
- 敵配置は既存 `ScenarioGenerator` の70%警戒ポイント、20%周辺、10%合法ランダムを維持した。目撃段階では生成済みの敵を合法な視線で現在視認させ、ability-retakeでは実際のRECONを事前実行する。
- 学習専用teacherは `CoachObservation` と action mask のみからラベルを作る。固定マップ上の最短距離計算はteacherラベル生成時だけに使用し、actor、coordinator、推論入口には経路探索や戦術規則を追加していない。
- 段階ごとの一括模倣では後半段階が前半の移動方策を忘れるため、8段階を1 episodeずつ巡回する学習順にした。不採用の蒸留・retake限定実験分岐は正式実装へ残していない。
- defender専用のtrain／learning対応を追加し、attacker側や他モデルのtrainファイルはimportしていない。checkpointは attacker と分離した。
- Task 13の全評価項目を同一seedの学習済み／合法ランダム／actor公開情報だけの簡易規則で集計する評価器を追加した。
- coreファイルは変更していない。

## checkpoint と学習条件

- actor: `checkpoints/coach/defender/task13/latest.pt`
- 学習再開用: `checkpoints/coach/defender/task13/training_latest.pt`
- SHA-256: `3bd06216926910ee67150449ff1e053abb0de5256e12e9b3a98dad94f429724b`
- schema: `coach-checkpoint-v1`、観測: `coach-observation-v2`、side: `defender`
- episode 500、training step 1657、seed 11。actor checkpointにcritic stateはなく、optimizer stateも `None`。critic・optimizer・履歴は学習再開用checkpointだけに保持する。
- Task 11.6 defender observation-v2 checkpoint（episode 200）から、初期配置／保持／目撃／rotationを各20、group-up／ability-retake／full-roundを各60、defuse-escortを40 episode、段階巡回方式で模倣学習した。

## 未使用seed 600–609の評価

詳細は `reports/task13_final_holdout10.json`。値は学習済み / ランダム / 簡易規則。

| 段階 | 平均報酬 | plant阻止率 | retake成功率 | 集合率 | 無効移動率 |
|---|---:|---:|---:|---:|---:|
| hold-watch | 0.211 / 0.180 / 0.231 | 0.10 / 0 / 0 | 0 / 0 / 0.25 | 0 / 0.186 / 0 | 0 / 0.004 / 0.020 |
| sighting-response | 0.353 / 0.070 / 0.245 | 0.20 / 0 / 0 | 0.333 / 0 / 0.50 | 0.102 / 0.113 / 0.274 | 0.041 / 0.013 / 0.030 |
| rotate | 1.015 / 0.605 / 0.901 | 0.70 / 0.50 / 0.50 | 0.667 / 0 / 0.60 | 0.325 / 0 / 0.146 | 0.030 / 0.009 / 0.028 |
| group-up | 0.308 / 0.104 / 0.738 | 0 / 0 / 0 | 0.20 / 0 / 0.60 | 0.159 / 0.029 / 0.119 | 0.013 / 0.006 / 0 |
| ability-retake | -0.521 / -0.990 / 0.525 | 0 / 0 / 0 | 0.20 / 0 / 0.70 | 0.270 / 0.309 / 0.533 | 0.013 / 0.008 / 0.003 |
| defuse-escort | 1.117 / -0.799 / 1.109 | 0 / 0 / 0 | 1.00 / 0.10 / 1.00 | 1.00 / 0.981 / 1.00 | 0.013 / 0.024 / 0 |
| full-round | -0.677 / -0.532 / -0.270 | 0.10 / 0.20 / 0 | 0 / 0 / 0.30 | 0.180 / 0 / 0.169 | 0 / 0.033 / 0.015 |

- defuse中の平均護衛人数は defuse-escort で3.817人、full-roundではdefuse到達がなく0人。
- full-roundの単独retake率は0.166、集合率は0.180、過剰rotation率は0.024、未確認エリア対応率は0.110、無効移動率は0。
- 初期配置のwatch coverageは学習済み0、ランダム1.88、簡易規則0。初期配置を含む全段階を評価したが、この指標は未達である。

## 観測境界とテスト

- 未視認敵の実位置だけを2地点へ変更するテストで、actorのgrid/vector、teacher行動、評価用簡易規則が完全一致し、`critic_enemy_truth` だけが変化することを確認した。
- enemy truthの直接参照は学習環境のシナリオ配置、学習専用critic入力、評価指標に限定される。teacher、模倣学習入力、learning入口は公開観測だけを使う。
- 味方の合法視認共有、tick付きclear情報、tickごとの5人一括計算は既存 observation/coordinator 契約をそのまま使用した。
- Task 13専用テスト: 6件成功。
- `python -X utf8 -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q`: 162件成功。
- `python -X utf8 -m compileall -q coach_v1`、`git diff --check`: 成功。

## 変更ファイル

- `training/coach_environment.py`
- `training/defender_teacher.py`
- `training/defender_imitation.py`
- `train_coach_defender.py`
- `learning_coach_defender.py`
- `evaluate_task13.py`
- `test_coach_v1_task13_defender.py`
- `checkpoints/coach/defender/task13/latest.pt`
- `checkpoints/coach/defender/task13/training_latest.pt`
- `reports/task13_final_holdout10.json`
- `HANDOFF-13.md`

## 残課題と次の推奨タスク

- 局所のrotate、ability-retake、defuse-escortではランダムを上回り、無効行動も抑制できたが、full-roundではランダムと簡易規則を上回っていない。特にfull-round retake 0/10、初期配置watch coverage 0はTask 13の品質上の残課題であり、完成度を過大評価しないこと。
- 追加改善する場合は、推論側へ規則を足さず、段階巡回を維持したまま初期配置と「plant後に遠距離から集合してdefuseへ遷移する」軌跡の比率、validationによるearly stoppingを見直す。今回確認した単一段階の連続追加学習は壊滅的忘却を起こすため避ける。
- 次の推奨タスクは **Task 14: フルマッチ統合**。統合時はこの残課題を既知のbaselineとして扱い、side切替、ラウンドreset、checkpoint切替、未視認情報境界を先に検証する。Task 14へ進む前に品質閾値を要求する場合は、上記2点をTask 13の追加学習として先に改善する。
