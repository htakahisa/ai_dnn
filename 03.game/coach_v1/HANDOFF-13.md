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

## 2026-09-29 継続作業: full-round集約とplant前後分割の検証

### 実施内容

- 関連設計、Task 12/13引き継ぎ、defender trainer／teacher／imitation／評価器、観測encoder、既存checkpoint履歴を再確認した。coreファイルは変更していない。
- 現行正式版が教師軌跡上ではdefuseできる一方、自身の方策で到達したplant後状態では止まる分布ずれを確認し、既存のactor-safe on-policy full-round集約を10/30 episodeで比較した。
- `evaluate_task13.py` に評価専用 `PhaseSplitDefenderPolicy` と `--postplant-checkpoint` を追加した。切替条件はactor公開観測の `spike_planted` だけであり、敵実位置、critic truth、経路探索、戦術規則は参照しない。2つのrecurrent stateはラウンド開始時に両方resetする。
- plant後局面だけを20 episodeずつ追加した専用候補、および現行plant前＋各plant後候補の組合せを実ゲームで評価した。
- phase routerの公開情報だけによる切替と両policy resetを自動テストへ追加した。

### 単一モデルの追加学習結果

seed 700--719は候補選定、seed 800--819は最終holdout。値は現行正式版／30 episode集約候補。

| holdout | 平均報酬 | plant阻止 | retake成功 | 集合率 | 過剰rotation | 無効移動 |
|---|---:|---:|---:|---:|---:|---:|
| 700--719 | -0.852 / -0.264 | 0/20 / 2/20 | 0/20 / 3/18 | 0.097 / 0.228 | 0.023 / 0.585 | 0.005 / 0.034 |
| 800--819 | -0.684 / -0.570 | 1/20 / 0/20 | 0/19 / 3/20 | 0.139 / 0.190 | 0.023 / 0.610 | 0.003 / 0.034 |

- 集約候補はfull-round retakeと平均報酬を改善したが、plant前の阻止、過剰rotation、無効移動を悪化させたため正式昇格していない。
- 集約を10 episodeに減らした候補はseed 700--719でretake 0、過剰rotation 0.756。30 episode集約後に全8段階を10 episodeずつ再巡回した候補もretake 0、無効移動0.062となり、どちらも不採用。
- 30 episode候補の別seed全8段階5 episode評価は、ability-retake 3/5、defuse-escort 4/5、full-round 1/5。詳細は `reports/task13_aggregation30_all_stages5.json`。

### plant前／plant後の2モデル案

- plant後局面だけを追加学習したcheckpointとの分割はseed 700--719で平均報酬-0.906、retake 0、集合率0となり不採用。局所教師状態とfull-roundから到達する状態の分布差が大きい。
- 現行正式版をplant前、30 episode集約候補をplant後にした組合せはseed 800--819で、plant阻止1/20、retake 2/19、平均報酬-0.681。現行よりretakeは改善したが、過剰rotation 0.457、無効移動0.089へ悪化したためproduction採用していない。
- メリットは前半の安定した方策を保持しながら後半だけ更新できること、モデル容量と最適化目標を分離できること。デメリットはcheckpoint・学習・昇格判定が倍増すること、plant時のGRU state切替、両モデル間の到達状態分布ずれ、Task 14でのロード／reset／side切替が複雑になること。
- 現時点では2モデル化の利益が副作用を上回らず、`03.DESIGN.md` の「sideごとに1モデル」を維持した。正式checkpoint `3bd062...` も置換していない。

### 観測境界

- phase routerは `CoachObservation.vector` の公開 `spike_planted` だけを読む。`game.chars`、敵座標、`critic_enemy_truth` を参照しない。
- 既存の未視認敵2地点差し替えテストは引き続き、actor grid/vector、teacher、ruleが同一でcritic truthだけが変化することを確認する。
- actor checkpointにはcritic／optimizer／敵truthを保存しない既存契約を変更していない。

### 追加・更新ファイル

- `evaluate_task13.py`
- `test_coach_v1_task13_defender.py`
- `checkpoints/experiments/task13_aggregation10/`
- `checkpoints/experiments/task13_aggregation30/`
- `checkpoints/experiments/task13_aggregation30_rehearsal10/`
- `checkpoints/experiments/task13_postplant20/`
- `reports/task13_baseline_full_round_20.json`
- `reports/task13_aggregation10_full_round_20.json`
- `reports/task13_aggregation30_full_round_20.json`
- `reports/task13_aggregation30_rehearsal10_full_round_20.json`
- `reports/task13_phase_split_postplant20_full_round_20.json`
- `reports/task13_baseline_final_holdout20.json`
- `reports/task13_aggregation30_final_holdout20.json`
- `reports/task13_phase_split_aggregation30_final_holdout20.json`
- `reports/task13_aggregation30_all_stages5.json`
- `HANDOFF-13.md`

### テストと残課題

- `python -X utf8 -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q`: **163件成功**。
- `python -X utf8 -m compileall -q coach_v1`、`git diff --check`: 成功。
- full-roundでは現行正式版も候補も簡易rule（seed 800--819でretake 9/20）を下回る。次のTask 13改善では、推論分割よりも、単一モデルのままon-policy状態をvalidation付きで収集し、plant前の低rotation／低衝突サンプルとplant後の遠距離retakeサンプルを同一mini-batchで均衡学習する必要がある。
- 品質改善をTask 13内で続けるなら、次はepisode末尾checkpointだけでなく固定validation seedごとのbest保存を実装する。Task 13をここで区切る場合の次タスクは引き続き **Task 14: フルマッチ統合** だが、full-round性能は未達baselineとして扱う。

## 2026-09-29 継続作業2: 単一モデル均衡学習と正式昇格

### 実装

- `training/defender_imitation.py` を整理し、従来の段階別模倣と同じactor-safe収集・最適化処理を共通化した。
- `fit_balanced_imitation` を追加した。1サイクルで指定した全段階を収集し、full-roundは公開観測の `spike_planted` だけでpreplant/postplantバケットへ分ける。各バケットから同数を復元抽出し、1つのshuffle済みmini-batch集合として学習する。
- full-roundだけは現在actorのgreedy行動で到達した状態へteacherラベルを付ける。初期配置、保持、目撃、rotate、group-up、ability-retake、defuse-escortはteacher軌跡を使い、希少なsetup／DEFUSEラベルを保持する。
- `validate_defender_actor` を追加した。固定seedのfull-roundをactorだけで評価し、平均勝敗を主軸にretake／plant阻止を加点、過剰rotation／無効移動を減点するscoreを保存する。各サイクルのbest actorは `best.pt`、対応する再開状態は `training_best.pt` へ分離保存する。
- `train_coach_defender.py` に `--balanced-cycles`、`--balanced-samples`、`--validation-seed`、`--validation-episodes` を追加した。既存のPPO、段階別模倣、full-round集約の累積targetと再開契約は維持した。
- 自動テストで、8段階混合、バケット同数抽出、固定validation seed、best／training_best保存、actor checkpointへのcritic・optimizer非混入を検証した。
- core、推論actor、coordinator、観測仕様、checkpoint schemaは変更していない。

### 学習とbest選定

- 現行正式版開始と、30 episode on-policy集約版開始を同一条件で4 balanced cycle比較した。各cycleは各バケット24 sample、validation seed 1000--1003。
- 正式版開始系は4 cycle内でretakeを得られなかった。集約版開始系は第1cycleでvalidation平均勝敗0.5、retake成功率0.5、selection score 0.598を記録し、その後3 cycleはいずれも悪化した。best保存により第1cycleを保持した。
- 採用モデルはepisode 538、training step 1990、seed 11。SHA-256は `0a5dd1245970e9555a34cea6008ebaa5989c4983b3cea49a5b66d7459fc666dd`。
- 正式保存先は `checkpoints/coach/defender/task13/latest.pt` と `best.pt`。再開状態は `training_latest.pt` と `training_best.pt`。actorファイルにcritic／optimizerは含まない。

### 未使用holdout

seed 1100--1119、full-round 20 episode。値は旧正式版／新best／ランダム／簡易rule。

| 指標 | 旧正式版 | 新best | ランダム | 簡易rule |
|---|---:|---:|---:|---:|
| 平均報酬 | -0.853 | **0.075** | -0.872 | 0.421 |
| 平均勝敗 | -0.95 | **-0.20** | -0.95 | 0.30 |
| plant阻止率 | 0 | **0.05** | 0 | 0 |
| retake成功率 | 0 | **0.316 (6/19)** | 0 | 0.60 |
| 集合率 | 0.169 | **0.279** | 0 | 0.237 |
| defuse中護衛人数 | 0 | **1.273** | 0 | 1.618 |
| 過剰rotation率 | **0.036** | 0.512 | 0.415 | 0.167 |
| 無効移動率 | **0.006** | 0.017 | 0.035 | 0.011 |

- 新bestは旧正式版とランダムを勝敗・報酬・retakeで明確に上回った。簡易ruleにはまだ届かない。
- seed 1200--1209の全8段階評価では、初期配置watch coverage 7.6（ランダム2.78）、group-up retake 5/10、ability-retake 4/10、defuse-escort 10/10、full-round plant阻止1/10・retake 3/9・平均報酬0.190。full-roundの簡易rule平均報酬-0.095を上回った。
- 詳細は `reports/task13_balanced_best_holdout20.json`、`reports/task13_balanced_baseline_holdout20.json`、`reports/task13_balanced_best_all_stages10.json`。

### 情報境界と残課題

- teacherラベル、バケット分類、validation actorはいずれも `CoachObservation` とaction maskだけを使う。バケット分類は公開 `spike_planted` のみで、未視認敵座標やcritic truthを参照しない。
- 未視認敵2地点差し替え時にactor grid/vectorとteacherが同一でcritic truthだけが変わる既存テストを維持した。
- 最大の残課題は過剰rotation率0.512。勝敗改善と引き換えに5人同時移動が増えている。推論規則で抑制せず、Task 15のopponent poolまたは追加学習時に、勝敗を維持したまま低rotation validation制約を強める。
- Task 13の8段階実装、学習、best選定、正式checkpoint昇格は完了。次の推奨タスクは **Task 14: フルマッチ統合**。

### 最終検証と追加ファイル

- `python -X utf8 -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q`: **164件成功**。
- 正式 `latest.pt` を `learning_coach_defender` でloadし、`training_latest.pt` からepisode 538／step 1990へresumeできることを確認した。
- 正式actorのSHA、actor／training metadataのstep一致、`latest.pt == best.pt`、actorへのcritic／optimizer非混入を確認した。
- `python -X utf8 -m compileall -q coach_v1`、`git diff --check`: 成功。
- 追加・更新: `training/defender_imitation.py`、`train_coach_defender.py`、`test_coach_v1_task13_defender.py`、正式checkpoint 4ファイル、`checkpoints/experiments/task13_balanced_official/`、`checkpoints/experiments/task13_balanced_aggregation30/`、上記3評価JSON、`HANDOFF-13.md`。
