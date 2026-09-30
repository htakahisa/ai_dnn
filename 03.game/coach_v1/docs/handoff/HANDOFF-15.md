# HANDOFF-15: self-play と opponent pool

## 2026-09-29 対応範囲

`01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、`03.DESIGN.md`、Task 12～14 の HANDOFF、
既存 `run_game._build_team_ai`、`DualRoleTeamAI`、`run_headless_full_match`、side 別
checkpoint loader を確認して実装した。core ファイル（`abilities_los.py`、
`battle_logic.py`、`game_core.py`、`map_data.py`、`map_data_defender_setup.py`、
`run_game.py`）は変更していない。

## 実装

- `config/opponent_pool.json` に omoko_v1、touyama_v2、gc_v1、ランダム性を持つ既存簡易
  AI、Task 11.6 の過去 coach checkpoint を登録した。過去 checkpoint は attacker／defender
  とも SHA-256 を固定し、ファイルが後から差し替わった場合は評価開始前に拒否する。
- `opponent_pool.py` に schema 検証、必須 opponent 検証、難易度1～3の絞り込み、seed固定
  重み付き抽選を追加した。評価済み round point rate が低い相手ほど次回抽選weightを増やし、
  どの相手もpoolから除外しない。
- `task15_self_play.py` に既存AI／過去coachの生成、毎試合の新規controller生成、同一seed・
  開始陣営交互の全相手評価、相手別／全体集計、推奨weight、paired regression判定を追加した。
- 過去coachとのself-playでは両陣営が固定Gorigonsを使う。通常対戦のロスター重複拒否は維持し、
  self-play指定時だけ既存 `TeamPlayerKey` で同じ表示名をteam-qualified identityへ分離する。
  これによりability owner、match stats、analyticsを衝突させず、coachの固定slot名も変えない。
- 昇格基準は相手ごと最低2試合、候補とincumbentの完全に同じschedule、全体win rate非悪化、
  全体round point rate非悪化、相手別round point rate低下最大0.05、過去coach勝率0.5以上。
  一つでも失敗すれば `PromotionDecision.promoted=False` と理由を返す。自動的なproduction
  checkpoint上書きは行わない。
- `evaluate_task15.py` を追加した。pool評価JSONを保存し、`--incumbent-report` 指定時は保存済み
  incumbentをmatch recordから再集計・整合確認して昇格判定も出力する。
- `full_match.py` は self-play用の明示opt-inと、同名ロスターでも曖昧にならないTeam AI実体に
  よるscore判定だけを追加した。通常のTask 14実行契約は変更していない。

## 正式checkpointのbaseline評価

実行:

```text
python -X utf8 -m coach_v1.evaluate_task15 --seed 1500 \
  --matches-per-opponent 2 --output coach_v1/reports/task15_opponent_pool.json
```

seed 1500／1501でcoach開始陣営を attacker／defender に交互化した。全10試合で
`memory_reset_ok`、`side_checkpoint_switch_ok`、`replay_ok` はすべてtrue。

| opponent | 勝敗 | round point rate | 平均score差 | 次回推奨weight |
|---|---:|---:|---:|---:|
| omoko_v1 | 0/2 | 0.037 | -12.5 | 1.463 |
| touyama_v2 | 0/2 | 0.071 | -12.0 | 1.429 |
| gc_v1 | 0/2 | 0.133 | -11.0 | 1.367 |
| simple_random | 1/2 | 0.511 | +0.5 | 0.495 |
| coach_v1 Task 11.6 | 2/2 | 1.000 | +13.0 | 0.500 |
| **全体** | **3/10** | **0.361** | - | - |

この結果は現行正式 checkpoint の回帰baselineであり、新checkpointの昇格結果ではない。
過去版だけに偏って強い一方、3つの既存主要AIへの汎化性能は未達であるため、新モデルを
採用したとは扱っていない。正式 attacker／defender checkpointは変更していない。

## 観測境界と自動テスト

新規 `test_coach_v1_task15_self_play.py` の6件で次を確認した。

- 必須3 AIとSHA固定の過去checkpointがpoolに含まれる。
- opponent抽選がseed再現可能で、弱点weightと難易度filterを反映する。
- 評価scheduleが開始陣営を交互化し、mirrored roster許可は過去coachだけに付く。
- 過去coach opponentについて、未視認敵の実位置だけを2地点間で変えてもactorの
  coach grid/vectorが完全一致する。
- 実ゲームの短縮mirrored self-playがside swap、memory reset、replay監査を通って完走する。
- 同一scheduleの非悪化候補だけが昇格し、全体・相手別・過去coach基準の悪化を拒否する。

全テスト:

- `python -X utf8 -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q`:
  **174件成功**。
- `python -X utf8 -m compileall -q coach_v1`: 成功。
- `git diff --check`: 成功。

## 変更ファイル

- 更新: `full_match.py`
- 新規: `config/opponent_pool.json`、`opponent_pool.py`、`task15_self_play.py`、
  `evaluate_task15.py`
- 新規テスト: `test_coach_v1_task15_self_play.py`
- 新規結果: `reports/task15_opponent_pool.json`
- 新規引継ぎ: `HANDOFF-15.md`

## 残課題と次の推奨タスク

- 2試合／相手は昇格パイプラインの動作確認と初期baselineには使えるが、統計的な実力推定には
  少ない。実際の候補昇格時は同じ未使用seedで最低10試合／相手を推奨する。
- 現行モデルは過去coachには勝つが omoko_v1、touyama_v2、gc_v1 に0/6。次回の追加学習では
  保存した推奨weightでこの3相手を優先し、推論側へ戦術規則や経路探索を足さずに候補を作る。
  候補は同一scheduleで現baselineと比較し、昇格gateを通るまで正式checkpointを置換しない。
- Task順の次は **Task 16: 苦手地点の自動収集**。ただしTask 15の絶対性能を完了条件として
  求める場合は、Task 16へ進む前に上記opponent分布を用いた追加学習をTask 15継続として行う。

## 2026-09-29 追加学習

主要AIに対する絶対性能を上げるため、Task 15の継続として実戦状態を用いた安全なDAgger経路を
追加した。`training/opponent_pool_imitation.py`は現在のactorで実戦行動しつつ、既存のside別
teacherをactor観測と合法action maskだけに適用する。保持する学習sampleはgrid、vector、actor
hidden、mask、teacher labelだけであり、game object、critic用enemy truth、未視認敵の実座標は
保存しない。`train_task15_self_play.py`は正式Task 12／13 checkpointを隔離候補へコピーしてから
resumeし、正式checkpointを直接更新しない。`team_ai.build_coach_v1_team`には学習中actorを注入する
任意引数を追加したが、未指定時の本番loaderと推論挙動は変更していない。coreファイルは変更して
いない。

### round 1

seed 1600--1602、omoko_v1／touyama_v2／gc_v1を各1試合、bucketあたり64 sample、2 epochで
学習した。収集戦scoreは1-13、0-13、5-13。評価済み候補は次に保存した。

- attacker: `checkpoints/experiments/task15_pool_dagger_round1/attacker/latest.pt`
  (`76cdb9af8109e74b4a8e6a2de1e68e17198351e775083de195d056e425c7c3ab`)
- defender: `checkpoints/experiments/task15_pool_dagger_round1/defender/latest.pt`
  (`f09a3151da6195b09d1f55ea396a72e754ed9187491cddd354911aa1762c3698`)
- 学習結果: `reports/task15_pool_training_round1.json`
- 固定seed 1500--1501再評価: `reports/task15_pool_dagger_round1_reproduced_eval.json`

| opponent | baseline | round 1候補 | 判定 |
|---|---:|---:|---|
| omoko_v1 point rate | 0.037 | 0.000 | 維持範囲内だが改善なし |
| touyama_v2 point rate | 0.071 | 0.071 | 同等 |
| gc_v1 point rate | 0.133 | 0.071 | 0.062低下 |
| simple_random point rate | 0.511 | 0.523 | 微増 |
| Task 11.6 point rate | 1.000 | 1.000 | 維持 |
| 全体win rate | 3/10 | 3/10 | 同等 |
| 全体point rate | 0.361 | 0.370 | 微増 |

GCの許容低下0.05を超えたため`promoted=false`。主要3相手の勝敗も0/6のままであり、正式
checkpointへ昇格していない。

### round 2と打ち切り判断

round 1候補へ主要3相手を各2回、simple／historicalを各1回混ぜ、1 epochへ弱めて継続した。
前半4試合はomoko 2-13、touyama 4-13、gc 5-13、simple 18-20。historical同士の試合は
延長が数分以上終了しなかったため、その未完試合だけを採用せず中断した。後半を別実行すると
touyama 0-13、omoko 1-13、simple 1-13、gc 1-13となり、teacher imitation loss低下と実戦性能が
一致しないことを確認した。悪化候補は昇格・保存せず、後半結果だけ
`reports/task15_pool_training_round2b.json`に残した。これ以上同じ模倣更新を重ねることは停止した。

正式checkpointのSHA-256は作業後もattacker
`c3a9cbf17a334f5fb372cc2505c11785dd22e849ee09792ef922bc725f9a908b`、defender
`0a5dd1245970e9555a34cea6008ebaa5989c4983b3cea49a5b66d7459fc666dd`であり、baselineから不変。

### 追加テストと残課題

- 未視認敵の実位置だけを2地点間で変更しても、保持する8要素のsample／teacher labelがすべて
  同一であることをテストした。
- 候補作成が正式checkpointを変更せず、再開可能であること、bucket均衡化とactor-only更新が
  候補だけを変更することをテストした。
- Task 15専用テストは8件成功。全テスト、compileall、diff checkの最終件数は下記の最終検証へ
  追記する。
- 主要3相手への改善は未達。次は同じteacher模倣を増やさず、設計どおりround勝敗を主報酬にした
  実戦on-policy更新、またはTask 16の苦手地点収集を先に行う。historical self-playの長期延長には
  Task 15側の学習試合上限も必要だが、通常match/coreの終了規則は今回変更していない。

### 追加学習後の最終検証

- `python -X utf8 -m unittest coach_v1.test_coach_v1_task15_self_play -v`: **8件成功**。
- `python -X utf8 -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q`:
  **176件成功**。
- `python -X utf8 -m compileall -q coach_v1`: 成功。
- `git diff --check`: 成功（改行コード予告warningのみ）。

追加学習で増えた変更は、更新`team_ai.py`、新規`train_task15_self_play.py`、
`training/opponent_pool_imitation.py`、上記2件を含むTask 15テスト、round 1候補checkpoint 4件、
学習／評価report 3件、および本節である。既存core、正式checkpoint、他モデルのtrain／learning対応は
変更していない。

## 2026-09-30 中断後の確認とround勝敗学習

上記のDAgger候補は昇格しなかった。その後、`training/opponent_pool_policy_gradient.py`と
`train_task15_policy_gradient.py`で、実戦のラウンド勝敗を報酬とするactor更新を追加した。
移動は学習時だけ確率的に選択し、facingとアビリティは従来のキャラクターモデルが担当する。
学習sampleはactorのgrid、vector、hidden、合法移動mask、選択した移動だけで、敵実位置や
critic truthを含めない。試合終了後、各ラウンドの勝敗をチーム名と陣営から対応付け、
ラウンド長で正規化した重みで移動actorを更新する。推論時の戦術規則や経路探索、coreファイルは
変更していない。

正式Task 12／13 checkpointを別ディレクトリへコピーしてから、seed 1700--1702で
omoko_v1、touyama_v2、gc_v1を各1試合学習した第1候補は、固定評価でGCの得点率を
0.133から0.297へ上げたが、touyama_v2とsimple_randomが許容幅を超えて悪化した。
その候補を`checkpoints/experiments/task15_pool_round_reward_round1/`に保存し、
弱化したtouyama_v2とsimple_randomをseed 1710--1711で各1試合追加学習した。
学習結果は`reports/task15_pool_round_reward_training.json`と
`reports/task15_pool_round_reward_training_round2.json`、両候補の固定評価は
`reports/task15_pool_round_reward_eval.json`と
`reports/task15_pool_round_reward_round2_eval.json`に保存した。

| 相手 | 旧正式版の得点率・勝利数 | 第2候補の得点率・勝利数 |
|---|---:|---:|
| omoko_v1 | 0.037、0/2 | 0.103、0/2 |
| touyama_v2 | 0.071、0/2 | 0.037、0/2 |
| gc_v1 | 0.133、0/2 | 0.161、0/2 |
| simple_random | 0.511、1/2 | 0.591、2/2 |
| 過去coach Task 11.6 | 1.000、2/2 | 1.000、2/2 |
| 全体 | 0.361、3/10 | 0.389、4/10 |

評価は両方ともseed 1500--1501、相手ごとに攻守開始を交互にした同一10試合で行った。
第2候補は全体勝率と得点率が向上し、相手別低下幅も許容0.05以内だったため、
`promotion.promoted=true`になった。昇格前の正式checkpointは
`checkpoints/experiments/task15_pre_pool_incumbent/`へactorとtraining stateを対で退避した。
昇格後の正式ファイルはattacker `checkpoints/coach/attacker/task12/latest.pt`と
`training_latest.pt`、defender `checkpoints/coach/defender/task13/latest.pt`、`best.pt`、
`training_latest.pt`、`training_best.pt`。actor SHA-256はattacker
`b4c6a609b8364cec348fdea3587e446a8ae943de41099175bc0481bbd7a737de`、defender
`4a3d216266d2b588f1e0f67165665df6d51a543452c8efa9dedb7296974ea792`で、評価レポートの
候補SHAと一致する。退避版SHAは旧baselineレポートと一致する。正式loaderと両sideの
`CoachTrainer.resume()`は成功し、actor／training stateのstepもattacker 1841、defender 1995で
一致した。

### 2026-09-30 最終検証と残課題

- `python -X utf8 -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q`:
  **178件成功**。Task 15専用は10件成功。未視認敵の実位置を変えても、固定乱数下で
  round報酬学習sampleの全5要素が同一であるテストを含む。
- `python -X utf8 -m compileall -q coach_v1`、`git diff --check`: 成功。
- 主要3相手との勝敗は依然0/6。今回の2試合／相手は昇格ゲートとしては有効だが、
  十分な実力推定には少ない。次は未使用seedで相手ごとに試合数を増やし、特にtouyama_v2で
  失う局面をTask 16の苦手地点収集へ渡すことを推奨する。
- historical同士の学習試合が長期延長になる場合がある。Task 15の学習運用側に試合上限を
  設ける余地があるが、今回coreの終了規則は変更していない。

上にある「正式checkpointは不変」「主要3相手への改善は未達」はDAgger実験終了時点の記録であり、
この節のround勝敗学習と正式昇格が最新状態である。

### 未使用seedでの追加確認

候補選択に使ったseed 1500--1501とは別に、seed 1510--1511で退避版と昇格版を各10試合
評価した。保存先は`reports/task15_holdout_incumbent_1510.json`と
`reports/task15_holdout_promoted_1510.json`。同じ相手順・開始陣営で比較すると、全体勝利数は
2/10から3/10、得点率は0.305から0.346へ上がった。omoko_v1は両者0.037で同等、
touyama_v2は0から0.037、gc_v1は0.037から0.133、simple_randomは0.409から0.469、
過去coachは0.963から1.000。未使用seedでも`promotion.promoted=true`で、相手別の
許容低下幅を超える退行はなかった。主要3相手への勝利はこのseedでも0/6のため、
「勝てる状態」まで達したとは判断しない。さらに多くの未使用seedによる検証と、
苦手局面を狙った学習が必要である。
