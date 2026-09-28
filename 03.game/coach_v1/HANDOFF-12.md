# HANDOFF-12: attacker coach の学習

## 2026-09-28 実装範囲

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、`03.DESIGN.md`、Task 11/11.5/11.6 の引き継ぎ、既存 trainer・環境・観測・coordinator を確認してから作業した。作業開始時の `git status` は空だった。
- attacker 専用の9段階（集合と進行、侵入、ability 後の侵入、複数方向、護衛、回収、plant、post-plant、1ラウンド全体）を `training/coach_environment.py` に追加した。全段階で既存の実ゲーム、凍結済み character checkpoint、警戒ポイント中心の `ScenarioGenerator` を使用する。utility 付き侵入では学習局面の直前に合法な味方 SMOKE を使用して公開 smoke と消費済み charge を作る。回収・plant・post-plant は公開 spike 状態を生成する。
- 1ラウンド開始局面ではゲームの LIVE tick は0のまま、敵配置の到達可能性だけ defender setup の20 tick を反映した。経過 tick 0 を配置器に渡すと警戒ポイントが到達不能となり、指定の70/20/10分布から外れるため。固定 seed 727 の500敵配置は point 718/1000、周辺183/1000、合法ランダム99/1000（200局面）だった。
- `train_coach_attacker.py` と対応する `learning_coach_attacker.py` を追加した。Task 12 checkpoint は `checkpoints/coach/attacker/task12/` に分離し、既存 v1/v2 attacker および defender checkpoint を変更しない。学習の再開、段階別累積 episode 指定、rollout 保存指定を備える。learning は別モデルや train ファイルを import しない。
- 学習専用の `training/attacker_teacher.py` と `training/attacker_imitation.py` を追加した。固定マップの距離計算は**教師ラベルの作成時だけ**使い、推論 actor と coordinator には経路探索・戦術分岐を追加していない。教師も公開 actor 観測と合法 action mask だけを入力とする。
- `evaluate_task12.py` で同一 seed の学習済み・合法ランダム・簡易ルールを比較する。plant、勝利、単独侵入、味方衝突、trade 距離、同じ敵への合法な二方向射線、ability 後の侵入、無効移動、spike 回収を測る。二方向射線の敵実位置は**学習・評価用指標だけ**に使う。
- 1ラウンドの長い rollout で、先行キャラクターの処理後に後続味方が手番前に死亡するケースを修正した。実際に行動した slot だけをログ件数・移動命令数に含める。
- core ファイルは変更していない。

## checkpoint と評価

- 正式 actor: `checkpoints/coach/attacker/task12/latest.pt`。SHA-256 `c3a9cbf17a334f5fb372cc2505c11785dd22e849ee09792ef922bc725f9a908b`、観測 `coach-observation-v2`、training step 1836、episode 428。critic・optimizer・敵実位置は actor ファイルに含まれず、学習状態は `training_latest.pt` に分離した。
- 学習経路: Task 11.6 の attacker v2 checkpoint（190 episode）から段階別 PPO 128 episode → 1ラウンド教師軌跡50 episode → モデル軌跡への教師ラベル30 episode → defender setup 20 tick を反映した1ラウンドモデル軌跡30 episode。正式 checkpoint は最後の分布に対する追加学習版。`checkpoints/experiments/task12_*` は比較用の非採用 checkpoint と、直前版の保存先。
- 未使用 seed 500～519、1ラウンド全体、既存 default defender AI 相手の同条件比較。学習済み **14/20勝・15/20 plant**、ランダム **0/20勝・0/20 plant**、簡易ルール **3/20勝・3/20 plant**。平均報酬はそれぞれ **0.738 / -0.999 / -0.449**。詳細は `reports/task12_setup20_candidate_full_round_20.json`。比較前の checkpoint も同じ20 seed で14勝・14 plantだったため、新分布版を採用した。
- 採用判断に使用していない seed 600～609 の追加確認では、学習済み **8/10勝・8/10 plant**、ランダム **0/10勝・0/10 plant**、簡易ルール **1/10勝・3/10 plant**。`reports/task12_final_holdout_full_round_10.json`。
- 採用版の同じ10 seed で、utility 付き侵入は **8/10 plant**、ability 後の侵入率 **0.96**。回収は **7/10でspike回収、7/10 plant**。回収の簡易ルールは9/10で、局所的にはまだ下回る。`reports/task12_setup20_candidate_utility_retrieve_10.json`。
- 全9段階の同条件評価（seed 100～103）は `reports/task12_final_all_stages_setup20.json`。学習済みの plant は rally 3/4、entry 2/4、utility-entry 2/4、multi-peek 3/4、escort 1/4、retrieve 3/4、plant 4/4。post-plant は4/4勝、full-round は2/4勝。4件という小標本のため、採否判断には上記20件・別10件を用いた。採用前の分布や checkpoint の評価は `reports/task12_pre_*` とその他 `task12_*` に残した。最終評価の SHA と seed は各 JSON の `sha256` と `seeds` を確認する。

## 観測境界・テスト

- `test_coach_v1_task12_attacker.py` を追加した。未視認敵の実位置だけを変えた full-round・utility-entry・post-plant 状態では actor grid/vector が完全一致し、critic truth だけが変わることを検証した。学習教師の指示も同一である。実際の Task 12 actor checkpoint を `learning_coach_attacker` で読み、観測 v2 と side を検証した。
- 9段階の実ゲーム実行、公開目的状態、utility SMOKE、合法 PLANT、setup 時間を含む敵配置比率、手番前死亡、actor/critic checkpoint 分離、教師あり・モデル軌跡学習、合法な二方向射線を自動テストに含めた。
- `python -X utf8 -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q`: **156件成功**。`python -X utf8 -m compileall -q coach_v1` と `git diff --check` も成功。

## 残課題と次の推奨タスク

- spike 回収単独局面は簡易ルール9/10に対し学習済み7/10。1ラウンドの単独サイト侵入率は0.146で、比較前 checkpoint の0より高い。局所局面の改善時にはこの二指標を監視する。
- 対戦相手は既存 default defender AI。複数の既存 AI・過去 checkpoint を混ぜる頑健性検証は Task 15 の opponent pool で行う。今回の20件は固定マップ・固定ロスターに限る。
- 次は **Task 13: defender coach の学習**。同じ観測境界と side 別 checkpoint を維持する。
