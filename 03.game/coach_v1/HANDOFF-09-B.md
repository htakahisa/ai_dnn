# HANDOFF-09-B: ごんごん（slot 1）

## 2026-09-27 実施結果

### 範囲と能力

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、Task 09/09-A/10 の handoff、既存の学習・観測・checkpoint・実ゲーム評価を確認した。
- ごんごんの HUNT には通常の能動的 ability 使用行動がない。使用は既存 action mask に従って常に無効とし、今回は facing のみを学習・評価した。
- 作業前から変更されていた `02.ALL_TASKS.md` と `10.task_template.md` には触れていない。core、coach、他キャラ checkpoint は変更していない。

### 学習と採否

- `training/gongon_rollout.py` で両陣営の合法な5v5共有目撃から slot 1 の観測を収集した。近距離・西・東・挟撃・自然配置を使用。敵の実座標は評価環境内の配置だけに使い、actor観測と facing 教師には渡していない。教師はそのtickの `snapshot.sightings` から45度以内の全方向を許容する。共有目撃がない5v5局面は8方向を許容して facing 損失を0にした。
- 3条件を比較した。初期の均等配分、attacker近距離を180例に増やした配分、評価側と同じ合法な近距離配置を180例加えた配分。いずれも警戒ポイント由来1400例＋5v5補助600例で、主分布70%を維持。警戒ポイント由来の敵配置は既存70/20/10 samplerを使用した。
- 学習 seed 500–539、validation 600–604、凍結5v5 holdout 700–709。正式候補は `checkpoints/experiments/task09b_gongon/fixed_near_priority/epoch80_best.pt`。旧正式版は `.../baseline_best.pt` に退避済み。
- 単一の新checkpointでは defender が改善した一方、attacker が後退した。このため `GongonPolicy()` は合法な観測の side フラグだけで旧 `best.pt`（attacker）と新 `defender_best.pt`（defender）を読み分ける。`defender_latest.pt` も同じ採用候補を保存した。明示的な `GongonPolicy(path)` は従来どおり指定された単一checkpointを使う。旧 `best.pt` は退避版とSHA-256一致、新 `defender_best.pt` は選択候補とSHA-256一致する。ゲーム側の `CharacterPolicy(1)` を直接使う場合は旧単一モデルのままなので、coordinatorへ渡すslot 1 actorには `GongonPolicy()` を使う。

### 検証結果

実ゲームは Task 10 の固定STAY/HOLD coach、相手 `default`、合法な近距離配置。指標は**強制 facing を除いた共有目撃への45度以内／合法目撃行動数**。数値は勝率や戦術的優位を示さない。旧モデルと候補で目撃機会数が変わるため分子・分母を併記する。

| seed | side | 旧正式 | 新候補 |
|---|---|---:|---:|
| 0–19 | attacker | 87/121 | 79/112 |
| 0–19 | defender | 29/282 | 246/276 |
| 独立20–39 | attacker | 72/101 | 66/109 |
| 独立20–39 | defender | 36/280 | 226/263 |

採用後の side 別ロードでは attacker は旧列、defender は新候補列を使う。seed 0–4 の実ゲームでは、`GongonPolicy()` による結果が各sideの明示的checkpoint指定と一致した。全実ゲーム比較でごんごんの HUNT 使用要求は0件。

- 凍結5v5 holdoutの合法目撃あり局面では旧モデル39.8%、採用した defender 候補65.6%。これは両sideを含む教師への許容方向率で、実ゲームattacker改善の根拠には使わない。
- 別 seed 4001 の警戒ポイント由来600例では、defender側の point が旧138/215→候補151/215、jitter が35/56→38/56、合法random が18/29→19/29。attacker は旧モデルを維持し、point 154/206、jitter 36/60、random 27/34。randomの件数は少なく、改善の確度は限定的。
- 自然配置では共有目撃がほぼ得られず、実戦での一般的な性能を測れない。今回の5v5は接敵を人為的に増やした検証であり、移動する学習済みcoachと組み合わせた勝率は未測定。

### 情報境界とテスト

- 未視認敵の実位置だけを2通りに変え、slot 1 のgrid、vector、ability target mask、facing教師集合が一致することを自動テストで確認。学習用の各サンプルは `CharacterObservation` と合法な教師ラベルだけを保持する。
- 追加テストは5v5配置別の収集、slot/能力mask、未視認敵の不変性、陣営別checkpointロード、警戒ポイント・ランダム分布を確認した。`python -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q` は **106件成功**。`python -m compileall -q coach_v1` と `git diff --check -- coach_v1` も成功。
- 推論側には戦術ルールや経路探索を追加していない。learningから他モデルのtrainファイルはimportしていない。

### 変更ファイル

- 新規: `training/gongon_rollout.py`、`experiment_task09b_gongon.py`、`experiment_task09b_gongon_near.py`、`evaluate_task09b_gongon.py`、`validate_task09b_gongon.py`、`test_coach_v1_task09b_gongon.py`、本書。
- 更新: `learning_character_gongon.py`、`evaluate_task10_rollout.py`（評価時のみの任意 side-routing フラグ）、`training/README.md`。
- 生成: `checkpoints/characters/gongon/defender_best.pt` と `defender_latest.pt`、`checkpoints/experiments/task09b_gongon/` の退避・実験 checkpoint、`reports/task09b_gongon_*.json`。旧正式checkpointと他キャラの正式checkpointは変更していない。

### 残課題・次の推奨タスク

- 実際の学習済みcoachとの連続対戦で、自然な接敵時の facing と勝率を再評価する。attacker旧モデルの西・東・挟撃の凍結観測精度は低く、単一モデル更新では実ゲーム退行があった。必要なら attacker 専用の追加学習を独立検証する。
- 次は **Task 09-C（ごんた、slot 2）**。RECON の仕様を個別に確認し、共有目撃に対する facing と能力の使用・対象・実効果を分けて評価する。
