# HANDOFF-09: 残りのキャラクターモデル

## 2026-09-24 Task 09 完了記録

### 範囲と事前確認

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、`03.DESIGN.md`、`HANDOFF-08.md`、既存の観測・知覚・学習・checkpoint実装を確認した。
- Task 09-A～Eとして固定ロスター5人を独立に学習・検証した。ゲーム接続coordinator、coach本体、coreは変更していない。
- 作業前から存在した `coach_v1/10.task_template.md` と `omoko_v1` 側の変更には触れていない。

### 実装

- ごんごん、ごんた、くんた、くりまるの `train_character_*.py` と `learning_character_*.py` を追加した。各trainは固有slotを指定して既存 `CharacterTrainer` を呼ぶ。learningは共通actor loaderを使い、trainファイルをimportしない。
- learningは観測version、固定マップ・警戒ポイントhash、形状、slotを確認してから、当該キャラクターのcheckpointだけをロードする。移動headはない。
- `training/character_curriculum.py` に固定マップの訓練専用1v1/2v1局面を追加した。`ScenarioGenerator` の敵配置を訓練ゲームに設定し、`TeamPerceptionBuilder` → `BeliefMemory` → `CharacterEnvironment.prepare` を通した安全な観測を作る。教師ラベルは合法な共有目撃、警戒ポイント、coach intent、action maskから作る。敵の実座標は局面配置後の教師ラベル計算やactor入力に渡さない。
- `train_task09.py` で5人を別seed・別checkpointへ学習し、point/jitter/random別のvalidationと全体値を `reports/task09_character_curriculum.json` に保存する。訓練120例、検証80例、16 epoch、各人128更新。訓練と検証は異なるseedを使う。
- `checkpoints/characters/{gorimaru,gongon,gonta,kunta,kurimaru}/` の `best.pt` と `latest.pt` を生成した。ごんごんのHUNTには通常active abilityがないため、使用actionは常にmaskされる。

### 検証結果

5人の検証集合はいずれも警戒ポイントと合法ランダム位置を含む。値は **best checkpointの訓練用局面に対する評価** であり、実戦性能ではない。

| character | point/random例数 | facing正解率 | ability判断正解率 | 使用ラベル再現率 |
|---|---:|---:|---:|---:|
| ごりまる | 51/9 | 17.5% | 86.3% | 68.8% |
| ごんごん | 56/7 | 52.5% | 100% | 対象外 |
| ごんた | 52/11 | 17.5% | 88.8% | 82.4% |
| くんた | 57/5 | 25.0% | 91.3% | 84.2% |
| くりまる | 59/6 | 30.0% | 92.5% | 66.7% |

- Task 00～09の `unittest` 81件成功。Task 09の新規テスト3件には、両sideとpoint/random分布、全slotのmask、ごんごんのHUNT、checkpoint取り違え拒否、未視認敵の実座標だけを変えた時の5slotの観測不変性を含む。
- `python -m compileall -q coach_v1` 成功。`git diff --check -- coach_v1` 成功。5つの既定 `best.pt` をロードして推論できた。

### 未視認敵情報の監査

- `Scenario.enemies` は局面設定にだけ使用する。学習サンプルに含むのは `CharacterObservation` と教師ラベルだけで、game、scenario、criticの参照はない。
- 敵の実位置を変え、味方がblindで合法な目撃がない2状態から作ったgrid、vector、ability target maskが全5slotで一致した。
- 共有目撃とclear履歴は既存の `TeamPerceptionBuilder` と `BeliefMemory` の契約を使用する。

### 制約と残課題

- 訓練は短い **staged supervised curriculum** であり、実ゲームの連続rolloutやability結果を収集していない。`ability_effective_rate` は未測定で `null`。特にfacing精度は低く、実戦投入可否の根拠にはならない。
- 敵の初期配置は70/20/10 samplerを使うが、観測時には1v1/2v1の訓練局面へ配置する。point/random両群で検証したが、ランダム群は5～11例と少ない。
- Task 07から継続する、ability行動payloadのfacingが既存ゲームで適用されない問題は未解決。非coreのゲーム接続で確認が必要。

### 変更ファイル

- 新規: `learning_character_base.py`、4人分の `train_character_*.py` / `learning_character_*.py`、`training/character_curriculum.py`、`train_task09.py`、`test_coach_v1_task09_characters.py`、本ファイル。
- 更新: `training/README.md`。
- 生成: `checkpoints/characters/` 配下の5人それぞれの `best.pt` / `latest.pt`、`reports/task09_character_curriculum.json`。

### 次の推奨タスク

**Task 10: チーム実行coordinatorのゲーム接続**。共通snapshotからcoachを1tickに一度だけ計算して5人に配り、characterモデルのfacing・abilityを接続する。実ゲームのrolloutでability結果とfacing適用を記録し、今回のstagedモデルの性能を再評価する。

## 2026-09-24 追加実験: epoch数だけ増加

- `experiment_task09_epochs.py` を追加し、Task 09と同じ訓練/検証seed、120/80例、network構成、学習率、batch sizeを維持して16→80 epochへ延長した。実験checkpointは `checkpoints/experiments/task09_epochs80/` に分離し、既存の5人分の正式checkpointは上書きしていない。
- `reports/task09_epoch_experiment.json` に16/40/80 epoch、validation loss最小のbest、訓練精度、point/random別精度を保存した。

| character | 元のbest facing | 80 epoch時点 facing | loss最小checkpoint facing | loss最小epoch |
|---|---:|---:|---:|---:|
| ごりまる | 17.5% | 52.5% | 45.0% | 51 |
| ごんごん | 52.5% | 51.3% | 47.5% | 25 |
| ごんた | 17.5% | 51.3% | 45.0% | 53 |
| くんた | 25.0% | 51.3% | 40.0% | 35 |
| くりまる | 30.0% | 47.5% | 51.3% | 62 |

- 4人は80 epoch時点でfacing精度が大きく改善したため、16 epoch不足は主要因の一つと考えられる。ただしごんごんは改善せず、全員でvalidation loss最小epochとfacing最高epochが一致しない。後半にlossが悪化したモデルもあり、epochだけを増やして正式bestを置き換えるのは適切ではない。
- 訓練例と検証例の差も残る。ゼロからの設計し直しが必要とは現時点で言えないが、facingを重視するcheckpoint選択、データ量・局面の増加、実ゲームでのfacing評価をTask 09の追加作業として検討する。Task 10の接続だけで精度が改善する根拠はない。

## 2026-09-24 追加実験: 訓練例とcheckpoint選択

- `experiment_task09_data_selection.py` を追加。80 epochと検証80例・モデル構成・seedを固定し、訓練例を120→360へ増やした。検証集合で選んだcheckpointを、選択に使わない別seedの160例でも評価した。
- 360例では総合loss最小とfacing精度最大のcheckpointを別々に保存・比較した。facingだけで選ぶ方法は5人で一貫して優位ではなかったため、正式checkpointの選択基準は総合loss最小のままとした。

| character | 120例・80epochのloss最小 | 360例・80epochのloss最小 | 360例・facing最大 |
|---|---:|---:|---:|
| ごりまる | 48.7% | 56.9% | 53.7% |
| ごんごん | 45.6% | 71.3% | 69.4% |
| ごんた | 44.4% | 54.4% | 55.6% |
| くんた | 35.0% | 63.7% | 61.3% |
| くりまる | 36.9% | 68.1% | 68.1% |

表は独立した160例でのfacing正解率。正式 `checkpoints/characters/` を360例・80epochの総合loss最小モデルで更新し、`train_task09.py` の既定値も一致させた。正式checkpointの検証80例でのfacingは、ごりまる60.0%、ごんごん72.5%、ごんた52.5%、くんた63.8%、くりまる61.3%。独立160例では表の中央列と一致した。

- 独立160例の合法ランダム配置群は各13～19例。正式checkpointのfacingは、ごりまる52.6%、ごんごん69.2%、ごんた35.7%、くんた50.0%、くりまる64.3%。特にごんたの警戒ポイント外への対応は追加検証が必要。
- 学習例とepochを増やすことで全員の独立集合の精度は向上した。ゼロからの再設計が必要という証拠はない。一方、これらは訓練専用の教師ラベルに対する精度であり、実ゲームのfacing/ability有効性は未測定。Task 10接続後の実戦評価を経て判断する。
- 更新後もTask 00～09の自動テスト81件、compileall、`git diff --check -- coach_v1` が成功した。元からの他領域の変更は保持した。

## 2026-09-25 Task 09-A 追補

ごりまるのみ、Task 10 の実ゲーム評価を受けて再学習し、正式checkpointを更新した。epoch延長だけでは実ゲームfacingが改善せず、合法な5v5観測の追加で改善した。詳細な比較、情報境界、残課題は [HANDOFF-09-A.md](HANDOFF-09-A.md) を参照する。

次スレッドのTask 09-B（slot 1・ごんごん）以降へ渡すfacing教師、目撃なしの扱い、実ゲーム評価、採否条件、smoke評価の限界は [HANDOFF-09-A.md](HANDOFF-09-A.md) 冒頭の「次スレッド向け引き継ぎ」に整理した。ごりまるのTask 09-Aはユーザー確認のうえ、いったん完了。smokeの戦術的な改善はcoachとの実戦評価後に再検討する。
