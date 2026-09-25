# HANDOFF-09-A: ごりまる facing 再学習

## 次スレッド向け引き継ぎ: Task 09-B 以降の進め方（2026-09-25）

### 現在の結論と対象

- **Task 09-A／ごりまる（slot 0）はユーザー確認のうえ、いったん完了。** 正式checkpointは `checkpoints/characters/gorimaru/best.pt` と `latest.pt`。採用元は `checkpoints/experiments/task09a_gorimaru_attacker_adjustment/near_priority/epoch90_latest.pt`。本書の古い節にある「現行」「次にsmokeを評価する」は当時の状態であり、この節と上のsmoke評価が最新。
- 次の **Task 09-B は固定ロスターの slot 1「ごんごん」**。能力は `HUNT` で能動的な使用・対象選択がない。ごりまる固有のsmoke教師・smoke評価をコピーせず、能力仕様を確認してからfacingを個別に検証する。Task 09-C以降も各キャラの能力に応じて同じ判断を行う。
- smokeの仮教師は「合法に目撃した敵の報告位置に最も近い使用可能セル」であり、5v5追加学習では旧モデルの使用・対象を模倣した。これは戦術的な正解ではない。ごりまるのsmokeは動作成功と明確な後退がないことまで確認した。**良いsmokeかどうかのブラッシュアップは、coachとの実戦統合で効果を集めた後に、キャラクターモデルへ学習結果を戻す段階へ延期する**。coachのみを学習しても固定されたキャラクターモデルの対象判断そのものは更新されない。

### facingで引き継ぐ方法

1. まず対象キャラの既存checkpoint、能力、Task 09の独立検証、Task 10の実ゲーム結果を確認する。epochを増やすだけで頭打ちかは決めつけない。ごりまるでは同じ360例を160/320 epochまで延長しても実ゲームfacingは改善せず、5v5の合法な共有目撃局面を加えて改善した。
2. 教師・評価に使う敵情報は `TeamPerceptionSnapshot.sightings` の**現在の合法なチーム共有報告**に限定する。未視認敵の実位置は局面生成や評価側だけで扱い、actor入力にもfacingラベルにも使わない。共通処理は `training/facing_labels.py` の `acceptable_facings_from_snapshot(snapshot, slot=対象slot)` を参照する。複数の敵報告が同時にある場合、いずれかに45度以内の方向をすべて正解として扱い、単一セル・単一方向への完全一致を主指標にしない。
3. 5v5補助データで現在の合法な目撃がない時に「直前の自分の向き」を唯一の正解とすると、目撃時の学習を妨げる可能性があった。ごりまる採用実験では、その場合だけ8方向すべてを許容してfacing損失を0とし、能力の教師は維持した。既存の警戒ポイント由来の局面では推奨方向の教師を残した。`training/gorimaru_rollout.py` の `mask_unsighted_facing=True` は**ごりまる固有の実装例**であり、次キャラでは自身のデータ収集経路に適用するか検証して決める。
4. 警戒ポイントを主要分布とし、追加の5v5局面は近距離・西・東・挟撃・自然配置を両陣営で集める。ごりまる採用実験は警戒ポイント由来1400例＋5v5補助約600例で、警戒ポイント由来を約70%維持した。これは次キャラの固定ハイパーパラメータではない。訓練・検証・holdoutのseedを分け、各side／接敵方向／警戒ポイント外の不足例を診断して配分を決める。
5. 採否は凍結観測の許容facing率だけで決めない。`evaluate_task10_rollout.py` の **`unforced_aligned_45 / unforced_sighting_actions`** をslot別・attacker/defender別・同じseedブロックで比較し、強制facingは分ける。旧モデルと候補で目撃機会数が変わるため分子・分母も併記する。目撃が0件ならその条件のfacingは評価不能。ごりまるでは凍結holdout改善候補がattacker実ゲームで退行したため却下し、近距離attacker例を増やした別候補を採用した。独立holdoutと選定後の新seedで再確認する。
6. 能力使用・対象はfacingと別の評価軸にする。使用要求／実行成功、誤使用、合法対象、対象一致と実際の効果を混同しない。ごりまるの仮のsmoke対象一致率や射線遮断数は戦術的な正解率ではない。他キャラも能力ごとの効果指標を必要に応じて追加する。正式checkpointへ反映する前に旧checkpointを保存し、そのキャラだけを更新する。

### 再現・境界・次の作業

- 参照実装: `training/character_curriculum.py`（70/20/10と合法目撃からの教師）、`training/character_trainer.py`（複数許容方向の損失）、`training/facing_labels.py`、`training/gorimaru_rollout.py`、`experiment_task09a_gorimaru_attacker_adjustment.py`、`evaluate_task09a_gorimaru_attacker_adjustment.py`。実験スクリプト内の `CURRENT` は**採用前のごりまるcheckpoint**を指すため、次キャラにそのまま使用しない。新キャラ用のtrain/learning対応を保ち、別モデルのtrainファイルをlearningからimportしない。
- ごりまる採用時の近距離STAY/HOLD・強制facing除外・seed 0–59合計は、attacker **180/294（61.2%）→203/289（70.2%）**、defender **234/275（85.1%）→189/197（95.9%）**。採用後のsmoke評価はstaged 80/80設置、同一状態で旧モデルと対象が違ったのは4/80。詳細は後続節と `reports/task09a_gorimaru_*.json`。評価再現時は `PYTHONHASHSEED=0` を指定する。
- 次スレッドでは `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、`HANDOFF-09.md`、本書、`HANDOFF-10.md` を読んでから、**ごんごん固有のTask 09-B** を開始する。coreや他4人のcheckpointは明確な必要性なしに変更しない。ごりまる固有スクリプトを共通化する場合も、新キャラの観測・能力・学習境界をテストする。

## 2026-09-25 追加: smoke 実ゲーム評価と Task 09-A 完了判定

- `evaluate_task09a_gorimaru_smoke.py` で固定マップ・STAY/HOLD coach・相手 default・20 tick・seed 60–69 を両陣営の「近距離／西／東／挟撃／自然配置」に適用した。旧正式モデル (`task09a_gorimaru_rollout/epoch70_latest.pt`) と採用モデル (`task09a_gorimaru_attacker_adjustment/near_priority/epoch90_latest.pt`) を各100ラウンド評価した。再現時は `PYTHONHASHSEED=0` を設定する。
- staged 配置80ラウンドでは、両モデルとも smoke 要求80/80、実際の設置80/80。自然配置20ラウンドでは両モデルとも設置0回。採用モデルの設置直後に味方が煙内にいた人数は合計0、敵は合計159。旧モデルはそれぞれ30、166。対象セルの合法性と残りtick、能力チャージ消費を実ゲームで確認した。
- 各tickの実ゲーム射線判定で、ごりまるの煙だけを一時的に外し、同じ生存者・位置・向きで可能だった射線を測定した。採用モデルは敵の射線629/3176組・tick、味方の射線408/3534組・tickを遮断。旧モデルの別試合では敵509/2900、味方267/3106。採用モデルは敵への遮断も味方への遮断も増えており、別試合の総数だけで smoke 対象の優劣は決められない。
- `--shadow-prior` で採用モデルが実際に受けた **同じ合法観測** を旧モデルにも渡し、旧モデルの対象セルだけを同じ試合状態で仮測定した。80回とも旧モデルも使用を選択し、対象が違ったのは4回（すべて attacker）。対象差による累計射線遮断は双方とも敵629・味方408組・tickで同数。差が出た個別2件では、一方は採用モデルが敵3・味方3組・tickを多く遮断し、もう一方は旧モデルが同数多く遮断した。残り2件は射線への差がなかった。別試合で defender の対象が大きく違った例は、同じ状態での対象予測差ではなく、その前の行動や位置の差によるものだった。
- 以上から、現在の採用モデルに **smoke の使用失敗や対象選択の明確な後退は確認されなかった**。ごりまるの Task 09-A はこの評価をもって完了とし、正式checkpointを維持する。煙による味方射線の遮断もあるため、煙の戦術的な最適性や勝率向上まで証明したわけではない。
- 観測境界: actor に渡るのは従来の `TeamPerceptionSnapshot` を通した合法な共有視認情報。敵の実位置は staged 評価と採点だけに用い、旧モデルの shadow 判断にも渡していない。core、推論側の戦術ロジック、他キャラモデルは変更していない。
- 変更: `evaluate_task09a_gorimaru_smoke.py`、`test_coach_v1_task09a_smoke.py`、`reports/task09a_gorimaru_smoke.json`、`reports/task09a_gorimaru_smoke_shadow.json`、`training/README.md`、本書。評価専用 probe レポートも生成した。
- 検証: `python -m unittest coach_v1.test_coach_v1_task09a_smoke -q` は4件成功。`python -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q` は102件成功。`python -m compileall -q coach_v1` 成功。`git diff --check -- coach_v1` は空白エラーなし（既存ファイルの改行コード警告のみ）。
- 残課題: 固定 STAY/HOLD と人工的な敵配置による射線測定は、実際の発砲数・撃破・勝率への効果を直接示さない。自然配置では使用例が得られなかった。次は Task 09-B のキャラクター別検証へ進み、後の coach 統合時に両陣営の自然な試合で smoke と戦果を再評価する。


## 2026-09-25 調査と採用結果

### 範囲

- `HANDOFF-10.md` の実ゲーム評価を出発点に、ごりまる（slot 0）だけを再学習・比較した。他の4人のcheckpoint、coach、ゲームcoreは変更していない。
- 作業前から変更されていた `attacker_guard_gc_debug.log` は触れていない。

### 80 epochで頭打ちだったか

- **確認されていなかった。** 旧360例学習の検証facingは50 epochで60.0%、74 epochで66.3%、80 epochで63.8%。総合loss最小の正式checkpointは50 epochだった。
- 360例・同じ検証80例のまま、旧 `latest.pt` から160、320 epochまで継続した。独立した320局面のfacingは、旧best 56.6%、旧80 epoch時点 58.8%、160 epoch時点 59.4%、320 epoch時点 59.1%。総合loss最小のcheckpointは50 epochから更新されなかった。
- Task 10と同じ固定STAY coach・人工接敵seed 0～9で、ごりまるの共有目撃への45度以内を比較した。旧bestはattacker 33/44、defender 3/42。160 epoch時点は16/37、1/30、320 epoch時点は17/37、1/30。**同じ局面のepoch追加だけでは実ゲームfacingは改善しなかった。**

### 学習局面の追加

- 旧カリキュラムは1v1/2v1のみだったため、`training/gorimaru_rollout.py` で実ゲームの5v5・近距離局面から actor-safe な観測を収集した。訓練seed 100～139、検証seed 200～209、独立確認seed 300～309。敵の配置・進行にはゲーム実体を使うが、教師facingは `TeamPerceptionSnapshot.sightings` の合法な共有報告だけから作る。未視認敵の実座標をラベルやactorへ渡さない。
- 旧baselineモデルの通常ability判断・対象を教師として維持し、facingの教師は最も近い共有目撃の方向とした。共有目撃がない場合は直前の合法な自分の向きを教師とする。これは評価用教師であり、推論時ロジックではない。
- 固定マップの警戒ポイント中心の旧360例を残し、5v5観測774例（共有目撃595例）を追加した。検証192例（目撃147例）、独立確認238例（目撃181例）は別seed。旧bestの50 epochから20/40/80 epoch追加した候補を保存した。

### 比較と採用

| ごりまるcheckpoint | 5v5独立確認・目撃ありの8方向教師一致 | 実ゲームattacker・45度以内 | 実ゲームdefender・45度以内 |
|---|---:|---:|---:|
| 旧正式best | 14.9% | 33/44 = 75.0% | 3/42 = 7.1% |
| 追加学習20 epoch時点 (`epoch70_latest.pt`) | 33.1% | 42/51 = 82.4% | 35/35 = 100.0% |
| 追加学習40 epoch時点 | 34.8% | 40/58 = 69.0% | 32/38 = 84.2% |
| 追加学習80 epoch時点 | 30.9% | 35/51 = 68.6% | 31/37 = 83.8% |

- 実ゲーム比較は全候補で同じseed 0～9、相手`default`、固定STAY/HOLD coach、最大20 live tickを使用した。分母がモデルごとに変わるのは戦闘進行と生存が変わるため。共有目撃への45度以内は戦術的な最善行動・勝率を示すものではない。
- `epoch70_latest.pt` をごりまるの正式 `best.pt` と `latest.pt` に採用した。旧正式bestは `checkpoints/experiments/task09a_gorimaru_rollout/baseline_best.pt` に、旧80 epoch時点は `checkpoints/experiments/task09a_gorimaru_epochs/baseline_epoch80.pt` に保存済み。
- ability発動成功は実ゲームでattacker/defenderとも各40/40のまま。旧モデルのsmoke対象教師との独立5v5一致率は100%→80%に低下した。警戒ポイント中心の旧検証では対象一致率100%を維持した。smokeの戦術効果や自然移動時の成果は未測定である。

### テストと情報境界

- 新規テスト3件で、未視認敵の実座標を変えても教師facingが変わらないこと、合法な複数共有目撃の扱い、実ゲーム収集例が安全な観測だけを含むことを確認した。
- `python -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q`: **93件成功**。`python -m compileall -q coach_v1` 成功。`git diff --check -- coach_v1` は改行警告のみでwhitespace errorなし。正式bestは選択候補とSHA-256一致し、`GorimaruPolicy()` でロードできた。
- core変更なし。他モデル用trainのlearningからのimportなし。推論時の経路探索・戦術ルール追加なし。

### 変更ファイルと再現

- 新規: `training/gorimaru_rollout.py`、`experiment_task09a_gorimaru.py`、`experiment_task09a_gorimaru_rollout.py`、`test_coach_v1_task09a_gorimaru.py`、本書。
- 更新: `evaluate_task10_rollout.py` に評価用のごりまるcheckpoint overrideを追加。`checkpoints/characters/gorimaru/best.pt` と `latest.pt` を選択候補で更新。`HANDOFF-09.md` に本書への参照を追記。
- 生成: `reports/task09a_gorimaru_epochs.json`、`task09a_gorimaru_real_game.json`、`task09a_gorimaru_rollout_training.json`、`task09a_gorimaru_rollout_real_game.json`、実験用checkpoint。
- 再現: `python -m coach_v1.experiment_task09a_gorimaru`、`python -m coach_v1.experiment_task09a_gorimaru_rollout`。後者は保存済み旧baselineから再開するため、正式checkpoint更新後も同じ出発点を使う。

### 次の作業

- Task 09-Bへ進む前に、各キャラクターを同じ手順で個別に検証する。実ゲームの自然移動での接敵数が少ないため、固定STAY評価だけで最終性能を断定しない。
- ごりまるのsmoke対象品質は別途効果指標で確認する。今回の成果はfacing改善の検証であり、smokeの戦術価値の証明ではない。

## 2026-09-25 追記：評価範囲と教師ラベルの改善

- ごりまる（slot 0）のみを対象に、当tickの合法なチーム共有目撃すべてについて、報告位置から45度以内の向きを正解集合にした。複数の目撃方向を同時に許容する損失と評価を追加した。他キャラクターは従来の単一ラベルを既定値として維持した。未視認敵の実座標は教師にもactor観測にも使っていない。
- 訓練は警戒ポイント由来1400例（point 962、jitter 294、random 144）と、5v5補助600例。警戒ポイント由来が70%。5v5は近距離・西・東・挟撃・自然配置を両陣営で収集し、方角ごとに上限を設けた。検証seed 600–604、独立holdout seed 700–709。
- 凍結した5v5 holdout観測1496例のうち合法目撃あり671例では、現行モデル63.3%、候補 `epoch80_best.pt` 85.8%（許容方向は平均3.43/8）。警戒ポイント由来の独立200例では60.0%→63.5%。ability使用一致は両者100%、5v5の擬似smoke対象一致は100%→95.0%。この許容精度は単一方向の完全一致率と混同しない。
- 実ゲームのSTAY/HOLD・近距離配置・seed 0–19では、**強制facingを除いた**ごりまるの45度以内は、現行attacker 66/92（71.7%）、defender 81/94（86.2%）。候補80はattacker 50/83（60.2%）、defender 91/92（98.9%）。attacker退行のため候補は正式checkpointに採用しない。`checkpoints/characters/gorimaru/best.pt` と `latest.pt` は今回変更していない。
- `reports/task09a_gorimaru_labels.json` は教師・凍結観測の比較、`reports/task09a_gorimaru_labels_real_game.json` は実ゲーム比較。候補は `checkpoints/experiments/task09a_gorimaru_labels/` に保存した。実ゲーム報告は `PYTHONHASHSEED=0` で作成した。先行報告の42/51は強制facing除外の値であり、全行動61/82とは分母が異なる。
- `python -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q`：97件成功。複数方向の損失、合法目撃のみの教師、未視認敵位置変更時の教師・actor観測不変、各配置の収集を確認した。core変更なし、推論への戦術ロジック・経路探索追加なし。

### 残課題と次の作業

- 凍結観測の改善がattacker実ゲームへ転移していない。次はattackerの強制facing除外失敗例を方角・目撃数で集計し、警戒ポイント比率を保ったまま訓練例の配分を調整する。再学習後も同じseed 0–19の両陣営と独立holdoutで採否を決める。
- 自然配置ではdefenderの合法目撃が0例であり、この設定だけでは一般的な対戦性能を判定できない。smoke対象の実際の戦術効果も未測定。残りのキャラクターTask 09-B以降へ、この候補を流用しない。

## 2026-09-25 追記：attackerの失敗分析と再学習

- 診断は推論に触れず、評価時に合法なチーム共有目撃の相対位置のみを記録した。前回の複数正解候補は、実ゲームattackerの複数目撃時33/54（現行56/71）で、失敗33件中18件が西向きだった。補助データの「目撃なし」に直前facingを正解として与えることと、attacker近距離例の不足を原因候補として切り分けた。これは観察からの推定であり、単独原因の確定ではない。
- `training/gorimaru_rollout.py` に `mask_unsighted_facing` を追加した。今回のみ目撃なしの8方向をすべて許容し、facing損失を0にする。ability教師は継続する。既定値は従来どおりで、前回の実験を再実行する条件は維持した。
- 比較した2条件は `mask_only` と `near_priority`。後者はattacker近距離の補助例を60→180例へ増やし、その他の配置を調整した。両条件とも警戒ポイント由来1400例＋5v5補助600例で、主分布70%を維持。後者の補助例には合法目撃あり385例が含まれる。訓練seed 500–539、検証600–604、凍結観測holdout 700–709。
- 採用候補 `near_priority/epoch90_latest.pt` は、独立した合法目撃holdout 671例でfacing許容率81.97%（旧正式63.34%）、警戒ポイント由来200例で65.5%（旧60.0%）。ability使用一致は両者100%。旧モデルのsmoke対象を擬似教師とした完全一致は100%→95%となり、実際のsmoke効果は未測定。
- 強制facingを除く実ゲーム近距離STAY/HOLDでは、seed 0–19のattackerは旧66/92→候補66/81、defenderは81/94→71/71。seed 20–39のattackerは52/97→53/87、defenderは77/87→75/80。選定後の独立seed 40–59ではattacker 62/105→84/121、defender 76/94→43/46。合計はattacker 180/294（61.2%）→203/289（70.2%）、defender 234/275（85.1%）→189/197（95.9%）。候補によって目撃機会数も変わるため、率と件数を併記する。
- ごりまる本人のsmoke使用要求はseed 0–9で両陣営とも旧・候補それぞれ10/10成功。`GorimaruPolicy()` で正式checkpointをロードできる。採用候補を `checkpoints/characters/gorimaru/best.pt` と `latest.pt` に反映した。旧正式版は `checkpoints/experiments/task09a_gorimaru_rollout/epoch70_latest.pt` に保持し、SHA-256の一致を採用前に確認した。他4人のcheckpointは変更していない。
- 診断と比較は `reports/task09a_gorimaru_attacker_diagnostics.json`、`task09a_gorimaru_attacker_adjustment.json`、`task09a_gorimaru_attacker_adjustment_real_game*.json`。実ゲームは `PYTHONHASHSEED=0`。`python -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q` は98件成功。`python -m compileall -q coach_v1` と `git diff --check -- coach_v1` も成功（後者は改行コード警告のみ）。未視認敵の実座標は教師・actor・診断へ渡していない。core変更や推論側の戦術ロジック・経路探索追加はない。

### 残課題・次の推奨作業

- smoke対象の擬似教師一致が95%へ下がった。使用自体は成功したが、smokeの遮蔽効果と味方への影響を別指標で測る必要がある。固定STAY評価では自然なcoach移動中の性能までは保証しない。
- ごりまるのfacingに関してはこのcheckpointを基準とし、次はTask 09-Bの次キャラクターを同様に一人ずつ検証する。実ゲーム統合時には両陣営・各接敵方向を再評価する。
