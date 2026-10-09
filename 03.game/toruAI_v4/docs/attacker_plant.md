# Toru AI v4：attacker plant の設計・学習手順

## 責務と採用方針

学習済み `attacker_analysis_best.pt` を固定してサイトと経路を選び、相手AIごとのプラント用モデルが局所移動、警戒方向、アビリティ使用、回収、プラントを実行する。相手AIごとに1モデルを保存し、味方5人がその重みを共有する。左右サイトは目標サイト入力で区別する。

**左サイトに偏っても構わない。左右の選択率を均等化する報酬・強制探索は入れない。** 右へ行けばほぼ倒される相手に、学習の都合で右へ行かせない。左右別の成績は記録するが、analysisが選ばない側は「評価対象なし」と表示し、架空の0%や100%を付けない。

学習中はanalysisの重みを固定するが、経路・配置予測は公開観測に応じて更新する。`preserve / supported` は行動モデルへの参考情報とし、アビリティ使用をこのmodeで禁止・強制しない。analysisのオンライン収集器と同じ `AdaptiveAttackPlanner` を使う。設計と学習ラベルは [attacker_online_design.md](attacker_online_design.md) を参照する。局所行動はplantの学習済み方策が選ぶため、成功率は別に独立評価する。

guardは別モデル。今回の学習区間と成績はプラント完了tick終了時まで、失敗した場合は未プラント終了時までとする。プラント後の配置や守備用アビリティを残す目的は含めない。

## 入力とアクション

FRCのattacker側公開観測、目撃履歴、味方の位置・状態・戦闘能力・アビリティ種類、スパイク所持・設置進捗、残り時間、選択経路・進捗・サイト、analysisの敵別領域確率と不確かさを入力する。各味方の射撃可能な接触、進むと増える接触、援護人数、停止・退避・援護待ちなど10個の公開戦闘特徴も加える。隠れた敵位置・敵の内部目標・敵AIの内部状態は入力しない。

既存defender policyの公開観測エンコードと合法アクション検証を利用し、attackerの設置・スパイク状態・配置予測を追加する。DQNの出力は58候補で、局所移動／待機×8方向facing、通常アビリティとULTの候補、PLANT、オーブ回収を選択する。候補はFRCの合法マスクで検証する。

アビリティ標的には、目撃地点、最新の観測履歴、analysisで敵が多いと推測される分岐領域の地形上の地点、目標サイト、落ちたスパイク、自分の位置を用いる。配置予測は領域単位なので、隠れた敵の実座標を標的として渡さない。公開確認できた無人地点は予測標的から除く。敵がいそうな地点と現在位置の関係を入力し、移動前からのfacingを学習させる。

## 経路の実行と停滞防止

analysisが選んだ経路の少し先を局所移動目標にする。味方には別のサイト内到達地点を割り当て、スパイク所持者の到達マスを塞がないようにする。

- 通常は経路周辺2マスと後方3マスまでを局所移動に含める。公開交戦・新しい目撃履歴・失明・退避判断時は経路外への退避も許可する。別入口担当は、その担当入口への経路を使う。
- 未視認でも待機・向きの調整を候補から除かない。前進を強制せず、合法候補から学習したQ値で選ぶ。
- 選択サイト内の所持者にはPLANTを候補として渡す。教師は失明や接触・HP・援護人数を確認して設置を勧める。推論ではPLANTを強制せず、危険なら中断も選べる。
- 同じ移動辺の往復はプレイヤーごとに最大2回までに制限する。スパイク回収は戻る必要があるため例外とする。
- 通常3tickごとに公開観測から再判断する。同じ敵IDの移動や無人確認も扱い、被害・死亡時は即時、停止6tickでは別の進路を検討する。変更回数の上限はなく、設置開始後は目標経路と援護役の別入口を維持する。情報が乏しい間は短距離の観測目標と偵察役の進路も更新する。schemaはversion 4、実行条件は `adaptive_combat_plant_v5`。旧モデルを新条件として再開しない。

局所移動の制約と公開アクションマスクは実行時も同じ。学習・評価で隠れた敵占有をマスクに使わない。

## 学習と報酬

相手別のDouble DQNとtarget network、共有プレイヤーreplayを使う。1セットは同じ対戦AI・ゲームで12ラウンド。セット終了後に更新し、相手AIとanalysisの重みは変更しない。

報酬の主目的は生存してプラント成功すること。成功+8を生存割合で縮小し、味方生存・生存者のアビリティ温存を小さく加点する。既に死亡したキャラの過去行動には、後の設置成功加点を付けない。未プラント終了-6。通常の前進、設置進捗、交戦での与ダメージ・撃破・停止射撃命中を補助とし、被ダメージ・死亡・時間・アビリティ消費を減点する。敵全滅でゲームが終わり設置できなかった場合も、実際のプラント成功には数えない。

予測していた方向へ先にfacingしたことは、各プレイヤーと敵の幾何的な初接触時に最大1回だけ小さく加点する。敵の方向を向いて待つだけでは繰り返し報酬を得られない。敵の実位置は、この学習時の答え合わせ・交戦結果計算だけに使い、公開入力と行動を確定した後に読む。

アビリティは使用しただけでは加点せず、小さな消費コストを付ける。FLASHの追加失明・RECONの追加リビール・SMOKEの射線遮断見込みなどを効果発生後に記録し、投げた元のtransitionへ加点する。観測効果がなければ小さく減点する。有効な攻撃は与ダメージ・撃破、安全な進入は前進・生存・設置でも評価する。効果記録は入力・教師判断に戻さない。追加リビールなしでも戦術的に無価値とは限らず、SMOKEも全射撃の反実仮想を再実行した評価ではない。実測診断と限界は [attacker_combat_review.md](attacker_combat_review.md) を参照する。

初期には公開情報だけで動く教師方策を95%の確率で選び、60セットかけて10%まで減らす。教師を選ばなかった行動に対する探索率を20%から3%に減らす。両者は加算する率ではない。教師は経路の前進、予測・目撃に基づく警戒とアビリティ、設置・回収を教える。学習lossにも小さな教師模倣項を加える。**実力評価・推論は教師選択とランダム探索を両方0にして、学習したQ値と同じ実行制約だけで動く。**

プラント後は待機のみでラウンドを実エンジンの最後まで進め、相手AIのラウンド間状態と公開勝敗履歴を保つ。この区間には学習transitionや報酬を追加しない。評価の被害・生存・アビリティ使用もプラント時点で固定する。

## 通常実行

設定は **`tv4_train_attacker_plant.py` 冒頭の名前付き定数**を変更する。

| 定数 | 既定値・意味 |
|---|---|
| `TRAINING_SETS` | 相手ごとの追加100セット。1セット=12ラウンド |
| `TARGET_OPPONENTS` | 現在は `("frc_v1",)`。対象を冒頭の定数で変更可能 |
| `TRAINING_PRESETS` | 複数の味方編成。セットごとに切り替え、敵との選手名重複を除外 |
| `EVALUATION_PRESETS` | 学習とは別の味方編成。評価seedごとに編成を固定して比較 |
| `RESUME_TRAINING` | Falseで新規、Trueで同じ保存先のlatestから再開 |
| `EVALUATION_ONLY` | Trueなら再開モデルの評価だけ。RESUMEもTrueにする |
| `EVALUATION_INTERVAL` | 10セットごと。今回の最終セットも必ず評価 |
| `EVALUATION_SEED_COUNT` | 独立3seed×12ラウンド=36評価ラウンド |
| `OPTIMIZER_UPDATES` | 1セット後の更新200回 |
| `ANALYSIS_DIRECTORY` | 採用済みanalysis bestの親ディレクトリ |

seed、学習率、探索・教師の減衰、報酬、保存先、コンソール色も冒頭で確認できる。CLIは一時的な確認用上書き手段。

toru AI v4は汎用AIであり、味方をGorigons等へ固定しない。相手AIごとに1モデルを持ち、異なる味方の戦闘能力・アビリティ・位置・状態を入力して5人で重みを共有する。実際の学習編成はログ・ラウンド結果に保存し、best評価には学習外の編成を使用する。

旧版の `ATTACKER_PRESET = "Gorigons"` で作ったモデルは固定編成の条件で学習・評価したモデル。そのcheckpointを複数編成の新条件として通常再開することはできない。汎用版の学習は `RESUME_TRAINING = False` で新規に行う。コード修正時に既存のモデル・データは変更しないが、通常保存先で新規学習を実行すると同名latest・bestは新条件で更新される。

attacker analysis は現在入力version 3、実行器は `public_online_combat_attack_v7`。旧入力や戦闘修正前のbestは新条件のplantへ読み込めない。先に `tv4_train_attacker_analysis.py` を新規学習し、そのbestでplantを新規学習する。analysis更新後はhashが変わるので、旧plant・guardケース・guardモデルを同条件の再開には使わない。

作業ディレクトリは `03.game/toruAI_v4/`。

```powershell
Set-Location C:\Users\ronet\MyProject\git\AI_dnn\03.game\toruAI_v4
python tv4_train_attacker_plant.py
```

```powershell
# 保存・学習を開始せず、schemaと使用analysisのhashを確認
python tv4_train_attacker_plant.py --describe

# 一時的に1種類・1セットを検証
python tv4_train_attacker_plant.py --opponents fnatic_v3 --sets 1 --eval-seeds 1
```

## 評価・best・再開

通常学習は `MAX_PARALLEL_WORKERS = 3` で相手AIごとに並列実行する。ログは `logs/attacker_plant/<相手AI>/training.log`。並列数・停止・他学習の設定は [並列学習](parallel_training.md) を参照する。

コンソールの緑色 `[実力評価][教師・探索なし]` がプラントモデルの性能。学習行は教師・探索を含む「収集プラント成功率」と直近の実力評価を表示する。成功率、敵味方の生存率・平均生存人数、通常アビリティ温存率・使用率、被害、時間切れをanalysis学習と同じ定義で表示する。左・右の内訳も出す。保存ログには色の制御文字を入れない。

bestは共通の独立評価seedに対して、3人以上生存かつ開始時チーム最大HPの30%以上を残した設置率 → 通常設置率 → 味方の平均生存人数 → 少ない被害 → 高い通常アビリティ温存率 → 少ない使用回数 → 短い成功時間、の順に改善した場合だけ更新する。残りHPも表示する。この生存条件は設置後勝率ではない。最終セットを無条件でbestにしない。

新規学習は同名latestを更新し、最初の評価で今回のbestを保存する。再開はpolicy、target、optimizer、replay、抽出乱数、完了セット数、直近評価を復元し、完了セット数から探索減衰も継続する。中断時は完了したセットまで保存済み。評価専用実行はpolicy・replay・bestを更新しない。

analysisのファイルhashをplantのlatest/bestに記録する。再開時と推論時は同じanalysisであることを確認し、後からanalysisを更新したモデルへ無言で差し替えない。新しいanalysisを使う場合はplantを新規学習する。

## 配置と推論

```text
toruAI_v4/
  tv4_learn_attacker_plant.py          # 入力、合法候補、DQN、学習更新、読込
  tv4_attacker_plant_controller.py     # analysis経路＋学習した行動の実行
  tv4_train_attacker_plant.py          # 実ゲーム学習、評価、保存
  test/test_tv4_attacker_plant.py
  docs/attacker_plant.md
  data/attacker_plant/<相手AI>/latest.pt
  data/best/<相手AI>/attacker_plant_best.pt
  logs/attacker_plant/training.log
  logs/attacker_plant/<相手AI>_rounds.jsonl
```

replayはlatestに学習データとして保存する。確認用の少量モデルは `data/attacker_plant_smoke_v1/`、ログは `logs/attacker_plant_smoke_v1/` などの専用保存先に分ける。本学習のanalysis、defender、plant bestには書き込まない。

```python
from toruAI_v4.tv4_attacker_plant_controller import ToruV4AttackerPlantController

# toruAI_v4を作業ディレクトリにする必要はない。既定パスはモジュール基準。
controller = ToruV4AttackerPlantController.from_best("fnatic_v3")
controller.set_game(game)
```

`run_game.py` で「Toru AI v4」を選ぶと、相手AI別のanalysis・plantのbestを読み、設置完了tickの後にguardのbestへ切り替える。保存先と読み込み条件は [通常ゲームのbestモデル](game_best_models.md) を参照する。

既存analysisで97〜100%だったとしても、新しいplantモデルの性能を保証しない。少量の動作確認結果と本学習のbest評価は区別する。
