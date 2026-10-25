# GC v3 の初期化と学習

GC v3はv2のモデル・戦術・教師を引き継がない独立した方策です。攻撃の設置前・設置後、防衛の初期配置・サーチ・リテイクも新規のネットワークが行動をサンプリングします。

新規学習ではランダムな重みから開始します。以前のモデル・学習データ・コードは `data/archive/before_scratch_20261010_205508/`、ログは `logs/archive/before_scratch_20261010_205508/` に退避しています。これらを通常のv3で読み込むことはありません。既存のscratch学習モデルは保存したままです。

## マップ

ユーザーが編集したマップの内容は変更していません。

- 設置後: `gc3_map_attacker_postplant.py` の `NEW_MAZE_STR`
- 防衛初期配置: `gc3_map_defender_initial.py` の `DEFENDER_SETUP_POSITION_MAP_GC_STR`

0は道、1は壁、2～4は表示用でAIは道として扱います。5は配置候補、6は警戒位置候補です。候補の周辺にも柔らかい指標を与え、候補外の合法な位置も選べます。警戒候補からアビリティ対象を手動で決める処理はありません。

ネットワークが毎回の移動・警戒位置・アビリティ対象・回復対象・行動種別の確率分布を出し、その分布からサンプリングします。移動はゲーム上の現在位置と隣接マスの選択を繰り返し、どの配置候補を使うかも学習します。特定候補への自動経路移動、到着判定、固定分担はありません。警戒位置はゲームの8方向の向きに変換します。

防衛初期配置は設置後よりランダム性を残します。設定は `gc3_spatial_policy.py` の冒頭です。

| 設定 | 設置後 | 防衛初期配置 |
| --- | --- | --- |
| 温度 | 1.0 | 1.6 |
| 合法行動の一様分布の混合率 | 0% | 15% |
| 候補指標の重み | 0.5 | 0.25 |
| 候補周辺の広さ | 1.5 | 2.5 |

壁、初期配置可能範囲、拘束、封印、アビリティ残数など、ゲームの物理的な制約は適用します。自動射撃やキャラクターのパッシブ効果は共有ゲーム側の処理です。

## 学習

学習方法はPPOです。報酬は勝敗に加え、戦闘・索敵・アビリティの実際の成果、配置・警戒の改善、設置・解除の進行を評価します。設定は `gc3_reward_phase.py` 冒頭の名前付き定数です。数値と計算方法は [報酬の仕様](gc3_rewards.md) にまとめています。報酬は学習時だけ計算し、行動の強制・教師への追従・模倣学習・アビリティ使用優先順位は追加していません。

サーチは設置を防いでラウンドに勝つことを成功とし、設置された時点で失敗として終了します。設置後は攻撃側のラウンド勝利を成功とします。設置後の開始状態は、既存AIで実際に設置まで進めた試合から取得します。位置・生死・HP・アビリティ残数・効果・タイマーを保持し、攻撃側のコントローラーだけをGCに交代します。設置前の行動はGCの学習データに含めません。開始状態の生成AIは固定され、重みの継承や模倣学習は行いません。詳しくは [今回の修正](gc3_training_fixes.md) を参照してください。

対戦相手はTYG・OMG・FRC・FNC・GG・SPSの既存AIです。相手の重みは更新しません。初期評価との比較も新規の未学習モデルを使います。

通常設定は各学習スクリプト冒頭の名前付き定数で変更します。`TRAINING_STEPS`、`OPPONENT_CODES`、`SEED`、`EVALUATE_EVERY_STEPS`、`EVALUATION_ROUNDS_PER_OPPONENT`、`RUN_NAME`、`RESUME` などを確認してください。新規学習は `RESUME = False`、継続は `True` です。以前の学習形式のチェックポイントは読み込みません。

通常設定は修正後の `RUN_NAME = "attacker_postplant_fixed_v1"` / `"defender_search_fixed_v1"` です。攻撃側は中断した4,363 stepから再開するため `RESUME = True`、累計目標は現在の実行指定に合わせて80,066 stepです。防衛側は新規学習の `RESUME = False`、20,000 stepに設定しています。下記のオプションなしコマンドを使います。既存の `*_scratch`・`*_reward_v1` のモデル・ログは保持しています。ネットワーク構造と開始状態が変わったため、それら旧runの再開は拒否します。こちらでは学習を起動していません。

作業ディレクトリは `D:\git\ai_dnn\03.game\ghost_champions_v3` です。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3
python gc3_train_defender_search.py
python gc3_train_attacker_guard.py
```

今回の学習モデルは `data/defender_search_fixed_v1/` と `data/attacker_postplant_fixed_v1/`、ログは同名の `logs/` 配下です。`initial.pt` は未学習モデル、`latest.pt` は再開用、`best.pt` は評価で選んだモデルです。通常のゲームは修正後のrunを優先し、まだ存在しない場合は保存済みモデルを描画用に読みます。攻撃側は報酬変更後の `attacker_postplant_reward_v1/best.pt`、防衛側は `defender_search_scratch/best.pt` が現在の代替保存先です。参照先は `__init__.py` 冒頭の定数で変更できます。古いモデルは古いネットワーク構造を再現して重みをそのまま読み、新規学習へは流用しません。

設置前のConCon v1教師データ収集 → BC → 固定ロジック相手の強化学習は [preplant_training.md](preplant_training.md) に実行手順と設定・報酬を記載しています。通常ゲームの攻撃側は設置前と設置後の2モデルです。

## 描画と確認

ポストプラントだけを見る場合は、AIディレクトリから専用スクリプトを実行します。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3
python tools/gc3_watch_attacker_postplant.py
```

`tools/gc3_watch_attacker_postplant.py` 冒頭の `CHECKPOINT`・`OPPONENT`・`SEED`・`TICK_MS`・`START_PAUSED`・`SITE_MODE`・`SPREAD_MODE` で通常設定を変更します。既定は修正後の `best.pt`、相手はOMG、seedは実行ごとに抽選です。設置までは非表示で実際のゲームを進め、設置直後にGC v3へ切り替えます。選手の位置・HP・死亡・アビリティ残数・効果・知覚を引き継ぎ、学習時と同じ確率サンプリングで行動します。画面の「再開」で再生し、一時停止や速度変更もできます。学習・モデル更新は行いません。

画面を閉じずに「次の場面」で設置直後の新しい場面へ切り替えられます。ラウンド終了後だけでなく再生途中でも切り替えられ、準備中も画面を操作・終了できます。画面上部の「相手」「サイト」「配置」は次の場面に適用します。サイトはA（左）・B（右）・A/B交互、配置はランダム・攻撃側が分散・防衛側が分散・双方が分散から選べます。既定はA/B交互とランダム配置です。「分散」は、生存している同じチームの選手間の最大マンハッタン距離が `SPREAD_MIN_CELLS`（既定8マス）以上の実際の設置状態です。選手を強制移動する処理はありません。直近 `RECENT_SCENES`（既定12）の配置との重複を避けて探します。最大 `SCENE_SEARCH_MAX_ATTEMPTS`（既定96）候補で条件に合う場面が見つからなかった場合は、元の場面を一時停止したまま案内を表示します。条件変更や再試行が可能です。整数の `SEED` を設定すると、同じ操作順で同じ場面の列を再現できます。

作業ディレクトリを `D:\git\ai_dnn\03.game` に戻し、通常のゲームを起動します。

```powershell
Set-Location D:\git\ai_dnn\03.game
python run_game.py
```

Ghost Championsの選手プリセットを適用したあと、AIを「Ghost Champions v3」に変更してください。初期モデルは未学習です。

テストはゲームのルートから実行します。

```powershell
python -m unittest discover -s ghost_champions_v3/test -v
```

## 学習グラフ

開いているグラフを自動更新するには、次のモニターを起動します。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3
python tools/gc3_watch_training_metrics.py
```

初期表示は修正後の `attacker_postplant_fixed_v1` です。防衛への切り替えでは `defender_search_fixed_v1` を表示します。修正前の `*_reward_v1` も対象欄から別に選べます。画面にrun名を表示し、まだ記録がないrunはデータ待ちになります。5秒ごとにログを確認し、新しい記録があれば画面の報酬・return・エントロピー・成功率を更新します。学習を別のPowerShellで実行しながら表示できます。自動更新を一時停止して拡大表示することもできます。以前から開いているモニターは、一度閉じてこのコマンドで起動し直してください。

通常設定は `tools/gc3_watch_training_metrics.py` 冒頭の `RUNS`、`DEFAULT_RUN`、`REFRESH_SECONDS`、`ROLLING_EPISODES`、`LOG_ROOT` にまとめています。モニターはログを読むだけで、学習や対戦は実行しません。PNG/SVGが必要な場合は、下記の保存用スクリプトを使います。

既存ログのグラフを生成する場合、作業ディレクトリはAIのディレクトリです。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3
python tools/gc3_plot_training_metrics.py
```

保存用も修正後の2runを表示します。学習途中や片方が未開始でも、現在の記録だけで生成できます。設定は `tools/gc3_plot_training_metrics.py` 冒頭の `RUNS`、`ROLLING_EPISODES`、`OUTPUT_ROOT` で変更します。グラフとCSVは `logs/training_graphs/` に保存します。学習や対戦は実行しません。

学習時に収集した行動データから、実際の `reward`、価値学習目標の `return`、`advantage`、`value`、割引率、経過tickを `rollout_transitions.jsonl` に保存します。収集バッチの平均と、PPO更新前の行動分布のエントロピーを `training_metrics.jsonl` に保存します。保存先は各学習実行のログディレクトリです。

グラフの報酬には、実測した行動ごとの `reward` の平均を使います。勝敗から代用しません。計測追加前の未記録区間は復元できません。新しい報酬では、各行に `reward_version` と割引後の `reward_components`、バッチに内訳の合計・平均も記録します。内訳の合計とPPOに渡した値の一致を検証します。ログ記録自体は方策・乱数の状態を変更しません。

別の `RUN_NAME` で学習した結果を表示するときは、モニターまたは保存用スクリプト冒頭の `RUNS` にその名前を設定してください。旧報酬と新報酬は別runとして表示し、曲線を接続しません。
