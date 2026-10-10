# Touyama Gaming v3 の実行方法

味方は Touyama Gaming の5人専用です。コピーされた旧モデルは新入力と互換性がないため削除済みです。best未作成のフェーズはゲームで標準controllerへ切り替え、理由を表示します。その結果は新モデルの性能ではありません。

## 設定

各 `tv3_train_*.py` / `tv3_collect_*.py` 冒頭の名前付き定数で、セット数・対象AI・評価頻度とseed・新規/再開・入出力先を変更します。通常はオプションなしで実行し、CLIは一時的な検証用にします。

学習の `MAX_PARALLEL_WORKERS` の既定値は6です。同じ学習フェーズの相手AIを最大6種類同時に処理します。1にすると順次実行します。

対象は `gc_v1`、`concon_v1`、`omoko_v1`、`fnatic_v3`、`frc_v1`、`toru_ai_v4`。ConCon v1の相手編成は `Gorigons`。Touyama v2と自分自身は対象に含めません。学習・評価とも味方は `Touyama Gaming` で、評価seedを学習と分けます。

Toru v4の対戦基準は `tv3_opponents.py` 冒頭の `TORU_BASELINE_OPPONENT = "touyama_v2"` と `TORU_BEST_DIRECTORY`。v2向けの7種類のbestを読み、v3向け学習済みとは扱いません。best不足・互換性不一致・途中更新では学習・収集を停止します。Toru v4の本学習と並行して入力bestを書き換えないでください。

この `touyama_v2` はToru v4が使用する保存済み重みの識別キーで、Touyama v2との対戦を起動する指定ではありません。対象変更前のデータ・モデル・実行中の学習は今回の修正では変更していません。再学習時は冒頭の新規学習設定を使い、後段も新しいbestに合わせて再収集・学習してください。

## 通常の学習順序

ユーザーが実行してください。今回の修正では学習・収集を実行していません。作業ディレクトリは `03.game/touyama_v3` です。

```powershell
Set-Location C:\Users\ronet\MyProject\git\AI_dnn\03.game\touyama_v3

# defender：前段bestの確定後に後段へ
python tv3_train_defender_analysis.py
python tv3_train_defender_search.py
python tv3_collect_defender_retake.py
python tv3_train_defender_retake.py

# attacker：plant完了後にguardの状態を収集
python tv3_train_attacker_analysis.py
python tv3_train_attacker_plant.py
python tv3_collect_attacker_guard.py
python tv3_train_attacker_guard.py
```

analysis/search/plantの1セットは12ラウンド。保存ケースからのretake/guardは相手のケースを各1回学習する単位です。retake収集は左右各50件、guard収集は左右合計50件が初期設定です。足りない側を人工配置で補いません。

retake収集は、片側が目標件数に達し、もう片側が20ブロック試しても0件なら、0件のサイトをスキップして次の相手AIへ進みます。1件以上あるサイトは目標件数まで収集を続け、上限で不足していれば未完了になります。retake学習は片側のみの保存ケースも受け入れ、0件のサイトの学習をスキップします。

収集設定は `tv3_collect_defender_retake.py` 冒頭の `DEFAULT_CASES_PER_SITE`、`DEFAULT_EMPTY_SITE_SKIP_BLOCKS`、`DEFAULT_MAX_BLOCKS` を変更します。既存の収集データを引き継ぐ場合は `DEFAULT_RESUME = True` にします。作業ディレクトリは `03.game/touyama_v3`、通常実行は `python tv3_collect_defender_retake.py` です。

`latest` は再開用、`best` は教師・探索なしの独立評価で改善した候補だけです。最終評価も行い、最終セットを無条件でbestにしません。

plantの集団維持報酬は、実行した行動のtickの経路計画を使います。次tickの再計画で経路がなくなっても例外を起こしません。この修正前のlatestは判定条件が異なるため、`tv3_train_attacker_plant.py` 冒頭の `RESUME_TRAINING = False` で新規学習してください。

defender_analysis更新後はsearchを更新し、search更新後はretake再収集・新規学習を行います。attacker_analysis更新後はplantを新規学習し、plant更新後はguard再収集・新規学習を行います。schema・マップ・前段hash・ケース条件が違うlatestの再開は拒否します。

## マップと行動

- `tv3_map_defender_init.py`：a-eが初期移動先、A-Eがfacing目標。
- `tv3_map_attacker_branch.py`：攻撃経路・予測の分岐領域。
- `tv3_map_retake_L.py` / `R.py`：合流位置と突入口。

setup制限ではa-dへ到達できず、eも20tick内に届かないため、開始後も初期監視位置へ向かいます。共通地形・setup制限は変更していません。

停止射撃・短距離の複数射線とfacing・能力による侵入妨害・スモーク内の解除への接近は入力・教師例・報酬へ反映しています。独立評価と本番は教師・探索なしです。実際の習得は新モデルの評価で確認してください。

plantでは、同じ公開敵への2〜3人の射線、入口ごとの準備人数、フラッシュ飛行中の露出を学習入力へ追加しました。教師は別入口隊の準備を最大6tick待ち、能力の効果と合わせた突入を教えます。`preserve` モードでも有効な能力を使えます。能力の全消費・左右均等選択・固定座標への戦術を強制しません。

bestは「5人生存・開始時チーム最大HPの90%以上を残した設置率」を最優先し、従来の生存条件付き設置率、通常設置率、生存・被害・能力温存などを続けて比較します。設定は `tv3_train_attacker_plant.py` 冒頭の `LOW_DAMAGE_*`、`PLANT_HP_RETENTION_REWARD`、`COOPERATIVE_HIT_REWARD`、`SHARED_FIRE_PROGRESS_REWARD`。入口待ちの設定は `tv3_attacker_entry_coordination.py` 冒頭にあります。

plantのschemaはFRC調査後の修正でversion 6、入力628次元になりました。壁の角を通り抜ける射線判定も共通エンジンと高速視界計算で修正しました。LOS版をscenario署名へ含めたため、修正前のanalysisを含むv3モデル・ケースは新条件と互換性がありません。学習中のプロセスには反映されません。新条件ではattacker analysis→plant→guard収集・学習、defender analysis→search→retake収集・学習の順で更新してください。analysis/plant/searchの新規学習は `RESUME_TRAINING = False`。収集はretakeの `DEFAULT_RESUME = False`、guardの `RESUME_COLLECTION = False` で旧ケースを混ぜずに行います。後段学習も各スクリプト冒頭の新規学習設定を確認します。既存best・latest・収集データは今回書き換えていません。

FRCだけ調べ直す場合は `tv3_train_attacker_analysis.py` と `tv3_train_attacker_plant.py` 冒頭の `TARGET_OPPONENTS = ("frc_v1",)`、`RESUME_TRAINING = False` に変更します。作業ディレクトリは `03.game/touyama_v3`。通常実行は次の順です。全6相手を更新するときは両スクリプトの `TARGET_OPPONENTS` を6相手へ戻します。

```powershell
py tv3_train_attacker_analysis.py
py tv3_train_attacker_plant.py
```

FRCは `FRC_DEMONSTRATION_WEIGHT`、`FRC_DEMONSTRATION_KIND_WEIGHT`、`FRC_CARRIER_SAMPLE_FRACTION`、`FRC_OPTIMIZER_UPDATE_MULTIPLIER` で教師損失・設置役の経験割合・更新回数を調整します。設置役の不要停止には `CARRIER_STALL_PENALTY` を使い、戦闘・設置中・教師の合流/フラッシュ待ち・合法前進不可の場合は対象から外します。往復回数制限は `tv3_learn_attacker_plant.py` の `EDGE_TRAVERSAL_WINDOW = 8` tickで解除し、公開交戦・失明・退避では制限しません。推論時の前進・能力・PLANTは学習モデルが選びます。

独立評価の各ラウンドは `logs/attacker_plant/<相手>_evaluation_rounds.jsonl` にseed・setとともに追記します。FRCの読取専用診断は `py tv3_review_attacker_plant_frc.py`。設定はスクリプト冒頭、出力は `logs/attacker_plant_review/`。旧version 5重みの診断用変換は追加入力をゼロ重みで拡張するだけで、学習・モデル保存・本番読込への転用はしません。

## 保存先とゲーム

bestは `data/best/<相手AI>/`、latestは `data/<フェーズ>/<相手AI>/`（defenderは `data/defender/search/` と `retake/`）。ケースは `data/retake_cases/` と `data/attacker_guard_cases/`、ログは `logs/` 以下です。

ゲームの作業ディレクトリは `03.game`。

```powershell
python run_game.py
```

編成画面で `Touyama Gaming v3` を選びます。相手と実設置サイトに合わせてbestを読み、search→retake、plant→guardを切り替えます。味方編成・モデル条件の不一致は理由を表示し、該当フェーズを標準controllerへ切り替えます。

非学習の検証は `03.game/touyama_v3` から行います。

```powershell
python -m unittest discover -s test -v
```
