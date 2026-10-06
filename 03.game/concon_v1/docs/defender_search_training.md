# defender search の追加学習

concon_v1 ディレクトリで実行します。

```powershell
python co1_train_defender_search.py
```

スクリプトは既存の `data/defender_search_data/co1_defender_search_positioning.pt` を読み込み、
配置用の重みを固定し、新しい search 用 DQN を追加学習します。
基本モデルのファイルは上書きしません。
基本モデルの作り直しは `python co1_train_defender_search.py --mode positioning` で行います。

## 学習対象

- 敵が射線上にいる場合の停止・敵方向への facing・射撃。
- 通常の交戦では停止射撃を +0.20、移動を -0.25 とし、全ての敵への射線を切る移動にはさらに -0.20 を与えます。
- フラッシュ中の敵に対しても退避は減点します。射線を保ち、敵に近づき、隣接距離を避ける前進だけを例外として評価します。
- 敵の撃破後は 6 tick、その敵の方向を向いて停止し、後続の敵に備える行動。新しい敵に射線が通れば交戦を優先します。
- 味方が発見した敵への射線を増やせる位置が BFS 距離 6 以内にある場合の加勢。味方の射撃開始は条件にしません。
- 同じ距離の支援位置では、すでに射線が通る味方と異なる角度から撃てる位置を評価します。自身に射線が通った後は停止射撃を評価します。
- 支援位置までの距離が遠い味方の持ち場の維持。
- リテイクに備えたスモークの温存と、味方の交戦を支援するフラッシュの使用。
- フラッシュで弱った敵へ射線を通す前進と、隣接距離への接近を避ける行動。

新しい DQN はマップ・味方位置・IQ 知覚で開示された敵位置・facing と、
停止時間、射撃後の状態、HP、アビリティ残数、敵の妨害状態を使用します。
controller は開示された敵位置のみを 12 tick 共有・記憶します。
移動・待機・facing・アビリティの使用はモデルが選択します。
BFS は学習時の報酬と基本モデルの教師データに使い、推論で移動経路を強制しません。
通常の交戦では、既存の IQ 観測を使った補助損失により、敵を向く停止行動の Q 値が合法な移動より高くなるよう学習します。
撃破後の警戒中にも同じ補助損失を適用します。既存の撃破状態の入力は、射線が通る6 tickの警戒中に継続して有効になります。観測の次元は変更していません。
敵が現在見えず、最後の位置の記憶だけがある帰還中は、配置用モデルの合法な移動を教師としてsearch部分を学習します。教師の判定は学習・評価時だけ行い、本番の行動選択はsearchモデルが行います。
死亡した敵の記憶は、生存敵の記憶と区別します。死亡記憶だけでチーム全員のsearchを継続することはありません。撃破者の警戒が終了し、生存敵の情報もない場合は配置用の出力に戻ります。
配置・援護の目標から遠ざかるか距離を縮めない移動には追加で -0.15 を与え、敵方向を向く加点によって往復移動で報酬を稼げないようにしています。
bestの選択では、交戦中の動作に加えて、撃破後の停止率と記憶だけの帰還時の配置動作との一致率が悪化していないことを確認します。
補助損失の設定は `COMBAT_ACTION_MARGIN=0.25`、`COMBAT_LOSS_WEIGHT=0.5` です。フラッシュ中の敵への前進にはこの補助損失を適用しません。

往復移動への修正は `reward_version=7` です。学習済みの旧checkpointの重みは自動更新しません。現在のマップとcheckpointが一致する場合は、次のコマンドでsearchモデルから追加学習できます。

```powershell
python co1_train_defender_search.py --resume data/defender_search_data/co1_defender_search_best.pt --episodes 1000
```

マップや配置を変更してcheckpointが一致しない場合は、配置モデルを作り直してからsearchを学習します。

```powershell
python co1_train_defender_search.py --mode positioning
python co1_train_defender_search.py --episodes 1000
```
フラッシュ候補は attacker と同じ実際の飛翔・着弾判定を使います。
敵方向への facing 報酬は、行動後の位置からその敵へ射線が通る場合だけ与えます。
壁・スモーク・他キャラクターで射線が遮られる場合、共有情報や記憶だけを理由にその敵を向いても加点しません。
ピークして射線が通るマスへ移動する tick から facing を評価します。別の敵へ射線が通る場合は、その敵の方向を評価します。
スモークは使用時の報酬を -0.45 に設定し、リテイクに向けた温存を学習します。
撃破後の方向は本人が視認した位置から記録し、その位置へ射線が通る間は警戒期間中の方向維持を評価します。

敵情報がない場面と setup では、モデルは固定した基本モデルの配置行動を使用します。
学習・評価の episode は敵全滅またはプラント時点で終了します。
味方全滅や未設置の時間切れでも episode を終了します。
プラント時の敵・味方の生存人数を、その時点で記録して終端報酬に使用します。
通常ゲームのスパイク設置後は、既存の DefaultDefenderController がリテイクします。

## 初期設定

基本配置・戦闘学習の設定値は `co1_train_defender_search.py` の先頭にまとめています。

| 設定 | 初期値 | 実行時の引数 |
| --- | --- | --- |
| 追加の対戦 episode 数 | 1000 | `--episodes` |
| 保存・ログ表示間隔 | 50 episode | `--checkpoint-interval` |
| 評価回数 | 相手ごとに 12 ラウンド | `--eval-rounds` |
| 探索率 | 1.0 → 0.05 | ファイル内の `EPSILON_*` |
| 加勢の距離上限 | BFS 距離 6 | `--support-distance` |

相手の attacker は omoko_v1、touyama_v2、fnatic_v3、gc_v1、toru_ai_v3.1 です。
学習の対戦相手はログ表示区間ごとに均等に割り当て、順序をシャッフルします。
既定の50 episodeでは5チームと各10試合を行います。区間の長さがチーム数で割り切れない場合や
最後の区間が短い場合も、相手ごとの試合数の差は最大1試合です。
`Training summary` は探索と重み更新を伴う学習中の戦績です。
best比較用の `Candidate search policy` はモデルを固定し、探索率0で各相手と
`--eval-rounds` 回（既定12回）対戦した評価です。こちらは探索率が下限に達した後に行います。
学習・評価はすべて通常の 5 対 5 の試合形式で、ゲーム本来のスポーン地点から開始します。
defender の setup も通常どおり進めます。配置済み・接敵付近からの開始や、開始時の人数削減は行いません。
ログは対戦相手ごとの設置前の defender 勝利数・割合と、プラント時の平均生存人数を表示します。
人数の平均はプラントした episode のみで計算し、プラントがない場合は n/a と表示します。

```text
omoko_v1: defender wins 4/10 (0.400) end={'attacker_eliminated': 4, 'planted': 6}
  plant_alive_avg: defender=3.00 attacker=2.00 (plants=6) plant_search_score=0.600 advantage_rate=66.7%
```

defender は味方、attacker は敵です。学習・評価の終了理由に defused は出ません。

```powershell
python co1_train_defender_search.py --episodes 3000 --eval-rounds 12
python co1_train_defender_search.py --episodes 1000 --opponents omoko_v1 touyama_v2
```

## 保存・再開・ゲームへの反映

保存先は `data/defender_search_data/` です。

- `co1_defender_search_positioning.pt`: 固定する基本モデル。
- `co1_defender_search_latest.pt`: 今回の追加学習の最新モデル。
- `co1_defender_search_best.pt`: 配置の検証に合格し、比較評価の採用条件を満たしたモデル。
- `battle_training_log.jsonl`: episode ごとの報酬、相手、勝敗、行動指標。

探索率が 0.05 になった後、スクリプトは候補を基本モデルまたは既存 best と同じ条件で再評価します。
best の採用には、配置到達・待機・facing の検証合格、平均 search_score と最低相手 search_score の維持が必要です。
移動射撃率 `moving_fire_rate` と停止・敵方向 facing 率は診断用に表示しますが、best の採用判定には使用しません。
両指標は同点時の `behavior_error` からも除外しています。プラント時の人数差や設置阻止を含む
平均・最低相手の search_score が改善した候補は、移動射撃率の上昇だけでは不採用になりません。
評価ログの `normal_stop_aim` は通常交戦で停止し、敵を向いた割合です。
学習時の停止射撃報酬と補助損失は継続して使用します。
敵全滅・未設置時間切れの防御成功は 1、味方全滅は 0 として評価します。
プラント時の評価値は `0.5 + (defender生存人数 - attacker生存人数) / 10` です。
評価は人数有利を優先し、同人数は常に 0.5、人数不利は 0.5 未満、人数有利は 0.5 より大きくなります。
3 対 4 は 0.400、3 対 3 は 0.500、4 対 3 は 0.600 です。5 対 5 と 1 対 1 はどちらも 0.500 です。
終端報酬は `3 × (2 × search_score - 1)` とし、プラント時の生存人数を学習に返します。
プラント時の終端報酬は人数差の 0.6 倍となり、人数不利は負、同人数は 0、人数有利は正になります。
`plant_alive_avg` 行の `plant_search_score` はプラントした試合だけの平均評価です。
`advantage_rate` はプラントした試合のうち defender の人数が attacker より多かった割合です。
`overall_search_score` は設置前の勝利 (1)・敗北 (0) も含む全試合の平均評価で、best の採用に使います。
総合評価には設置阻止による勝利も含まれるため、プラント時の人数評価とは別に表示します。
評価値が同じ場合は、撃破後の静止警戒・記憶位置からの帰還・加勢・スモーク温存などの行動指標を比較します。
行動ログの `post_kill_hold` は撃破後に停止し、敵がいた方向を向いていた割合です。
既存 best も現在の設置前の終了条件で再評価し、保存済みの旧勝率は比較に使用しません。
採用条件を満たす候補がなければ、スクリプトは既存 best または基本モデルを維持します。
配置マップを変更して基本モデルを再学習した場合、スクリプトは旧マップの best を
`co1_defender_search_best_incompatible_日時.pt` にバックアップし、今回の学習開始モデルを比較対象にします。
新しい best が採用されるまでは、ゲームは現在の基本モデルを使用します。
旧マップの latest などを `--resume` に指定した場合は、マップ不一致として学習を停止します。

再開時の episode 数も「追加で学習する数」です。
今回の報酬バージョンは 6 です。既存の search モデルの重みはそのまま読み込めるため、`--resume` で新方針の追加学習ができます。

```powershell
python co1_train_defender_search.py --resume data/defender_search_data/co1_defender_search_latest.pt --episodes 1000
```

モデル単独の評価:

```powershell
python evaluate_co1_defender_search.py --model data/defender_search_data/co1_defender_search_latest.pt --rounds 12
```

`python ../run_game.py` で defender 側を Gorigons / ConCon v1 にすると、
controller は `co1_defender_search_best.pt` があれば読み込み、なければ基本モデルを読み込みます。
起動ログに絶対パスと `skill=search` または `skill=positioning` を表示します。

テスト:

```powershell
python -m unittest discover -s test -p "test_co1_defender*.py"
```

テストは観測・報酬・モデル接続・重みの固定・本番 IQ の短い動作を検証します。
追加学習スクリプトの実行は手動で行ってください。
