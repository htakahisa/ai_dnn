# Toru AI v4：新規汎用defenderの学習

通常は `MAX_PARALLEL_WORKERS = 3` で相手AIごとに並列学習する。searchの相手別ログは従来と同じ。retakeのログは相手別に分離する。設定は [並列学習](parallel_training.md) を参照する。

選手名・チーム名を判断入力に含めない、新しく作成したDouble DQNです。任意の5人編成で重みを共有し、位置・HP・命中や回避等の能力値、スキル種別と残数、味方の能力構成、公開された敵の観測履歴から行動を選びます。マップは現在の26×44のマップです。編成に依存しない入力設計であり、あらゆる編成で勝てるという保証ではありません。

## モデルの構成

- **解析**：`data/best/<attacker AI>/defender_analysis_best.pt` を相手別に読み、重みを固定。旧`analysis_best.pt`・`best.pt`も読み込み可能。学習方法は[defender解析の学習](training.md)を参照。
- **search**：相手attacker AIごとの6モデル。各モデルはその相手だけとの対戦で、複数の味方編成を使って学習。
- **retake**：味方編成は共通化したまま、**相手attacker AI × 左右サイト**の12モデルに分割。

既存の他チームのdefender重みは使いません。retakeの初期値に使うのは、ここで新しく学習したToru v4 searchだけです。

旧構成の共通searchは廃止し、checkpointも整理済みです。相手別モデルを使用してください。学習用controllerは、別の相手用searchや旧共通モデルを指定するとエラーになります。ゲーム用controllerは未作成・条件不一致のフェーズを汎用controllerへ切り替え、理由を表示します。

## ゲーム画面からの動作確認

`03.game` をカレントディレクトリにして `python run_game.py` を実行し、編成画面のAIプルダウンで **Toru AI v4** を選びます。味方defenderには任意の5人編成、相手には6種類の対象AIのいずれかを選べます。

相手の制御AIを自動判定し、`data/best/<相手AI>/search_best.pt` と同じディレクトリの `retake_L_best.pt`・`retake_R_best.pt` を読み込みます。解析は同じディレクトリの `defender_analysis_best.pt` です。移行前は旧`best.pt`も読み込めます。searchが使用可能でretakeが未作成なら、プラント前は学習済みsearch、プラント後は標準defenderで動きます。retakeが片側だけ学習済みの場合も、実際の設置サイトに合わせて切り替えます。モデル未対応の相手、未作成search、解析モデルとの不一致、異なるマップでは標準defenderを使います。標準defenderにも通常のIQによる知覚制限を適用します。Toru AI v4のattacker側は当面、標準attackerです。サイド交代にも対応します。

起動・相手切替時のコンソールに、相手AI、search・retake L/Rの読込先、汎用controllerへの切替理由を表示します。実行中に学習モデルを更新した場合は、次の試合開始で読み直します。`tv4_run_defender.py --render` も同じゲーム用controllerを使用します。

## 実行順序

作業ディレクトリは `03.game/toruAI_v4/`。通常のゲームが動くPython環境で実行します。

```powershell
Set-Location C:\Users\ronet\MyProject\git\AI_dnn\03.game\toruAI_v4

# 1. 相手別searchを全6AI分、新規学習。各モデルはセット1から開始
python tv4_train_defender_search.py

# 2. searchのbestを固定し、相手AI別・左右別のretakeを学習
python tv4_train_retake.py

# 3. 学習後、任意の編成で実行（敵味方で選手名が重ならない編成）
python tv4_run_defender.py --opponent fnatic_v3 --defender-preset "Gorigons" --render
```

`tv4_train_defender_search.py`上部の `TRAINING_SETS` が各相手モデルの今回の学習セット数です。既定100なら、オプションなしでは全6AI分を順番に、新規に1〜100セットずつ学習します。**各モデルの1セットは、その相手との12ラウンド**です。`--opponents fnatic_v3` で1相手、`--opponents fnatic_v3 touyama_v2` で複数を選べます。選択内容によってモデルの共有や保存先は変わりません。`--resume`を明示した場合だけ各相手のlatestから再開し、保存済みセット数に今回の学習セット数を追加します。評価は `EVALUATION_INTERVAL = 1` により毎セット、対応する相手のみと同じ3seed×12ラウンド（36ラウンド）・編成で実施し、改善した場合だけbestを更新します。全6AI分の評価は1セットあたり合計216ラウンドです。重み・optimizer・replay・完了セット数・best選定は相手ごとに独立しています。

学習は複数編成を切り替え、評価には学習で使わない編成（Eine Kleine / SUPES / BBL）を使います。各編成の5人の能力・IQ・コンボ等は既存エンジンのルールで反映されます。

```powershell
# 短い確認・特定相手に限定
python tv4_train_defender_search.py --opponents fnatic_v3 --sets 1 --eval-seeds 1
python tv4_train_retake.py --opponents fnatic_v3 --sets 1 --eval-seeds 1

# 評価だけ。重み・replay・bestは更新しない
python tv4_train_defender_search.py --eval-only
python tv4_train_retake.py --eval-only
```

retakeはsearchと同じ解析重みを使います。searchはそのまま固定してretakeを学習してください。searchを学習し直してbestが変わった場合は、retakeの継続条件が変わるため、明示的に新しいretake学習を開始します。

## searchの目的

プラントは許容します。プラントされたこと自体に罰は付けず、**生存・早めの寄り・スキル温存・敵の撃破**を両立させます。

予測サイトの確率が65%以上になると、遠いキャラほど早めの移動を強く評価します。本人に割り当てられた合流地点への歩行距離を、プラント直後の到着予算（55 − 解除6 − 突入交戦20 − 余裕5 = 24tick）で割って重みにし、上限2倍で前進に追加加点、足踏み・後退に追加減点します。近いキャラの圧力は小さく、到着済みなら追加減点はありません。設置までの残り時間を予測する処理ではなく、設置が早くても間に合う位置への移動を促します。

公開観測で敵と本人の射線が通る場合（敵位置がスモーク内なら除外）、そのtickに被ダメージまたは撃破があった場合は、この追加報酬を外します。味方が別の場所で交戦しているだけでは除外しません。従来の移動・戦闘報酬は継続するため、交戦中の行動も学習できます。

調整する定数は `tv4_train_defender_search.py` 冒頭の `ROTATION_PROGRESS_REWARD`、`ROTATION_DELAY_PENALTY`、`ROTATION_URGENCY_CAP` です。作業ディレクトリを `03.game/toruAI_v4/` にして、通常は `python tv4_train_defender_search.py` で新規学習します。既存の重みには再学習で反映されます。`--resume` を使う場合、旧報酬のreplayを除外して重みと完了セット数を引き継ぎます。

確率が低い間は監視位置を目標にし、左右の確率が65%以上になれば、その側の合流地点を目標にします。合流地点は `tv4_map_retake_L.py` / `tv4_map_retake_R.py` の小文字 `a`・`b`・`c` の全マスです。`c` は記載のある側で使用します。生存者には位置に応じて異なる合流マスを割り当てます。大文字 `A`・`B`・`C` は対応する突入口として読み込みますが、searchの移動目標や到着評価は小文字の集合場所です。能力定点は使用しません。1tickに1マスの移動、向き、能力、ult、解除、オーブ回収が行動候補です。移動目標は最終行動を強制せず、学習したQ値が行動を選びます。

フラッシュ・スモーク・リコンを含む、その選手が持つ合法な能力を選べます。能力の消費だけでは加点せず、最後の1回をsearchで使う場合は追加コストを付けます。プラント時に生存者が残した能力にも報酬があります。危険時に最後の1回を使う行動は禁止しません。

最初の5セットは、移動・公開情報に基づく能力使用・解除の補助行動を徐々に減らしながら学習します。searchの補助行動は最後の能力を温存します。retakeでは温存制約を外します。学習済みフェーズの評価・本番では補助行動や探索を使わずQ値を使います。Setup中の配置は地形と進入許可マップに基づく共通ルールです。

**searchの学習・評価はプラント完了tickで終了し、retakeを実行しません。** プラント後の勝敗は不明なので`PLANTED`と表示し、負けには数えません。プラント前に決着した場合だけ`WIN/LOSS`を表示します。過去ラウンドの履歴には観測されたプラント先を残し、プラント後の勝敗は捏造しません。したがってsearchのみの学習では、通常試合のスコアやプラント後の結果に応じた相手の適応までは再現しません。

retakeを学習する段階では、固定したsearchでプラントまで進めてからretakeを実行し、通常のラウンド終了まで継続します。ゲームでの動作確認は未作成のフェーズを汎用controllerで補完します。

searchのbest比較は、生存・資源・リテイクまでの距離を合わせた準備スコアを優先します。安全なスポーン地点に居続けるだけで有利にならないよう、距離も評価します。プラントなしで勝つことも認め、プラントなしで全滅した試合を高く評価しません。

プラント時の`味方生存人数−相手生存人数`にも明示的な加点・減点を付けます。例えば5対3は5対5より高く評価します。人数差だけを優先して味方を失わないよう、生存人数・スキル温存・距離の評価も併用します。各ラウンドの人数差と評価時の平均人数差をログに表示します。

プラント時には、生存している味方全員の「実際に設置された側のa・b・cのいずれかまでの最短通路距離」を個別に測ります。爆発までの実際の残りtickから、合流までの移動時間・合流後の突入交戦用20tick・解除用6tickを引き、5tick以上の余裕があるかを判定します。残り55tickなら合流まで24tick以内が基準です。交戦時間は固定の見積もりで、戦闘結果を保証するものではありません。`tv4_train_defender_search.py` の `RETAKE_COMBAT_RESERVE_TICKS` と `RETAKE_SAFETY_MARGIN_TICKS` で調整できます。旧チェックポイントからは学習済み重み・optimizer・完了セット数を継続し、旧報酬のreplayを空にして新基準で収集し直します。集合場所の変更時もbestを新しい配置で再評価します。

1人でも余裕が不足すると、チーム共通のプラント時報酬とbest選定用準備スコアから `min(5, 2×遅延人数 + 0.1×最大不足tick)` を減点します。相対的に遠くても必要な余裕を満たしている人には、この減点を付けません。死亡者は到着判定から除き、生存人数・死亡の報酬で別途評価します。各ラウンドのログには参加可能人数・遅延人数・最遠距離・最小余裕・減点を、評価サマリには全生存者が余裕を満たした割合と平均遅延人数を出します。既存bestは次の学習時のbest比較で同じ新基準に再評価します。

## retakeの目的

保存した設置状態から繰り返し学習する収集・学習経路は [retake_training.md](retake_training.md) を参照してください。`tv4_collect_retake.py` で各相手・各サイド50件を収集し、`python tv4_train_retake.py` で開始できます。通常のリテイク設定は `tv4_train_retake.py` 冒頭の定数を編集します。

実際のsearchが動いた5対5の試合を続行し、実際のプラント後の行動だけを学習します。敵の隠れた配置へ移動したり、架空のプラント位置を生成したりしません。既知のattackerごとの傾向は、専用の左右ネットワークに学習させます。

モデルは実際のプラントサイトで切り替えます。予測が外れた場合も、プラント後は公開された設置位置へ切り替えます。retakeのbest比較はリテイク勝率→解除回数→平均報酬の順です。

**ある相手が片側にほとんどプラントしない場合、その側のデータ・評価は少なくなります。** サイト別のプラント数・サンプル数をログに表示し、学習も評価もない側のbestは生成しません。実行時に左右どちらかのbestがない場合も、別サイトの重みへ黙って置き換えません。

## 保存先

```text
data/best/<attacker AI>/
  defender_analysis_best.pt
  search_best.pt
  retake_L_best.pt
  retake_R_best.pt

data/defender/
  search/<attacker AI>/latest.pt
  retake/<attacker AI>/L_latest.pt
  retake/<attacker AI>/R_latest.pt

logs/defender/
  search/<attacker AI>/training.log
  retake/training.log
```

ログは各段階でコンソールと同じ `training.log` のみで、起動時に上書きします。生存人数・プラント時のリテイク距離・資源温存率・スキル使用回数・解除・勝敗と、bestの保存先を出します。資源温存率は開始時の全員の能力残数を分母、プラント時に生存者が持つ残数を分子にします。死亡した味方が持っていた能力は残った資源に数えません。

相手の制御AIは`相手AI=touyama_v2`、味方の選手編成は`学習編成=Fnatic2023`と明記します。評価時は`評価編成=...`、retakeのモデル別集計は`サイト=L/R`を表示します。

latestにはoptimizer・target network・replay・完了セット数を保存します。オプションなしでは既存latestを読み込まず、新規学習のチェックポイントで上書きします。再開する場合のみ `python tv4_train_defender_search.py --resume` または `python tv4_train_retake.py --resume` を使用します。`--resume`でlatestがない場合はエラーになります。`--fresh`は従来コマンドとの互換用で、オプションなしと同じ新規学習です。別の設定で試す場合は `--data-dir` / `--best-dir` / `--log-dir` で出力先を分けられます。既存bestも評価して比較し、改善した場合だけ更新します。

既存の解析モデルとbestは更新しません。search・retakeのcheckpointには解析モデルと固定searchのハッシュを記録し、学習途中で入れ替わった場合は継続を拒否します。現在の解析モデルは以前の監視ルールで学習したものなので、新しいsearchが取得する観測でも精度が維持されるかは今後確認が必要です。

## 検証

```powershell
python -m unittest discover -s test -v
```

選手名や味方の順序に依存しない入力、未知の敵位置とスパイク保持者の非公開性、合法な能力、最後の能力の温存、プラント時のモデル切り替え、左右別のサンプルとネットワーク、実際のDQN更新を検証します。
