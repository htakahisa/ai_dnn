# GC v3 設置前：Omoko BC → 目標サイトを使うRL

学習は下記のスクリプトを自分で実行した場合だけ開始します。設置前は5人共通の1モデル、設置後は既存モデルです。

現在の実行方法は [回収・設置報酬の修正とミクロ再学習](gc3_attacker_micro_objectives.md) を参照。
既存BCを再利用して、修正後の報酬でミクロRLをST0から始める設定。
収集・BCは再実行しない。以下は以前のrunの説明で、現在の通常設定とは異なる。

## 以前の継続方法：ULT持ち越し・オーブ報酬の修正

2026-10-12の修正では、収集済みOmokoデータとBCを保持し、学習済みRLのbest（ST11,438）の重みから続けます。教師データの再収集・BCの再実行は不要です。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3
python gc3_train_attacker_preplant_omoko_rl.py
```

設定はこのスクリプト冒頭です。`RUN_NAME="attacker_preplant_omoko_rl_resources_v2"`、`INITIAL_CHECKPOINT` は旧RL best、`BC_CHECKPOINT` は同じOmoko BC best、`TRAINING_STEPS=20000`（引き継いだ11,438を含む累計）、`RESUME=False` を設定済みです。旧runは変更せず、新しいrunへ重みを引き継ぎます。報酬が変わるためoptimizerは新規です。修正版runの2回目以降は `RESUME=True` と累計step目標を変更します。

- 同じ文脈群の味方・相手それぞれのULTポイントを次ラウンドに引き継ぎます。死亡しても保持し、実際にULTで消費した分は消費後の残量を使います。新しい文脈群（既定7ラウンド）では0から始めます。
- 設置できた場合は、既存の凍結した設置後モデルで実際のラウンド終了まで進めてから残量を保存します。設置後に消費・獲得するポイントも反映します。設置後の観測・実行行動・確定した結果も履歴に残します。PPOの学習判断・報酬は設置前だけです。
- 回収完了で実際に増えたULTポイント1点につき、回収した本人へ `+0.20` の報酬（`orb_ultimate_gain`）を与えます。通常の2点獲得は `+0.40`、残り容量が1点なら `+0.20`、満タンなら0です。途中中断・回収要求だけ・相手の回収は加点しません。設定場所は `preplant/gc3_rollout_attacker_preplant.py` 冒頭の `ORB_ULTIMATE_REWARD_PER_POINT` です。
- ポイントは観測の既存フィールドを使うため、モデル構造や旧教師データの形式は変えません。目標サイトの距離報酬、確率サンプリング、固定ロジックとの対戦を続けます。
- 各ラウンドの `ultimate_points_start` / `ultimate_points_preplant_end` / `ultimate_points_round_end` と回収で得たポイントをログへ保存します。報酬の内訳にも実際にPPOへ渡す `orb_ultimate_gain` が記録されます。

グラフは `attacker_preplant_omoko_rl_resources_v2` を参照し、累計11,438からの変化を表示します。新ログがない間は記録待ちです。設定は `tools/gc3_watch_attacker_preplant_metrics.py` 冒頭の `RL_RUN`。PNG/SVGは `logs/training_graphs/gc3_preplant_omoko_resources_training_progress.*` に保存します。既に開いている旧グラフは起動し直してください。

実戦描画のモデル選択は `Omoko RL修正版` です。修正版が作成されれば既定で採用し、作成前は旧Omoko RLを表示します。通常ゲームも修正版を優先します。

以下は初回の収集・BCの説明です。今の継続作業では繰り返しません。

## 今回の変更

- 同じ設置前モデルがラウンド初めにA/Bの目標サイトを確率分布から選び、5人で共有します。その選択もBC・PPOの学習対象です。移動先やルートを固定する規則は追加しません。
- 接近報酬は、選んだサイト内の実際に設置できるマスまでの壁を考慮した距離を使います。Bへの接近はAから離れても加点されます。目標を選ぶだけで接近した扱いにならないよう、選択前後を同じサイトで比較します。
- 教師は既存Omoko Gaming v1の凍結モデルです。名前だけを対応させ、実際の能力・ステータス・スパイク所持者はGCと同じです。教師が選んだサイトをサイト選択の正解として記録します。教師の役割分割や内部ルートは生徒に移植しません。
- BCの初期重みは現在のGC v3設置前RLのbest（ST26753）です。既存部分の重みを引き継ぎ、同じモデルにサイト選択・目標入力を追加します。optimizerは新規です。元のモデル、ConConデータ、設置後モデル、候補マップを上書きしません。
- 観測本体は従来どおりGC v3です。IQ・視認制限を通った公開情報、実行した行動、確定した過去ラウンドの結果を全件保存し、全過去イベントへのattentionを続けます。自分で選んだサイトも判断に使います。サイト選択時にはその判断自身の正解ラベルを入力から除きます。
- セットアップ中も、ゲームの移動許可マップ内でアタッカーが動く判断を記録・学習します。設置、アビリティ、オーブ回収はセットアップ中にはゲーム仕様どおり不可です。
- 教師が要求した移動が味方や壁で止められた場合は、実際の停止・向きを正解として記録します。向きの出力は壁の方向も含みます。足元にオーブがないなど、実行できない回収要求は正解にしません。
- オーブの正解ラベルはBC損失の重みを5倍にします。試合中に回収を強制するルールではありません。教師データには回収行動と実際の回収完了の両方が必要です。
- 各BC epochで固定ロジック・7ロスター各4ラウンドを確率サンプリングで評価します。bestは回収完了条件を満たすモデルを優先し、その中で設置率・成功率・検証損失で選びます。
- `completed_orbs` は5 tickの回収が完了し、実際にオーブが消えた回数です。回収行動を選んだだけでは増えません。既定では28ラウンド中1回以上の回収完了と1回以上の設置で `RL_ready=True` になります。これは回収を始めた確認であり、安定した回収を保証する基準ではありません。
- RLは開始直前にもBCを再評価し、回収ゼロならPPOを開始しません。BCからRLを自動実行する処理はありません。RLではBCとのKL罰則を使い、行動とサイト選択を確率分布からサンプリングします。

## 通常の実行

作業ディレクトリは `D:\git\ai_dnn\03.game\ghost_champions_v3`。通常設定は各スクリプト冒頭の名前付き定数を変更し、オプションなしで実行します。

まず新しい教師データを収集し、終了後にBCを実行します。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3
python gc3_collect_attacker_preplant_omoko.py
python gc3_train_attacker_preplant_omoko_bc.py
```

BCの `completed_orbs` と `RL_ready` を確認してから、別にRLを実行します。開始時の再評価に通らなければ学習を開始せず停止します。

```powershell
python gc3_train_attacker_preplant_omoko_rl.py
```

| スクリプト | 主な冒頭定数と既定値 |
|---|---|
| `gc3_collect_attacker_preplant_omoko.py` | `COLLECTION_ROUNDS=280`, `DATASET_NAME="attacker_preplant_omoko_site_v1"`, `TEACHER_KIND="omoko"`, `REQUIRE_ORB_LABELS=True`, `ROUNDS_PER_MATCH=7` |
| `gc3_train_attacker_preplant_omoko_bc.py` | `EPOCHS=6`, `RESUME=False`, `INITIAL_CHECKPOINT` は旧RL best、`BC_ORB_LABEL_WEIGHT=5.0`, `MIN_BC_ORB_COLLECTIONS=1`, `MIN_BC_PLANT_RATE=0.01` |
| `gc3_train_attacker_preplant_omoko_rl.py` | `TRAINING_STEPS=20000`, `RESUME=False`, `INITIAL_CHECKPOINT` はST11,438のRL best、`BC_CHECKPOINT` はOmoko BC best、`MIN_BC_ORB_COLLECTIONS=1`, `BC_KL_WEIGHT=0.10`, `EVALUATE_EVERY_STEPS=1000` |

相手のAIは `FIXED_OPPONENT_AI="default"` で固定。FNC・FRC・TYG・OMG・SPS・GRG・GCのロスターをシャッフルした一巡ごとに全て使います。設置前の評価は設置・殲滅・時間切れで止まり、設置後までの勝率ではありません。

## 保存先と継続

| 種類 | 場所 |
|---|---|
| 新Omoko教師データと履歴 | `data/attacker_preplant_omoko_site_v1/` |
| 新Omoko BC | `data/attacker_preplant_omoko_bc_site_v1/` |
| 目標サイト報酬の新RL | `data/attacker_preplant_omoko_rl_site_v1/` |
| ULT持ち越し・オーブ報酬修正版RL | `data/attacker_preplant_omoko_rl_resources_v2/` |
| ログ | 上記run・データ名と同じ `logs/` 配下 |
| 通常対戦の全公開履歴 | `logs/match_history/*.jsonl.gz` |
| 旧教師データ・モデル（保持） | `data/attacker_preplant_concon_history_v1/`, `data/attacker_preplant_bc_history_v2/`, `data/attacker_preplant_rl_history_v2/` |

BCでまだ回収できなければ、同じ収集データを使い、BCスクリプトの `RESUME=True` と累計 `EPOCHS` を増やして続けます。重み・optimizer・乱数状態を再開します。

RLの継続もRLスクリプトの `RESUME=True` と累計 `TRAINING_STEPS` を変えて実行します。stepは全員の方策判断の合計で、セットアップ中の判断も含みます。7ラウンド単位で集めるため、指定stepを超える場合があります。

今回は設置前の報酬とモデルを変更したので、旧history_v2 runへのそのままの再開はできません。上記の新runを使います。既存runの上書きは拒否します。学習開始後に入力モデル、データ、コード、通常設定を変える場合も新runが必要です。

## グラフと実戦描画

同じ作業ディレクトリで、別のPowerShellから実行します。

```powershell
python tools/gc3_watch_attacker_preplant_metrics.py
python tools/gc3_watch_attacker_preplant.py
```

グラフはOmoko BCと修正版RLのログを5秒ごとに読みます。損失、実reward・return、エントロピー、設置率・成功率に加え、オーブ回収が完了したラウンドの割合を表示します。設定は冒頭の `BC_RUN`・`RL_RUN`・`REFRESH_SECONDS`。新ログがない間は記録待ちとなり、旧値は使いません。PNG/SVGは `logs/training_graphs/gc3_preplant_omoko_resources_training_progress.*`。既に開いた旧グラフは起動し直してください。

実戦描画では `Omoko RL修正版` / `Omoko RL` / `Omoko BC` を切り替えられます。修正版があれば既定で採用し、作成前は旧Omoko RLを表示します。新モデルの目標サイトを画面に表示します。`RL best` / `RL最新` / `BC best` は以前のrun比較用です。設定は冒頭のrun名・`DEFAULT_MODEL`・`POSTPLANT_CHECKPOINT`。通常ゲームのGC v3も修正版RL bestを優先します。

これらの描画・グラフコマンドは学習しません。教師モデルの読み取り専用動作確認は `logs/validation/gc3_omoko_teacher_frozen_probe.json` に記録しています。確認中のoptimizer更新は0で、教師データは作成していません。
