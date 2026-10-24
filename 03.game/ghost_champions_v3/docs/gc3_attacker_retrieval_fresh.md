# 設置前ミクロを重みの引き継ぎなしで学習

過去のBC・RL・マクロの重みを読み込まず、ランダム初期化からBCを行います。その新しいBCで設置とオーブ回収を確認してから、RLをST0から開始します。既存のOmoko教師データ40試合を再利用し、収集し直しません。設置後は了承済みの既存モデルを凍結して使います。学習は以下のコマンドを実行した時だけ開始します。

## 修正した入力と報酬

公開観測にある落下スパイク位置から、壁を考慮した歩行距離、各移動先での距離の変化、味方それぞれの距離を追加します。IQによる推定位置が壁に入った場合は、その周囲の通行可能マスまでの距離を入力します。実際の隠れた位置を入力には使いません。既存の教師データからも同じ入力を計算できます。

回収への接近報酬を味方全員へ共有する方式から、近づいた本人へ渡す方式に変更しました。設置完了も設置した本人へ渡します。行動の強制や役割別モデルの追加はなく、引き続き確率分布からサンプリングします。

| 内容 | 報酬と配分 |
| --- | --- |
| 落下スパイクへの接近 | 本人の過去最高の近さを更新した差分。1回の落下につき1人最大+6 |
| 実際のスパイク回収 | 本人へ+3。初期所持や単なる回収要求では加点しない |
| 有効な設置進行 | 本人へ1tickにつき+0.25。4tick完了で計+1 |
| 設置キャンセル・死亡中断・未完了で終了 | その試行の進行報酬を割引補正して返還し、さらに−0.05 |
| 実際の設置完了 | 設置者本人へ+5、1ラウンド1回 |
| ラウンド勝敗 | 既存設定の+10／−10 |
| オーブ | 既存のまま。回収で実際に増えたULTポイント1点につき本人へ+0.20 |

接近して戻る往復では繰り返し加点しません。回収・設置完了やフェーズ終了による接近報酬のリセット減点も行いません。回収後に実際に再落下した場合は新しい回収対象として扱います。

報酬バージョンは `gc3-preplant-personal-retrieval-v6`。係数は `preplant/gc3_reward_attacker_objectives.py` と `preplant/gc3_rollout_attacker_preplant.py` 冒頭の名前付き定数です。キャンセル時の割引補正により、進行してはキャンセルする反復の割引後合計はマイナスになります。

## 実行

通常設定は各スクリプト冒頭の定数で変更します。作業ディレクトリは `D:\git\ai_dnn\03.game\ghost_champions_v3` です。

BC開始前にゲーム本体と評価処理の読み込みを確認します。読み込みに失敗した場合は、重みの更新やrunの作成より前に停止します。AIの実装は選択されたものだけを読み込み、今回の固定ロジックとのBC評価には旧AI用のgymnasium・matplotlib・TensorBoardを要求しません。

まず、新しいBCを12epoch実行します。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3
python preplant/gc3_train_attacker_retrieval_bc.py
```

`INITIAL_CHECKPOINT=None`、`REQUIRE_FRESH_INITIALIZATION=True`、`RESUME=False` により、旧モデルを読み込まずに開始します。教師データは `DATASET_PATH`、回数は `EPOCHS` で確認できます。学習データとモデルは `data/attacker_preplant_retrieval_bc_v2/`、ログは同名の `logs/` 配下へ保存します。

BC終了後に、新しいBCからRLを累計ST100,000まで実行します。

```powershell
python preplant/gc3_train_attacker_retrieval_rl.py
```

このRLは新しいBCの重みのみを使います。旧BC・旧RL・旧マクロからの開始は拒否します。RLのstepとoptimizerは0から開始します。`TRAINING_STEPS=100000`、`RESUME=False` が初回設定です。同じ新runを後から続ける場合は、このスクリプトの `RESUME=True` に変更します。

RL開始前に、確率サンプリングで28ラウンドを評価します。設置率50%以上かつオーブ回収完了1回以上の既存基準に届かない場合は、PPOを開始せず結果を保存して停止します。BCを続ける場合はBCスクリプトの `RESUME=True` と `EPOCHS` を増やします。RL開始後は凍結したBC参照を変えずに使ってください。

相手AIは固定ロジック `default`。ロスターはFNC・FRC・TYG・OMG・SPS・GRG・GCをシャッフルして回します。設置前は5人共通の1モデル、設置後は凍結した既存モデルです。マクロは使用しません。

RLモデルは `data/attacker_preplant_retrieval_v2/`、ログは `logs/attacker_preplant_retrieval_v2/` に保存します。過去のモデル・教師データ・ログは保持します。ソースや報酬の変更を検出するチェックも維持します。

## グラフと描画

同じ作業ディレクトリで、別のPowerShellから実行します。

```powershell
python tools/gc3_watch_attacker_preplant_metrics.py
```

新しいBCとRLだけを表示し、5秒ごとに更新します。未学習の間は記録待ちになります。旧runへ切り替えて表示しません。`tools/gc3_watch_attacker_objective_metrics.py` も同じ新runを表示します。

```powershell
python tools/gc3_watch_attacker_preplant.py
```

描画の初期選択と通常ゲームのGC v3は、今回のRL best → RL最新 → BC best → BC最新の順に使用します。まだ新しいモデルがなければ「未学習ミクロ」のランダム初期化を使います。描画で「回収修正best」「回収修正最新」「回収修正BC」「回収修正BC最新」を選べます。旧モデルは比較用の明示的な選択肢として残っています。

改善の程度は新しいBC・RLを実行した後に評価します。実装テストや未学習モデルの試合確認は、学習成果の評価ではありません。
