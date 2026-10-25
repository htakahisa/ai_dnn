# アタッカーミクロ：回収・設置報酬の修正

現在の学習は、重みを引き継がない [新しいBC→RLの手順](gc3_attacker_retrieval_fresh.md) を使います。以下は以前のv5と旧runの記録です。

報酬バージョンは `gc3-preplant-objective-completions-v5`。学習はコマンドを実行した時だけ開始する。

## 今回の報酬

| 内容 | 報酬 | 配分 |
|---|---|---|
| 落ちたスパイクを実際に回収 | +1 | 回収者本人 |
| 有効な設置が1tick進行 | +0.25（4tickで合計+1） | 設置者本人 |
| 設置をキャンセル・死亡中断・未完了でラウンド終了 | その試行の進行報酬を割引補正して返還、さらに−0.05 | 設置者本人 |
| 設置完了 | +5、ラウンド内で1回 | 味方全員に同額 |

係数は `preplant/gc3_reward_attacker_objectives.py` 冒頭の名前付き定数。
設置完了は勝敗とは別のイベントとして加点する。既存の実勝利+5／敗北−5、戦闘・能力・オーブ・時間の報酬も維持する。

目的サイトと落下スパイクへの接近は、歩行距離による近さを使い、これまでの最高値を更新した分だけ加点する。
目的サイトは自分たちが選んだサイト内の設置可能マスを使う。目標を決めたこと、所持者が変わったこと、落下したことだけでは加点しない。
同じ場所へ戻って再接近しても加点しない。回収・設置完了やフェーズ終了で接近報酬をゼロへリセットして減点する処理は廃止。
回収後に実際に再落下した場合は新しい回収対象として扱う。単なる滞在や初期所持では回収ボーナスを払わない。

キャンセル時、時刻`t_i`に受け取った進行報酬`b_i`を `b_i / gamma^(t_cancel - t_i)` だけ返す。
このため、割引を含めた同一試行の進行＋キャンセルの合計は、追加ペナルティ分だけマイナスになる。
待機して減点を先送りしても稼げない。次の試行では債務をリセットし、実際の完了では返還しない。
無効な場所でのPLANT、設置要求のみ、敵の設置では加点しない。進行は実際の設置タイマーの増加から判定し、完了tickの増加も数える。
回収は射撃の直前に確認するので、回収した同じtickに死亡してもイベントを取りこぼさない。

本人のイベント報酬と共有報酬は従来のRecorderを通してPPOへ渡し、`reward_components`に本体と同じ割引で記録する。
イベント内訳は `spike_pickup` / `plant_progress` / `plant_cancel` / `plant_complete`。
`episodes.jsonl`には `spike_pickups` / `plant_cancellations` / `plant_completions` も記録する。
観測・モデル構造・行動マスク・確率サンプリングを変更したり、回収・設置を強制したりはしない。

## ミクロの再学習

作業ディレクトリは `D:\git\ai_dnn\03.game\ghost_champions_v3`。
通常設定は `preplant/gc3_train_attacker_objective_rl.py` 冒頭の名前付き定数。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3
python preplant/gc3_train_attacker_objective_rl.py
```

- 既存のOmoko BC `data/attacker_preplant_omoko_bc_site_v1/best.pt`を再利用。教師データ収集・BCを自動でやり直さない。
- 初回は`INITIAL_CHECKPOINT=None`・`RESUME=False`で既存BCから開始した。現在は`RESUME=True`で、同じrunの`latest.pt`（ST22,547）とoptimizer・乱数状態を復元する。
- `TRAINING_STEPS=100000`。ST22,547を含む5人の方策判断数の累計目標。収集する7ラウンドの境界で超過する場合がある。
- `RUN_NAME="attacker_preplant_objectives_v1"`を継続する。新たにBCから学習を始めたり、ST0のbestへ戻したりはしない。
- 相手は固定ロジック`default`、ロスターは7つのメジャーチームをシャッフルして一巡する。
- 学習対象は全員共有の設置前モデル。設置後は既存モデルを凍結して実際の勝敗まで進める。マクロは使わない。

新しいモデルと履歴は `data/attacker_preplant_objectives_v1/`、ログは `logs/attacker_preplant_objectives_v1/`。
報酬が変わったので旧runの直接再開は拒否する。保存済みモデル・データ・ログは書き換えない。
最初に凍結BCを評価し、実設置・オーブ回収の既存ゲートを満たしてからPPOを開始する。

グラフは別のPowerShellで同じ作業ディレクトリから実行する。5秒ごとに新runのログを更新し、旧RLのログへ切り替えない。

```powershell
python tools/gc3_watch_attacker_objective_metrics.py
```

ミクロの描画は同じディレクトリから実行する。学習後は「ミクロ再学習best」または「ミクロ再学習最新」を選ぶ。
保存前の既定表示は以前の設置前ミクロ「勝敗RL best」。旧マクロは明示的な比較用の選択肢として残している。

```powershell
python tools/gc3_watch_attacker_preplant.py
```

## 学習なしの検証

ゲーム全体の作業ディレクトリ `D:\git\ai_dnn\03.game` から実行する。

```powershell
python -m unittest ghost_champions_v3.test.test_gc3_attacker_objective_rewards
```

実エンジンの設置完了、無効PLANT、回収と同tickの死亡、報酬のPPO受け渡し、往復、繰り返しキャンセル、待機後キャンセル、死亡・時間切れ中断を検証する。
