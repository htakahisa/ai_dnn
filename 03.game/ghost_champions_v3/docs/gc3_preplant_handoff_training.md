# GC v3 設置前：実際の勝敗と停止対策

このページは以前のhandoff runの記録です。現在のミクロ再学習は [回収・設置報酬の修正](gc3_attacker_micro_objectives.md) を参照してください。報酬変更後はこの旧runを直接再開しません。

2026-10-12の修正。既存のOmoko教師データとBCを再利用し、設置前RLのlatest（ST32,484）から続ける。
教師データの再収集・BCの再実行は不要。学習はコマンドを実行した時だけ開始する。

## 通常実行

作業ディレクトリは `D:\git\ai_dnn\03.game\ghost_champions_v3`。
通常設定は `gc3_train_attacker_preplant_omoko_rl.py` 冒頭の名前付き定数にある。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3
python gc3_train_attacker_preplant_omoko_rl.py
```

| 定数 | 現在の設定 |
|---|---|
| `RUN_NAME` | `attacker_preplant_omoko_rl_handoff_v3` |
| `INITIAL_CHECKPOINT` | `data/attacker_preplant_omoko_rl_resources_v2/latest.pt` |
| `BC_CHECKPOINT` | `data/attacker_preplant_omoko_bc_site_v1/best.pt` |
| `POSTPLANT_CHECKPOINT` | `data/attacker_postplant_fixed_v1/best.pt`（凍結） |
| `NAVIGATION_FEATURES` | `True` |
| `TRAINING_STEPS` | `50000`（ST32,484を含む累計） |
| `RESUME` | `False`（初回） |
| `FIXED_OPPONENT_AI` | `default`（固定ロジック） |
| `OPPONENT_CODES` | FNC / FRC / TYG / OMG / SPS / GRG / GC |
| `EVALUATE_EVERY_STEPS` | `1000` |
| `BC_KL_WEIGHT` | `0.10` |

初回は別runへ既存の重みを引き継ぐ。報酬と入力処理が変わるためoptimizerは新規。
2回目以降は **`RESUME=True`** にして累計 `TRAINING_STEPS` を変更し、同じコマンドを実行する。
再開時はモデル・optimizer・乱数状態を復元する。コード・入力モデル・通常設定が変わったrunの再開は拒否する。
旧run・教師データ・BC・設置後モデルは上書きしない。
stepはセットアップを含む5人の方策判断数の合計。7ラウンド単位で収集するため目標を超える場合がある。

## 報酬とモデル

- モデルは設置前・設置後の2つ。キャリア・エスコート・リトリーブには分けない。
- 設置しただけで成功+5を与える処理を廃止。設置後は既存の凍結モデルで実際のラウンド終了まで進める。
  勝利+5 / 敗北-5を設置前モデルの最後の判断に返し、設置後に実際に経過したtick数でも割り引く。
  殲滅・時間切れなど設置前に終わった場合は、その実際の結果を1回だけ返す。
  設置後の判断をPPOへ混ぜたり、設置後モデルを更新したりはしない。
- `success` と評価の `success_rate` は実際のラウンド勝利。`plant_rate` は別に表示する。
  bestは実勝率を最優先し、同率なら設置率で選ぶ。過去runの成功率は従来の意味のまま残す。
- キャリアの接近報酬は、自分たちが確率的に選んだ目的サイト内の設置可能マスまでの歩行距離。
  接近・設置進行のpotentialは設置前の終了で閉じる。A/Bの近い方を目的にする報酬には戻さない。
- 設置前の実時計1tickにつき `-0.01` の共有報酬（`battle_time`）。
  移動・待機・回収・設置に共通で、セットアップ中は0。有限の残り時間を学習で考慮させる。
  設定は `preplant/gc3_rollout_attacker_preplant.py` 冒頭の `BATTLE_TIME_COST_PER_TICK`。
- 実オーブ回収で増えたULTポイント1点につき本人へ `+0.20`（`orb_ultimate_gain`）。
  回収途中・要求のみ・敵の回収では加点しない。満タンなら獲得ポイント0なので報酬0。
  戦闘・有効なアビリティの既存報酬も維持する。実際にAIへ渡した報酬内訳をログに残す。
- 目的サイトへの歩行距離と増減、隣接する壁・味方・認識済み敵、移動許可、設置可能マス、
  味方とキャリアの距離、残り時間、本人の直近の停止・往復を、既存の公開観測と過去履歴から導く。
  これらを学習可能な入力として追加し、行動マスク・経路・配置を強制しない。
  追加部分の出力は初期状態で0なので、移行直後の行動分布は従来モデルと同じ。
- 観測の保存形式は従来のGC v3のまま。収集済み教師データも使える。
  敵の非公開位置、教師の内部経路、現在の正解行動、未来の勝敗を入力しない。
  サイト選択とそのcriticには、その判断自身が選んだサイトの情報を入れない。
  全ラウンドの公開観測・実行行動・確定結果を保存し、全履歴へのattentionを続ける。
- ULTポイントは設置後の消費を含む実残量を次ラウンドへ引き継ぐ。新しい文脈群では0から開始する。

## 保存先

| 種類 | 場所 |
|---|---|
| 再利用するOmoko教師データ | `data/attacker_preplant_omoko_site_v1/` |
| 再利用するOmoko BC | `data/attacker_preplant_omoko_bc_site_v1/` |
| 今回の初期重み・以前のRL | `data/attacker_preplant_omoko_rl_resources_v2/`（保持） |
| 修正後RL | `data/attacker_preplant_omoko_rl_handoff_v3/` |
| 修正後RLログ | `logs/attacker_preplant_omoko_rl_handoff_v3/` |
| 通常対戦の全公開履歴 | `logs/match_history/*.jsonl.gz` |

## グラフと実戦描画

同じ作業ディレクトリで、別のPowerShellから通常はオプションなしで実行する。

```powershell
python tools/gc3_watch_attacker_preplant_metrics.py
python tools/gc3_watch_attacker_preplant.py
```

グラフは今回のRLログを5秒ごとに更新。実reward・return・エントロピー・実勝率・設置率・実オーブ回収率を表示する。
新ログがない間は記録待ちとなり、旧runの値へ置き換えない。
BCパネルは再利用する既存BCの記録であり、以前の成功率は当時の定義。
参照先の設定は `tools/gc3_watch_attacker_preplant_metrics.py` 冒頭の `BC_RUN` / `RL_RUN`。
PNG/SVGは `logs/training_graphs/gc3_preplant_omoko_handoff_training_progress.*`。
既に開いているグラフは起動し直す。

実戦描画では **`勝敗RL best` / `勝敗RL最新`** を選ぶ。
新モデルが保存されれば既定で採用し、作成前は旧モデルを表示する。
通常ゲームも新モデルを優先する。比較用に旧モデルの選択肢は残す。
相手は固定ロジックのまま。メジャーチームのメインAIへの切り替えは行っていない。

## 学習しない検証

```powershell
python tools/gc3_verify_attacker_handoff.py
```

凍結した移行モデルで、解除負け・爆発勝ち・設置前時間切れを再生する。
以前の同じseedと試合結果が変わらないこと、実勝敗が報酬へ1回だけ反映されること、
サンプリング時とPPO再計算の確率・valueが一致すること、モデルとファイルを更新しないことを検証する。
記録は `logs/validation/gc3_preplant_handoff_fix.json`。

今回の通常手順では収集やBCを実行しない。初回収集用スクリプトは残しているが、
既存データを使ったBCが必要になった場合は、別runとスクリプト冒頭の設定を確認してから実行する。
