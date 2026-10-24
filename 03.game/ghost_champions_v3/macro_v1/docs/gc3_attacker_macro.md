# アタッカーマクロ v1

固定ロジック相手の初段はST20,621で完了。次段階は[メジャーチームのメインAIとの継続学習](gc3_attacker_macro_major.md)を参照。
以下は初段の仕組みと実行方法。

既存の設置前・設置後モデルの上に、チーム共通のマクロモデルを置く。
既存モデルは凍結し、マクロだけを新規のPPOで学習する。教師データ収集やBCは実行しない。

マクロはアタッカーの各tickの移動前に1回動く。セットアップ・設置前・設置後を含む。
出力は「現在の目的を維持」「Aを目的にする」「Bを目的にする」の3つで、確率分布からサンプリングする。
初回は未設定なのでA/Bを選ぶ。以降は毎tick自由に切り替えられ、クールダウンや切替罰則はない。
選択した目的サイトは5人が共有し、設置前モデルの従来の目的サイト入力へ渡す。
設置前モデル自身が最初にサイトを選ぶ処理は、マクロ接続時には使わない。

ユーザー指定により、設置後モデルは実際のスパイク位置を使う従来のまま。
設置後もマクロは判断・記録を続けるが、目的サイト変更は設置後モデルの行動を変更しない。
設置後のキル・デス・解除開始・勝敗もマクロの報酬へ反映する。

## 報酬

| 実際のイベント | マクロへの報酬 |
|---|---:|
| 味方のキル | +1 / kill |
| 味方のデス | -1 / death |
| 設置完了 | +10（ラウンドに1回） |
| ラウンド勝利 / 敗北 | +30 / -30（終了時に1回） |
| 設置済みで、敵が一度も実際に解除を開始せず、ラウンドに勝利した | 追加+10（終了時に1回） |

キル・デスは5人の合計をマクロ1モデルへ渡し、人数分重複させない。
最後の追加報酬は設置したラウンドに限る。未設置の殲滅勝利や時間切れでは付かない。
ラウンド勝利が必須。設置後に敗北した場合は、解除開始がなくても追加+10を付けない。
解除を一瞬始めて中断した場合も「開始あり」として記録する。
同じtick中に解除者が倒され、tick末に進捗が0へ戻っても追加報酬は付けない。
実行不能な解除要求だけでは「開始あり」にしない。
接近・時間経過・オーブ・アビリティ・サイト切替の追加報酬はマクロには付けない。
報酬係数は `gc3_reward_attacker_macro.py` 冒頭の名前付き定数。
PPOには通常のgamma割引とGAEを使う。実イベントの未割引合計と、実際にAIへ渡した割引後reward/returnを別に記録する。

## 観測と履歴

GCv3と同じ公開観測を使い、現在の目的と設定後の経過tickも入力する。
IQ・視認制限を通した情報だけを使う。敵の非公開位置や未来の結果を入力しない。
現在の選択を行う前の履歴だけを読み、選択の正解ラベルをその判断へ入れない。
全ラウンドの公開観測・下位行動・過去マクロ判断・確定結果を保存し、全履歴へのattentionを使う。
マクロの追加イベントとencoderキャッシュは独立させ、凍結した下位モデルの履歴へ余分なイベントを混ぜない。

## 学習を開始するコマンド

作業ディレクトリは `D:\git\ai_dnn\03.game\ghost_champions_v3\macro_v1`。
通常設定は `gc3_train_attacker_macro.py` 冒頭の名前付き定数。
実装・import・描画では学習を開始せず、次のコマンドを明示して実行した場合だけ開始する。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3\macro_v1
python gc3_train_attacker_macro.py
```

既定は `TRAINING_STEPS=20000`、`RESUME=False`、`RUN_NAME="attacker_macro_v1"`。
STはマクロの判断回数で0から開始する。設置前モデルのST46,540とは別。
通常は1tickに1判断。死亡して動ける味方がいなくなった設置後も、マクロの判断と報酬記録は続く。
7ラウンド単位で収集するため、目標STを超える場合がある。
2回目以降は `RESUME=True` と累計 `TRAINING_STEPS` を設定し、同じコマンドを実行する。
再開時にはソース・通常設定・下位チェックポイントのハッシュを検証し、異なるrunへの継続は拒否する。

下位モデルは `PREPLANT_CHECKPOINT` のhandoff_v3 best（ST46,540）と、`POSTPLANT_CHECKPOINT` の既存guard best。
両者と相手の重みは更新しない。相手AIは `default` の固定ロジック。
FNC / FRC / TYG / OMG / SPS / GRG / GCのロスターを一巡ごとにランダムな順で使う。
bestは固定評価の指定報酬のラウンド平均を優先し、同率なら実勝率・設置率の順で選ぶ。
保存先は `data/attacker_macro_v1/`、ログは `logs/attacker_macro_v1/`（いずれもmacro_v1内）。
重み・optimizer・乱数状態に加え、全公開履歴を `data/attacker_macro_v1/histories/` に保存する。
イベント報酬内訳は `episodes.jsonl`、PPOの実reward/return/entropyは `training_metrics.jsonl` と `rollout_transitions.jsonl`。

## 実戦描画

マクロ学習のグラフはmacro_v1を作業ディレクトリにして、別のPowerShellで実行する。
現在のマクロrunの実reward・return・エントロピー・実勝率・設置率・追加+10の獲得率を5秒ごとに自動更新する。
新ログがない項目は記録待ちとし、旧設置前モデルの値は表示しない。学習の開始や設定変更は行わない。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3\macro_v1
python tools/gc3_watch_attacker_macro_metrics.py
```

参照run・更新間隔の設定は `tools/gc3_watch_attacker_macro_metrics.py` 冒頭の名前付き定数。
PNG/SVGは `logs/training_graphs/gc3_attacker_macro_training_progress.*` に保存する。

学習モデルが保存された後、GCv3のディレクトリから通常どおり実行する。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3
python tools/gc3_watch_attacker_preplant.py
```

「マクロ best」「マクロ最新」を選ぶ。画面には目的サイト、マクロの判断、切替回数を表示する。
モデルの保存前は従来の設置前モデルを表示する。
通常ゲームもマクロbestが作成されたら上位モデルを使う。
学習時と同じ下位モデルのハッシュを確認するため、別の下位モデルへの無断差し替えは拒否する。

## テスト

学習せず凍結モデルの実試合と報酬を検証する場合は、macro_v1を作業ディレクトリにして実行する。

```powershell
python tools/gc3_verify_attacker_macro.py
```

設定はこのスクリプト冒頭の名前付き定数。新しい学習モデルや教師データは作成しない。
検証結果は `logs/validation/gc3_macro_frozen_probe.json`。

作業ディレクトリを `D:\git\ai_dnn\03.game` として実行する。

```powershell
python -m unittest discover -s ghost_champions_v3/macro_v1/test -p test_gc3_attacker_macro.py
```
