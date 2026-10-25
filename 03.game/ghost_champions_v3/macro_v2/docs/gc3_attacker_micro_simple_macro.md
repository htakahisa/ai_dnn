# 簡単なロジックによるマクロ指示テスト

**この文書は完了済みの撤退なしv3 runの説明です。現在は[撤退練習を追加したv4 run](gc3_attacker_micro_simple_retreat.md)が通常設定です。**

`gc3_train_attacker_micro_directives.py` の `MACRO_DIRECTIVE_MODE='simple'` が通常設定です。マクロ自体は学習せず、下位の指示入力層だけを追加学習します。

1. ラウンドseedの偶数でA、奇数でBを選び、設置予定マスをラウンド中固定します。
2. 設置予定マスは、実際に設置可能でユーザーの5の候補でもある座標を優先して選びます。該当候補がなければ指定済みの設置可能座標から選び、保持者の目的地はそれに最も近い5の候補を使います。
3. スパイク保持者は設置予定マスに最も近い5の候補へ進軍します。味方は、その周囲の別々の5の候補へ進軍し、到着後は待機します。保持者は到着しても進軍指示のまま、ミクロが設置を判断します。
4. ランダムな目的地変更、待機、撤退は出しません。ミクロの経路・戦闘・回収・設置の出力は、従来通り確率的にサンプリングします。設置後は既存ガードのままです。

現在の候補マップは両サイトとも設置可能マスを5で指定しているため、保持者の目的地と設置予定マスが一致します。目的地候補は追加・変更していません。敵の非公開状態で計画を立てません。

評価のbestは勝率→設置率→測定できた指示対応率で選びます。撤退指示はこのテストでは出さないため、撤退習得の評価には使えません。`scenario_instruction_ready` は今回測定した指示の到達基準、`instruction_ready` は3指示全部の基準であり、撤退が未測定なら後者はFalseです。指示対応率のグラフでは未測定の指示を表示しません。

## 実行と設定

通常設定は `gc3_train_attacker_micro_directives.py` 冒頭の名前付き定数です。`DIRECTIVE_CHECKPOINT` は停止した前回の `attacker_micro_directives_v2/latest.pt`、`ADDITIONAL_TRAINING_STEPS=20000`、新runは `attacker_micro_directives_v3` です。重みを引き継ぎ、旧ログ・モデルを上書きしません。BCや教師データ収集はやり直しません。

作業ディレクトリは `D:\git\ai_dnn\03.game\ghost_champions_v3\macro_v2` です。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3\macro_v2
python gc3_train_attacker_micro_directives.py
```

新モデルは `data/attacker_micro_directives_v3/`、ログは `logs/attacker_micro_directives_v3/` に保存します。中断再開は `RESUME=True` にします。`ORDER_DURATION_TICKS`、`ORDER_WARMUP_TICKS`、`ORDER_PROBABILITIES` は検証用の旧randomモードでのみ使い、simpleモードでは使いません。

グラフは同じ作業ディレクトリで別のPowerShellから実行します。

```powershell
python tools/gc3_watch_attacker_directive_metrics.py
```

既定表示はv3で、5秒ごとに更新します。実行中の古いグラフはスクリプトを再起動して新しい選択肢を読み込んでください。通常の実戦モデルは自動で切り替えません。スクリプトの編集・読み込みやテストではユーザーの学習を開始しません。
