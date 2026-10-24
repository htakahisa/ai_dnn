# 左の目標再配置：修正前後72ラウンド比較

2026-10-12。左の詰まり修正候補を全左ラウンドで有効にし、元の処理と各72ラウンド比較した。対象はfrc_v1、revision3 set5 best。学習・optimizer更新は0、保存モデルのハッシュは評価前後で一致。通常対戦への接続は行っていない。

seedは804000042 / 804000142 / 804000242、開始roundは1 / 13、各ブロック12ラウンド。プリセットはTouyama Gaming。teacher、epsilon、route explorationは0。修正前の67設置・5timeout・有利39・同人数以上55は既存set5評価の集計と一致した。

## 結果

人数条件は設置成功時の味方・敵の生存人数で集計する。「同人数以上」は有利人数と同人数の合計。

| 指標 | 全体・修正前 | 全体・候補 | 左・修正前 | 左・候補 | 右・修正前 | 右・候補 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 評価数 | 72 | 72 | 34 | 34 | 38 | 38 |
| 設置 | 67 | 68 | 32 | 33 | 35 | 35 |
| 有利人数で設置 | 39 | 39 | 29 | 30 | 10 | 9 |
| 同人数で設置 | 16 | 17 | 3 | 3 | 13 | 14 |
| 同人数以上で設置 | 55 | 56 | 32 | 33 | 23 | 23 |
| 不利人数で設置 | 12 | 12 | 0 | 0 | 12 | 12 |
| 時間切れ | 5 | 3 | 2 | 1 | 3 | 2 |
| 設置前の全滅 | 0 | 1 | 0 | 0 | 0 | 1 |

全体の平均残HPは347.78→342.50、右の平均残HPは261.05→251.05。設置時の平均生存人数は全体3.955→3.956、右3.229→3.200。

## 変わった結果と限界

修正の発動はseed804000042 round9の左1件だけ。47tickにslot0の目標を8,3から7,3へ譲り、100tick時間切れ・5v3・HP500が、56tick設置・5v3・HP500に変わった。

同ブロックの後続右ラウンドも結果が変わった。

| round | 修正前 | 候補 |
| --- | --- | --- |
| 10 | 66tick設置、5v3、HP450 | 68tick設置、3v5、HP210 |
| 11 | 100tick時間切れ、3v3、HP220 | 94tick全滅、0v3、HP0 |
| 12 | 66tick設置、3v4、HP220 | 78tick設置、4v4、HP300 |

右では修正処理は適用していない。ただしrolloutはブロック開始時にseedを設定し、12ラウンドを連続実行する。前のラウンドの履歴と乱数の消費が後続に影響し得るため、左の結果変更後も右の結果が必ず一致するとは限らない。この比較だけで後続差分の原因をそれぞれ特定したとは言えない。

記録した最終結果の項目（サイト・設置・tick・生存人数・敵生存人数・残HP・終了理由）では残り68ラウンドに差はなかった。全ラウンドの行動が同一だったという確認ではない。

この72ラウンドでは全体の設置数・同人数以上の設置数は改善し、大幅な悪化は見られなかった。一方、右の有利設置の減少・全滅の増加・残HPの低下はある。未知のseedや他の相手まで悪化がないことは未確認。時間切れ減少2件のうち、設置成功への改善は左1件、残り右1件は全滅への変更である。

## 再確認用ファイルと実行

- 実行スクリプト：`tv3_review_attacker_plant_left_goal_yield_evaluation.py`
- 集計・変更一覧：`logs/custom_extensions/left_goal_yield_72round_comparison.json`
- 修正前の各ラウンド：`logs/custom_extensions/left_goal_yield_72round_comparison_baseline.jsonl`
- 候補の各ラウンド：`logs/custom_extensions/left_goal_yield_72round_comparison_candidate.jsonl`

設定はスクリプト冒頭のSOURCE_CHECKPOINT、EXPECTED_SOURCE_SHA、EXPECTED_SET、EVALUATION_SEEDS、EVALUATION_START_ROUNDS、PRESET、OUTPUT_PATHで確認・変更する。既存結果の上書きを禁止しているため、再実行時はOUTPUT_PATHを新しい名前へ変更する。

作業ディレクトリは`03.game/touyama_v3/`。

```powershell
python tv3_review_attacker_plant_left_goal_yield_evaluation.py
```

単体テスト11件通過。実行ディレクトリは`03.game/`。

```powershell
python -m unittest touyama_v3.test.test_tv3_plant_left_goal_yield
```
