# 左の入口占有連鎖：目標再配置の修正確認

2026-10-12。ユーザー指示により左の詰まりだけを修正確認した。右専用のコード・学習設定・保存モデルは変更していない。学習・optimizer更新も行っていない。

## 修正候補

`tv3_learn_attacker_plant_left_goal_yield.py`に、公開の味方位置と目標から占有の連鎖を調べる処理を追加した。通常のcontrollerへの接続はまだ行わず、診断スクリプト内でだけ目標割当へ適用して検証した。

- 保持者の左設置目標を味方が占有している状態が3tick続いた場合に限定する。集合待ち・setup・右サイト・設置中は新たな再配置を行わない。
- 占有者自身の目標と最短経路の次の床をたどり、通路を塞いでいる味方の連鎖を調べる。
- 既に自分の目標に到着している味方を優先し、2歩以内の到達可能な床へ目標を変える。現在の占有と全員の目標を避ける。設置床が埋まっている場合は近傍の通常床も候補とする。右設置床は使わない。
- 現在機能している公開敵への射撃支援を移動対象にせず、公開被射線を増やす候補も除く。
- 譲った位置の目標は保持者が入口を通るまで維持する。1tickで元の目標へ戻って塞ぎ直すことを避ける。

座標・選手名・対象roundに依存した条件は修正候補へ入れていない。実行する移動・停止・設置は既存ネットワークが選ぶ。maskや予約、壁、移動阻害などの禁止を解除しない。

## 同じ失敗場面での非学習確認

入力モデルはrevision3 set5の専用best、左側のパラメータ・encoderは同checkpointに固定された共通v16。固定analysis、seed804000042、start_round1、target_round9。先行8ラウンドは元の処理に固定し、結果・対象開始RNGの一致を確認した。

| 条件 | 結果 |
| --- | --- |
| 元の処理 | 100tick timeout、5v3、HP500 |
| 左目標再配置の修正候補 | **56tick設置、5v3、HP500** |

47tickに入口の占有連鎖を検出し、slot0の目標を8,3から7,3へ変更した。この座標はその場の空き・未割当・到達可能性から選ばれた結果で、固定値ではない。行動の直接上書きは0件。モデルが目標変更に応じて移動し、その後の味方と保持者の通路が空き、設置まで完了した。

成功していた同seed round1でも診断した。再配置イベント0件、元と同じ59tick・5v3・HP460設置。これは1成功場面の回帰確認であり、左全体の不変や全詰まり解消の証拠ではない。

単体テスト11件通過。占有連鎖、直接の占有者に出口がないケース、待機閾値、譲った目標の維持・解除、右/setup/集合待ちの対象外、設置中、blind、既存射撃支援、公開被射線増加、round・tick欠落リセット、右への切替時の状態解除、controller間の状態分離を確認した。

その後、既存評価と同じ条件で修正前後各72ラウンドの比較を実施した。結果は [72ラウンド比較](tv3_left_goal_yield_72round_review.md) を参照。

## ファイルと実行

- 修正候補：`tv3_learn_attacker_plant_left_goal_yield.py`
- 検証スクリプト：`tv3_review_attacker_plant_left_goal_yield.py`
- テスト：`test/test_tv3_plant_left_goal_yield.py`
- 失敗場面の検証結果：`logs/custom_extensions/left_goal_yield_seed042_round9_final.json`
- 成功場面の回帰結果：`logs/custom_extensions/left_goal_yield_seed042_round1_regression.json`

通常の検証設定はスクリプト冒頭のSOURCE_CHECKPOINT・SEED・START_ROUND・TARGET_ROUND・OUTPUT_PATHで変更する。OUTPUT_PATHが既に存在する場合は、新しいファイル名を指定する。

作業ディレクトリは`03.game/touyama_v3/`。

```powershell
python tv3_review_attacker_plant_left_goal_yield.py
```

単体テストは`03.game/`から実行する。

```powershell
python -m unittest touyama_v3.test.test_tv3_plant_left_goal_yield
```

通常実行へ適用する際は、既存共通v16の保存契約を黙って変えず、左目標処理の識別と保存・復元を設計する必要がある。今回のモデル・analysis・scenario・LOSの保存値は変更していない。通常対戦への採用や全体再学習は今回の確認に含めていない。
