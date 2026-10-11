# FRC右専用 revision 3：時間切れを最優先にする学習修正

追記：ユーザーの5セット学習が完了。[set5の実力評価](tv3_frc_right_timeout_revision3_set5_review.md)で時間切れ12→5（右12→3）、設置60/72→67/72。今回の修正は改善したが、左2・右3の時間切れが残る。以下の学習未実行は実装時点の記録。

2026-10-12。ユーザーから「時間切れ解消が最優先、全滅より意味がない」という指示を受けて実装した。学習・optimizer更新は未実行。既存の学習済みモデルが時間切れを解消したとは扱わない。

## 確認した停滞

revision2 set5評価の右timeoutは12件。6件は味方3人以上を残しており、うちseed804000142 round11は5v3・HP380、round23は4v4・HP220。残り6件は味方1〜2人。全12件を同一原因と断定していない。

round11の実トレースでは、所持者が51tick以降に11,40へ前進しては12,40へ退避し、recent_edge制限で待機を繰り返した。残り時間が減っても教師が同じ往復を指示した。carrier_blocked_ticksは45。既存の停滞罰は、接敵・退避・教師STAYなどの条件で免除されていた。

終盤に保持者だけを10,40へ押し出し、設置教師を実行する対照は死亡を増やし、2v3 timeoutのままだった。この対照だけを解決策とは扱わない。

## 変更内容

新しい専用ファイルは`tv3_custom_train_attacker_plant_frc_v1_R_v3.py`。revision2のソースはそのまま保持し、旧専用checkpointは旧契約で読める。

- 右目標で、最寄り設置床までの移動＋残り設置作業に対する余裕が12tick以下になった場合、公開敵の射線・記憶・煙から、期限内に到達できる設置床を比較する。特定座標へ固定しない。
- 保持者が選んだ床を味方が占有している場合、終盤に限定して味方の目標を近傍の空き設置床へ変える。自分の目標へ既に到着している味方も対象にし、保持者の目標が同じ間は譲った位置へ戻らない。
- 期限内完遂が物理的に可能なら、その設置床への合法な前進と、床到着後の合法PLANTを教師例にする。退避・接敵教師より期限内完遂を優先する。設置進捗を考慮して残作業を計算する。
- 終盤の保持者のSTAY・能力使用などへ0.25、目標までの距離を減らさない移動へ0.15の追加学習罰を付ける。推論時の行動を強制せず、mask・予約・エンジン禁止は維持する。候補が塞がれていれば不正に通過させない。

公開のbounded threatsと、そのtickに割り当てた保持者の目標をExtensionContextへ追加した。敵の非公開座標・向き・blindは渡さない。入力次元・行動種類・LOS・scenario・配置・analysisは変更していない。

通常学習設定は、時間切れ罰を-10から-20へ変更した（全滅等の失敗は-6）。bestは時間切れ率の低さを最優先にし、次に設置率、同人数以上設置率、人数優位、能力優位を比較する。全滅と設置は別に記録し、時間切れ減少だけを設置成功と呼ばない。

初回専用学習でも、学習前の72ラウンド評価を実施・保存する。学習後に初期評価を上回らなければ、新しい専用bestを自動採用しない。

## 非学習検証

271テスト通過。期限・残り設置進捗、mask、対象外scope、公開射線による設置床選択、停止中の味方の譲り・譲った位置の維持、時間切れ最優先の選定、旧専用重みだけの初期化を確認した。

同じ旧revision2 set5モデルを使い、先行ラウンドは旧処理に固定、対象ラウンドのみ新しい目標hookと期限教師を診断実行した。

| ケース | 旧モデル・旧処理 | 新しい期限教師の診断実行 |
| --- | --- | --- |
| seed142 round11、start1 | 100tick timeout、5v3、HP380 | 90tick設置、5v3、HP380 |
| seed142 round23、start13 | 100tick timeout、4v4、HP220 | 89tick設置、4v4、HP220 |

round11では先行10ラウンド結果と対象開始RNG hashの一致を確認した。教師を実行しない旧重み＋新hookのround11は、依然として5v3 timeout。**学習が必要であり、この診断を学習済み性能や全12件の解決と扱わない。**

記録：`logs/custom_extensions/revision3_complete_seed142_round11_teacher.json`、`revision3_complete_seed142_round11_model.json`、`revision3_complete_seed142_round23_teacher.json`、`revision3_unit_tests.txt`。教師診断は通常運用へ入れた強制ルールではない。

## 次の学習と評価

`tv3_train_attacker_plant.py`冒頭を次の設定にした。

```python
TRAINING_SETS = 5
TOTAL_TRAINING_SET_LIMIT = 5
RESUME_TRAINING = False
PLANT_CUSTOM_MODULES = ('tv3_custom_train_attacker_plant_frc_v1_R_v3',)
TIMEOUT_FIRST_SELECTION = True
PLANT_CUSTOM_INITIAL_CHECKPOINT = HERE/'data'/'best'/'custom_profiles'/'3e2fa49fec166d75'/'frc_v1'/'attacker_plant_best.pt'
```

今回ユーザーが学習したrevision2 set5の右専用重みを初期値として保持する。共通v16は凍結したまま、optimizer/replayは新規、revision3 profileの通算セットは0から。旧学習を再開する操作ではない。入力・scenario・analysis・固定共通の一致を確認したうえで重みだけをコピーし、新しい拡張契約は保持する。SHAや版を書き換えて旧checkpointを新モデルに見せない。

入力元revision2 set5のSHAは`d371eb2cf610ad4a23da5d1092146ed8e72802408d90fe444ce1ff2f4218392a`。共通v16のSHAは`cd9079b8813a14e05edd4751bef61751e48e3aea0a9db688aecab401d369d453`。実ファイルの読込とコピー後の重み一致・共通凍結を検証した。

作業ディレクトリは`03.game/touyama_v3/`。学習はユーザーが実行する。

```powershell
Set-Location C:\Users\ronet\MyProject\git\AI_dnn\03.game\touyama_v3
python tv3_train_attacker_plant.py
```

初期評価とset5の72ラウンド評価で、全体・右timeoutを最初に比較する。設置数、全滅、同人数以上設置、左への影響も併記する。revision2の専用bestと共通bestを上書きしない。新profileのbest選定と通常対戦への採用は別。

追加学習が必要なのは今回のFRC右plant。analysis・defenderの再学習は不要。通常対戦へ採用するときは専用plantに合わせてguardケース・guard依存を更新する。

診断スクリプトは`tv3_review_attacker_plant_frc_v1_R_timeout.py`。同じ作業ディレクトリから`python tv3_review_attacker_plant_frc_v1_R_timeout.py`で非学習の教師対照を実行する。冒頭のSOURCE_CHECKPOINT・TARGET_ROUND・START_ROUND・OUTPUT_PATHを確認し、出力が既にある場合は新しいOUTPUT_PATHを指定する。
