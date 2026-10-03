# FRC専用AI

## 実戦用の行動制約

学習済みモデル (`frc_v1`) の推論と、`attack` / `defense` / `match` 段階の PPO ロールアウトには同じ行動制約を適用します。攻撃では左サイトを5ラウンド中3回、右サイトを2回選び、右サイトへの攻撃のうち1回は外周ロングを指定し、もう1回は最短経路を使います。セットアップ中から移動し、Lohen を前に、Furina（スパイク所持者）と支援役を後ろに配置します。Lisa を含む全員の移動は、壁と味方の占有を考慮した経路で補正します。狭い通路では味方を避ける往復を抑え、サイト入口に着いた支援役は奥へ進んでキャリアーの通路を空けます。味方の ASH エリアは進入の妨げとして扱わず、同じ地点への重複投下を避けます。

守備ではセットアップ中から両サイトの入口を見通せる前寄りの位置へ向かい、ラウンドごとに担当と立ち位置を変えます。照準は入口の奥の進入路へ向け、敵を見ても単独で追いません。ミッドで早く敵を確認した場合は Lisa の SMOKE で進行を遅らせ、入口に近づいた敵やサイト付近の味方が倒れた場合には Lisa の SMOKE と Arlecchino の ASH を投げます。スモークは味方やサイトを覆う着弾点を避けます。設置後はスパイクの手前で味方の合流を待ち、2人以上が近づいたらリテイクします。解除中の味方がいる場合は攻撃側の進入口へ照準を向けます。

攻撃側が設置した後は、守備側スポーンからサイトへ入る主な方向を推定し、味方をスパイク周辺の別々の位置に配置して各方向へ照準を向けます。スパイク付近のスモークまたは解除開始の通知を観測すると、Jean が使用可能なら RECON を投げ、近い味方がスパイクへ寄って確認します。解除通知中は遠くで見えた敵に釣られず、スパイク側へ照準を向けます。Lisa が味方側からスパイク付近を覆うスモークを使う行動も抑えます。この判断は公開観測のみを使い、攻撃側 `attack / match` の PPO ロールアウトにも適用します。

この制約を追加する前の checkpoint も読み込めますが、そのモデルは新しい行動分布で学習していません。行動と勝率は別々に確認し、再学習後に `evaluate` と `watch` で評価してください。

固定ロスターは `Furina, Lisa, Lohen, Jean, Arlecchino`。
[承認済み設計](DESIGN.md) に沿って、観測、履歴、行動mask、専用controller、
攻守別の共同policy、実ゲームの学習環境、PPO学習・評価コマンドを実装しています。

GUIと大会画面には次の2種類を登録しています。編成は「Furina Classic」を選びます。

| 表示 | controller key | 用途 |
| --- | --- | --- |
| FRC v1（基礎ルール） | `frc_v1_baseline` | Lohen先行・回復・索敵・追撃の教師AI。モデルなしで使用可能 |
| FRC v1（学習モデル） | `frc_v1` | 攻撃 `runs/selfplay_01/A_policy.pt`、守備 `runs/tactics_finetune_20260930/D_policy.pt` を読み込む |

学習モデルのcheckpointがない場合は理由を表示して停止します。
基礎ルールやランダムな重みへ自動的には切り替えません。
`runs/smoke/` の重みは短時間の動作確認用で、実戦用モデルではありません。

## 観測と行動

画面と共通の `../public_effects.py` で飛翔物の現在位置・表示済み軌跡、警告、発動領域をコピーします。
未通過path、内部着弾先、隠れたowner、警告の内部残時間はactorへ渡しません。
警告の開始を継続して観測できた場合だけ、公開ルールと観測ageから残時間を推定します。
途中から観測した警告は残時間不明のまま渡します。
飛翔物の方向を渡しますが、v1では確定した着弾予測の専用fieldはまだ持たせていません。

敵の目撃はfacing、壁・煙、blind、revealに従います。v1の目撃座標には追加のIQ誤差を掛けません。
味方の有効IQと戦闘能力は入力に含めます。効果の敵味方が観測から判別できない場合はunknownです。
自チームの使用履歴は、ゲームで資源が消費されたことを確認してからownとして記憶します。

盤面grid、味方状態vector、最大64個の効果tokenを入力にします。
token超過時も全警告・領域をgridへ残します。5slotは同じ行動前snapshotを使います。
学習時の全状態criticは別入力で、実戦actorへは渡しません。

DANCEの味方対象、ASHの射程、RAIDの向き、各ウルト、設置・解除の継続/中断、オーブ取得を扱います。
攻撃側の学習モデルには経路制御を併用します。ローエンが先にスポーンを出てから
スパイク所持者が続き、ラウンド中は同じサイトへ向かいます。味方の現在位置は
次の一歩だけを塞ぐものとして扱い、先の通路を恒久的な壁とはみなしません。
付近に敵がいなければローエンとキャリアーの足踏み・逆行を経路制御で補正し、
設置可能なマスではキャリアーに設置を継続させます。
SERENADEは現行ゲームの死亡後自動発動を使います。
BALEMOONはFurina生存中でも合法な行動として残し、使用か温存かを学習対象にします。

## 学習

リポジトリルートから実行します。既存環境のPythonは `D:\git\python\python.exe` です。
次のコマンドは初期課題を始める例であり、十分な実戦性能を保証する学習量ではありません。

```powershell
python -m frc_v1.train --side A --stage threats --steps 10000 --output frc_v1/checkpoints/A_threats.pt
python -m frc_v1.train --side A --stage support --steps 10000 --resume frc_v1/checkpoints/A_threats.pt --output frc_v1/checkpoints/A_support.pt
python -m frc_v1.train --side A --stage match --teacher-steps 512 --steps 10000 --resume frc_v1/checkpoints/A_support.pt --output frc_v1/checkpoints/A_policy.pt
```

守備は `--side D` で別モデルとして学習します。
curriculumは `threats / support / entry / attack / defense / balemoon / match`。
`defense` は守備用です。`--resume` は重みを引き継ぎ、optimizerは各実行で作り直します。
`--imitation-weight 0.03` はPPO中に実際に訪れた局面でも基礎ルールの行動を少量教師として使い、
移動・支援の行動が消えるのを抑えます。launcherは `attack / defense / match` でこれを有効にします。
攻撃側の `attack / match` では、ローエンとスパイク所持者がサイトへ近づく歩数にも小さな報酬を与えます。
守備側の `defense / match` では、初期配置から守備サイト周辺へ進む歩数に小さな報酬を与えます。
攻撃側の `attack / match` と守備側の `defense / match` のPPOロールアウトでは、実戦と同じ行動制約を通してゲームへ行動を渡します。
PPOには補正前のサンプル行動とその確率を保存し、実際の報酬は補正後の行動で計算します。
`--opponent` と `--opponent-ai` で編成と対戦controllerを変更できます。
`--seed` は訓練用、評価用には別の値を使います。

全段階を順に実行するlauncherも用意しています。

```powershell
.\frc_v1\train_curriculum.ps1 -StepsPerStage 10000 -TeacherSteps 512
```

旧モデルを残して新しい候補を作る場合は、`-OutputDir frc_v1/runs/retrain_20260930` を指定します。
このフォルダに攻撃・守備の各段階と最終の `A_policy.pt / D_policy.pt` が保存されます。

学習環境は `VisualFPSBattle` の移動・効果・戦闘処理を呼び出します。
`stop_after_round` で終了状態を保持し、実ゲームの勝敗で終了報酬を作ります。
通常のGUI・大会・headless試合は従来どおり次ラウンドへ進みます。
各checkpointのmetadataにロスター、マップhash、field順、公開範囲、能力定数を保存し、読み込み時に検証します。

## 評価

```powershell
python -m frc_v1.evaluate --mode baseline --rounds 20 --output frc_v1/evaluation/baseline.json
python -m frc_v1.evaluate --mode learned --rounds 20 --opponents Fnatic2023 "Ghost Champions" "Touyama Gaming" "Omoko Gaming"
python -m frc_v1.smoke --mode baseline
python -m frc_v1.smoke --mode learned --checkpoints frc_v1/runs/tactics_finetune_20260930 --opponent "Touyama Gaming" --seed 100003 --output frc_v1/runs/tactics_finetune_20260930/full_match_touyama_seed100003.json
```

`evaluate` は攻守それぞれの独立ラウンドを測定し、勝率・95% Wilson区間とFRCの指標をJSON出力します。
候補モデルを本番checkpointに置き換えず比較するときは、
`python -m frc_v1.evaluate --mode learned --side A --checkpoint frc_v1/runs/repair/A_policy_1024.pt --rounds 20`
のように指定します。評価が改善するまで本番checkpointを更新しないでください。
`smoke` は通常の攻守交代を含むフルマッチを完走させます。
候補checkpointを画面で観戦するには
`python -m frc_v1.watch --checkpoints frc_v1/runs/retrain_20260930 --seed 100001`
を実行します。描画ありの通常試合として攻守交代まで進み、ウィンドウを閉じた後に
候補フォルダの `visual_match.json` へ結果を保存します。
`python -m frc_v1.diagnose --side A --mode learned` で1ラウンドの各ロールの行動回数と
開始・終了位置を表示できます。学習済みモデルが移動を繰り返しても実際には動かない場合などに使います。
警告被害・トレード・孤立死などの詳細な指標は今後拡張します。
現時点の出力には初回接敵へのLohen参加、回復、ASH/BALEMOON使用、契約付与、警告領域滞在を含みます。
初回接敵の参加者は実ゲームのengagementから取得し、同tickに複数人参加した場合も記録します。

効果観測の比較には `--effects all / none / flight / warning` を使います。
通常のスモーク盤面は各構成に共通です。異なる構成で再学習する場合は出力先を分けます。
評価だけで効果入力を消す比較では `evaluate --effects none` のように明示します。

## 検証

```powershell
python -m unittest test_frc_v1 test_frc_v1_learning test_idol_system test_contractor_system test_combat_facing_rules test_replay_viewer_visibility test_game_tick_time coach_v1.test_coach_v1_task10_coordinator
```

情報漏洩、同一tickのsnapshot、DANCE/BALEMOON/SERENADEの実ゲーム挙動、
actorとcriticの分離、checkpoint往復、PPO確率の再計算、終了報酬を確認します。
短時間の更新・評価は配線確認であり、モデルの強さの評価には複数seedでの十分な訓練と対戦が必要です。
