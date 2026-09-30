# FRC専用AI

固定ロスターは `Furina, Lisa, Lohen, Jean, Arlecchino`。
[承認済み設計](DESIGN.md) に沿って、観測、履歴、行動mask、専用controller、
攻守別の共同policy、実ゲームの学習環境、PPO学習・評価コマンドを実装しています。

GUIと大会画面には次の2種類を登録しています。編成は「Furina Classic」を選びます。

| 表示 | controller key | 用途 |
| --- | --- | --- |
| FRC v1（基礎ルール） | `frc_v1_baseline` | Lohen先行・回復・索敵・追撃の教師AI。モデルなしで使用可能 |
| FRC v1（学習モデル） | `frc_v1` | 攻撃 `checkpoints/A_policy.pt`、守備 `checkpoints/D_policy.pt` を読み込む |

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
`--opponent` と `--opponent-ai` で編成と対戦controllerを変更できます。
`--seed` は訓練用、評価用には別の値を使います。

全段階を順に実行するlauncherも用意しています。

```powershell
.\frc_v1\train_curriculum.ps1 -StepsPerStage 10000 -TeacherSteps 512
```

学習環境は `VisualFPSBattle` の移動・効果・戦闘処理を呼び出します。
`stop_after_round` で終了状態を保持し、実ゲームの勝敗で終了報酬を作ります。
通常のGUI・大会・headless試合は従来どおり次ラウンドへ進みます。
各checkpointのmetadataにロスター、マップhash、field順、公開範囲、能力定数を保存し、読み込み時に検証します。

## 評価

```powershell
python -m frc_v1.evaluate --mode baseline --rounds 20 --output frc_v1/evaluation/baseline.json
python -m frc_v1.evaluate --mode learned --rounds 20 --opponents Fnatic2023 "Ghost Champions" "Touyama Gaming" "Omoko Gaming"
python -m frc_v1.smoke --mode baseline
```

`evaluate` は攻守それぞれの独立ラウンドを測定し、勝率・95% Wilson区間とFRCの指標をJSON出力します。
`smoke` は通常の攻守交代を含むフルマッチを完走させます。
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
