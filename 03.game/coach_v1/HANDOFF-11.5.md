# HANDOFF-11.5: Task 11 までの残課題監査

## 2026-09-28 調査範囲と判定

`01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、`HANDOFF-00.md`～`HANDOFF-11.md`（09-A～Eを含む）を確認し、現在の知覚、観測、coordinator、学習環境、モデル、推論 entry point、関連テストと照合した。Task 11.5 の完了条件は残タスク一覧の作成であり、下表は Task 11.6 が更新できる作業台帳とする。`完了` は既存実装または既存の評価・テストで確認できた項目を指し、実戦性能が証明されたという意味ではない。

| ID | 引き継ぎ元 | 残課題・確認結果 | 状態 | 対応時期 |
|---|---|---|---|---|
| A01 | 00, 01 | 警戒ポイント実データ、schema、固定マップ検証と hash。`config/watch_points_map.py`、`common/watch_points.py` と Task 02 テストで対応。 | 完了 | Task 02 |
| A02 | 00, 01, 02 | 合法視認、味方共有、未視認敵二状態比較。`perception/team_perception.py` と Task 03 テストで対応。 | 完了 | Task 03 |
| A03 | 00～04 | clear age、最終目撃と警戒ポイント履歴、round reset。`perception/belief_memory.py` と Task 04 テストで対応。 | 完了 | Task 04 |
| A04 | 00, 01, 03, 04 | 固定 shape と正規化、actor/critic 入力分離。`observation/coach_encoder.py`、`models/coach_model.py` と Task 05/11 テストで対応。 | 完了 | Task 05/11 |
| A05 | 01, 02, 05 | 70/20/10 の合法配置、固定 seed、学習環境への適用。`training/scenario_generator.py` と `training/coach_environment.py` で対応。 | 完了 | Task 06/11 |
| A06 | 01, 03, 04, 05 | 1 tick 1回の5人分 coach 計算、character 観測・能力接続、round reset。`coordinator.py` と Task 10 テストで対応。 | 完了 | Task 10 |
| A07 | 06～09 | 5人それぞれの学習・推論 entry point と独立 checkpoint、凍結 character の実ゲーム接続。Task 09-A～E と Task 11 で対応。 | 完了 | Task 09～11 |
| A08 | 07～09 | ability payload の facing がゲームで適用されない件。`coordinator.py` が検証後の facing を適用し、Task 10 実ゲーム評価は非強制時1130/1130の適用を確認。core 変更不要。 | 完了 | Task 10 |
| A09 | 08, 09, 10 | 固定 STAY/HOLD による実ゲーム評価不足。Task 10 で人工接敵の facing・ability イベントを測定し、Task 11 で学習済み coach の限定局面を評価した。ただし自然な一連の試合での効果は B03 に残る。 | 一部完了 | Task 10/11、残り Task 12～14 |
| A10 | 10, 11 | attacker/defender 別 coach actor/checkpoint、4段階の限定局面、再開。Task 11 の実装・評価・テストで対応。 | 完了 | Task 11 |
| A11 | 11 | 味方占有 mask、終了 tick 報酬、再開時 RNG の不整合。`HANDOFF-11.md` 最終追補と Task 11 テストで修正済み。 | 完了 | Task 11 |
| B01 | 11 | 学習環境は `_move_order()` を自チーム先行に並べ替えるが、実ゲームは spike 保持者優先の `_move_order()` をそのまま使う。相手が先に動く tick では actor が見る味方/敵の合法観測、衝突結果、報酬分布が学習時と異なり得る。事前計算した rollout 観測と actor 実入力を一致させる現在の仕組みを保ちながら、通常順での学習・評価を検証する必要がある。 | **未完了** | **Task 11.6、Task 12 前** |
| B02 | 05 | round/detonation timer は actor DTO・観測にない。Task 12 の plant/post-plant と Task 13 の defuse に公開残時間が必要かを、観測契約と学習対象に照らして決定する。必要なら Task 03/05 境界・観測 version・checkpoint 再学習を一括で扱う。 | **未完了・採否判断** | **Task 11.6、Task 12 前** |
| C01 | 02 | 警戒ポイント追加候補は実戦の苦手地点レポートを人が確認してから更新する。現時点で候補レポートなし。 | 未完了 | Task 16、必要時に固定マップ設定更新と再学習 |
| C02 | 09-A～E, 10 | 自然な移動・配置での各 slot/side の facing、smoke遮蔽と味方への影響、RECON/FLASH の実効果・戦果・勝率への寄与。固定 STAY と人工接敵の数値だけでは判定できない。低精度 slot の追加学習は自然対戦の結果で採否を決める。 | 未完了 | Task 12/13 の1ラウンド評価、Task 14/17 の自然試合・比較評価 |
| C03 | 09-C/D/E | RECON の先行索敵価値、FLASH の投射時間・遮蔽込みの対象品質、くりまるの FLASH 効果を独立指標で検証する。効果イベントの延べ件数だけでは十分でない。 | 未完了 | Task 12/13 の能力を伴う行動、Task 17 の比較評価 |
| C04 | 11 | attacker の plant・護衛・spike回収・post-plant・自然な1ラウンド、残る移動衝突を評価する。 | 未完了 | Task 12 |
| C05 | 11 | defender 5v5 の探索不足、2v2/3v3 の衝突傾向、retake・defuse・自然な1ラウンドを評価する。 | 未完了 | Task 13 |
| C06 | 10, 11 | 通常メニュー登録、setup、陣営交代、round 越え、フルマッチの動作・性能評価。現在の Task 11 限定局面は setup を固定 STAY で通過する。 | 未完了 | Task 14 |
| C07 | 11 | opponent pool と複数の既存 AI・過去 checkpoint に対する頑健性。 | 未完了 | Task 15 |
| D01 | 01～04 | 当時記録されたリポジトリ全体の既存6～7テスト失敗。原因・現時点の再現性は本 Task の coach 専用テストからは判定できず、後続タスクにも担当がない。 | 現状未判定・本 PJ 外の履歴 | リポジトリ全体のテスト整備時に再判定 |

`C01` は Task 16 の苦手地点収集を待つ条件付きの設定改善であり、Task 12 前に警戒ポイントを恣意的に追加しない。`D01` も過去の記録を未確認のまま「解決」とはしない。`A09` の「一部完了」は Task 10/11 の評価実施という事実と、自然な1ラウンドでの性能評価が残ることを分けた判定である。

### Task 11.6 への引き継ぎ

1. **B01**: 実ゲーム順の defender と spike 保持者が先行する attacker/defender の tick を含む、合法観測・rollout・衝突の比較テストを作る。必要な修正は学習環境/coordinator側で検討し、推論側に戦術規則や自動経路探索を増やさない。
2. **B02**: Task 12/13 の目的行動に残時間が必要か決定する。必要な場合のみ合法な公開値を sensor から DTO/encoder へ渡し、version と checkpoint 互換性を更新する。不要と判断した場合も理由をこの台帳に追記して完了にする。
3. B01/B02 を完了と記録してから Task 12 に進む。core 変更が必要と分かった場合は、変更前に理由・影響・非core代替案を提示する。

### 今回の変更と検証

- 新規 `HANDOFF-11.5.md`（本台帳）と `test_coach_v1_task11_5_audit.py`。既存実装、core、正式 checkpoint、既存のユーザー変更 `10.task_template.md` は変更していない。
- 新規テストは、実ゲーム学習環境の両 side で未視認敵を合法な2地点に置いたとき actor grid/vector が不変で critic 専用 truth だけが変わること、learning entry point が `train_*.py` を import しないことを確認する。
- `python -X utf8 -m unittest coach_v1.test_coach_v1_task11_5_audit -q`: **2件成功**。
- `python -X utf8 -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q`: **142件成功**（追加前140件）。

## 2026-09-28 Task 11.6 追記

上表の B01・B02 は `HANDOFF-11.6.md` の実装、v2 再学習、両 side の評価と自動テストにより**完了**した。C01～C07、A09 の自然試合評価、D01 の扱いは上表の予定どおり。次は Task 12。
