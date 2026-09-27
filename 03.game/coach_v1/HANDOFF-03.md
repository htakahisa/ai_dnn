# HANDOFF-03: Team Perceptionの実装

## 2026-09-24 Task 03 完了記録

### 事前確認

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、`03.DESIGN.md`、`HANDOFF-00.md`～
  `HANDOFF-02.md`、Task 01／02実装を確認した。
- 既存coreの `AbilityLosMixin.check_cell_line_of_sight()`、Bresenham線、壁・smoke規則、
  `reveal_remaining`、blind、facing、spike状態と、既存 `IQPerceptionEngine` の座標誤差処理を
  確認した。
- Task 03は現在tickの合法な知覚生成までとし、clear age、最終目撃履歴、警戒ポイント確認履歴、
  round resetはTask 04へ残した。actor tensor化、critic観測、runtime cacheも実装していない。
- 既存変更中の `coach_v1/10.task_template.md` は変更せず保持した。

### 実施内容

- 生ゲームへ触れる唯一のactor側センサー境界として `TeamPerceptionBuilder` を追加した。
- actorへ渡す `TeamPerceptionSnapshot` と、その構成要素をすべてfrozen dataclass、enum、tuple、
  数値、文字列だけで定義した。live game、Character、legacy perception proxyへの参照は保持しない。
- 固定Gorigons rosterを名前で照合して設計上のslot順へ並べ、死亡者もslotを詰めずに保持する。
- 生存かつ非blindの味方ごとに、既存coreのBresenham壁・smoke LOSへfacing前方半角90度条件を
  加えた通常視認マスクを生成する。壁とsmokeセル自体はclear対象外とした。
- 5人の通常視野を統合し、`currently_visible` と各セルの `visible_viewer_count` を生成する。
- 通常視認できた生存敵、または `reveal_remaining > 0` のRECON共有対象だけについて、
  `EnemySighting` を1敵1件生成する。通常視野はblind、壁、smoke、後方条件をすべて検査し、
  RECONは設計どおりそれらに依存せず共有する。
- 複数viewerが資格を持つ場合は `effective_iq` 最大、同値ならslot最小を選ぶ。視認資格確定後にだけ
  既存 `IQPerceptionEngine` と同じ最大誤差設定で報告座標を生成する。IQによる敵省略は行わない。
- 未視認敵は `enemy_id` と公開済み生死だけを `EnemyPublicState` へコピーし、位置、HP、spike所持、
  距離、方向など現在位置由来のfieldを持たせない。
- 自チームのspike所持slot、公開済みdrop位置、plant状態とplant位置を `SpikeSharedInfo` へコピーした。
  敵spike所持者は視認中でも公開しない。
- LIVEでは `(current_round, live, battle_tick)`、defender setupでは
  `(current_round, defender_setup, defender_setup_ticks_remaining)` のtick keyを生成する。

### 未視認敵情報の監査結果

- 合法な味方状態と公開状態が同一で、未視認敵の実位置とfacingだけが異なる二状態から、
  `TeamPerceptionSnapshot` が完全一致することを自動テストで確認した。
- 同テストでは未視認敵に対してIQ座標補正が呼ばれた時点で失敗するstubも使用し、誤差付きの実座標を
  一旦生成して破棄する経路がないことを確認した。
- 壁越し、smoke越し、後方、blind中に通常目撃・clearが生成されないことを確認した。
- `EnemyPublicState` に位置・HPがなく、spike情報にも敵carrier fieldがないことを確認した。
- 実敵座標への参照は `TeamPerceptionBuilder._sighting_for()` 内の視認資格判定に限定され、資格のない
  敵についてsnapshotへ派生値を出力しない。

### core変更判断

coreファイルは変更していない。既存 `check_cell_line_of_sight()` を読み取り専用で利用すれば、
Bresenham壁・smoke規則を重複実装せず、観測側でfacing条件を追加できた。smoke一覧、blind、
RECON、spike状態も既存公開状態から取得できたため、core interface追加や代替案は不要だった。

### 変更ファイル

- `coach_v1/perception/__init__.py`（新規、Team Perception公開API）
- `coach_v1/perception/team_perception.py`（新規、安全なDTOとセンサー境界）
- `coach_v1/test_coach_v1_task03_team_perception.py`（新規、自動テスト13件）
- `coach_v1/HANDOFF-03.md`（新規、本記録）

既存変更中の `coach_v1/10.task_template.md` と、core、既存AI、Task 00～02成果物は変更していない。

### テスト結果

- Task 00～03＋関連視認・team回帰:
  `python -m unittest -v coach_v1/test_coach_v1_task03_team_perception.py coach_v1/test_coach_v1_task02_watch_points.py coach_v1/test_coach_v1_task01_foundation.py coach_v1/test_coach_v1_task00_design.py test_replay_viewer_visibility.py test_team_names.py`
- 結果: 47件すべて成功（`OK`）。
  - Task 03 Team Perception: 13件
  - Task 02警戒ポイント: 10件
  - Task 01基本構成: 11件
  - Task 00設計契約: 7件
  - replay視認回帰: 2件
  - team名／preset回帰: 4件
- `python -m compileall -q coach_v1`: 成功。
- `git diff --check`: errorなし。
- 全体確認 `python -m unittest discover -v`: 209件中203件成功、6件失敗。
  - analytics: enum整形、planted round集計、環境に`flask`がないimport errorの3件。
  - gc_v1: 既存checkpointのfacing入力shape、teacher用mock、screening action期待の3件。
  - Task 03の13件は全体確認内でもすべて成功した。失敗箇所はTask 03モジュールの変更箇所と
    依存・重複せず、今回以前から存在する範囲のためタスク外として未修正。

### 残課題

- 現在tickのsnapshotだけを実装している。clear age、敵の最終目撃位置／age、警戒ポイント確認tick、
  round reset、死亡敵の履歴処理はTask 04。
- snapshotとbeliefをCNN／非グリッド入力へ変換する固定shape・dtype・正規化はTask 05。
- actor／criticの入力分離を含む各encoderはTask 05、Task 07。
- 1tickに一度だけsnapshotと5人分出力を計算するcoordinator cache、および既存gameへの接続は
  Task 10。
- repository全体の既存6テスト失敗はTask 03の範囲外として未修正。

### 次の推奨タスク

`Task 04: Belief Memoryの実装`。今回のimmutable snapshotだけを入力にして、clear ageと敵最終目撃を
tick単位で更新する。履歴更新時も生ゲームやCharacterを受け取らず、未視認敵の実位置へ自動追従しない
テストを先に固定する。
