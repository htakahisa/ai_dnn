# HANDOFF-04: Belief Memoryの実装

## 2026-09-24 Task 04 完了記録

### 事前確認

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、`03.DESIGN.md`、`HANDOFF-00.md`～
  `HANDOFF-03.md`とTask 01～03実装を確認した。
- Task 03の `TeamPerceptionSnapshot` が、通常視野、viewer数、合法な敵目撃、公開済み生死だけを
  不変DTOとして保持することを確認した。
- Task 04はそのsnapshotだけから履歴を更新する層とし、生ゲーム、`Character`、
  `game_state["chars"]`、critic情報を受け取らない境界を維持した。
- 最終的な観測channel、shape、dtype、正規化上限はTask 05の責任であるため、Task 04ではraw tick値と、
  呼び出し側が上限を明示する汎用的なage正規化関数までを実装対象とした。
- 既存変更中の `coach_v1/10.task_template.md` と `omoko_v1` 配下は変更せず保持した。

### 実施内容

- `BeliefMemory` を追加し、`TeamPerceptionSnapshot` だけを受け取る `update(snapshot)` APIを実装した。
- 現在視認マップと現在viewer数を保持し、歩行可能かつ合法的にclearされたセルについて、
  最終clear確認memory tickとclear経過tickを更新する。
- 同じセルを再確認するとclear ageを0へ戻し、未確認中は入力tick差の分だけageを増加させる。
- 敵ごとに最終目撃報告位置、最終目撃memory tick、経過tick、目撃方向を保持する。
  目撃方向は合法な報告位置と報告viewerのsnapshot内味方位置から8方向へ量子化する。
- 敵最終目撃地点をグリッド化した `last_seen_enemy_count` と、同一セルでは最も新しい報告のageを使う
  `last_seen_age` を生成する。
- 未視認中は最後の合法な報告位置を維持し、敵の現在実位置へ追従しない。再び合法な
  `EnemySighting` が届いた時だけ位置を更新する。
- 公開済み生死が死亡へ変化した敵は、古い目撃位置、tick、age、方向を消去し、最終目撃マップからも
  除外する。
- 警戒ポイントのセルが通常clearマスクに入った時だけ最終確認memory tickを更新し、確認ageを保持する。
  RECON目撃だけでは警戒ポイントをclear扱いにしない。
- round番号が進んだ時はclear、敵目撃、警戒ポイント履歴を自動resetする。明示的な `reset()` も追加した。
- live tickの増加、defender setup countdownの減少、setupからliveへの遷移を、単調増加する
  round内 `memory_tick` へ変換した。同一snapshotの再入力は冪等、同じtickで内容が違う入力、
  tick逆行、side混在、敵ID集合変更は入力エラーとする。
- actorへ渡す結果は `BeliefSnapshot`、`EnemyBelief`、`WatchPointBelief` のfrozen dataclassと、
  tuple、enum、数値だけで構成した。
- `normalize_age(age, maximum_age_ticks)` を追加した。0～1へclipし、未確認の `None` は1.0とする。
  未確認と古い履歴の区別は、位置／確認有無を表す別fieldと組み合わせて維持できる。

### 未視認敵情報の監査結果

- `BeliefMemory.update` の引数は `snapshot` だけで、生ゲームやcharacter引数を持たない。
- snapshotに現在目撃がないtickでは、保存済みの合法な最終目撃位置とageだけを更新する。
- 合法な履歴が同じ2つのmemoryへ同一の未視認snapshotを渡した場合、actor側の
  `BeliefSnapshot` が完全一致するテストを追加した。
- Task 03の「未視認敵の実位置だけが異なる2状態から同一snapshotを生成する」テストと合わせ、
  未視認敵の現在座標、HP、距離、方向、spike所持、実体参照がTask 04へ混入する経路がないことを確認した。

### core変更判断

coreファイルは変更していない。Task 03の安全なsnapshotを履歴入力に使うことで必要な情報をすべて
更新できたため、core interface追加も代替案の採用も不要だった。

### 変更ファイル

- `coach_v1/perception/belief_memory.py`（新規: belief DTO、履歴更新、reset、age正規化）
- `coach_v1/perception/__init__.py`（Task 04公開APIを追加）
- `coach_v1/test_coach_v1_task04_belief_memory.py`（新規: 自動テスト12件）
- `coach_v1/HANDOFF-04.md`（新規: 本記録）

### テスト結果

- Task 00～04関連回帰:
  `python -m unittest -v coach_v1/test_coach_v1_task04_belief_memory.py coach_v1/test_coach_v1_task03_team_perception.py coach_v1/test_coach_v1_task02_watch_points.py coach_v1/test_coach_v1_task01_foundation.py coach_v1/test_coach_v1_task00_design.py`
- 結果: 53件すべて成功（`OK`）。
  - Task 04 Belief Memory: 12件
  - Task 03 Team Perception: 13件
  - Task 02警戒ポイント: 10件
  - Task 01基本構成: 11件
  - Task 00設計契約: 7件
- `python -m compileall -q coach_v1`: 成功。
- `git diff --check`: errorなし。既存ファイルのLF→CRLF warningのみ。
- 全体確認 `python -m unittest discover -v`: 222件中216件成功、6件失敗。
  - analytics: enum整形error、環境に`flask`がないimport error、planted round集計failureの3件。
  - gc_v1: 既存checkpointのfacing入力shape error、teacher用mock error、screening action failureの3件。
  - Task 04の12件は全体確認内でもすべて成功した。6件は先行HANDOFFにも記録されたTask 04外の
    既存範囲であり、今回の新規モジュールと依存・変更箇所が重ならないため未修正。

### 残課題

- beliefをCNN用グリッドchannelと非グリッドvectorへ変換するshape、dtype、presence mask、
  正規化上限の確定はTask 05。
- critic専用の敵実情報はTask 07以降でactor入力と別builder／別APIとして実装する。
- 1tickに一度だけTeam PerceptionとBelief Memoryを更新し、5人分のcoach出力をcacheする接続はTask 10。
- repository全体の既存6テスト失敗はTask 04の範囲外として未修正。

### 次の推奨タスク

`Task 05: coach観測エンコーダー`。Task 03の現在snapshotとTask 04のbelief snapshotを、固定shapeの
CNN channel／非グリッドvectorへ変換する。未確認を示すpresence maskとage値を分け、正規化上限、
dtype、channel順、固定ロスター順を観測version契約としてテストで確定する。
