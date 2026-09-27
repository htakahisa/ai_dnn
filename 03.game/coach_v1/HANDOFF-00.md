# HANDOFF-00: 仕様確定と設計文書作成

## 2026-09-24 Task 00 完了記録

### 実施内容

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、現在の固定マップ、既存controller実行経路、
  LOS／smoke／facing、通常ability、ULTIMATE／orb、PLANT／DEFUSE、party preset、
  IQ知覚経路を確認した。
- `03.DESIGN.md` を正式なv1実装契約として作成した。
- 行動優先順位、担当モデル、facing視野角、固定ロスター、checkpoint分割、file構成、
  version規則、v1対象外を確定した。
- Task 00の設計契約を検証する自動テストを追加した。

### 確定事項

- 固定ロスターは `Gorigons` の
  `ごりまる, ごんごん, ごんた, くんた, くりまる` とし、この順序をslot 0～4へ固定する。
  ごんごんは既存ゲームに通常active abilityがないタイガーのため、通常ability使用actionを
  maskし、facingだけを学習対象とする。
- 通常視野は正面半角90度（合計180度、境界を含む）とする。
- coachが移動、停止、PLANT、DEFUSE、戦術意図を担当する。
- キャラクター固有モデルがfacingと通常abilityの使用／対象を担当する。
- ULTIMATEとorbはv1対象外とする。
- attacker coachとdefender coachは別checkpoint、キャラクターは1人1checkpointとする。
- actorへ渡すのは専用センサーが作るコピー済みDTO／数値配列だけとする。

### 未視認敵情報の監査結果

coach_v1にはTask 00時点でactor観測実装がないため、coach_v1内部からの漏洩はまだない。
一方、既存の汎用経路はcoach actorの入力として安全ではない。

- `battle_logic.py` がcontrollerへ渡す生の `game_state["chars"]` は敵実座標を含む。
- `IQPerceptionEngine.build_game_view()` は全生存敵をproxy化し、未視認敵も現在座標を
  IQに応じてぼかして残す。これは座標そのものが不正確でも、現在位置由来の情報漏洩になる。
- `PerceivedGameView.real_game` と `PerceivedCharacter.real_character` から実体へ到達できる。
- `PrivateInfoController` は敵のspike所持だけを隠し、敵位置は隠さない。

したがって、これらをcoach actorへ渡すことを `03.DESIGN.md` で明示的に禁止した。
Task 03では、センサー境界だけが実状態を読み、合法に視認できた敵だけを新しい
`TeamPerceptionSnapshot` へコピーする。必須の漏洩テストは、合法履歴が同じで未視認敵の
実位置だけを変えた二状態のactor観測が完全一致することとした。

### core変更判断

coreファイルは変更していない。既存のper-character controller API、action戻り値、LOS情報を
使い、非coreのteam wrapperとcoordinator cacheで接続できるため、現時点でcore変更は不要と
判断した。coreへteam callbackを追加する代替案は影響範囲が大きいため不採用とした。

### 変更ファイル

- `coach_v1/03.DESIGN.md`（新規）
- `test_coach_v1_task00_design.py`（新規）
- `coach_v1/HANDOFF-00.md`（新規、本記録）

### テスト結果

- 実行コマンド:
  `python -m unittest -v test_coach_v1_task00_design.py test_replay_viewer_visibility.py test_team_names.py`
- 結果: 13件すべて成功（`OK`）
- 内訳: Task 00設計契約7件、既存replay視認回帰2件、既存team名／preset回帰4件
- 追加3ファイルの行末空白検査でerrorなし。

### 残課題

- 警戒ポイントの座標とmetadataはTask 02で作成する。
- 実際の `TeamPerceptionSnapshot` と情報漏洩runtime testはTask 03で実装する。
- 観測channelのshapeと正規化値はTask 05で確定する。
- network規模、報酬重み、学習parameterは担当する学習タスクで決める。

### 次の推奨タスク

`Task 01: coach_v1の基本構成作成`。共通定数、型、version、hash、checkpoint metadata、
設定読み込みと最小import testを、今回確定した契約に従って実装する。

## 2026-09-24 設計改訂: Gorigons固定roster

新設された `Gorigons` presetをv1の固定rosterへ採用した。既存checkpointはまだ作成して
いないため、観測・行動versionは `v1` のまま据え置く。今後ロスターを変更する場合は、
`03.DESIGN.md` の互換性規則に従ってversionを上げ、全checkpointを再学習する。
