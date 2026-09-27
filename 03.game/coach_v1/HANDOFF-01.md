# HANDOFF-01: coach_v1の基本構成作成

## 2026-09-24 Task 01 完了記録

### 実施内容

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、`03.DESIGN.md`、`HANDOFF-00.md`、
  現在の固定マップ、`Gorigons` preset、既存checkpoint実装を確認した。
- `coach_v1` と `coach_v1.common` をimport可能なpackageとして作成した。
- 固定マップ寸法、固定ロスターとslot順、移動差分、facing順、coach／character別の
  checkpoint・log・report保存先を定義した。
- side、model family、facing、移動、目的行動、戦術意図、roster slot、model targetの
  共通型を定義した。
- checkpoint schema、coach／characterの観測versionと行動versionを確定値で定義した。
- 改行差だけを正規化するmap SHA-256、canonical JSON SHA-256、UTF-8 JSON設定読込を
  実装した。設定読込ではroot型、重複key、NaN／Infinityを検査する。
- checkpoint metadataの生成、厳格なkey／型／hash／roster／target検証、互換性検証、
  model stateとoptimizer再開情報を含む最小payload生成を実装した。
- import時にディレクトリ作成やtorch読込を行わず、trainerが明示的に開始したときだけ
  出力先を作成できるようにした。

### 未視認敵情報の監査結果

Task 01ではactor観測、知覚DTO、encoderをまだ実装していない。今回追加した共通型と
checkpoint metadataには敵座標、実ゲーム、Character、critic観測を保持するfieldがなく、
`game_state["chars"]`、`PerceivedGameView.real_game`、
`PerceivedCharacter.real_character`への参照もない。

自動テストで `coach_v1/common` に上記実体参照、`enemy_position`、critic観測参照がないことを
検査した。実状態を使う視認資格判定と、合法な情報だけをコピーする
`TeamPerceptionSnapshot`、および二状態比較による本格的な漏洩テストはTask 03の範囲である。

### core変更判断

coreファイルは変更していない。Task 01は独立した定数、型、hash、metadata、設定読込だけで
完結し、既存AIやcontrollerとの接続を必要としないため、core変更も代替案の採用も不要だった。

### 変更ファイル

- `coach_v1/__init__.py`（新規）
- `coach_v1/common/__init__.py`（新規）
- `coach_v1/common/constants.py`（新規）
- `coach_v1/common/types.py`（新規）
- `coach_v1/common/versions.py`（新規）
- `coach_v1/common/hashing.py`（新規）
- `coach_v1/common/checkpoint.py`（新規）
- `test_coach_v1_task01_foundation.py`（新規）
- `coach_v1/HANDOFF-01.md`（新規、本記録）

既存の変更済みファイルとTask 00成果物は変更していない。

### テスト結果

- Task 01＋関連回帰:
  `python -m unittest -v test_coach_v1_task01_foundation.py test_coach_v1_task00_design.py test_replay_viewer_visibility.py test_team_names.py`
- 結果: 24件すべて成功（`OK`）
  - Task 01基本構成: 11件
  - Task 00設計契約: 7件
  - replay視認回帰: 2件
  - team名／preset回帰: 4件
- `python -m compileall -q coach_v1`: 成功
- 今回の追加ファイルの行末空白検査はerrorなし。`git diff --check`は既存変更中の
  2ファイルにGitのLF→CRLF warningのみ。
- 全体確認 `python -m unittest discover -v`: 187件中181件成功、6件失敗。
  失敗は既存のanalytics 3件、gc_v1 3件で、今回追加したpackageへの依存や変更箇所はない。
  そのうち1件は実行環境に`flask`がないことによるimport error。他5件は既存変更中の実装と
  既存期待値の不一致である。Task 01の11件は全体確認内でもすべて成功した。

### 残課題

- 警戒ポイントのJSON実データ、schema、座標・重複検証、可視化、実設定hashはTask 02で作る。
- `TeamPerceptionSnapshot`と実ゲームからの安全なコピー境界、未視認敵二状態比較はTask 03で作る。
- belief memoryはTask 04、actor／critic encoderの実観測shapeはTask 05以降で作る。
- modelとtrain／learning entry pointは担当する後続タスクまで作成しない。
- repository全体の既存6テスト失敗はTask 01の範囲外として未修正。

### 次の推奨タスク

`Task 02: 警戒ポイント設定と検証機能`。今回追加した`load_hashed_json_config()`と
`canonical_json_sha256()`を使用し、現在の固定マップ専用データとして座標、方向、重要度、
陣営、状況、タグを定義・検証する。

## 2026-09-24 追記: テスト配置変更への対応

Task 00／Task 01のテストをリポジトリ直下から `coach_v1/` 配下へ移動したため、テスト内の
設計書、ログ保存先、commonモジュール検査の相対パスを新配置基準へ更新した。実装本体、core、
既存AIは変更していない。

再確認コマンド:
`python -m unittest -v coach_v1/test_coach_v1_task00_design.py coach_v1/test_coach_v1_task01_foundation.py`

結果: 18件すべて成功（Task 00: 7件、Task 01: 11件）。
