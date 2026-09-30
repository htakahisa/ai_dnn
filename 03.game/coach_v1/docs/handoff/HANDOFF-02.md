# HANDOFF-02: 警戒ポイント設定と検証機能

## 2026-09-24 Task 02 完了記録

### 事前確認

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、`03.DESIGN.md`、`HANDOFF-00.md`、
  `HANDOFF-01.md`、Task 01の共通型・hash・checkpoint実装を確認した。
- 現在の `NEW_MAZE_STR` は26行×44列で、ゲーム上は値`1`だけが移動不能な壁、
  `0/2/3/4/5`は移動可能であることを既存coreから確認した。
- Task 02では観測DTO、Team Perception、belief memory、シナリオ生成を実装しない責任境界を
  維持した。

### 実施内容

- 固定マップ専用の `watch-points-v1` JSON形式を定義し、`watch_points_map.py` の`4`を座標の
  正本として35地点を登録した。
- `watch_points_map.py` をASTで安全に読み込み、`4`の座標からJSONを再生成する
  `convert_watch_points_map.py` を追加した。JSONと同じ座標の既存metadataは保持し、新規座標は
  安定ID・幾何学的に決定的なfacing・初期重要度／タグを生成する。
- 各地点に一意なID、`[row, column]`、重要度1～5、8方向facing、対象陣営、対象状況、
  周辺ランダム化半径1～3、タグを設定した。
- attacker／defenderの両方と、`carry/retrieve/guard/search/retake` の全状況を登録した。
- JSONのroot、map、pointごとの必須fieldを固定し、未知fieldも拒否する厳格なloaderを実装した。
- 固定マップの寸法・SHA-256照合、範囲外、壁上、重複ID、重複座標、無効な方向・重要度・
  陣営・状況・半径・タグ、陣営と状況の不整合を検査する。
- 推奨facingの直近が壁またはマップ外になる設定も拒否する。
- マップへ8方向矢印を重ね、座標・重要度・半径・陣営・状況・タグを一覧表示するCLIを追加した。
  `--side`、`--situation`で絞り込み、`--no-map`で検証とhash表示だけを実行できる。
- 設定形式と検証・表示方法を `config/README.md` に記録した。

現在のhash:

- map SHA-256: `8f90e7c0f596e4f235b5a60d5a2350ca4bfa4ab63d82fb6efa8437f8255d6528`
- watch points canonical JSON SHA-256:
  `991eb802749627f8576faec2ba8335e93b9f3e8033b7a57e8d467979dc5844fc`

検証・表示コマンド:

```powershell
python -m coach_v1.validate_watch_points
python -m coach_v1.validate_watch_points --side attacker --situation guard
python -m coach_v1.validate_watch_points --no-map
```

### 未視認敵情報の監査結果

Task 02の設定・loader・可視化は、固定マップ文字列と静的な警戒ポイントだけを入力とする。
live game、Character、actor/critic観測、敵ID・敵座標を受け取るinterfaceは追加していない。
設定JSONにも敵実情報を表すfieldはない。

自動テストでTask 02実装に `game_state["chars"]`、`PerceivedGameView.real_game`、
`PerceivedCharacter.real_character`、`enemy_position`、`enemy_pos`、`critic_observation` の参照が
なく、設定JSONにenemy fieldがないことを検査した。Task 02ではactor観測自体をまだ生成しないため、
未視認敵の実情報が観測へ混入する経路は追加されていない。未視認敵の実位置だけが異なる二状態の
同一観測テストは、設計どおりTeam Perceptionとactor観測境界を実装するTask 03で行う。

### core変更判断

coreファイルは変更していない。固定マップは読み取り専用入力として検証CLIとテストから参照するだけで、
設定検証・可視化は `coach_v1` 内で完結した。core interface追加や代替案の採用は不要だった。

### 変更ファイル

- `coach_v1/common/constants.py`（既存、設定ファイルpath定数を追加）
- `coach_v1/common/watch_points.py`（新規、型・loader・検証・text可視化）
- `coach_v1/config/watch_points_map.py`（入力、`4`が警戒ポイント。ユーザー作成の入力ファイル）
- `coach_v1/config/watch_points.json`（変換結果、固定マップ用35地点）
- `coach_v1/config/README.md`（設定形式と変換・利用方法）
- `coach_v1/convert_watch_points_map.py`（新規、map.py→JSON変換）
- `coach_v1/validate_watch_points.py`（新規、検証・表示CLI）
- `coach_v1/test_coach_v1_task02_watch_points.py`（新規、自動テスト9件）
- `coach_v1/HANDOFF-02.md`（新規、本記録）

既存変更中の `coach_v1/10.task_template.md` は変更していない。core、既存AI、Task 00／01成果物は、
上記の `common/constants.py` への1定数追加を除いて変更していない。

### テスト結果

- Task 00～02＋関連視認・team回帰:
  `python -m unittest -v coach_v1/test_coach_v1_task02_watch_points.py coach_v1/test_coach_v1_task01_foundation.py coach_v1/test_coach_v1_task00_design.py test_replay_viewer_visibility.py test_team_names.py`
- 結果: 33件すべて成功（`OK`）
- Task 02警戒ポイント: 10件（map.py→JSON座標一致を含む）
  - Task 01基本構成: 11件
  - Task 00設計契約: 7件
  - replay視認回帰: 2件
  - team名／preset回帰: 4件
- `python -m compileall -q coach_v1`: 成功
- `python -m coach_v1.convert_watch_points_map --write`: 成功、35地点を生成
- `python -m coach_v1.validate_watch_points --no-map`: 成功、35地点と上記2 hashを確認
- `git diff --check`: errorなし。既存追跡ファイルのLF→CRLF warning 1件のみ。
- 全体確認 `python -m unittest discover -v`: 196件中189件成功、7件失敗。
  - analytics: 3件（enum表示、planted round集計、環境に`flask`がないimport error）
  - gc_v1: 4件（既存checkpoint shape、teacher用mock、ultimate学習期待、screening action期待）
  - Task 02の9件は全体確認内でもすべて成功した。失敗箇所はTask 02モジュールをimportしておらず、
    今回の変更との依存・変更重複はないため、タスク外として未修正。

### 残課題

- 警戒ポイントは現固定マップ用の初期設定であり、学習・評価結果に基づく追加候補は自動採用せず、
  後続の苦手地点レポートを人が確認して更新する。
- 70%警戒ポイント／20%周辺／10%合法ランダムの配置生成はTask 06の範囲であり未実装。
- Team Perception、壁・smoke・facingを考慮した合法視認、チーム共有、未視認敵二状態比較はTask 03。
- 警戒ポイント最終確認tickを含むbelief memoryはTask 04。
- repository全体の既存7テスト失敗はTask 02の範囲外として未修正。

### 次の推奨タスク

`Task 03: Team Perceptionの実装`。今回の警戒ポイント設定には敵情報を混在させず、専用センサーだけが
実状態を読んで視認資格を判定し、合法に視認された情報だけを `TeamPerceptionSnapshot` へコピーする。
壁越し、smoke越し、後方、チーム共有、未視認敵実位置差分の必須漏洩テストを先に固定する。
