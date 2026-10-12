# 2026-10-12：収集サイト不一致とplant依存モデル不一致の修正

## 守備retake収集

GC v2の`preplant()`は`choose_site()`より先に`objective_action()`を呼ぶ。収集側が`choose_site()`を左へ限定しても、期限・設置継続処理は全サイトの設置床から目標を選び、右へ設置できた。収集コンテキスト内で`objective_action()`へ渡す設置床も指定サイトへ絞った。通常対戦のGC v2コード・重み・盤面は変更していない。例外時にも一時差替えを復元する。

保存済みケースは実際の設置サイトで記録され、不一致は保存前に拒否されていた。既存ケース604件（6相手の左右各50件＝600件、GC v2左4件）のファイル・requested_siteとsiteの一致・現行scenario・analysis/searchハッシュを確認した。これらを保持して不足分を収集できる。

`tv3_collect_defender_retake.py`冒頭の`DEFAULT_RESUME=True`を通常設定にした。同条件のcollection.jsonがあれば再開、空ディレクトリなら新規作成する。設定・依存ハッシュが異なる場合は拒否し、ケースを消さない。manifestのない非空ディレクトリも拒否する。新規に取り直す場合だけ`DEFAULT_RESUME=False`にする。

実際のGC v2/search bestを使用し、失敗ブロックのseed6000142で左右各12ラウンドを非学習で実行した。左右とも12設置すべて指定サイトと一致。テストに反対サイトの最寄り床へ期限設置する失敗例と、例外時の復元確認を追加した。

## 攻撃plant

FRC共通plant v16 set10の依存analysisは`dae22732e282113222b69f2bcff32d39066eeea1220b623479554f73791d7680`だった。今回の再学習後analysisは`52533feec0ce8ac23f258d0b5dd3ef35a345980d1258dd0af55b5c9a901d0e69`。旧共通モデルを固定する右専用学習と、新analysisの組合せは互換性がなく、従来のチェックで停止した。保存済みハッシュ・schemaの書換えや検査の無効化は行っていない。

`tv3_train_attacker_plant.py`冒頭に名前付きの学習モードを追加した。

- `PLANT_TRAINING_PROFILE='common'`：通常の全体再学習。`COMMON_TARGET_OPPONENTS`の共通plantを現在のanalysisから学習する。旧右専用拡張・固定共通重み・専用初期モデルを読み込まない。
- `PLANT_TRAINING_PROFILE='custom'`：既存の右専用設定を使用。`PLANT_CUSTOM_MODULES`、`PLANT_COMMON_DIRECTORY`、`PLANT_CUSTOM_INITIAL_CHECKPOINT`、`TARGET_OPPONENTS`、専用5set上限は保持している。共通モデルとanalysisの互換性を引き続き厳密に確認する。
- 共通モードの回数は既存の`TRAINING_SETS=70`、総数上限は`COMMON_TOTAL_TRAINING_SET_LIMIT=None`。専用モードの`TOTAL_TRAINING_SET_LIMIT=5`とは分離した。CLIの対象・総数上限の明示指定は優先する。
- `tv3_train_attacker_all.ps1`冒頭の`$PlantTrainingProfile='common'`をplant段階に渡す。一括analysis再学習後に旧共通plantへ依存する専用学習を実行しない。

新analysisに非対応の旧bestは新学習の重み初期化に使わず、新optimizer/replayで作成する。右専用モデル・guardには旧依存が残るため、新共通plantの採用後はguardケース再収集→guard学習が必要。右専用を再学習する場合も新analysisと共通plantに対応した初期モデルが必要で、旧専用初期モデルの依存不一致を黙って許可しない。

現行analysisを使う共通モードの`--describe`が成功し、対象6相手の拡張なし・固定重み分岐なし・専用初期モデルなしを確認。学習は実行していない。関連単体・統合テスト62件通過、PowerShell構文確認済み。

## 失敗した段階からの実行

analysisとsearchを取り直す必要はない。作業ディレクトリは`03.game/touyama_v3/`。以下はユーザーが順に実行する。各コマンドの成功を確認してから次へ進む。

守備：冒頭`DEFAULT_RESUME=True`のまま収集を続け、完了後にretakeを学習する。

```powershell
python tv3_collect_defender_retake.py
python tv3_train_defender_retake.py
```

攻撃：冒頭`PLANT_TRAINING_PROFILE='common'`のままplantから開始し、終了後にguardを収集・学習する。

```powershell
python tv3_train_attacker_plant.py
python tv3_collect_attacker_guard.py
python tv3_train_attacker_guard.py
```

一括スクリプトは先頭のanalysisから実行するため、今回の復旧には上記の段階別コマンドを使う。
