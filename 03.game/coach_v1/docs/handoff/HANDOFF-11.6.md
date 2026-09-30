# HANDOFF-11.6: Task 12 前の残課題解消

## 2026-09-28 対応範囲

Task 11.5 の台帳で Task 12 前に必要とされた **B01** と **B02** を対応した。`01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、Task 11/11.5 HANDOFF、知覚・観測・coordinator・学習環境・評価経路を確認してから変更した。core ファイルは変更していない。既存の `10.task_template.md` のユーザー変更と、v1 正式 checkpoint は変更していない。

| ID | 結果 | 状態 |
|---|---|---|
| B01 | 学習環境がゲームの `_move_order()` を保持する。先行する相手の移動後に coach 観測を作り、同じ tick の残りを元の順序で処理する。rollout と coordinator の実入力を照合し、敵が空けたマスへの移動・衝突をテストした。 | 完了 |
| B02 | 公開 round/detonation 残時間を sensor DTO にコピーし、有効な方だけを coach 観測 v2 の2特徴に正規化して追加した。両 side を v2 で4段階再学習した。旧 v1 checkpoint は旧観測で読め、character 観測は106特徴の v1 契約を保つ。 | 完了 |

### 観測・checkpoint 境界

- `coach-observation-v2` は 27 grid channel、86 vector 特徴。追加した `round_time_remaining` は設置前、`detonation_time_remaining` は設置後だけ有効で、他方は0。基準はゲームの `ROUND_DURATION_TICKS` と `SPIKE_DETONATION_TICKS`。ゲーム内で公開される timer だけを sensor から渡す。
- `coach-observation-v1` は84特徴のまま保持。checkpoint metadata を厳密照合して該当 encoder を選び、coordinator も actor に合う encoder を使用する。character checkpoint は v1 の共有観測部分を使い続け、再学習を要しない。
- v2 trainer の既定保存先は `checkpoints/coach/{side}/observation_v2/`。異なる観測 version の `latest.pt` の上書きを拒否する。既定の推論 loader は従来の正式 v1 checkpoint を指す。v2 を使う場合は checkpoint path を明示する。正式昇格は Task 12/13 の目的行動・自然ラウンド評価後に判断する。
- actor checkpoint には critic weight、optimizer、敵実位置を含めず、学習状態を `training_latest.pt` に分離した。v2 の両 actor ファイルも `coach-observation-v2` metadata と actor 専用 payload を確認した。

### v2 再学習と限定局面評価

固定 map、警戒ポイントの既存 70/20/10 シナリオ、seed 11、各 episode 最大25 live tick、rollout 保存なし。Task 11 と同じ段階数で新規学習し、attacker は `2v1:30 → 2v2:20 → 3v3:50 → 5v5:90`（計190 episode、380 update）、defender は `2v1:30 → 2v2:50 → 3v3:30 → 5v5:90`（計200 episode、400 update）。各 side の actor/training checkpoint は `checkpoints/coach/{attacker,defender}/observation_v2/` に保存した。旧 v1 正式 actor の SHA-256 はそれぞれ `6a4b7a3257e38f6b297d4c0fdbe5971064f8158eca6435f2b8c697e2ab08c1b6`、`426794b4d775360d401e99555aa5407672a238e013a16dadc1828deb00126c08` のまま。

未使用 seed 100～109 の10 episode、最大25 tick で合法ランダム coach と比較した。下表は学習済み / ランダムの順。`round_win` は一部を除いて0であり、勝率や自然な1ラウンド性能を示すものではない。各 JSON は `reports/task11_6_{side}_{stage}.json`。

| side | stage | 報酬 | 新規クリアマス | 無効移動 |
|---|---|---:|---:|---:|
| attacker | 2v1 | 0.0540 / 0.0442 | 27.0 / 26.6 | 0.0 / 0.4 |
| attacker | 2v2 | 0.0540 / 0.0494 | 27.0 / 26.2 | 0.0 / 0.3 |
| attacker | 3v3 | 0.0552 / 0.0382 | 27.6 / 24.3 | 0.0 / 1.0 |
| attacker | 5v5 | 0.0770 / 0.0182 | 38.5 / 22.1 | 0.0 / 2.6 |
| defender | 2v1 | 0.4328 / 0.1536 | 130.0 / 33.3 | 0.0 / 0.7 |
| defender | 2v2 | 0.2470 / 0.0498 | 138.5 / 34.5 | 0.0 / 0.7 |
| defender | 3v3 | 0.1772 / 0.0420 | 119.5 / 32.7 | 0.7 / 1.1 |
| defender | 5v5 | 0.1834 / 0.0038 | 124.9 / 38.1 | 1.8 / 4.5 |

attacker の2v1～3v3はランダムとの差が小さい。defender 3v3/5v5 の無効移動も残る。両方とも Task 12/13 で目的行動と自然なラウンドを学習・評価する必要がある。今回の限定局面では設置後残時間の戦術的な使い方をまだ検証できない。

### 自動テスト・情報境界

- 新規 `test_coach_v1_task11_6_order.py`: spike 保持者優先を含む両 side の元の移動順、1 character 1回処理、rollout と actor 実入力の一致、先行した敵が空けたマスへの移動を確認。
- 新規 `test_coach_v1_task11_6_clock.py`: 両 side の設置前後 timer、v1/v2 checkpoint 経路、旧 checkpoint 上書き防止、character の106特徴維持、設置後の未視認敵位置を2地点に変えた actor grid/vector の不変性を確認。critic 専用 truth のみ変化する。Task 11.5 の設置前・両 side の同等テストも成功。
- `python -X utf8 -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q`: **148件成功**。`python -X utf8 -m compileall -q coach_v1` と `git diff --check` も成功。
- 旧 v1 正式 checkpoint を明示した短い実ゲーム評価は両 side とも成功した。新 v2 checkpoint の8件の限定局面評価も成功した。

### 変更ファイル

- 移動順・学習と評価: `training/coach_environment.py`、`training/coach_trainer.py`、`evaluate_task11.py`、`diagnose_task11.py`。
- 公開観測・互換性: `perception/team_perception.py`、`perception/__init__.py`、`observation/coach_encoder.py`、`observation/character_encoder.py`、`common/versions.py`、`common/checkpoint.py`、`models/coach_model.py`、`coordinator.py`、`learning_coach.py`、`learning_character_gongon.py`。
- テスト: 新規 `test_coach_v1_task11_6_order.py`、`test_coach_v1_task11_6_clock.py`。既存の `test_coach_v1_task01_foundation.py`、`test_coach_v1_task05_coach_encoder.py`、`test_coach_v1_task11_coach.py` は v2 の期待値に更新。
- 記録と生成物: 本 HANDOFF、`HANDOFF-11.5.md` への完了追記、両 side の `observation_v2/{latest.pt,training_latest.pt}`、8件の `reports/task11_6_*.json`。作業開始前の `10.task_template.md` 変更は触れていない。

### 残課題と次の推奨タスク

Task 12 前の B01/B02 は完了。Task 11.5 の C01～C07、A09 の自然試合評価、PJ 外の履歴 D01 は台帳どおり後続で扱う。次は **Task 12: attacker coach の plant・護衛・spike 回収・post-plant と自然な1ラウンド**。`checkpoints/coach/attacker/observation_v2/` を指定して再開し、時間が異なるシナリオを含めて学習する。続く Task 13 は defender の retake・defuse・1ラウンドと3v3/5v5衝突を扱う。旧 v1 と新 v2 の限定局面報酬を、観測と移動順が異なる実験の直接比較として扱わない。
