# HANDOFF-09-E: くりまる（slot 4）

## 2026-09-27 実施結果

### 対象と能力

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、Task 09-A/B/C/D/10 の引き継ぎと既存実装を確認した。くりまるの通常能力は FLASH。ゲームの能力処理では、発射可能な対象への投射で1チャージを消費し、爆発時に遮蔽のない敵を盲目にする。くんたと同じ能力契約だが、slot 4 固有 checkpoint を使う。
- `training/kurimaru_rollout.py` で既存 coordinator と固定 STAY/HOLD coach に接続した5v5観測を収集する。敵の実位置は局面の配置だけに使い、actor には `CharacterObservation` だけを渡す。facing 教師は現在の合法なチーム共有目撃から45度以内の全方向。FLASHの暫定教師は共有報告位置に最も近い合法投射先。目撃なしの5v5局面では facing 損失を0、FLASHを不使用とする。
- `train_character_kurimaru.py` と `learning_character_kurimaru.py` の対応は既存のまま。core、coach、他キャラ checkpoint、推論側の戦術ロジックや経路探索は変更していない。作業前から変更されていた `10.task_template.md` は保持した。

### 学習と採用

- 旧正式 `best.pt`（epoch 76）と `latest.pt`（epoch 80）を `checkpoints/experiments/task09e_kurimaru/baseline_best.pt`、`baseline_latest.pt` に退避し、各正式ファイルとの SHA-256 一致を確認した。
- 警戒ポイント由来1400例と合法な5v5補助600例を使用し、前者を70%とした。前者の敵配置は既存の70/20/10 sampler。5v5補助は両陣営の近距離・西・東・挟撃・自然配置から各60例。訓練 seed 500–511、validation 600–604、凍結5v5 holdout 700–709。旧 checkpoint から epoch 81/86/96 の候補を作った。
- 最初の全層追加学習は凍結観測の facing が向上したが、実ゲームの FLASH 敵効果が減った。次の facing head のみの追加学習は能力重みを完全に保持したが、警戒ポイント分布の複数群と defender の実ゲーム facing が後退した。いずれも正式採用しなかった。
- 最終候補は旧モデルの**同一の合法 actor 観測**に対する FLASH 使用・対象出力を能力教師として用い、共有目撃に基づく facing 教師とともに追加学習した。`experiment_task09e_kurimaru_facing_only.py --distill-ability` で再現できる。選択元は `checkpoints/experiments/task09e_kurimaru/ability_distill/epoch86_latest.pt`。これをくりまるの正式 `best.pt` と `latest.pt` の両方へコピーし、3ファイルの SHA-256 一致を確認した。ほかのキャラクターの正式 checkpoint は更新していない。

### 独立観測での検証

- 凍結5v5 holdout の合法な共有目撃あり743例で、facing許容方向率は旧 **39.0%** → 採用候補 **84.9%**。共有報告に最も近い合法対象という暫定教師との完全一致は **18.5%** → **14.8%**。後者は戦術的な FLASH 効果を表さず、旧モデルの能力出力を保持した候補では採用指標にしなかった。
- 独立 seed 4004 の警戒ポイント由来1200例と seed 5004 の2400例で、attacker/defender × point/jitter/合法random の6群すべてにおいて候補の facing 許容方向率が旧版以上だった。後者の旧→候補は attacker point 60.5→64.0%、jitter 58.7→61.9%、random 58.7→62.2%、defender point 59.5→63.5%、jitter 62.4→65.7%、random 51.2→52.0%。能力使用判断は各群で両者100%。同じ2400観測に対する能力使用は **2400/2400一致**、両者が使用した500例の対象は **500/500一致**し、候補の対象はすべて合法だった。この一致は検証した分布に限る。

### 実ゲーム比較

Task 10 の固定 STAY/HOLD coach、相手 `default`、合法な近距離配置、`PYTHONHASHSEED=0`、seed 0–39、各最大20 live tick で旧版と候補を同一 seed で比較した。facing 指標は強制 facing を除く、共有目撃への45度以内／合法な目撃行動数。分母はモデルごとの戦闘経過で変わる。FLASH 効果はくりまるを発動者とする被影響敵の延べイベント数で、ユニーク敵数や勝率ではない。

| side | facing 旧 | facing 候補 | FLASH発動成功 旧→候補 | くりまるの敵効果イベント 旧→候補 |
|---|---:|---:|---:|---:|
| attacker | 47/186 = 25.3% | 131/168 = 78.0% | 40/40→40/40 | 35→40 |
| defender | 212/322 = 65.8% | 261/319 = 81.8% | 40/40→40/40 | 120→120 |

独立 seed 20–39 だけでも attacker facing 20/103→71/96、FLASH効果12→15、defender facing 117/169→134/163、FLASH効果60→60。全層追加学習候補の `epoch96_best.pt` は初期20 seed で attacker FLASH効果23→18、defender 60→31だったため却下した。採用候補でも FLASH 効果イベントの維持・増加が戦術的な最適性や勝率改善を証明するものではない。

### 情報境界・テスト

- 未視認敵の実位置だけを2通りに変更し、slot 4 の grid、vector、能力対象 mask、facing・FLASH 教師が一致する自動テストを追加した。共有報告位置を変えると合法な対象教師が変わることも確認した。学習サンプルに game、敵の実座標、critic は保持しない。能力蒸留の教師も旧 `KurimaruPolicy` が受け取る安全な観測から作る。
- 新規テスト5件で、両陣営の5v5収集、slot/能力 mask、未視認敵の不変性、共有報告に基づく対象、警戒ポイントと合法random、異なるslotへのcheckpoint誤ロード、実ゲームのFLASH効果集計を確認した。
- 正式checkpoint更新後の `python -X utf8 -m unittest discover -s coach_v1 -p test_coach_v1_task*.py -q` は **120件成功**。`python -X utf8 -m compileall -q coach_v1`、`git diff --check -- coach_v1`、正式checkpointのSHA-256照合、`KurimaruPolicy()` による両sideの推論も成功した。

### 変更ファイル

- 新規: `training/kurimaru_rollout.py`、`experiment_task09e_kurimaru.py`、`experiment_task09e_kurimaru_facing_only.py`、`evaluate_task09e_kurimaru.py`、`validate_task09e_kurimaru.py`、`test_coach_v1_task09e_kurimaru.py`、本書。
- 更新: `training/README.md`、`checkpoints/characters/kurimaru/best.pt` と `latest.pt`。
- 生成: `checkpoints/experiments/task09e_kurimaru/` の退避・実験 checkpoint、`reports/task09e_kurimaru_*.json`。coreと他キャラの正式 checkpoint は変更していない。

### 残課題・次の推奨タスク

- 現在の実ゲームは固定 STAY/HOLD coach と人為的な近距離接敵であり、学習済みcoachとの連続対戦や自然配置での勝率は未測定。FLASH 効果は投射時間、遮蔽、味方位置、試合の展開にも左右されるため、延べイベント数だけで対象選択の良さは判断できない。
- 次は **Task 11（coach 学習環境とモデル）**。凍結した5人の character checkpoint と Task 10 coordinator を用い、attacker/defender の学習を進める。その後、自然な移動条件で facing と能力効果を再評価する。
