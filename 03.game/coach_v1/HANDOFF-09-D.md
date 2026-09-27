# HANDOFF-09-D: くんた（slot 3）

## 2026-09-27 実施結果

### 対象と変更

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、Task 09-A/B/C/10 の引き継ぎ、既存の観測・学習・ゲーム接続を確認した。くんたの通常能力は FLASH。作業前から変更されていた `10.task_template.md` は保持した。
- `training/kunta_rollout.py` は既存 coordinator と固定 STAY/HOLD coach を通した5v5観測を収集する。敵の実配置は訓練・評価環境にだけ使う。actorへ渡すのは既存の `CharacterObservation` のみ。facing教師は現在の合法なチーム共有目撃から45度以内の全方向、FLASH使用・対象教師は現在の共有目撃と既存の合法投射maskから作る。目撃がない5v5局面では facing 損失を0にし、FLASHは不使用とする。
- ゲーム本体、推論側の戦術ロジック、経路探索、coach、他キャラの checkpoint は変更していない。`learning_character_kunta.py` と `train_character_kunta.py` は既存の対応したまま。別モデルの train からの import はない。評価用 `evaluate_task10_rollout.py` に発動者別 FLASH 被影響敵集計だけを追加した。

### 学習と採用

- 旧正式 `best.pt` と `latest.pt` を `checkpoints/experiments/task09d_kunta/baseline_best.pt` と `baseline_latest.pt` に退避した。旧 `best.pt` と退避版の SHA-256 一致を確認した。
- 旧 checkpoint（epoch 55）からくんただけを追加学習した。訓練は警戒ポイント由来1400例と5v5補助600例で、前者70%。警戒ポイント由来は既存の敵配置70/20/10 samplerを使う。補助例は両陣営の近距離・西・東・挟撃・自然配置から各60例。訓練 seed 500–511、validation 600–604、凍結5v5 holdout 700–709。epoch 60/65/75 の best/latest を比較し、追加の独立分布評価と実ゲーム結果から `epoch75_latest.pt` を選んだ。
- 選択候補を `checkpoints/characters/kunta/best.pt` と `latest.pt` の両方へコピーし、候補を含む3ファイルの SHA-256 一致を確認した。くんたの単一正式 checkpoint のみ更新した。

### 凍結観測と分布検証

- 凍結5v5 holdout の合法な共有目撃あり743例で、facing許容方向率は旧 **42.3%** →候補 **85.9%**、FLASH対象の教師ラベルとの完全一致は **17.3%** → **74.1%**。能力使用判断の一致は両者 **99.9%**。訓練用5v5例600件の能力使用正例は32件なので、最後の値だけで能力の良さは判断しない。これらは教師との一致であり、敵への実効果ではない。
- 独立 seed 4003 の警戒ポイント由来600例は point 396、jitter 128、合法random 76。facing許容方向率は attacker point 72.2%→77.8%、jitter 61.3%→77.4%、random 52.5%→70.0%。defender point 61.6%→65.2%、jitter 57.6%→62.1%、random 55.6%→52.8%。最後の群は36例中1例の差で、追加の独立サンプルで検証した。
- seed 5003 と6003 の警戒ポイント由来2400例ずつを合わせた4800例では、旧→候補のfacing許容方向率が attacker point 61.8%→75.7%、jitter 64.7%→72.3%、random 64.2%→72.1%、defender point 60.6%→66.5%、jitter 59.6%→59.8%、random 59.8%→66.8%。6群すべてで候補が旧版以上。能力使用判断は各群で両者100%。個々のseed・小群の変動まで単調改善を保証する結果ではない。

### 実ゲーム比較

Task 10 の固定 STAY/HOLD coach、相手 `default`、合法な近距離配置、`PYTHONHASHSEED=0`、seed 0–39、各最大20 live tick で旧版と候補を同一seedで比較した。facing指標は強制facingを除く、共有目撃への45度以内／合法な目撃行動数。分母はモデルごとの戦闘経過で変わる。FLASH効果は発動者をくんたに限定した被影響敵の延べイベント数で、ユニーク敵数や勝率ではない。

| side | facing 旧 | facing 候補 | FLASH発動成功 旧→候補 | くんたの敵効果イベント 旧→候補 |
|---|---:|---:|---:|---:|
| attacker | 155/220 = 70.5% | 174/234 = 74.4% | 40/40→40/40 | 32→45 |
| defender | 2/63 = 3.2% | 298/401 = 74.3% | 40/40→40/40 | 0→120 |

独立の seed 20–39 でも attacker は facing 88/123→97/131、FLASH効果13→20、defender は facing 2/32→154/198、FLASH効果0→60。効果イベントの増加にはゲーム内の行動経過差も含まれるため、FLASH対象の単独因果効果や勝率改善までは断定しない。

### 情報境界・テスト

- 未視認敵の実位置だけを2通りに変え、slot 3 の grid、vector、能力対象mask、facing・FLASH教師ラベルが一致する自動テストを追加した。共有報告位置を変えると合法な対象が変わることも確認した。訓練サンプルに game、敵の実座標、critic は保持しない。
- 新規テスト5件は5v5配置、slot/能力mask、未視認敵の不変性、共有報告に基づく対象、point/randomと正式 checkpoint、実ゲームFLASH効果のslot別合計を確認する。正式 checkpoint 更新後の `python -X utf8 -m unittest discover -s coach_v1 -p test_coach_v1_task*.py -q` は **115件成功**。`python -X utf8 -m compileall -q coach_v1`、`git diff --check -- coach_v1`、正式 checkpoint のSHA-256照合と `KuntaPolicy()` の両side推論も成功。

### 変更ファイル

- 新規: `training/kunta_rollout.py`、`experiment_task09d_kunta.py`、`evaluate_task09d_kunta.py`、`validate_task09d_kunta.py`、`test_coach_v1_task09d_kunta.py`、本書。
- 更新: `training/README.md`、評価指標の `evaluate_task10_rollout.py`、`checkpoints/characters/kunta/best.pt` と `latest.pt`。
- 生成: `checkpoints/experiments/task09d_kunta/` の退避・実験 checkpoint、`reports/task09d_kunta_*.json`。core と他キャラの正式 checkpoint は変更していない。

### 残課題・次の推奨タスク

- 現在の5v5実ゲーム比較は固定 STAY/HOLD coach と人工的な近距離接敵であり、移動する学習済みcoachとの連続対戦・自然配置での勝率は未測定。自然配置では共有目撃が少ない。
- FLASHの対象教師は現在の共有目撃と合法投射maskを基準にする。爆発地点・敵の遮蔽・投射時間を踏まえた独立の効果評価が必要。実ゲームの延べイベント数だけで戦術的な最適性は判断できない。
- 次は **Task 09-E（くりまる、slot 4）**。能力仕様を個別に確認し、合法な共有目撃に対する facing、使用・対象、実際の敵への効果を分けて検証する。

## 2026-09-27 追加確認: 警戒ポイント分布の採用判断

最初の採用候補 `epoch65_latest.pt` では警戒ポイント群の一部が旧版を下回った。これを後続タスクへ送る計画はなかったため、Task 09-D 内で保存済み候補を再比較した。`epoch75_latest.pt` は独立seed 4003 の600例で6群中5群が旧版以上、残る defender random は36例中1例差。さらに別seed 5003/6003 の各2400例を合わせた4800例では6群すべてが旧版以上となった。実ゲーム40 seedの比較でも、両sideのfacingとくんた自身のFLASH効果イベントが旧版を上回り、発動成功数は維持した。これを根拠に正式checkpointを `epoch75_latest.pt` に更新した。

少数例の各群で常に単調改善する保証はないが、先の「警戒ポイント群の低下」を将来タスクへ残す判断は撤回した。分布の再測定は `validate_task09d_kunta.py` の `--seed`、`--count`、`--output` で再現できる。結果は `reports/task09d_kunta_distribution_validation.json`、`task09d_kunta_distribution_5003.json`、`task09d_kunta_distribution_6003.json` に保存した。checkpoint更新後に全115件の自動テスト、compileall、差分チェック、候補と正式2ファイルのSHA-256一致を再確認した。
