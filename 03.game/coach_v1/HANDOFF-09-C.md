# HANDOFF-09-C: ごんた（slot 2）

## 2026-09-27 実施結果

### 対象と実装

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、Task 09/09-A/09-B/10 の引き継ぎ、既存の観測・学習・ゲーム接続を確認した。ごんたの通常能力は RECON。作業前から変更されていた `10.task_template.md` は保持した。
- `training/gonta_rollout.py` で、既存 coordinator と固定 STAY/HOLD coach を通す5v5観測を収集する。敵配置は訓練・評価環境内だけで行う。actorへ渡すのは `CharacterObservation` のみ。facing 教師はその tick の合法なチーム共有目撃から45度以内の全方向、RECON使用・対象教師は現在の共有目撃と既存の合法対象maskから作る。目撃がない5v5局面では facing 損失を0にし、RECONは使用しないラベルとした。
- 推論側の戦術ロジック、経路探索、core、coach、他キャラの checkpoint は変更していない。`learning_character_gonta.py` は既存の slot 2 loader のままで、正式 checkpoint の差し替えだけで新モデルを読む。他モデル用 train ファイルの import はない。

### 学習と採用

- 旧正式 `best.pt` を `checkpoints/experiments/task09c_gonta/baseline_best.pt` に、旧 `latest.pt` を `baseline_latest.pt` に退避した。旧 `best.pt` と退避版の SHA-256 一致を確認してから更新した。
- 旧 checkpoint（epoch 36）からごんただけを追加学習した。訓練は警戒ポイント由来1400例と5v5補助600例で、前者70%。警戒ポイント由来は既存の敵配置70/20/10 samplerを使う。補助例は両陣営の近距離・西・東・挟撃・自然配置から各60例。訓練 seed 500–511、validation 600–604、凍結5v5 holdout 700–709。epoch 41/46/56 の best/latest を比較し、validation loss 最小の `epoch41_best.pt` を選んだ。
- 選択候補を `checkpoints/characters/gonta/best.pt` と `latest.pt` の両方へコピーし、SHA-256 一致を確認した。モデル入力・行動仕様は既存のまま。独立 checkpoint の対象はごんただけ。

### 凍結観測と分布検証

凍結5v5 holdout の合法な共有目撃あり907例で、facing 許容方向率は旧 **36.3%** →候補 **87.7%**、RECON対象の教師ラベルとの完全一致は **24.7%** → **69.1%**。能力使用判断の一致は両者 **99.9%**。能力使用の正例が少ないため、最後の値だけで能力の良さは判断しない。これらは教師との一致であり、実ゲームの勝率や索敵効果ではない。

別 seed 4002 の警戒ポイント由来600例は point 416例、jitter 119例、合法random 65例。facing 許容方向率は以下。能力使用判断の一致は各群で両者100%。

| side | 配置 | 件数 | 旧 | 候補 |
|---|---|---:|---:|---:|
| attacker | point | 207 | 61.8% | 65.7% |
| attacker | jitter | 61 | 54.1% | 62.3% |
| attacker | random | 32 | 40.6% | 50.0% |
| defender | point | 209 | 59.8% | 61.2% |
| defender | jitter | 58 | 58.6% | 62.1% |
| defender | random | 33 | 51.5% | 63.6% |

### 実ゲーム比較

Task 10 の固定 STAY/HOLD coach、相手 `default`、合法な近距離配置、`PYTHONHASHSEED=0`、seed 0–39、各最大20 live tick で旧版と候補を比較した。facing 指標は強制 facing を除く、共有目撃への45度以内／合法な目撃行動数。分母はモデルごとの戦闘経過で変わる。RECON効果イベントは被影響敵の延べ件数で、ユニーク敵数や勝率を示さない。

| side | facing 旧 | facing 候補 | RECON発動成功 旧→候補 | 敵への効果イベント 旧→候補 |
|---|---:|---:|---:|---:|
| attacker | 46/259 = 17.8% | 185/255 = 72.5% | 40/40→40/40 | 141→166 |
| defender | 445/640 = 69.5% | 292/403 = 72.5% | 40/40→40/40 | 200→200 |

seed 0–19 と独立20–39 も個別に記録した。defender の seed 0–19 では旧228/321、候補152/223で率はわずかに下がり、20–39では旧217/319、候補140/180で上がった。候補で目撃機会数が減っているため、単独の率だけで戦闘優位は断定しない。凍結観測・point/jitter/random・合計40 seedでの改善とRECON効果の維持を根拠に、単一の正式 checkpoint を採用した。

### 情報境界・テスト

- 未視認敵の実位置のみを2通りに変え、slot 2 の grid、vector、能力対象mask、facing・RECON教師ラベルが一致することを自動テストで確認した。教師が共有報告位置によって合法な対象を変えることも確認した。訓練サンプルに game、敵の実座標、critic は保持しない。
- 新規テスト4件は5v5配置、slot/能力mask、未視認敵の不変性、共有報告に基づく対象、point/randomと正式 checkpoint を確認する。正式 checkpoint 更新後の `python -X utf8 -m unittest discover -s coach_v1 -p test_coach_v1_task*.py -q` は **110件成功**。`python -X utf8 -m compileall -q coach_v1`、`git diff --check -- coach_v1`、正式checkpointのSHA-256照合と `GontaPolicy()` の両side推論も成功。

### 変更ファイル

- 新規: `training/gonta_rollout.py`、`experiment_task09c_gonta.py`、`evaluate_task09c_gonta.py`、`validate_task09c_gonta.py`、`test_coach_v1_task09c_gonta.py`、本書。
- 更新: `training/README.md`、`checkpoints/characters/gonta/best.pt` と `latest.pt`。
- 生成: `checkpoints/experiments/task09c_gonta/` の退避・実験 checkpoint、`reports/task09c_gonta_*.json`。core と他キャラの正式 checkpoint は変更していない。

### 残課題・次の推奨タスク

- 現在の5v5実ゲーム比較は固定 STAY/HOLD coach と人工的な近距離接敵であり、移動する学習済みcoachとの連続対戦・自然配置での勝率は未測定。自然配置では共有目撃が少ない。RECONの効果イベントが同数または増加したことは索敵価値や戦闘勝利の証明ではない。
- random群は各side約30例で精度の確度が限定的。対象教師は現在の共有目撃を狙うため、未確認領域の先行索敵の良さは別の対戦評価が必要。
- 次は **Task 09-D（くんた、slot 3）**。FLASHの仕様を個別に確認し、合法な共有目撃に対する facing、使用・対象、実際の敵への効果を分けて検証する。
