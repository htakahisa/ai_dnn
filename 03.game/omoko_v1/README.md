# omoko v1 の実行コマンド

## 攻撃経路の事前勝率を算出し直す場合

omoko v1 の経路評価スクリプトは、相手AIごと・攻撃経路ごとの勝率を事前に算出します。この処理は対戦による評価と集計であり、モデルの重みを再学習する処理ではありません。既定設定では、評価スクリプトが相手AI 6種類に対して、各経路を12ラウンド×8セット（96ラウンド）ずつ評価します。

carry は、集計済みの勝率を最初の6ラウンドの経路抽選に利用します。carry は、正規化した勝率に基づく確率50%と全候補の均等確率50%を混ぜて経路を選びます。carry は、7ラウンド以降には試合内の成績に基づく経路選択を行います。carry は、利用できるサマリがない場合やモデル・地形・相手AI・編成の条件が合わない場合には、従来の初期探索を行います。

作業ディレクトリ：`C:\Users\ronet\MyProject\git\AI_dnn\03.game\omoko_v1`（ユーザーがこのディレクトリへ移動済みの状態）。

ユーザーは、5つのスクリプトを順番に実行する場合、次の PowerShell コマンドを実行してください。このスクリプトは `omoko_v1` を作業ディレクトリとして使用し、各 Python スクリプトが正常終了した場合にだけ次のスクリプトへ進みます。いずれかのスクリプトが失敗すると、このスクリプトは後続のスクリプトを停止します。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_ov1_advisor_and_carry_routes.ps1
```

このコマンドは、新しい PowerShell プロセス内だけで実行ポリシーを `Bypass` にします。ユーザーの PowerShell 実行ポリシー設定は変更しません。組織のグループポリシーが実行を制限している場合は、ユーザーは管理者に許可された実行方法を確認してください。

ユーザーが5段階の処理を個別に実行する場合は、ユーザーは次のコマンドを順番に実行してください。

```powershell
python ability_advisor_v1/ov1_collect_attacker_ability_advisor.py
python ability_advisor_v1/ov1_train_attacker_ability_advisor.py
python ability_advisor_v1/ov1_evaluate_attacker_ability_advisor.py
python ov1_evaluate_attacker_carry_routes.py
python ov1_summarize_attacker_carry_routes.py
```

ユーザーは、実行前に各スクリプト冒頭の名前付き定数で設定を確認・変更してください。

| 対象スクリプト | ユーザーが編集する主な定数 |
| --- | --- |
| `ov1_evaluate_attacker_carry_routes.py` | `SETS_PER_ROUTE`（既定8）、`BASE_SEED`、`PARALLEL_WORKERS`（既定6）、`OPPONENTS`、`OUTPUT_PATH`、`RESUME`（既定`True`） |
| `ov1_summarize_attacker_carry_routes.py` | `INPUT_PATH`（評価結果の読込先）、`OUTPUT_PATH`（通常対戦用サマリの保存先）、`TABLE_PATH`（勝率表の保存先） |

評価スクリプトは、`RESUME = True` の場合、設定・ソースコード・モデルなどの条件が一致する既存結果から評価を再開します。評価スクリプトは、既存結果と現在の条件が一致しない場合、既存結果を同じ保存先で置き換えて最初から評価します。集計スクリプトは、既存のサマリJSONと勝率表を新しい内容で置き換えます。既定の保存先を使う場合、ユーザーは更新後も保存先を変更する必要はありません。

たとえば、ユーザーが評価結果を `data/carry_route_evaluation/route_win_rates_20261016.json` に保存する場合は、ユーザーは評価スクリプトの `OUTPUT_PATH` と集計スクリプトの `INPUT_PATH` を両方とも `AI_DIR / "data/carry_route_evaluation/route_win_rates_20261016.json"` に設定してください。

集計スクリプトは、全対象の評価が完了した結果から、既定では次のファイルを作成します。

- `data/carry_route_evaluation/route_priors_summary.json`：carry が経路抽選に使用する勝率サマリ。
- `data/carry_route_evaluation/route_win_rates_summary.md`：ユーザーが確認する相手AI別・経路別の勝率表。

carry は、コントローラーの初期化時にサマリJSONを読み込みます。ユーザーは、更新したサマリを反映するために通常対戦のコントローラーを生成し直してください。ユーザーがサマリの保存先を変更する場合は、ユーザーは `ov1_learning_attacker_carry.py` 冒頭の `ROUTE_PRIOR_SUMMARY_PATH` も同じ保存先へ変更してください。

## 定点アビリティアドバイザーを再学習する場合

2026-10-17 採用：通常対戦の定点アドバイザーは、FLASH・SMOKEを採用済みの再学習モデルで判定し、RECONを従来のモデルで判定します。ユーザーは、[採用内容と確認結果](docs/ov1_attacker_advisor_deployment.md) でモデルの保存先と設定を確認できます。ユーザーがゲームを再起動すると、通常対戦はこの組合せを使用します。

対象モデルは、omoko v1 の carry・escort が SMOKE・FLASH・RECON の定点を使用するか温存するかを助言する定点アビリティアドバイザーです。

作業ディレクトリ：`C:\Users\ronet\MyProject\git\AI_dnn\03.game\omoko_v1`（ユーザーがこのディレクトリへ移動済みの状態）。

ユーザーは、次の3コマンドを上から順番に実行してください。ユーザーは、各コマンドが正常終了してから次のコマンドを実行してください。

```powershell
# 1. 収集スクリプトが定点アドバイザー用の対戦データを再収集する
python ability_advisor_v1/ov1_collect_attacker_ability_advisor.py

# 2. 学習スクリプトが収集データから定点アドバイザーモデルを作り直す
python ability_advisor_v1/ov1_train_attacker_ability_advisor.py

# 3. 評価スクリプトが助言なし・現在のモデル・再学習候補を比較する
python ability_advisor_v1/ov1_evaluate_attacker_ability_advisor.py
```

収集・学習スクリプトは、現在のモデル・シナリオ・観測仕様と互換性のない旧収集データを自動削除し、互換性のある収集データを残します。学習スクリプトは、途中の判断データも保持して学習し、検証データで不要な使用許可を減らすモデルと区分ごとの閾値を選びます。学習スクリプトは、候補を `ability_advisor_v1/data/ov1_attacker_ability_advisor_candidate.pt` に保存します。通常対戦の `Ov1AttackerController` は、FLASH・SMOKE用の採用済み `_flash_smoke.pt` とRECON用の従来モデルを読み込みます。学習処理は、これらの採用済みモデルを上書きしません。

評価スクリプトは、助言なし・従来モデル・全種類を再学習候補で判定する条件・通常対戦の採用済み組合せを、同じseedで比較します。評価スクリプトは、実消費、投擲時の教師判定、着弾時の効果、勝率・設置率を記録します。ユーザーは、通常対戦の組合せと次の再学習候補の比較を `candidate_vs_deployed`、従来モデルと採用済み組合せの比較を `deployed_vs_advised` で確認できます。ユーザーは、候補の勝利数を確認してから次の採用を判断してください。

ユーザーが現在の仕様と互換性のある収集済みデータだけで定点アドバイザーモデルを学習し直す場合は、ユーザーは手順2と手順3を実行してください。ユーザーが carry・escort・retrieve・guard の重み、関連配置・定点、観測仕様、相手AIを更新した場合は、ユーザーは手順1から実行してください。

## ユーザーが設定を変更する場所

ユーザーは、各スクリプト冒頭の名前付き定数を編集してください。通常実行にはコマンドラインオプションを指定しません。

| 対象スクリプト | ユーザーが編集する主な定数 |
| --- | --- |
| `ov1_collect_attacker_ability_advisor.py` | `DEFAULT_MATCHES_PER_OPPONENT`（既定40）、`DEFAULT_ROUNDS_PER_MATCH`（既定12）、`DEFAULT_WORKERS`（既定6）、`DEFAULT_SEED`（既定0） |
| `ov1_train_attacker_ability_advisor.py` | `DEFAULT_EPOCHS`（既定80）、`DEFAULT_BATCH_SIZE`（既定256）、`DEFAULT_LEARNING_RATE`（既定0.001）、`DEFAULT_THRESHOLD_CANDIDATES`、`DEFAULT_OUTPUT` |
| `ov1_evaluate_attacker_ability_advisor.py` | `DEFAULT_MATCHES_PER_OPPONENT`（既定10）、`DEFAULT_ROUNDS_PER_MATCH`（既定12）、`DEFAULT_WORKERS`（既定6）、`DEFAULT_MODEL`、`DEFAULT_CANDIDATE_MODEL` |

ユーザーが追加収集で異なる対戦データを増やす場合は、ユーザーは収集スクリプトの `DEFAULT_SEED` を変更してください。

ユーザーは、保存先・検証条件・互換性の詳細を [定点アビリティアドバイザーの説明](docs/ov1_attacker_ability_advisor.md) で確認できます。
