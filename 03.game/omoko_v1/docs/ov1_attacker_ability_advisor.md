# carry・escort の定点アビリティ判定

2026-10-17 採用：通常対戦の `DeployedAbilityAdvisor` は、FLASH・SMOKEを採用済みの再学習モデルで判定し、RECONを従来モデルで判定する。両モデルは試合履歴・使用履歴を共有する。起動処理は、両モデルのcarry/escort・左右サイトの全4区分を検査する。ユーザーは、[採用内容](ov1_attacker_advisor_deployment.md) で保存先と検証結果を確認できる。

concon_v1 の判定機を omoko_v1 の配置・定点・モデルに合わせて移植しています。事前収集した対戦データで学習した1つのモデルが、SMOKE・FLASH・RECON の定点を使用するか温存するか助言します。

入力は carry/escort・左右サイト、相手AI、定点と投射終点、時刻、使用者の位置・向き・残数、味方とIQ知覚で公開された敵の位置・向き、同じ試合の直近5ラウンドの観測・使用履歴です。事前情報は学習済みの重みに反映され、試合中の履歴と合わせて使用されます。未発見の敵の実配置は教師ラベルの生成時だけ参照します。

carry・escort のモデルがアビリティを選ぶ仕組みを維持し、定点候補を判定機で絞ります。温存した定点を使用済みにはせず、別の定点も同じ判断内で確認できます。次回の判断では再判定します。視認敵への通常使用、定点以外の使用、retrieve・guard は従来の判断を続けます。

omoko v1 は、通常対戦で定点アドバイザーを必須とします。ゲームは、omoko v1 が attacker・defender のどちらで開始する場合も、コントローラー初期化時にモデルの読込と carry/escort・左右サイトの全4区分の互換性を検査します。定点アドバイザーは、モデル欠落・破損・入力仕様不一致・重みの異常・配置や重み署名の不一致を検出すると、`AbilityAdvisorError` で起動を停止します。検証済み区分がないモデルも読み込み、未検証の定点を使います。defender 側の初期化も attacker 用の検査を実行するため、ゲームは攻守交代まで異常を持ち越しません。

定点アドバイザーは、実行中に未知の相手や非有限スコアを検出すると例外を送出します。公開観測で敵の関与が確認できる定点と、学習データの判定基準を満たさなかった未検証の定点は使用します。その他の検証済み区分は学習スコアで使用・温存を判断します。

carry の RECON は、アドバイザー接続時に有効な定点も接敵情報もない場合、使用候補から外します。ランダムな設置地点への投射で定点判定を迂回しません。比較評価の助言なし条件は `enabled=False` で明示し、モデル未作成の通常対戦と区別します。

## 通常の実行手順

作業ディレクトリ：`03.game/omoko_v1/`（このディレクトリへ移動済み）。以下の順に実行します。

```powershell
python ability_advisor_v1/ov1_collect_attacker_ability_advisor.py
python ability_advisor_v1/ov1_train_attacker_ability_advisor.py
python ability_advisor_v1/ov1_evaluate_attacker_ability_advisor.py
```

各スクリプト冒頭の名前付き定数を変更して設定します。

| スクリプト | 主な定数と既定値 |
| --- | --- |
| collect | `DEFAULT_MATCHES_PER_OPPONENT=40`、`DEFAULT_ROUNDS_PER_MATCH=12`、`DEFAULT_OPPONENTS`、`DEFAULT_SEED=0`、`DEFAULT_WORKERS=6`、`DEFAULT_OUTPUT_DIR` |
| train | `DEFAULT_DATA_DIR`、`DEFAULT_OUTPUT`、`DEFAULT_EPOCHS=80`、`DEFAULT_BATCH_SIZE=256`、`DEFAULT_LEARNING_RATE=0.001`、`DEFAULT_SEED=0` |
| evaluate | `DEFAULT_MATCHES_PER_OPPONENT=10`、`DEFAULT_ROUNDS_PER_MATCH=12`、`DEFAULT_MODEL`、`DEFAULT_OPPONENTS`、`DEFAULT_SEED=100000`、`DEFAULT_WORKERS=6`、`DEFAULT_OUTPUT` |

相手は concon_v1、touyama_gaming_v3、fnatic_v3、ghost_champions_v2、toru_ai_v4、frc_v1 です。収集・比較評価は相手ごとに最大6プロセスで実行します。carry/escort・左右サイトの4区分は、通常の試合進行に従って一緒に収集します。経路は通常対戦と同じ選択処理です。

収集は通常のsetup・IQ知覚・射撃・設置後のguard・ラウンド遷移を実行し、既存のモデルを固定します。味方と敵に同名の選手がいても、チーム別の選手識別情報を使用します。出力は毎回新しいJSONLになり、全相手の完了後に公開します。追加データを増やす場合は収集seedを変更してください。

教師ラベルは concon_v1 と同じです。FLASH は投射終点から効果の射線が届く敵、RECON は終点中心の9×9内の敵、SMOKE は3×3内の敵または敵から味方への射線を遮れる場合を「必要」とします。この教師は幾何的な関与を表すため、勝利への効果は比較評価で確認します。

学習処理は、データを試合単位で60%・20%・20%に分割し、同じ定点の途中の判断も保持します。学習処理は、同じ試合・ラウンド・定点の合計重みをそろえて学習します。学習処理は、検証データだけを使って区分ごとの閾値を選びます。検証合格には、20ラウンド以上、温存したラウンド5件以上、必要な候補の見逃し率と温存候補の誤り率が5%以下、必要な候補を一度でも温存したラウンドの割合が5%以下であることを要求します。学習処理は、この条件を満たす閾値のうち、全候補が不要なラウンドでの使用許可を最も減らす閾値を選びます。学習処理は、全区分の実際の判定を集計して最良エポックを選び、テストデータは最終確認だけに使用します。旧モデルは、従来の共通閾値0.10で動作を続けます。

比較評価は、助言なし・従来モデル・再学習候補・採用済み組合せを同じseedで対戦させます。評価処理は、設置率・勝率・設置時の残数・助言回数に加え、実消費、投擲時に教師が不要と判定した消費、FLASH・RECONが着弾時に敵へ届かなかった回数を記録します。評価処理は、carry・escort・投擲先別の件数と、条件間の増減を保存します。Ghost Champions は内部の乱数により同一seedでも動きが異なるため、ユーザーは複数試合で判断してください。

## 保存先と互換性

- 収集データ：`ability_advisor_v1/data/datasets/`
- 判定モデル：`ability_advisor_v1/data/ov1_attacker_ability_advisor.pt`
- 再学習候補：学習処理は `ability_advisor_v1/data/ov1_attacker_ability_advisor_candidate.pt` に保存する。
- 学習レポート：モデルと同じ場所の `.report.json`
- 比較評価：`ability_advisor_v1/logs/ov1_attacker_ability_advisor_evaluation.json`

通常対戦の `Ov1AttackerController` は、採用済みFLASH・SMOKEモデルと従来のRECONモデルを自動読込します。学習処理は次の候補を別名に保存します。比較評価は、`DEFAULT_MODEL` の従来モデル、`DEFAULT_CANDIDATE_MODEL` の候補、`DEFAULT_DEPLOYED_FLASH_SMOKE_MODEL` と従来RECONを組み合わせた採用済み条件を比較します。ユーザーは、次の採用を比較結果で判断してください。

concon_v1 の収集データと判定モデルは流用できません。omoko_v1 の carry・escort・retrieve・guard の重みと、配置・定点の署名を保存して照合します。これらの入力を更新した場合は判定機の収集・学習・比較評価をやり直してください。相手AIの更新時も再収集・学習・評価してください。

2026-10-16 修正：旧署名は `ov1_map_data_*.py`・`ov1_learning_attacker_*.py`・roster の全ファイルのバイト列を含み、コメント修正や guard 配置の修正でも全4区分が失効していました。現在は `SCENARIO_CONTRACT_VERSION` と実際の共通グリッド、carry/escort の配置文字列、役割・サイトから署名を作成します。Pythonソース全体のハッシュは取りません。判定特徴量・教師・射線の仕様は従来の専用バージョンで照合し、重みの変更は別の重み署名で検出します。収集データ・モデルの `scenarios.*.manifest` に署名の材料も保存します。

定点アドバイザーは、旧形式の署名だけが残る既存モデルについて、元の配置を確認できないため、署名を書き換えて有効化しません。ユーザーは、新形式への移行時に、omoko v1 の定点アドバイザー用データの再収集と定点アドバイザーモデルの再学習を一度実行する必要があります。その後のコメント・推論コードの編集だけでは、定点アドバイザーはモデルを失効させません。収集スクリプトと学習スクリプトは、旧データの自動削除を維持します。ゲームは、旧形式のモデルが残っている場合も、起動時の互換性検査で停止します。

今回の追加では射線仕様、観測次元、配置、既存モデルの重みを変更していません。既存の carry・escort・retrieve・guard の再学習は必須ではありません。必要なのは omoko_v1 用の新しい判定モデルの学習です。定点使用の変化で経路ごとの成績も変わり得るため、採用後は経路評価の更新を推奨します。

## テスト

作業ディレクトリ：`03.game/omoko_v1/`（移動済み）。

```powershell
python -m unittest discover -s test -p test_ov1_attacker_ability_advisor.py -q
```

実装時には収集全体・学習・比較評価を実行していません。
