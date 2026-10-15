# carry・escort の定点アビリティ判定

concon_v1 の判定機を omoko_v1 の配置・定点・モデルに合わせて移植しています。事前収集した対戦データで学習した1つのモデルが、SMOKE・FLASH・RECON の定点を使用するか温存するか助言します。

入力は carry/escort・左右サイト、相手AI、定点と投射終点、時刻、使用者の位置・向き・残数、味方とIQ知覚で公開された敵の位置・向き、同じ試合の直近5ラウンドの観測・使用履歴です。事前情報は学習済みの重みに反映され、試合中の履歴と合わせて使用されます。未発見の敵の実配置は教師ラベルの生成時だけ参照します。

carry・escort のモデルがアビリティを選ぶ仕組みを維持し、定点候補を判定機で絞ります。温存した定点を使用済みにはせず、別の定点も同じ判断内で確認できます。次回の判断では再判定します。視認敵への通常使用、定点以外の使用、retrieve・guard は従来の判断を続けます。

モデル未作成・入力仕様不一致・未評価の相手や定点・モデルまたは配置の識別情報不一致では、従来どおり定点を候補に残します。現在のIQ知覚で敵の関与が確認できる定点も候補に残します。

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

学習データは試合単位で60%・20%・20%に分割します。温存の助言には、スコアが `DEFAULT_SKIP_THRESHOLD=0.10` 未満で、同じ役割・サイト・相手・種別・定点について検証とテストの両方で20ラウンド以上、温存候補5件以上、必要な定点の見逃し率と温存候補の誤り率が5%以下であることを要求します。条件を満たす区分は学習終了時の `validated_groups` と `.report.json` に出ます。十分な区分がない場合は収集試合数を増やしてください。

比較評価は助言なし／ありを同じseedで対戦させ、設置率・ラウンド勝率・設置時の残数・助言回数を保存します。Ghost Champions は内部の乱数により同一seedでも動きが異なるため、複数試合で判断してください。

## 保存先と互換性

- 収集データ：`ability_advisor_v1/data/datasets/`
- 判定モデル：`ability_advisor_v1/data/ov1_attacker_ability_advisor.pt`
- 学習レポート：モデルと同じ場所の `.report.json`
- 比較評価：`ability_advisor_v1/logs/ov1_attacker_ability_advisor_evaluation.json`

通常対戦の `Ov1AttackerController` は既定パスの判定モデルを自動読込します。別パスで比較したい場合は、train の `DEFAULT_OUTPUT` と evaluate の `DEFAULT_MODEL` を同じ別パスに変更し、評価後に採用するモデルを既定パスへ配置します。

concon_v1 の収集データと判定モデルは流用できません。omoko_v1 の carry・escort・retrieve・guard の重みと、配置・定点の署名を保存して照合します。これらを更新した場合は判定機の収集・学習・比較評価をやり直してください。相手AIの更新時も再収集・学習・評価してください。

今回の追加では射線仕様、観測次元、配置、既存モデルの重みを変更していません。既存の carry・escort・retrieve・guard の再学習は必須ではありません。必要なのは omoko_v1 用の新しい判定モデルの学習です。定点使用の変化で経路ごとの成績も変わり得るため、採用後は経路評価の更新を推奨します。

## テスト

作業ディレクトリ：`03.game/omoko_v1/`（移動済み）。

```powershell
python -m unittest discover -s test -p test_ov1_attacker_ability_advisor.py -q
```

実装時には収集全体・学習・比較評価を実行していません。
