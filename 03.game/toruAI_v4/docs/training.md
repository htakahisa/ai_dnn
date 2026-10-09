# Toru AI v4：defender の敵プラント先予測の学習

実際のゲームエンジンで **12ラウンド×相手AI別** に対戦し、プラント前の公開観測から最終プラント先を予測します。学習するのはdefender側の予測モデルです。attackerのモデルは更新しません。

## 実行

通常は `MAX_PARALLEL_WORKERS = 3` で相手AIごとに並列学習する。ログは `logs/defender_analysis/<相手AI>/training.log`。並列数と実行方法は [並列学習](parallel_training.md) を参照する。

作業ディレクトリは `03.game/toruAI_v4/`。既存のゲームが動くPython環境（PyTorch、NumPyなど）を使用します。

学習セット数は`tv4_train_defender_analysis.py`上部の`TRAINING_SETS`で指定します。`--sets`を省略するとこの定数を使います。1セットは12ラウンドで、相手AIごとの**追加**セット数です。通常は定数を編集し、動作確認など一時的に変えるときだけ`--sets`を指定します。実際のセット数・開始位置・終了予定はログとlatestの`training_plan`に保存し、bestの評価結果にも今回の予定セット数と採用時のセット番号を残します。

```powershell
Set-Location C:\Users\ronet\MyProject\git\AI_dnn\03.game\toruAI_v4

# 最初の確認：相手1種類、12ラウンド学習＋12ラウンド評価
python tv4_train_defender_analysis.py --opponents fnatic_v3 --sets 1 --eval-every 1

# 6種類すべて：TRAINING_SETS回学習。10セットごとと最終セットに3seed×12ラウンドで評価
python tv4_train_defender_analysis.py

# 相手を指定
python tv4_train_defender_analysis.py --opponents gc_v1 touyama_v2
```

相手のキー：`gc_v1`, `touyama_v2`, `omoko_v1`, `fnatic_v3`, `frc_v1`, `toru_ai_v3`。
最後のキーは既存ゲームの `toru_ai_v3.1`（attacker_v3）を呼びます。
相手AIは既存のパーティープリセットで出撃します。toru AI v4 の味方は固定しません。冒頭の `TRAINING_PRESETS` の複数編成をセットごとに切り替え、`EVALUATION_PRESETS` の学習に使わない編成で評価します。敵と選手名が重なる編成はその対戦から除外します。通常設定は `TARGET_OPPONENTS`、`RANDOM_SEED`、`RESUME_TRAINING`、`DATA_DIRECTORY`、`LOG_DIRECTORY` 等の名前付き定数を編集します。単一編成の旧モデルは、複数編成で学習済みとは扱いません。

初回は未学習で左右50%です。確率80%以上が3tick続くまで `WAIT` になります。少量の動作確認だけで精度が保証されるものではありません。

## 12ラウンドの扱い

- 同じゲームとattackerコントローラーを12ラウンド維持します。ラウンド番号、勝敗、相手のラウンド間の内部状態が通常のリセット処理を通じて引き継がれます。
- サイド交代は無効です。毎セット新しい対戦を開始し、予測側の過去ラウンド履歴もクリアします。
- 予測モデルはセット中に更新せず、12ラウンド終了後に過去最大600の**プラントしたラウンド**と合わせて学習します。
- 同じラウンドのtick数で重みが偏らないよう、ラウンドを均等に選んでからtickを選びます。
- プラントなしのラウンドは履歴・生存・勝敗のログに残し、左右の教師ラベルを付けません。
- `train_preupdate` は、その試合を学習する前の予測です。学習中の`eval`は学習とは別の共通3seedで12ラウンドずつ、合計36ラウンドを行い、重みと学習用replayを更新しません。seed別の率を単純平均せず、全ラウンドをまとめて正解率・猶予・生存人数を計算します。
- 現時点の入力は、分岐領域別の目撃・監視状況、敵別の最終観測・移動差分・情報の古さ、味方状態、ラウンド番号、過去ラウンド結果です。履歴特徴を持つMLPによる教師あり学習で、移動方策の強化学習ではありません。
- 落ちたスパイクの公開位置、最後に落下を確認した位置と経過tick、その分岐領域と左右サイトまでの通路距離、地面から消えた状態と経過tickも入力します。回収者の特定は行いません。保持者は既存ルールで非公開です。

## 情報収集の移動

3人が左・中央・右を監視し、2人が左右の遮蔽物で待機します。初期の監視位置は`e/f/g`付近から、3歩以内に射線を切れる退避位置のある場所を自動選択します。

情報収集役は既定で **4tick監視位置へ進む → 4tick壁裏へ戻る** を繰り返します。3人のタイミングをずらします。これは移動目標の切り替え周期であり、移動時間も含みます。位置によっては周期内に到着しないため、移動率とtickログで確認してください。敵を発見すると退避を優先し、最後の観測から8tickまではその位置を危険な候補として扱います。安全側に動いた後、最低5tickは退避を継続します。隠れた敵の実位置は移動計算にも入力しません。

```powershell
# 覗く／隠れる周期を調整
python tv4_train_defender_analysis.py --opponents fnatic_v3 --sets 20 --peek-ticks 3 --hide-ticks 5

# 全マップ・5人の配置・特徴の名前を表示（対戦しない）
python tv4_train_defender_analysis.py --describe
```

通常の射撃・命中・移動回避補正は既存エンジンのままです。敵の位置が射撃処理後に公開されるため、最初の接触時は被弾し得ます。射撃を禁止したり、移動中の回避率を追加したりしません。

公開観測は既存FRCのチームセンサーを使用します。IQ位置ノイズを追加せず、味方5人で共有される公開位置・視認範囲を使います。未公開の敵位置、スパイク所持者、attacker内部の目標サイト・次の分岐点は予測入力に含めません。

既定では予測に応じた寄りを実行せず、監視と予測性能を確認します。`--rotate` を付けると、左右の待機役と中央の情報収集役が予測先へ寄ります。左右の情報収集役は残します。プラント後のリテイクはこの学習の対象外です。

### 配置を編集する

`docs/posts.example.json` をコピーして、各位置を `[行, 列]`（0始まり）で編集し、`--posts docs/my_posts.json` を指定してください。

- `watch`：監視位置
- `retreat`：壁裏の退避位置
- `alternate`：接触した後の再監視位置。既定ではwatchと同じで、別の角にしたい場合は編集します。
- `look`：監視する方向を決める目標位置
- `site`：担当側（L/R）

配置・周期・判断条件を変更した場合は、新しい学習を開始します。既存モデルへの再開では設定の一致を確認します。

分岐マップの文字は既存のゲームマップへの注釈として読み、文字の下のオーブ等は元の地形から復元します。分岐間の接続やattackerの移動ルールは上書きしません。次の分岐先の指定がなくても、既存6種類のAIが実際に通った経路で学習できます。

## 出力とログ

**既定のログはコンソールと同じ内容の`training.log`だけです。** 正解率・平均残りtick・味方／相手の平均生存人数・平均撃破と損失・保存先はここに表示します。CSV・JSON・別のサマリ・エンジンの大量のデバッグ出力は保存しません。

保存先は固定の`logs/defender_analysis/training.log`です。**起動時に上書き**し、前回分は追記しません。日付・セッションIDのディレクトリは作りません。詳細モードの出力も`logs/defender_analysis/`以下です。`--log-dir`で変更できます。

詳細を保存したい場合だけ`--detailed-logs`を指定します。`--trace-ticks`は詳細ログも有効にしてtickごとの記録を追加します。以下の一覧でtraining.log以外のログファイルは詳細モードの出力です。

```text
toruAI_v4/
  data/defender_analysis/<相手AI>/
    latest.pt          # 重み・optimizer・replay・完了セット数・設定
    set_000001.npz     # プラントしたラウンドの特徴、ラベル、round番号、plant tick
  data/best/<相手AI>/
    defender_analysis_best.pt   # 解析モデル。改善時だけ上書き
  logs/defender_analysis/                     # 既定はtraining.logのみ。起動時に上書き
    training.log
    # 以下は--detailed-logs指定時のみ
    rounds.csv         # Excelで見られるラウンド別結果（UTF-8 BOM）
    sets.csv           # セット別集計（学習前予測と評価を区別）
    rounds.jsonl       # ラウンド別の機械可読ログ
    sets.jsonl         # セット別の機械可読ログ
    events.jsonl       # 覗き・隠れる・退避・判定変更
    ticks.jsonl        # --trace-ticks指定時のみ：確率、公開目撃、味方位置、移動人数
    summary.json       # 各AIの最新セット／評価結果
    summary.txt        # 正解率と正解時の平均残りtickを日本語で表示
    config.jsonl      # 設定・分岐点・配置・特徴定義
    engine.log         # 既存AIのロード警告やエンジンの詳細出力
```

ラウンドの表示：`OK`=正解、`MISS`=誤判定、`WAIT`=未判断、`NO_PLANT`=プラントなし。

- `accuracy`：判断したプラント試合の正解率
- `coverage`：プラント試合のうち判断できた割合。未判断が多いモデルを正解率だけで評価しないために併記します。
- `correct_all_plants`：未判断も含めたプラント試合全体での正解割合
- `correct_lead_ticks`：正解した場合の「プラント完了tick − 最後に寄り先を決定・変更したtick」。同じ側の確率を再出力しても判断時刻は変えません。
- `changes`：一度決めた側を反対側へ変えた回数
- `defenders_at_plant`：プラント時の生存人数
- `attackers_at_plant`：プラント時の相手生存人数。両チームともプラント完了tickの射撃処理後の人数です。
- `defenders_alive` / `attackers_alive`：ラウンド終了時の味方／相手生存人数
- `preplant_defender_kills`：プラント前に味方が記録した撃破数。プラントした場合は完了tickまで、プラントなしの場合はラウンド終了まで。
- `preplant_defender_losses`：同じ時点までの味方の脱落人数。5人からの生存人数の減少です。
- サマリのプラント時平均はプラントしたラウンドだけ、終了時平均とプラント前の撃破・損失平均は全ラウンドが対象です。相手の環境ダメージ等による脱落と味方による撃破を区別するため、`preplant_attackers_eliminated`も別に保存します。
- `movement_rate`：プラント前の味方生存tickに対する移動tickの比率
- `peeks`：監視目標への切り替え回数。実際の視認や到着を保証する数ではありません。
- `retreats`：危険への接触により退避を開始した回数
- `spike_drops` / `spike_disappearances`：公開された落下位置の出現・変更、落下状態の消失。詳細位置とtickはevents/ticksログに残します。

誤判定時の猶予も`lead_ticks`に残しますが、正解時の猶予平均には含めません。
セット終了後には、実際に保存したファイルとモデルの絶対パスをコンソールとtraining.logに表示します。残りtickの平均・中央値は小数1桁で表示します。

## 再開と評価だけの実行

**実戦で使う解析重みは`data/best/<相手AI>/defender_analysis_best.pt`、学習の再開は`data/defender_analysis/<相手AI>/latest.pt`です。** 学習データは`data/defender/`と並ぶ`data/defender_analysis/`以下に保存し、日付ディレクトリは作りません。`--data-dir`で変更できます。旧日付ディレクトリを`--resume`に指定することもできますが、新しく保存するチェックポイントと学習データは`data/defender_analysis/`へ出力します。今回の整理で、既存の`data/analysis/`・`logs/analysis/`をそれぞれ`defender_analysis/`へ移動し、`analysis_best.pt`を`defender_analysis_best.pt`へ改名しました。保存内容は変更していません。別の保存先に残る旧モデルも、新名 → `analysis_best.pt` → `best.pt`の順で読み込み、bestの比較にも使います。bestには推論用の重み、入力定義、設定、採用時の評価結果を保存します。

bestの選択は、①未判断を含む全プラント試合での正解割合、②判断した試合の正解率、③正解時の平均残りtickの順に比較します。同点・悪化なら既存bestを維持します。生存人数は固定された移動ルールにも依存するため、この予測モデルの選択基準には入れずサマリで確認します。

再開時は設定・入力定義が一致することを要求します。新規学習で条件が変わった場合は、旧条件のbestの評価値を比較に使わず、今回の初回評価を基準に同じ保存先を更新します。同じ入力・条件の旧評価方式だけを引き継ぐ場合は、既存bestを共通seed群・編成で再評価します。評価seedと実際の編成の組、集計結果、採用セットをbestに保存します。同じ条件で同点・悪化なら既存bestを維持します。

評価設定は`EVALUATION_INTERVAL = 10`と`EVALUATION_SEED_COUNT = 3`で変更できます。一時的な変更は`--eval-every`と`--eval-seeds`です。途中の評価頻度に関係なく最終セットでは必ず評価するため、`--eval-every 0`は途中評価のみ無効にします。旧bestの再評価が必要な初回比較だけ、追加で36ラウンドかかります。

`--eval-only`は再開先のlatestを共通seed群・評価編成で一度評価し、best・latest・学習用replayを更新しません。`--sets`を評価回数には使いません。コード変更だけでは既存モデルを評価し直したり上書きしたりしません。

```powershell
# 起動時に表示されるdataのパスを指定。追加でTRAINING_SETS回学習
python tv4_train_defender_analysis.py --resume data/defender_analysis

# fnaticだけ学習した実行を再開する場合は、同じ相手を指定
python tv4_train_defender_analysis.py --resume data/defender_analysis --opponents fnatic_v3

# 一時的な評価のみ。EVALUATION_SEED_COUNTで指定した共通seed群を評価
python tv4_train_defender_analysis.py --resume data/defender_analysis --opponents fnatic_v3 --eval-only
```

Ctrl+Cで終了した場合、完了済みの12ラウンドのチェックポイントから再開できます。途中セットは学習済みとして数えません。既存AIのロード警告やフォールバックを確認する場合は`--detailed-logs`でengine.logを保存してください。

現在の入力形式はversion 3です。味方の能力種類、射撃性能、最大HP、残り使用回数、ultの情報を公開snapshotから入力します。選手名は特徴に使いません。version 1/2や固定編成のモデルは今回の条件で再開せず、`RESUME_TRAINING = False`で新規学習してください。analysisを更新したら、依存するsearchとretake収集・学習も順に更新します。

## テスト

同じ作業ディレクトリから：

```powershell
python -m unittest discover -s test -v
```
