# 設置後の guard モデル

実行ディレクトリは `concon_v1` です。学習・評価・推論は同じマップ設定、観測、行動を使います。
通常対戦では保存済みモデルを読み込み、学習は実行しません。

## 配置・停止の先行学習

学習開始時に、固定地形の各マスから各配置点への移動・停止・向きを教師あり学習します。
通常時の行動値には学習済みの表形式モデル、敵への射線や解除通知がある時には戦闘ネットワークを使います。
味方による配置点の占有と、通路を譲る行動も先行学習に含めます。BFS は教師データの生成に使い、推論時に経路や停止・向きを上書きしません。
学習済みの配置行動は戦闘学習中に固定して、基本動作を忘れないようにします。

先行学習後は、単独100ケースで配置到達・8 tick の連続停止・指定方向がすべて成功し、往復がないことを確認します。
さらに5人同時の20ケースで到達・連続停止・指定方向が各95%以上、到着後の離脱が0であることを要求します。
これを満たさなければ戦闘学習を開始せず、勝率が高くても `best` に採用しません。
5人評価の95%という基準は、混雑した通路での未到達を完全に排除する保証ではありません。評価 JSON に実測値を保存します。

既存の戦闘モデルに配置学習を追加し、戦闘の追加学習をせずに評価・保存する場合は `concon_v1` 内で実行します。

```powershell
python co1_train_guard.py -map L --resume data/guard_L_data/co1_guard_L_best.pt --positioning-only
python co1_train_guard.py -map R --resume data/guard_R_data/co1_guard_R_best.pt --positioning-only
```

`positioning_only` でも通常の対戦評価を行います。旧 `best` が基本動作の検証に失敗し、新モデルが成功した場合は新モデルを採用します。
この操作で旧 `best` を更新する時は `co1_guard_L_before_positioning_v1.pt` のような名前で最初の旧モデルを残します。
新しいモデルには `positioning_version=1` と学習済みの行動値を保存し、推論・評価はその情報から対応するモデルを読み込みます。

### 回数の指定と追加学習

`DEFAULT_EPISODES=3000` は1回の実行で追加する対戦エピソード数です。最適な回数や、3000話で十分という保証を意味しません。
新規学習では0→3000話、2500話のモデルから `--resume` すれば2500→5500話になります。
最終モデルを自動採用するのではなく、配置の合格と対戦評価を満たす `best` を採用します。

配置の事前学習は `DEFAULT_POSITIONING_STEPS=250` 回の教師あり更新が既定値で、`--positioning-steps` で変更できます。
250回でL/Rの検証を通過した実績はありますが、最小回数とは確認していません。
習得済みのモデルから再開する時は、まず配置の検証を行い、合格すれば更新を0回にして既存の配置行動を使います。
再学習を明示する場合は `--retrain-positioning --positioning-steps 500` のように指定します。
`--positioning-steps 0` は検証に合格する既存モデルが必要で、不合格なら対戦学習に進みません。
対戦中の探索は、使用可能なアビリティ・ウルトがあれば移動・待機も含む有効な行動候補から選び、使えない通常時は配置行動を保持します。

### アビリティ・ウルトの使用

移動・待機・アビリティ・ウルトを同時に行動候補へ残し、推論・学習ともモデルの行動値で使用タイミングを選びます。使用可能なアビリティがあっても移動・待機を無効にしません。
通常時のアビリティを低い固定値にする抑制を撤廃し、ウルトの対象・方向を行動候補に追加しました。
行動マスクは実行可能性を判定し、使用するか温存するかと、使用対象・方向の選択をモデルが担当します。
残数・必要ポイント・壁・占有・移動阻害・発動中のポータルを確認します。死亡時の SERENADE は既存のゲーム処理が発動します。
残数がある間も配置・停止を選べます。RAID・ESCAPE 自体による移動は発生します。
既存112行動のモデルは読み込み時に互換変換し、既存の学習済み重みを保持します。使用タイミングは以前の強制使用マスク下では学べていないため、追加対戦学習で調整してください。
追加対戦学習にはウルト使用可能な開始状態も含め、使用対象・方向の改善を学びます。250回の配置先行学習には含めません。

報酬バージョン3では、敵への射線も解除通知もない時のアビリティ・ウルト使用に0.08の消費コストを付けます。配置点へ移動できるのに停止する減点は、使用中にも適用します。これにより、待機と同じ加点を得たり、待機の減点を免れたりする目的で使う行動を抑えます。交戦・解除対応時にはこの消費コストを付けず、実行可能な使用候補は維持します。

報酬の変更は追加学習から反映されます。保存済みモデルの行動値は自動では変わりません。探索率が0.05でもランダムな使用は残るため、探索なしの評価に表示する `quiet_utility`（射線・解除通知がない判断のうち使用を選んだ割合）で確認してください。この割合を `behavior_error` にも含め、勝率が同じモデルの採用判断に使います。平常時の使用すべてが無駄とは限らず、切り替え直後の使用を禁止する仕様ではありません。

```powershell
# concon_v1 内で、既存の配置行動を検証・保持して対戦学習を3000話追加
python co1_train_guard.py -map L --resume data/guard_L_data/co1_guard_L_best.pt --episodes 3000 --positioning-steps 0
```

開始時に事前学習の実更新回数、追加対戦数、開始・終了エピソード番号を表示します。
チェックポイントには `positioning_training`、`battle_training_run`、`battle_episodes_this_run` を保存し、回数を区別できるようにします。
追加学習の効果は固定条件での `best` の勝率・最低チーム別勝率・配置/停止の成績を比較してください。
改善しない場合は回数だけを増やさず、交戦時の照準・移動射撃・解除阻止の評価を見て学習対象を見直します。

## 学習と評価

`co1_train_guard.py` 冒頭の左右フラグで学習対象を選びます。

```python
TRAIN_LEFT_SITE = True
TRAIN_RIGHT_SITE = True
```

両方 `True` なら L、R の順でそれぞれ `DEFAULT_EPISODES` 回学習します。
左だけなら `TRAIN_RIGHT_SITE = False`、右だけなら `TRAIN_LEFT_SITE = False` にします。
両方 `False` は設定エラーです。回数・評価回数なども同じスクリプト冒頭の定数を変更します。
作業ディレクトリは `03.game/concon_v1/`、通常実行は次のとおりです。

```powershell
Set-Location concon_v1
python co1_train_guard.py
```

`-map L` / `-map R` は一時的に片側だけを実行する上書きとして残しています。
左右の保存先は既定では `data/guard_L_data/` と `data/guard_R_data/` です。
左右両方で保存先を指定した場合は、その下に `guard_L_data/` と `guard_R_data/` を作り、ログも分けます。
左右を再開する場合は `DEFAULT_RESUME_BY_MAP` に各サイトのチェックポイントを設定します。

```python
DEFAULT_RESUME_BY_MAP = {
    "L": Path("data/guard_L_data/co1_guard_L_best.pt"),
    "R": Path("data/guard_R_data/co1_guard_R_best.pt"),
}
```

各値が `None` なら新規学習です。単独サイトでは `DEFAULT_RESUME` / `--resume` が優先します。
左右両方に単一の再開モデルは指定できません。

```powershell
python evaluate_co1_guard.py -map L --rounds 36 --output data/guard_L_eval.json
python evaluate_co1_guard.py -map R --rounds 36 --output data/guard_R_eval.json
```

既定の対戦相手は attacker A1/A2/A3 と同じ6チームです。

| 指定名 | 対戦チーム |
| --- | --- |
| `omoko_v1` | Omoko Gaming |
| `touyama_v2` | Touyama Gaming |
| `fnatic_v3` | Fnatic2023 |
| `gc_v1` | Ghost Champions |
| `toru_ai_v3.1` | Team Elites |
| `frc_v1` | Furina Classic |

学習は既定で各サイト1000エピソード、50エピソードごとにチェックポイントを保存します。
対戦相手は全チームと1試合ずつ戦う巡回方式で、巡回ごとに順番をシャッフルします。
学習開始からの累計試合数の差は常に最大1試合です。6チームなら6試合ごとに同数になります。
1000試合のようにチーム数で割り切れない回数では、端数の分だけ1試合の差が残ります。
50試合ごとの表示は区間内の集計で、巡回をまたぐため完全には揃いません。累計のチーム別試合数も表示します。
開始状態もカリキュラムの各段階で、チームごとに使用回数の少ない状態から選び、状態間の回数差を最大1に抑えます。
探索率が下限0.05に達してから、探索なしで各チーム36試合を評価します。
`best` の更新には、6チームの平均勝率と最も低いチーム別勝率が両方とも以前以上であることを要求します。
配置・停止の検証を通過したモデル同士で比較します。両方の勝率が同じ場合は、配置点からの離脱・往復・不適切な facing・移動射撃の割合を平均した `behavior_error` が以前以下の場合だけ更新します。
既存の `best` がある場合は、同じ評価条件で再評価して比較します。

```powershell
python co1_train_guard.py -map L --episodes 5000 --eval-rounds 60
python co1_train_guard.py -map R --save-dir data/guard_R_trial
python evaluate_co1_guard.py -map R --model data/guard_R_trial/co1_guard_R_best.pt
```

`--opponents` にチーム指定を並べると対象を絞れます。指定しなければ6チームすべてを使います。
`--seed`、`--checkpoint-interval`、`--device cpu|cuda` も指定できます。
`--resume` は重みを引き継ぐ追加学習です。optimizer、リプレイ、探索率のスケジュールは新しく始まります。

## 学習の開始状態

学習環境は通常ゲームの設置後状態を合成し、通常のtick・射撃・アビリティ・IQ知覚・敵AIで進めます。
carry モデルによる設置前の戦闘は実行しません。生存人数は各チーム1〜5人、HPと使用済みアビリティも変えます。
敵AIは開始状態の設定後、各チームの通常のリテイク処理を使います。

| 開始状態 | 内容 |
| --- | --- |
| `hold` | attacker は割り当てられた防御点に配置され、対応する大文字の方向を向く。defender はサイトからBFS距離8〜20マスで開始。 |
| `transition` | attacker はスパイクからBFS距離2〜6マスに配置され、防御点への移動も学ぶ。defender は距離8〜20マスで開始。 |
| `smoke` | attacker は距離2〜6マスで開始。スパイク周囲のスモークと、解除を1tick進めたdefenderを設定し、解除阻止を学ぶ。 |
| `pressure` | attacker は距離2〜6マス、defenderは距離2〜8マス。attackerにフラッシュ・リコン・電撃のいずれかが残る状態から開始し、退避とカウンター使用を学ぶ。 |
| `pressure_tap` | スモーク内の解除が1〜4tick進み、attackerに被効果が残る状態から開始。退避と解除阻止の判断を学ぶ。 |

既定では前半20%が `hold`、次の20%が `hold` / `transition`、残り60%が5状態の混合です。
`smoke` の開始スモークは開始前に使用されたものを表す合成状態です。敵AIに毎回スモーク使用や解除継続を強制しません。
評価では5状態を順番に使い、チーム別・開始状態別に勝率、解除率、終了理由を出力します。
`--rounds 36` は各チーム合計36試合です。5状態へ順番に割り当てます。

```powershell
# スモーク内解除の場面を重点的に追加学習する例
python co1_train_guard.py -map L --resume data/guard_L_data/co1_guard_L_best.pt --start-modes smoke --save-dir data/guard_L_smoke_trial
python evaluate_co1_guard.py -map L --model data/guard_L_smoke_trial/co1_guard_L_best.pt --start-modes smoke
```

学習モードを絞っても、`best` の選出は5状態すべての評価で行います。
評価結果は設置後状態に対する成績です。設置前からのラウンド全体の勝率とは別です。

## 配置と行動

- `a〜e` はキャラの配置先、`A〜E` は対応するキャラの facing 目標座標です。大文字は移動先ではありません。
- マップは各小文字・大文字を1点ずつ含めます。数字部分は通常ゲームの地形と一致する必要があります。
- 設置時の生存者に重複しない配置先をランダムに割り当てます。5人未満なら、どの点が空いても構いません。
- 一度決めた割り当てはラウンド中変更しません。死亡した味方の配置先へ再割り当てしません。
- モデルは上下左右への移動、停止、8方向の facing、SMOKE / FLASH / RECON の使用を選びます。
- アビリティの対象候補は設置位置、割り当てられた facing 目標点、知覚できた敵の位置です。アビリティtickではゲーム仕様に従い現在の向きを維持します。
- 推論側は解除時の接近や2tick停止を固定制御しません。モデルは解除通知、射撃可能性、スパイクへの射線、停止tick数、アビリティ残数などから判断します。
- 敵位置は本番と同じIQ知覚と12tickの味方共有記憶を使います。未発見の敵の実座標はモデルへ渡しません。
- 移動と射撃、スモーク隣接時の視認、Reconによるスモーク越し射撃は通常ゲームの処理を使います。射撃は自動です。

報酬は起爆・敵全滅による勝利を +10、解除などの敗北を -10 とし、死亡、敵へのダメージ、配置への移動、facing、射撃可能な位置での連続停止を補助します。
死亡時の -0.5 は一度だけ加え、死亡したキャラの最後の行動もラウンド終了まで保持して勝敗報酬を反映します。死亡や移動不能で判断を行わない tick も割引期間に含めます。
通常時は配置点での停止を +0.04、移動を -0.02、配置点からの離脱を追加で -0.06 とします。配置点以外で移動可能なのに待機した場合は -0.03 です。
通常時の facing は正面を +0.02、背面を -0.02 として、その間は角度に応じて補間します。交戦時は正面 +0.04、背面 -0.04、移動 -0.04、連続停止時は向きに応じて最大 +0.03 です。
解除通知があり射撃できない場合は、モデルがスパイクへ接近するよう距離の報酬を切り替えます。
この場合は通常時の配置維持・離脱・facing の報酬を適用せず、移動できない場合やアビリティを使う場合には待機罰則も加えません。
評価出力には、解除通知も射撃可能な敵もない状態での `leave_goal` / `reversals` / `quiet_bad_facing` と、交戦中の `moving_fire` / `bad_fire_facing` を表示します。分母となる判断数も JSON に保存します。
地形の通行不可・占有・アビリティ残数は行動マスクで扱います。解除通知によって移動やReconを強制するマスクは使いません。
被射撃によって facing が固定される tick は、ゲーム側で実行できない旋回だけをマスクします。移動先の選択肢は維持します。
現在の行動にはウルト使用も含めます。

### 被効果中の退避とカウンター（報酬バージョン4）

guard の入力には自身のフラッシュ、リコン、電撃、命の契約、移動阻害の残り時間と、上下左右・現在地からの敵射線数を追加しています。射線はIQ知覚で共有された敵位置と12tick以内の記憶から算出し、未発見の敵座標は参照しません。遮蔽の判断は壁を優先し、自身がリコン・電撃で露見している場合はスモークを安全な遮蔽として扱いません。スモークに入っているだけでは退避状態になりません。

被効果中は通常の配置維持・交戦停止の補助報酬を外し、敵射線が減る移動を評価します。射線上で移動可能なのに待機する行動は減点し、カウンター使用には平常時の消費コストを適用しません。モデルには自己位置へのSMOKEと、直前に共有された敵位置へのアビリティ使用を残しています。使用そのものへの成功加点はなく、実際のダメージ・死亡・勝敗から使用の有効性を学びます。

解除の残り時間が自身のスパイクまでのBFS距離+2tick以下になる場合は、退避の射線減少報酬を外し、解除阻止の接近・照準を評価します。移動・停止・アビリティを強制する推論処理は追加していません。被効果中の退避は追加学習で習得する行動であり、旧モデルのままで必ず逃げるという仕様ではありません。ゲームでは通常の毎tickの移動を使い、移動速度の追加変更は行いません。

`pressure` / `pressure_tap` は被効果直後を表す合成開始状態です。効果前に通常のIQ知覚で見えた敵だけを記憶し、その後は本番のIQ知覚・敵AI・射撃・効果時間で進めます。評価には被効果中の判断数 `impaired`、射線上の待機数 `exposed_wait`、射線が減った移動数 `cover_moves`、アビリティ・ウルトを選んだ数 `counter_utility` を表示します。

旧112/136行動・旧観測のチェックポイントは既存の重みを保持して読み込めます。追加入力の重みは0から始めます。defender retake が共有する旧観測と行動番号は維持しています。

通常の追加学習では、`co1_train_guard.py` 冒頭の `TRAIN_LEFT_SITE`、`TRAIN_RIGHT_SITE`、`DEFAULT_RESUME_BY_MAP`、`DEFAULT_EPISODES`、`DEFAULT_OPPONENTS`、`DEFAULT_SAVE_DIR`、`DEFAULT_DEVICE` を変更します。片側だけ再開する場合は `DEFAULT_RESUME` でも指定できます。相手の既定値は上記6AI、開始状態の既定値は5状態を含むカリキュラムです。

作業ディレクトリは `03.game/concon_v1/` です。通常の実行コマンドはオプションなしです。

```powershell
# 03.game から作業ディレクトリへ移動
Set-Location concon_v1
python co1_train_guard.py
```

評価設定は `evaluate_co1_guard.py` 冒頭の `DEFAULT_MAP`、`DEFAULT_MODEL`、`DEFAULT_ROUNDS`、`DEFAULT_OPPONENTS`、`DEFAULT_OUTPUT` を変更し、同じ作業ディレクトリで `python evaluate_co1_guard.py` を実行します。学習はユーザーが手動で実行します。

### 報酬修正後の再学習

新しいチェックポイントには `reward_version=2` を保存します。旧モデルの重みは読み込めますが、コード修正だけで旧モデルの行動は変わりません。新しい報酬で再学習してください。
旧モデルを比較用に残して学び直す場合は、次のように別の保存先を指定できます。

```powershell
python co1_train_guard.py -map L --save-dir data/guard_L_reward_v2
python co1_train_guard.py -map R --save-dir data/guard_R_reward_v2
```

別の保存先で作ったモデルを通常対戦で使うには、登録 factory の `model_path` にその `best` を指定してください。既定の保存先で学習する場合は、既存の `best` と再評価して比較し、選定条件を満たすと通常対戦用のモデルを更新します。

## 保存先とパターン追加

| マップ | 配置定義 | 保存先 |
| --- | --- | --- |
| `L` | `co1_map_guard_L.py` | `data/guard_L_data/` |
| `R` | `co1_map_guard_R.py` | `data/guard_R_data/` |

各保存先には `co1_guard_L_best.pt` / `co1_guard_L_latest.pt` のようなファイル、エピソード別チェックポイント、`training_log.jsonl` を作成します。
モデルにはマップ名、配置・facing・地形の署名、観測・行動のサイズを保存します。別マップや変更前のマップのモデルは読み込みエラーになります。
マップ変更後に旧 `best` が残る場合は、別の `--save-dir` で再学習してください。

パターンを増やす場合は、たとえば `co1_map_guard_L2.py` を作り、`co1_guard_scenarios.py` の `SCENARIOS` に追加します。

```python
"L2": GuardSettings("co1_map_guard_L2", "left"),
```

共通ファイルを複製せず、次のコマンドで学習・評価できます。

```powershell
python co1_train_guard.py -map L2
python evaluate_co1_guard.py -map L2
```

## 通常対戦への接続

通常対戦・大会では、`co1_attacker_scenarios.py` の `CONCON_ATTACKER_POSTPLANT_MODELS` に左右の学習済みモデルを登録しています。
攻撃側はスパイク設置後、実際の設置サイトに応じて左なら L、右なら R の `best` を読み込んで使用します。
このモデルは攻撃側の guard 用です。防御側のリテイクは `ConconDefenderController` の標準動作です。
設定は以下の形です。
循環importを避けるため、推論クラスのimportはfactoryの中で行います。

```python
def left_guard():
    from concon_v1.co1_learn_guard import ConconGuardController
    return ConconGuardController(map_name="L")


def right_guard():
    from concon_v1.co1_learn_guard import ConconGuardController
    return ConconGuardController(map_name="R")


CONCON_ATTACKER_POSTPLANT_MODELS = {
    "left": (left_guard,),
    "right": (right_guard,),
}
```

同じサイトに複数factoryを設定すると、既存のラウンドコントローラが設置時に1つ選びます。
推論は既定で `best` を読み込みます。`latest` を試す場合はfactoryから `model_path` を指定してください。

## 確認

```powershell
python -m unittest discover -s test -p test_co1_guard.py -q
```

配置の維持、敵情報の非開示、スモーク／Recon射線、停止tick、モデル互換性、6チーム×左右サイトの通常ゲーム処理、評価時のRNG保持を確認します。
