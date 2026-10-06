# Defender retake 学習

`concon_v1` ディレクトリから実行します。

## 定数を変更して実行する場合

`co1_train_defender_retake.py` 冒頭の設定を変更し、`python co1_train_defender_retake.py` だけで実行できます。現在は収集データを使用し、基礎モデルから10周新しく学習する設定です。

| 定数 | 現在値 | 学習条件 |
| --- | --- | --- |
| `USE_COLLECTED_CASES` | `True` | 保存した設置直後から学習。`False`ならsetup/searchから対戦 |
| `CASES_DIR` | `data/defender_retake_cases` の絶対パス | 収集データの読込先 |
| `RESUME_TRAINING` | `False` | foundationから新規開始。`True`なら左右のlatestとoptimizerを引き継ぐ |
| `DEFAULT_CASE_EPOCHS` | `10` | データ学習の周回数。使用データ500件なら追加5000 episode |
| `DEFAULT_EPISODES` | `2000` | `USE_COLLECTED_CASES=False`の場合の追加episode数。データ学習では使わない |
| `CHECKPOINT_INTERVAL` | `100` | 追加100 episodeごとに評価・保存 |
| `DEFAULT_EVAL_ROUNDS` | `None` | 評価件数は全AI合計で評価間隔と同数。整数指定なら各AIあたりの評価件数 |
| `EVAL_START_EPSILON` | `0.05` | 学習のepsilonが0.05以下になった後の評価タイミングから実戦評価を開始。それ以前もlatestを保存 |
| `EPSILON_START` / `EPSILON_END` | `0.15` / `0.05` | 今回の追加学習の探索率。再開時にもstartから始める |
| `EPSILON_DECAY_RATIO` | `0.7` | 今回の追加学習の最初の70%で探索率をendまで下げる |
| `FORCE_SAVE` | `False` | latest/bestは保存。`True`なら評価時の番号付き重みも追加保存 |
| `FOUNDATION_LOSS_WEIGHT` | `0.5` | 非交戦時の集合・移動・解除を維持する補助損失の重み。`0`なら補助損失を無効化 |
| `FOUNDATION_MARGIN` | `0.5` | 基礎モデルが高く評価する合法な操作と、その他の操作のQ値の差の目安 |

例えば10000 episode追加したい場合、使用データが500件なら `DEFAULT_CASE_EPOCHS = 20` にします。基礎からデータ学習をやり直す場合は `RESUME_TRAINING = False` にします。通常のsetup/searchからの学習に戻す場合は `USE_COLLECTED_CASES = False` にし、`DEFAULT_EPISODES` で追加回数を指定します。収集途中で使用件数が500件未満なら、実際の追加回数は使用件数×周回数になります。

## コマンド引数で設定する場合

明示した引数は対応する定数より優先します。以下の例で基礎から始める場合は `--no-resume`、通常のsetup/searchから学習する場合は冒頭の `USE_COLLECTED_CASES = False` を設定してください。

```powershell
python co1_train_defender_retake_base.py
python co1_train_defender_retake.py --episodes 1000 --no-resume
```

リテイク学習スクリプトは左右のモデルを同時に管理し、実際にプラントされたサイトのモデルだけに経験を追加します。実戦学習では、サイトや生存人数を人工的に変更する開始モードはありません。基礎学習とリテイク学習は、それぞれ専用スクリプトから起動します。

## 収集データからの学習

`--cases-dir` を指定すると、学習は保存した設置直後の状態から始まります。相手AIの進行状態、双方の位置・HP・アビリティ・効果・残り時間、IQ知覚を復元し、defender の行動を現在の学習用モデルへ接続します。相手AIも通常どおり判断・行動します。保存状態の再生動画を模倣する学習ではなく、その状態から実際に試合を進める強化学習です。

```powershell
# 収集した500件を10周: 左右合計5000 episode
# 既存の latest と optimizer を引き継ぐ場合
python co1_train_defender_retake.py --cases-dir data/defender_retake_cases --case-epochs 10 --resume

# 基礎モデルから新しく始める場合
python co1_train_defender_retake.py --cases-dir data/defender_retake_cases --case-epochs 10 --no-resume

# 周回数ではなく追加 episode 数で指定する場合
python co1_train_defender_retake.py --cases-dir data/defender_retake_cases --episodes 5000 --resume
```

データ学習の既定値はスクリプト冒頭の `DEFAULT_CASE_EPOCHS = 10` です。`--cases-dir` を指定し、回数引数を省略すると「使用データ件数 × 10 episode」になります。500件なら5000 episode、左右250件ずつなら各サイトモデルが2500回のリテイクを経験します。私はまず10周を初回の目安とし、実戦評価が改善している場合に追加学習する方針を勧めます。500件だけでは最適な学習回数は確定できません。

スクリプトは各AI・左右のデータを同数選びます。収集完了時に各50件なら500件すべてを使用します。収集中などで件数が違う場合は、最少の組の件数にそろえて選び、保存済み件数・使用件数・使用しない追加件数を表示します。各AI・各サイトに少なくとも1件が必要です。読み込む一覧は開始時に固定し、収集スクリプトがその後追加した場面は次回起動時に取り込みます。

1周ごとに各組のデータと対戦順をシャッフルし、その周で各使用場面を1回ずつ使います。スクリプトは保存時の全体乱数系列へ毎回戻さず、学習時の探索・相手AIの乱数を進めます。データ学習の `--episodes` と `--checkpoint-interval` は「対戦相手数 × 左右2サイト」で割り切れる値にしてください。5チームなら10の倍数です。

評価は学習時のepsilonが `EVAL_START_EPSILON`（現在0.05）以下になってから、`CHECKPOINT_INTERVAL` ごとのタイミングで実行します。それ以前は評価を省略し、latestと指定時の番号付き重みを保存します。未評価の重みでbestを更新しません。5000 episode・減衰割合0.7・評価間隔100なら、3500 episodeから評価します。

評価はデータセットから開始せず、従来どおりsetup/searchから各AIと対戦し、評価中はepsilon=0です。評価数は既定で `CHECKPOINT_INTERVAL` と同数です。現在のスクリプト設定は評価開始後、100 episodeごとの合計100リテイク評価です。評価には設置前の処理時間もかかります。保存した場面への慣れだけで best を選ばず、実戦での解除率で比較します。

search 重みは収集時と同じ内容が必要です。異なる場合は収集時の重みを `--search-model` で指定します。既存の基礎モデルの検証、アビリティ定点・BFS距離、報酬、latest/best の保存はデータ学習でも共通です。`USE_COLLECTED_CASES = False` にして `--cases-dir` を省略すると従来のsetup/searchからの学習になり、回数の既定値は `DEFAULT_EPISODES`（現在2000）です。

## 基本移動・解除の事前学習

`co1_train_defender_retake_base.py` は敵のいない基礎課題を学習します。開始候補は `co1_retake_start_positions.py` の `START_CELLS` に明示してあります。defender search のライブフェーズの防御地点と、その地点から BFS 距離2以内の床マスを使用します。

| search 地点 | 基準位置（行、列） | 開始候補数 |
| --- | --- | --- |
| a | (11, 12) | 8 |
| b | (11, 16) | 8 |
| c | (10, 21) | 9 |
| d | (11, 32) | 8 |
| e | (14, 31) | 8 |

左右の設置可能位置それぞれについて、これらの開始候補から到達する経路を BFS の教師データにします。最短経路の分岐と、ランダム探索で経路から2歩程度外れた場合の復帰も含めます。IQ知覚でスパイク位置がずれる床マスも、本番と同じ誤差処理から列挙して教師データに含めます。教師データの順番と検証用の開始・設置位置は seed を使ってランダムに選びます。

移動4方向・待機・解除の6操作に対する値を学習します。解除可能な距離では解除を選び続けることを学びます。学習後は、本番の IQ 知覚でモデルが移動を選び、ゲームの解除処理で6 tick を継続して解除できることを確認します。検証に BFS の行動指示は使いません。左右とも検証に合格してから `co1_defender_retake_L_foundation.pt` と `co1_defender_retake_R_foundation.pt` を保存します。

```powershell
# 基礎学習の更新数・検証回数を変更する例
python co1_train_defender_retake_base.py --steps 250 --eval-trials 20

# 基礎モデルから実戦学習を開始
python co1_train_defender_retake.py --episodes 1000 --no-resume

# 新形式の戦闘モデルを再開
python co1_train_defender_retake.py --resume --episodes 1000
```

`co1_train_defender_retake.py` では保存済みの基礎モデルを読み込みます。基礎の値は固定して残し、戦闘・facing・アビリティのネットワークを実戦で更新します。射撃可能な敵がいない場面では、集合中にも基礎の値をネットワーク出力に加えます。基礎学習は各Aへの全到達可能経路と、Aで待機する操作も含みます。移動の基礎値は現在の目標（担当Aまたはスパイク）を参照します。敵の位置記憶があっても、射線が通らなければ基礎の値を使えます。射撃可能な敵がいる場面では、実戦で学習するネットワーク出力を使います。

戦闘学習のQ値によって基礎動作が崩れにくいよう、集合中を含む非交戦状態には補助損失を適用します。現在の行動マスクで合法な移動・待機・解除の候補だけを比較し、基礎モデルが高く評価する操作のQ値を保つ方向に学習します。絶対的なQ値は固定せず、向きやアビリティ・ウルトの教師行動は指定しません。基礎値が未学習の位置や、現在の合法候補がない旧形式の経験には補助損失を適用しません。推論側でBFSの移動経路や解除行動を強制する処理はありません。

A集合の導入でマップ観測を10から12チャネルへ増やし、基礎値にもAへの移動を追加しました。旧foundation・best・latestは新形式と互換性がありません。`python co1_train_defender_retake_base.py` で基礎を再作成し、`python co1_train_defender_retake.py --no-resume` で実戦学習を始めてください。既存の収集ケースはゲーム地形を変更していないため再利用できます。

既存の戦闘モデルの `best` / `latest` は、基礎学習では変更しません。`--resume` は新形式の戦闘モデルからの再開に使用します。基礎モデルがない場合や検証に合格していない場合には、実戦学習を始めずに理由を表示します。独自の保存先を使う場合は、基礎学習と実戦学習の両方に同じ `--save-dir` を指定してください。

## 試合と学習範囲

- 試合は通常の 5 対 5 の setup から開始します。
- 設置前は既存の defender search モデルで動きます。search の重みは固定し、更新しません。
- 設置後も同じ試合の位置・HP・生存人数・残りアビリティ・ウルトポイントを引き継ぎます。
- 設置後の行動だけを retake の replay に追加します。設置前に attacker が全滅した試合などはログに除外として記録します。
- episode は実際に defender が設置後の行動を行ったリテイク1回を数えます。リテイクが発生しない試合は episode に数えず、同じ相手と setup から再試合します。設置と同時に defender が全滅し、リテイク行動がなかった試合も除外します。リテイク後の敗北は1 episode に数えます。
- 設置された tick の途中でも、その後の defender 行動は retake に切り替わります。
- 学習・推論ともに本番の IQ 知覚を使用します。観測に入れる敵位置は、自身または味方が得た視認情報と短期記憶です。隠れた敵位置を待機場所の計算にも使いません。

既定の対戦相手は `touyama_v2`、`omoko_v1`、`gc_v1`、`toru_ai_v3.1`、`fnatic_v3` の5チームです。左右合計200回の学習リテイクごとに評価します。各チームの学習リテイク数は40回ずつです。左右の設置先は試合の自然な結果に任せ、そのサイトのモデルに経験を追加します。左右別の回数は揃えません。

一方のサイトにしか攻撃しないチームでも、左右合計の40回を終えたらその区間の学習を完了します。リテイクが発生しない試合だけを除外して再試合し、反対サイトの発生を待ち続けません。epsilon は左右合計の学習リテイク数に従って進みます。チームごとの件数を同じにするため、`--episodes` と `--checkpoint-interval` は対戦チーム数で割り切れる値にしてください。

## モデルの行動

モデルは移動、8方向の facing、アビリティ、ウルト、解除を選択します。射撃は既存のゲーム処理に任せます。

通常の敵への交戦時には停止と照準を報酬で評価します。敵がフラッシュなどで弱体化した場合は、前進を通常の移動射撃と同じように減点しません。交戦後の短い照準維持、共有情報の方向を向いた接近も学習対象です。

移動先から射線が通る視認・共有・短期記憶の敵がいる場合は、その敵を向くfacingだけを移動・待機候補に残します。壁・スモーク・味方に遮られる敵へは照準を強制しません。被弾反応でゲームがfacingを固定したtickは、その固定を優先します。敵情報がない移動には進行方向への照準報酬を与え、Aで待つ間はサイトへ進む最初の通路を向く旋回を評価します。敵情報がない場合は8方向の選択肢を残すため、実戦経験から敵が現れやすい場所への事前照準も学習できます。照準だけで待ち続ける報酬を得ないよう、停止中は照準の改善量を評価します。

解除完了の報酬は `+10`、設置後の敗北は `-10` です。敵へのダメージ・撃破は解除へ進むための補助報酬とし、敵の全滅だけを retake の成功とは扱いません。解除はゲームの `DEFUSE` 行動を選び、隣接距離と既存の同時解除制限に従います。

## アビリティのマップ

`co1_map_retake_L.py` と `co1_map_retake_R.py` のマーカーを使用します。

| マーカー | 対象 | 発射条件 |
| --- | --- | --- |
| A | 集合候補 | 上下左右に隣接するAを同じ入口の集合エリアとして扱う |
| S | スモークの定点 | 指定の BFS 距離以内。壁越しでも使用可能 |
| F | フラッシュの定点（現マップでは2か所） | 指定の BFS 距離以内、対象まで壁に遮られない射線がある |
| R | リコンの定点 | 指定の BFS 距離以内、対象まで壁に遮られない射線がある |

スモークには、実際のスパイク位置も発射先として追加します。この対象にも BFS 距離条件を適用します。スモークに隠れた解除を観測と報酬に含めます。

Flash・Recon・Smoke の BFS 距離は、左サイト用の `--flash-distance-l`、`--recon-distance-l`、`--smoke-distance-l` と、右サイト用の `--flash-distance-r`、`--recon-distance-r`、`--smoke-distance-r` で個別に指定します。現在の既定値は左が Flash=7、Recon=11、Smoke=16、右が Flash=7、Recon=6、Smoke=10 です。両スクリプトで共通設定ファイル `co1_retake_config.py` を使用します。`DEFAULT_FLASH_DISTANCE_L`、`DEFAULT_RECON_DISTANCE_L`、`DEFAULT_SMOKE_DISTANCE_L` と、末尾が `_R` の3定数でも変更できます。ULT には BFS 距離制限を設けません。

ウルトはポイントなどのゲーム側の使用条件を満たし、視認・共有された敵情報があるときに選択できます。対象指定のウルトは敵位置を使います。方向指定のウルトはモデルが facing を選びます。

アビリティが使用可能でも移動・待機・解除の行動候補を残します。推論で発射や突入を強制する処理はありません。

## 合流と残り時間

集合先にはマップのAだけを使用します。各味方に異なるAを割り当て、入口ごとの人数上限を生存人数と入口数から決めます。現在の左右マップは各2エリアで、5人生存なら最大3人ずつに分散します。担当Aは保持し、後続が先行者の位置を追いかけることや毎tickの集合先変更を防ぎます。観測には全Aと味方の担当Aを別チャネルで追加します。

集合経路はAからサイトへ突入する通路を通りません。各Aから同じサイトの設置可能マスへの最短経路と、その通路に囲まれた領域をサイト側とし、集合中は外側からサイト側への移動を行動マスクで禁止します。敵の実位置は参照しません。既にサイト側にいる味方は外側へ戻れます。Aの割り当て・到着距離・前方Aへの移動条件・基礎学習・移動報酬は同じ集合用距離を使用します。左の `(11,16)` から下側Aへは、`(12,16)`、`(15,16)`、`(18,12)`、`(18,9)` を経由する南側の経路を学習します。突入開始後はサイト側への移動を許可します。

集合経路の制約はチェックポイントの `coordination_version=2` で識別します。制約追加前の基礎・戦闘モデルは再作成してください。基礎学習、リテイク学習の順に同じコマンドを実行し、収集ケースは再利用できます。

ゲームで修正前の学習結果を確認する場合は、`coordination_version=1` のモデルも読み込めます。ゲームは保存バージョンに対応する旧集合経路を使用し、起動ログに `coordination_version=1` と表示します。モデルのマップ・観測・行動・報酬形式の検証は維持します。新しい経路制約を反映する学習ではバージョン2の基礎モデルを必要とし、バージョン1のモデルを新方式として再開することはありません。

先行者が一度Aへ到着した場合、後続の到着までのBFS距離に3マスを足した範囲で、スパイクに近く、人数上限を超えない別エリアのAへ進めます。各先行者は同じエリアへ戻らず、後続の担当Aは変えません。先行者が移動途中でも、後続の準備が整った時点で突入状態へ切り替えます。到着判定はIQの味方位置誤差を考慮して担当Aの経路距離1以内を許容し、到着履歴を保持します。

突入開始はチームで共有し、一度突入したら集合状態へ戻りません。集合・移動・解除6 tick・戦闘余裕3 tickが残り時間に収まらない味方は待機対象から外します。残り時間が少なければ未集合でも突入します。生存者が1人の場合や、味方が既に解除位置まで進んでいる場合も集合を切り上げます。突入後も担当入口へ向かう経路は使いますが、Aでの停止は不要です。時間不足の場合は直接解除へ向かう目標を使います。

集合中はSmoke・Flash・Reconの定点使用とスパイクへのSmokeを行動マスクで禁止し、突入開始時に許可します。敵情報に基づくウルトは従来の条件で使用できます。

これらの計算は観測と学習報酬のために使います。実際の移動・待機・突入は DQN の出力です。学習後の行動と勝率は手動学習後に確認してください。

## 引数と保存

```powershell
# BFS 距離や既存 search の重みを明示する例
python co1_train_defender_retake.py --episodes 1000 --flash-distance 6 --recon-distance 6 --smoke-distance 6 --search-model data/defender_search_data/co1_defender_search_best.pt

# 左右サイトの距離を個別指定する例
python co1_train_defender_retake.py --flash-distance-l 7 --recon-distance-l 11 --smoke-distance-l 16 --flash-distance-r 6 --recon-distance-r 10 --smoke-distance-r 15

# 既定の保存先から左右モデルの学習を再開（1000リテイク追加）
python co1_train_defender_retake.py --resume --episodes 1000

# 探索率と番号付き保存を指定する例
python co1_train_defender_retake.py --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay-ratio 0.7 --force-save

# 左右を一つのディレクトリに保存する場合
python co1_train_defender_retake.py --save-dir data/retake_run --episodes 1000
python co1_train_defender_retake.py --resume-dir data/retake_run --save-dir data/retake_run --episodes 1000
```

`--search-model` の省略時は本番と同じ search 重みを選びます。既定の保存先は `data/defender_retake_L_data` と `data/defender_retake_R_data` です。それぞれ `co1_defender_retake_L_latest.pt` / `co1_defender_retake_R_latest.pt` と、評価で採用した `best.pt` を保存します。latest には再開用の optimizer も保存します。再開時にはマップ、観測、行動、BFS 距離設定の一致を検証します。

距離設定は各サイトの重みに、そのサイトの3種類の値を保存します。本番の推論と評価も同じ設定を使います。以前の `--flash-distance`、`--recon-distance`、`--smoke-distance` は左右共通の指定として使用できます。優先順位は「サイト・アビリティ別引数」「左右共通のアビリティ別引数」「全体共通の `--ability-distance`」「サイト別の既定値」です。以前の共通距離形式の重みも読み込めます。再開時には各サイトの保存値と指定値が一致することを検証します。

探索率は `--epsilon-start`（初期値0.15）、`--epsilon-end`（初期値0.05）、`--epsilon-decay-ratio`（初期値0.7）で指定します。事前学習した移動・解除を利用できるよう、初期値を0.15に下げています。追加リテイクの最初に start から始め、指定割合まで線形に減らし、その後は end を維持します。除外試合では epsilon を進めません。再開時も今回追加するリテイク数を対象に減衰します。評価は常に epsilon=0 です。設定と保存時点の epsilon は重みに記録します。

スクリプト冒頭の `EPSILON_START`、`EPSILON_END`、`EPSILON_DECAY_RATIO`、`FORCE_SAVE` でも既定値を変更できます。コマンド引数を指定した場合は引数を優先します。`--force-save` は保存間隔ごとと最終試合に左右の番号付き重みを追加し、`--no-force-save` は番号付き保存を無効にします。latest と評価で採用した best は番号付き保存の設定に関係なく保存します。

`--episodes` は左右合計の追加学習リテイク数、`--checkpoint-interval` は評価までの学習リテイク数（既定200）です。評価リテイク数の既定値は全チーム合計で `--checkpoint-interval` と同じです。5チーム・既定200なら各チーム40回、合計200回を評価します。`--eval-rounds` を指定すると、各チームの左右合計の評価リテイク数を上書きします（例: `--eval-rounds 10` は5チーム合計50回）。`--episodes 1000` は左右合計1000回を学習し、5チームなら各チーム200回を学習します。

評価も各チームの左右合計回数を揃え、サイトは自然な設置先に従います。既定値では各チーム40回、5チーム合計200回の評価リテイクを集めます。設置前に終了した試合などの除外分は、この200回に含みません。成績は左モデルと右モデルを分けて表示します。各モデルの解除率と best の判断には、そのモデルの対象サイトでの評価結果だけを使います。そのサイトの評価リテイクが0件の場合は `N/A` と表示し、そのモデルの best 更新を行いません。

ログの `round` は除外を含めた実試合数、`episode` は学習対象のリテイク数、`counted_episode` はその試合を数えたかを表します。対戦相手別の `excluded_no_retake` には、設置前終了と設置後に行動できなかった試合を含めます。除外試合を学習経験に追加せず、評価・保存の区切りにも数えません。リテイクが発生しない間は再試合を続け、10試合ごとに待機状況を表示します。

学習ログの `episode` は左右合計の学習回数です。`training_episodes` に左右それぞれの内訳を保存します。保存モデルの `episode_counting` は `team_balanced_retake` です。

best は学習実行ごとに選択し、平均解除率、対戦相手別の最低解除率、通常の敵への移動射撃率の順で比較します。相手別の評価件数は `evaluation` に保存します。`retake_training_log.jsonl` は設置前の除外試合も含みます。`--force-save` を指定すると保存間隔ごとに番号付き重みを追加します。

通常の試合では `ConconDefenderController` が既定の保存先の左右 best を設置後に読み込みます。独自の `--save-dir` に保存した重みを単独で読み込む場合は `ConconDefenderRetakeController(model_path=...)` を使います。

`run_game.py` などで「ConCon v1」（Gorigonsの既定AI）を選ぶと、設置前はsearch、設置後は設置サイトに応じたL/Rのbestを使用します。既定の読込先は次の2ファイルです。

- L: `data/defender_retake_L_data/co1_defender_retake_L_best.pt`
- R: `data/defender_retake_R_data/co1_defender_retake_R_best.pt`

各サイトを初めて使った時に、`[ConCon defender retake] site=L/R model=... episode=... epsilon=0.000` と表示します。各モデルのアビリティ距離は、そのbestに保存したサイト別の設定を読み込みます。推論中は重みを固定し、探索を行いません。両モデルは同じコントローラー内で別々に保持し、ラウンドが変わると各モデルの記憶をリセットします。モデルファイルを更新した後に新しい重みを使う場合は、試合アプリを再起動してください。

bestが存在しないサイトでは、読込先を1度ログに出して標準defenderコントローラーを使用します。別ディレクトリの左右bestを指定する場合は、`ConconDefenderController(retake_model_paths={"L": 左モデルのパス, "R": 右モデルのパス})` として両方を指定できます。

## コンソールログ

学習集計の `elapsed_time=HH:MM:SS` は今回の起動からの経過時間（初期化・過去の評価・保存を含む）、`training_time=HH:MM:SS` はその区間の学習時間（前の評価・保存を除く）です。評価後は `evaluation_time` と、その時点の `elapsed_time` を表示します。終了時にも全体の経過時間を表示します。JSONLの各試合には `elapsed_seconds`、区間末には `window_training_seconds` を保存し、チェックポイントには評価の `evaluation_seconds` も保存します。時間は今回の実行ごとに0から計測し、過去の実行時間は加算しません。

学習区間と評価結果は、対戦相手ごとに成績を1行ずつ表示します。`defender wins` はリテイク中の解除成功数、`losses` はリテイク中の敗北数、`L` / `R` はサイト別の解除成功数 / リテイク数です。

成績行の `time_expired` は時間切れ（スパイク起爆を含む）の回数、`defender_eliminated` は defender 全滅の回数です。リテイクが発生した試合だけを集計し、学習・評価ともにチーム別・サイト別・全体の回数を表示します。search 中に defender が全滅した除外試合は、この敗因の回数に含めません。

```text
Training summary: episodes=1-200 attempted_rounds=250 total_rounds=250 epsilon=0.122
  omoko_v1: defender wins 20/40 (50.0%) | losses=20 | L=15/25 R=5/15 | time_expired=12 defender_eliminated=8
    retake_rate=40/50 (80.0%) | defuse_rate=20/40 (50.0%) | excluded=10
```

`retake_rate` は実試合に対するリテイク発生率、`defuse_rate` はリテイクに対する解除成功率です。この指標行、全チームの集計、サイト別の平均・最低解除率を defender search と同じ ANSI 緑色で表示します。リテイクが0件の解除率は `N/A` と表示します。待機中のログでもリテイク発生率を緑色で表示します。

評価ログは `L model evaluation:` と `R model evaluation:` の2グループに分けます。各グループのチーム成績の分母は、そのサイトで実際に発生した評価リテイク数です。チームごとの左右合計件数を揃え、各サイトの件数は表示された内訳で確認します。評価の `retake_rate` は対象サイトのリテイク数 / 実試合数、`excluded` は対象外の試合数、括弧内の `other_site` は反対サイトでリテイクが発生した回数です。

学習ログはチームごとに左右合計の成績を1行にし、`L` / `R` で内訳を表示します。5チーム・既定の200回区間では、各チームの成績の分母が40になります。

JSONL とチェックポイントのデータは引き続き構造化した形式で保存し、色の制御文字は含めません。

## テスト

```powershell
python -m unittest discover -s test -p test_co1_defender_retake.py
python -m unittest discover -s test -p test_co1_retake_foundation.py
```

テストは学習スクリプトを起動しません。観測・射線・BFS 距離・情報漏洩・合流と時間切迫・設置後の記録範囲・エンジンの解除完了を確認します。

基礎テストでは、教師が正しい場合の既知の値をメモリ内に置き、実際の IQ 知覚とゲーム処理で到達・解除できることを確認します。学習の更新処理は実行しません。実際の基礎学習結果は、ユーザーが手動で実行した際の検証で確認します。
