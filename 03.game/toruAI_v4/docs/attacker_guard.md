# Toru AI v4：guard の収集・学習

## 実行順序

guard学習は `MAX_PARALLEL_WORKERS = 3` で相手AIごとに並列実行する。学習ログは `logs/attacker_guard/<相手AI>/training.log`。収集は順次実行。設定は [並列学習](parallel_training.md) を参照する。

**plant学習完了 → 設置状態50件の収集 → guard学習**の順に実行する。plant学習中には同時実行しない。採用済みplant・analysisのbestを直接読み込み、bestファイルの別コピーやアーカイブは作らない。

作業ディレクトリは `03.game/toruAI_v4/`。通常設定は各スクリプト冒頭の名前付き定数を変更し、オプションなしで実行する。CLIは一時的な検証用の上書き手段。

```powershell
Set-Location C:\Users\ronet\MyProject\git\AI_dnn\03.game\toruAI_v4
# plant学習が終わってから実行
python tv4_collect_attacker_guard.py
# 収集が完了してから実行
python tv4_train_attacker_guard.py
```

## 収集

`tv4_collect_attacker_guard.py` の `CASES_PER_AI = 50` は相手ごとの左右合計50件。50セットや左右各50件ではない。6種類なら合計300件。左右を均等化せず、analysis・plantが実際に選んで設置したサイトを使う。

通常setupから12ラウンドを進め、実際の設置完了tick終了時、guardの最初の行動より前の状態を保存する。HP・位置・facing・アビリティ・ULTポイント・観測履歴・タイマー・飛行中の効果・相手コントローラー・乱数状態を維持する。未設置、終了済み、味方全滅、敵全滅は保存しない。配置・HP・生存人数は人工的に変更しない。

既存defenderのリテイクと同じゲーム状態シリアライザーと共有tensor保存を使う。ケース内にはシミュレーションに必要な非公開状態も含まれるが、行動入力は公開snapshotのみ。採用bestファイルを別の入力用ディレクトリにコピーする処理はない。保存後は公開情報による基準guardを使ってラウンドを最後まで進め、同じ相手AIのラウンド間状態を引き継ぐ。

収集時のbestのhash・採用セット・味方編成を記録する。収集再開・学習開始・推論時は一致を確認する。plantまたはanalysisが更新された場合は、plant学習完了後に収集し直す。

汎用plantのbestから収集するときは、その学習用編成をブロックごとに切り替え、相手と選手名が重ならない編成で50件を集める。guardの独立評価にはplantの学習外評価編成を使う。旧固定編成モデルのメタデータは参照・推論用に読めるが、通常の汎用guard収集・学習は固定編成のplantを入力として拒否する。固定編成のケース・評価だけで汎用AIの性能を確認したことにはしない。

収集設定は `CASES_PER_AI`、`TARGET_OPPONENTS`、`PLANT_BEST_DIRECTORY`、`ANALYSIS_BEST_DIRECTORY`、`RESUME_COLLECTION`、`MAX_COLLECTION_BLOCKS`、seed、保存先を冒頭で変更する。既定の試行上限は相手ごと100ブロック。足りなければ不足を表示して終了コード2を返す。同じ条件で `RESUME_COLLECTION = True` にすると続けられる。

新規収集は同じguard保存先の収集ファイルを置き換える。plant・analysis・defenderのデータやbestには書き込まない。`collection_summary.json` の `complete=true` を確認して学習を開始する。

## 行動・報酬

相手AIごとに1モデルを作り、味方生存者で重みを共有する。左右は実際の設置サイトを入力して区別する。

- 失明時は目撃履歴と地形から射線を切る退避地点を候補にする。合法なカウンターフラッシュも残す。表示中のFLASHの持ち主は非公開なので、その存在だけでカウンターを禁止しない。移動・発動不能の場合はエンジンの合法マスクを優先する。
- 設置周辺の遮蔽物・進入口・目撃地点から、味方を分散し、同じ脅威に複数方向の射線を通せる位置を考える。局所移動先の射線リスク・クロス射線を入力し、移動／待機と8方向facingを学習する。敵の隠れた位置やfacingは入力しない。
- 目的は解除阻止・爆発までの遅延・設置後勝利。撃破は補助に抑える。爆発または設置後勝利に+8、敗北に-8。解除されずに時間が進むことを小さく加点する。被害・死亡・無駄なアビリティ消費は減点し、終端の生存・温存加点は勝った場合だけにする。
- 失明中の射線切りとクロス射線の改善は行動前後の差で補助評価する。同じ場所で待つだけでクロス射線報酬を繰り返し得る方式にはしない。
- 公開の解除開始通知は設置地点の監視・妨害に使い、解除者の隠れた実位置を暴露しない。通常アビリティ・ULT・移動・待機・facing・オーブ回収は合法候補から選ぶ。attackerの解除候補は無効。

フラッシュだけでなく、持っている各通常アビリティを遅延・解除阻止につながる用途で学習する。教師方策にも種類別の例を用意する。

| 種類 | 守備中の用途 |
|---|---|
| FLASH | 突入・解除の妨害、合法な失明時カウンター |
| SMOKE | 侵入路の射線を切る。教師は解除者を隠すスパイク直上のスモークを避ける |
| ASH | 目撃・最終目撃地点、解除通知があれば設置地点を攻撃して妨害 |
| RECON | 敵が見えていないときに侵入方向を確認し、味方の警戒・妨害を助ける |
| RAMP | 侵入路や設置付近の現在地へ罠を置く。同じ場所への繰り返し要求を避ける |
| DANCE | 傷ついた味方を回復し、守備を継続できる時間を延ばす |

HUNT、SERENADEは既存エンジンの自動効果として扱い、架空の手動発動を追加しない。手動発動できるULTも合法候補に含める。使用そのものへの加点はせず、解除阻止・遅延・生存・最終勝利の結果で学習する。不要なら温存できる。ログには通常アビリティの種類別使用回数も出す。

## 学習・独立評価・best

`tv4_train_attacker_guard.py` の1セットは、相手AIの全保存ケースをランダム順に各1回学習する単位。50件なら1セット=50guardエピソードで、plantの1セット=12ラウンドとは異なる。

Double DQN、target network、共有replayと小さな教師模倣項を使う。ケースを復元したらguardモデルと探索設定だけを差し替え、ゲーム・相手・履歴・効果・タイマーをリセットしない。将来の行動・交戦乱数は学習用seedで変える。

学習設定は冒頭の `TRAINING_SETS`（既定30追加セット）、`TARGET_OPPONENTS`、`CASES_DIRECTORY`、`RESUME_TRAINING`、`EVALUATION_ONLY`、`EVALUATION_INTERVAL`（既定5セット）、`EVALUATION_SEED_COUNT`（既定3）、更新回数・batch・replay・seed・報酬・探索減衰・保存先・色で変更する。

実力評価は学習ケース再生ではなく、独立seedで通常setupから、固定plant・analysisと候補guardを使って12ラウンドずつ対戦する。既定は3seedで36ラウンド。教師・ランダム探索は無効で、重み・optimizer・学習replayを更新しない。

緑色 `[実力評価][通常setupから・教師/探索なし]` のguard勝率を見る。学習行は教師・探索を含む収集結果で、直近の実力評価セットも併記する。左右別結果も出し、設置されなかった側は対象なしと表示する。

- guard勝率：実際にguardへ入った設置後ラウンドの勝利数÷guardラウンド数。
- 全体勝率：設置前の決着も含む全評価ラウンドの勝利数÷全ラウンド数。
- 生存率：guard開始時を分母とし、終了時の敵味方生存人数を集計する。
- アビリティ率：guard開始時の通常アビリティ残数が分母。plant以前の消費・ULTは含めず、死亡者の未使用分を温存に数えない。

設定したxセットごとと最終セットに評価する。bestは同じseed・条件で、guard勝率→全体勝率→味方生存→少ない被害→アビリティ温存→少ない消費、の順に改善した場合だけ更新する。同点・悪化では以前の重みと採用セットを維持する。guard評価が0件ならbestを保存しない。

latestは毎セット保存する再開用で、bestとは別。モデル・target・optimizer・replay・抽出乱数・完了セット・累計guardエピソード・直近評価を復元する。収集データや使用bestが変わった状態では再開できない。

## 保存先

```text
toruAI_v4/
  tv4_collect_attacker_guard.py
  tv4_train_attacker_guard.py
  tv4_learn_attacker_guard.py
  tv4_attacker_guard_controller.py
  tv4_guard_runtime.py
  test/test_tv4_attacker_guard.py
  data/attacker_guard_cases/
    collection.json
    collection_summary.json
    cases.jsonl
    blocks.jsonl
    *.case.gz
    tensors/
  data/attacker_guard/<相手AI>/latest.pt
  data/best/<相手AI>/attacker_guard_best.pt
  logs/attacker_guard_collection/collection.log
  logs/attacker_guard/training.log
  logs/attacker_guard/<相手AI>_rounds.jsonl
```

`run_game.py` の「Toru AI v4」は、相手AI別のanalysis・plant・guardのbestを自動的に読み、設置完了tickの後にguardへ切り替える。`ToruV4AttackerPlantGuardController.from_best("gc_v1")` で相手を指定して読み込むこともできる。保存先と読み込み条件は [通常ゲームのbestモデル](game_best_models.md) を参照する。

学習中のplantを使う収集・guard学習は実行しない。通常ゲームへの接続は、学習完了後の保存済みbestを使う読み込み検証と、設置からguardへの実ゲームの切替で確認している。
