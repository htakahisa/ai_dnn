# toru AI v4 全体確認（2026-10-09）

対象はtoruAI_v4のdefender/attackerのanalysis、行動モデル、収集、評価、保存、推論の接続。既存モデルのメタデータを読み、ソースと単体テストを確認した。実対戦、データ収集、本学習はこの確認では実行していない。既存の学習データ・モデルをコード修正時に上書きしていない。

## 見つかった問題と修正

| 対象 | 問題 | 修正・確認結果 |
|---|---|---|
| defender analysis | 味方がGorigons固定だった | 複数学習編成を切替、学習外の複数編成で独立評価。編成とseedを保存 |
| attacker analysis / plant | 味方がGorigons固定だった | 複数学習編成と学習外評価編成に変更。旧固定条件での通常再開を拒否 |
| analysis共通入力 | 味方の能力種類・射撃性能を入力していなかった | 公開snapshotの最大HP、accuracy、hs、dodge、reaction、IQ、残数、ult情報と能力種類を追加。名前は入力しない |
| defender analysis評価のみ | セット数の分だけ単独編成を反復していた | 共通seed群・評価編成の組を使う一度の評価へ統一。モデルは更新しない |
| defender analysis / searchのbest | 前段モデル等を更新した新規学習でも旧bestの条件不一致で停止した | 新規条件の旧評価値は比較しない。同条件の再開では不一致を拒否 |
| plantのanalysis読込 | 重みとhashを別々のファイル読込から取得していた | 一度読んだbytesから両方を取得 |
| guard収集・評価 | plantの単独編成を引き継いでいた | 複数学習編成で実プラント状態を収集、plantの学習外評価編成で評価。旧固定plantから通常収集・学習しない |
| search / retake | 既に複数編成を使っていたが、CLIで学習・評価編成を重複指定できた | 共通入口で重複を拒否。前段analysisの更新後は依存hashが変わる |

編成プールは通常学習9編成、学習外評価3編成。敵と同じ選手名を含む編成はその相手との対戦から除外する。既定の全6相手に複数の有効学習・評価編成がある。任意編成への対応は能力・状態を入力する共有モデルによる設計であり、未知の全編成に対する高性能を検証したという意味ではない。

## 保存済みモデルの確認

確認時点ではdefender analysisの全6相手のbest/latestがGorigons固定、旧入力version 2だった。attacker analysisの5相手のbestは固定編成、gc_v1のbest/latestは複数編成だったが能力情報追加前のversion 1だった。通常保存先のplant best（gc_v1、touyama_v2、omoko_v1）も固定編成だった。

searchの6相手のbest/latestは複数編成だった。ただし前段は旧defender analysisなので、新analysisに無条件で差し替えて継続してはいけない。

入力形式はdefender analysis version 3、attacker analysis version 2へ更新した。旧形式の重みを新形式の入力に接続しない。起動済みのプロセスは既に読み込んだ旧コードを使い続けるため、実行中にソースを変更しても新しい入力形式へ切り替わらない。

## bestとlatest

全6学習入口で、設定されたxセットごとと最終セットに独立評価し、latestとbestを区別する。評価対象は通常のゲーム状態から開始する。guard/retakeの保存ケースでの学習成績だけをbestの根拠にしない。評価編成・seedを共通にし、同条件で同点・悪化ならbestを更新しない。新規学習で条件が変わる場合は新条件の評価を基準にする。

確認時点の既存bestには最終セットより前の採用例がある。例えばfnaticのattacker analysisは採用40セット目に対しlatestは60セット、defender analysisは採用20セット目に対しlatestは30セット。latestが無条件でbestになっているわけではない。ただしこれらは旧編成・入力条件の結果であり、今回の汎用版の性能値として使わない。

守備側の公開入力、敵の隠れた座標を入力しないこと、plantまでとguardの区切り、guardの失明時退避・カウンター・複数射線、各手動アビリティの合法候補を既存・追加の単体テストで確認した。敵の実座標や最終結果は学習ラベル・報酬・集計だけに使用する。

通常の実行確認用 `tv4_run_defender.py` のGorigonsは一試合の味方編成の選択であり、学習モデルの固定編成制約ではない。相手の既存AIに固定されている相手編成も、toru AI v4の味方固定とは区別する。

## 新しい条件での実行順

二つの学習系列は独立している。attackerのためにdefender analysisから始める必要はない。

| 系列 | 順番 |
|---|---|
| defender | `tv4_train_defender_analysis.py` → `tv4_train_defender_search.py` → `tv4_collect_retake.py` → `tv4_train_retake.py` |
| attacker | `tv4_train_attacker_analysis.py` → `tv4_train_attacker_plant.py` → `tv4_collect_attacker_guard.py` → `tv4_train_attacker_guard.py` |

作業ディレクトリは `03.game/toruAI_v4/`。各スクリプト冒頭の対象AI、編成、学習回数、評価頻度、seed、保存先を確認する。analysis/plant/guardは `RESUME_TRAINING = False`、search/retakeは `TRAINING_MODE = "fresh"` で新条件の学習を開始する。途中から同じ入力・編成・前段hashの学習を続けるときだけTrue/resumeにする。guard収集はplant学習が完了してから行う。新規収集はcollector冒頭の再開設定をFalseにする。

通常実行例（順番に、一つずつ完了を待つ）：

```powershell
Set-Location C:\Users\ronet\MyProject\git\AI_dnn\03.game\toruAI_v4
python tv4_train_attacker_analysis.py
# analysis学習完了後
python tv4_train_attacker_plant.py
# plant学習完了後
python tv4_collect_attacker_guard.py
# 収集が完了した後
python tv4_train_attacker_guard.py
```

guardは各相手AIにつき左右合計50件。左右比を均等にしない。defender retakeの既定は左右各50件であり、別の収集仕様。

通常保存先で新規学習・新規収集を実行すると、担当するlatest、best、ケースは新条件で更新される。他フェーズの保存物を削除しない。既存の固定編成モデルを汎用版として採用することはしない。

## 検証の範囲

作業ディレクトリ `03.game/` で `python -m unittest discover -s toruAI_v4/test -q` を実行し、93件すべて成功。合成状態・mockの評価・一時ディレクトリの保存を用いて確認した。指定の作業ディレクトリから両analysisスクリプトの `--describe` も成功。長時間学習や実対戦での到達性能は、この検証の対象外。guard収集・本学習の実エンジン通し動作はplant学習完了後に確認する。
