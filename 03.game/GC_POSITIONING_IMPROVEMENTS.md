# GCの指定プラント位置とポストプラント配置

読み込む本番モデルのversion/episodeはアプリ起動ログと実際のチェックポイントで確認してください。
以下の再学習候補は出力ディレクトリへ保存し、本番モデルには自動反映しません。

## Carrier先頭接敵を修正（2026-09-19、学習revision v7）

episode 1250は750より勝率・設置率・未エントリー率が悪化したため停止しました。
再開ディレクトリの`best_by_eval`はepisode 750を保持しており、1250の`latest`は使いません。

episode 750を100/60/40の相手に6 seed×10ラウンド再評価して、未エントリー理由を追加計測しました。
60ラウンドの未エントリー33.3%に対し、未エントリーかつCarrier死亡は31.7%でした。
その31.7%はすべてチーム全滅です。Carrier死亡時の護衛は4マス以内平均0.57人、
Carrier進行方向の前方8マス以内平均0.08人でした。初回脅威検知時でも前方護衛は平均0.16人です。
診断結果は`gc_v1/data/attacker_gc_curriculum_v6_resume_750/entry_diagnosis_60.json`に保存しています。

今回の変更:

- Carry入力を57から61、Escort入力を67から71へ拡張しました。最終設置位置までの距離、
  前方8マス以内の護衛、近距離護衛、護衛役フラグを自チームの観測から入力します。
- 設置位置まで9～24マスで前方護衛がいない場合、Carryは護衛役が生存し時間に余裕がある間だけ
  待つ例を学習します。EscortのMAIN/MAIN_LEAD/DEFAULT_MAIN/ROTATE/REHIT役には、
  Carrierの進行方向へ先行する例を学習させます。
- Carrierが前方護衛なしでサイトへ前進する行動を減点し、対象Escortには前方8マス以内を
  維持する報酬を与えます。撃ち合い中の停止学習、Macroの作戦、推論側の強制移動は維持します。
- `carrier_preentry_death_rate`、`no_entry_carrier_death_rate`、護衛人数、
  `screened_approach_rate`を評価ログに追加しました。
- Guardはepisode 750で配置率と設置後性能が良かったため重みを固定し、Carry/Escortだけ更新します。

既存のepisode 750ベストから入力層をゼロ拡張して開始します。新しい訓練seedを使い、
弱い相手50/10/0から1000エピソードで100/60/40へ上げ、残り1000エピソードを最終強度にします。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\retrain_gc_carrier_screening_v7.ps1
```

出力先は`gc_v1/data/attacker_gc_curriculum_v7`です。83件の関連テストが通過し、短期検証で
学習・評価・保存・best再読込・Holdoutが完了することと、Guard重みが変化しないことを確認しました。
短期検証は性能評価には使いません。最初の判断点はepisode 500です。

## v5 Holdoutの停滞を診断して修正（2026-09-19、学習revision v6）

現在は下記のv6起動スクリプトを使ってください。入力・行動空間はCarry v5（57入力）、
Escort v3（67入力）、Guard v2（34入力）を維持し、学習方法を変更しています。

今回のHoldoutが使ったbestセットはepisode 1750です。episode 750や最新episode 2000の評価とは
別の重みです。1750の選択時評価は勝率22.22%、未エントリー25.56%、時間切れ8.89%。
ユーザー実行のHoldout150ラウンドは勝率15.33%、未エントリー39.33%、時間切れ14.67%でした。

その1750のCarry/Escort/Guardを固定し、Holdoutの3 seedから各10ラウンド、計30ラウンドを再現しました。
モデル入力・マスク・全行動のQ値・選択時Macro目標・実移動・隣接占有を記録しました。
モデル3個のSHA256も記録しています。結果は勝利5/30、設置10/30、時間切れ4/30です。
Carryの1681tick中、非交戦かつ設置進捗0での停止334回のうち224回は合法な距離短縮先がありました。
移動選択の実行失敗0件、空き前進先の誤マスク0件。一方、前進を味方に塞がれて停止する場面もありました。
これは主に学習した行動選択と護衛の協調の問題を示します。Macro目標変更が主因という仮説は断定しません。

詳細は `gc_v1/data/diagnostics_v5_holdout/trace.json`、集計は同ディレクトリの `summary.json` です。

学習の変更:

- 序盤の教師による行動置換は従来通り80%から0へ減衰します。以後もモデル自身が到達した
  観測に教師ラベルを付け、模倣損失を最低0.15倍で継続します。DQNの更新も継続します。
  評価・Holdout・実戦には教師による選択や模倣更新を使いません。
- IQで見えている情報から実際に射撃できる場面を、Carry/Escort/Guardの停止例として学習します。
  見えない敵や敵配置を教師判断に渡しません。
- EscortがCarrierの次の進行マスを占有している場合は、合法な移動で進路を空ける例を学習します。
  自分の経由地点に着いていても、Carrierを塞ぎ続ける停止を教師にしません。
- Carryの最終登録設置位置については「距離1で到着扱い、待機」を教師にせず、最後の1歩と設置を
  教えます。中間の経由地点ではSplit等の同期のための待機を維持します。
- 未エントリー25%以下・時間切れ10%以下は各評価seedが満たす条件に変更しました。
  平均値で悪いseedが隠れることを防ぎ、`worst_carry_no_entry_rate` / `worst_timeout_rate`を出力します。
  Holdoutには採用セットの`selected_episode`も出力します。

以前のHoldout seedは診断に利用したため、今回は評価seedに組み込み、新しい3 seedをHoldoutに使います。
Macroの重み・ゲームcore・推論の行動選択には今回の修正を加えていません。

プロジェクトルートのPowerShellで実行:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\retrain_gc_navigation_v6.ps1
```

v5の同時保存best3モデル（episode 1750）から、弱い相手50/10/0で再開し、1000エピソードで
100/60/40まで上げ、残り1000エピソードを最終強度で学習します。新しい出力先は
`gc_v1/data/attacker_gc_curriculum_v6`。既に使用した出力先では起動を拒否します。
別の出力先が必要ならスクリプトに `-OutputDir gc_v1/data/attacker_gc_curriculum_v6_retry` を指定してください。

### Ctrl+Cでepisode 750に停止した場合

`attacker_gc_curriculum_v6` の`latest` 3モデルと履歴はepisode 750まで保存されています。
次のスクリプトは保存済みepisodeを読み、残り1250エピソードをglobal episode 751から開始します。
相手強度・探索率・教師置換率・訓練seedも元の2000エピソード計画上の位置を使います。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gc_v1\resume_gc_navigation_v6.ps1
```

出力先は`gc_v1/data/attacker_gc_curriculum_v6_resume_750`です。元の750までのファイルは変更しません。
チェックポイントにAdam optimizerの内部状態とreplay bufferは含まれていないため、それらだけは空から再開します。
モデル重みは`best_by_eval`ではなく、停止直前の`latest` 3モデルを読み込みます。

評価は6 seed×30、Holdoutは新規3 seed×50です。今回の修正で勝率向上が確認済みという意味ではありません。
`[EP] teacher=0.000 demo=0.150` は教師で行動を置換せず、教師ラベルの学習を継続している状態です。
履歴にはphase別の教師ラベル件数も保存します。20エピソードの短期検証では学習・評価・保存・
best3モデル再読込・Holdoutまで正常に完了しましたが、評価勝率向上は確認できていません。
採用判断は教師なし評価と新規Holdoutの結果で行います。下記v5コマンドは過去の実行記録です。

## v4の2000エピソード後の停滞を修正する再学習

現在の推奨版はCarry v5（57入力）・Escort v3（67入力）・Guard v2（34入力）です。
v4最終モデルを実エンジンで診断し、seed 3026091701/3026091702では撃ち合いが0tickでも
100tickで時間切れになり、どちらも2マス往復が53回ありました。
seed 5026091702ではサイト付近まで進んだ後も `(11,10)` と `(11,11)` を往復しました。
選んだ行動・実際の座標・Macro目標は `gc_v1/reports/curriculum_v4_latest_navigation.json` にあります。

Carryには隣接4方向の距離改善量・壁・占有、直前の実移動、停滞時間、再訪、残り時間の余裕を追加しました。
Escortも同じ情報を使い、Carrierの進行方向は最終設置位置ではなく現在のMacro経由地点に揃えました。
新Escortの行動マスクは壁と現在観測できる占有マスを除外し、進路を塞ぐ配置にも学習側で減点します。
Carryの旧「移動＋能力」行動は、標的がある場合は実エンジンで停止・能力使用、ない場合は移動となっていました。
v5では0が待機、1が停止して能力使用、2/4/6/8が移動、10が設置です。
旧3/5/7/9は曖昧な重複行動として使用せず、能力には観測済みの有効な標的が必要です。
自動のエントリー前能力使用も新Carryでは行わず、モデルが選んだ能力使用を実行します。
v4以下のCarry・v2以下のEscortは旧入力・旧行動のまま読み込めます。

学習序盤の500エピソードだけ、非戦闘時に観測上合法な前進や目標位置での設置を教師例として使います。
使用率は80%から0%へ減らし、DQN更新と合わせて教師行動の価値を高める学習を行います。
教師は各自のMacro目標に従い、戦闘・能力はDQNに任せます。隠れた敵情報は参照しません。
評価・holdout・実戦では教師を使わず、常にモデル自身の出力を使います。
500エピソードを過ぎた訓練もDQNと探索のみです。
5行動分の将来報酬を同じキャラ・同じphase区間で反映し、設置等の報酬が経路の行動にも早く伝わるようにしました。
非戦闘時の後退・往復と、設置せず時間切れになる終了には追加の減点を与えます。
撃ち合い中には前進の正の報酬を与えません。旧設定では前進+0.20が交戦時の移動減点-0.12を上回り、
撃ち合い中でも進み続ける誘因が残っていました。交戦中の勝敗・被ダメージ・キルの報酬は引き続き反映します。
Macroの重み・ゲームcore・本番モデルはこの修正で変更しません。

v4の同じbestセット（episode 750）から入力を拡張して開始します。最終episode 2000からは開始しません。
以下をプロジェクトルートのPowerShellから実行してください。出力先は未使用の空ディレクトリが必要です。

```powershell
D:\git\python\python.exe -X utf8 gc_v1/train_attacker_gc_real_curriculum.py `
  --init-carry gc_v1/data/attacker_gc_curriculum_v4/dqn_attacker_carry_gc_best_by_eval.pt `
  --init-escort gc_v1/data/attacker_gc_curriculum_v4/dqn_attacker_escort_gc_best_by_eval.pt `
  --init-guard gc_v1/data/attacker_gc_curriculum_v4/dqn_attacker_guard_gc_best_by_eval.pt `
  --output-dir gc_v1/data/attacker_gc_curriculum_v5 `
  --episodes 2000 --curriculum-episodes 1000 --navigation-bootstrap-episodes 500 `
  --navigation-demo-weight 1.0 --n-step 5 --final-mix 0.20 `
  --start-stats 50 10 0 --final-stats 100 60 40 `
  --eval-seeds 3026091700 5026091700 6026091700 `
  --holdout-seeds 4026091700 7026091700 8026091700 `
  --eval-interval 250 --eval-episodes 30 --holdout-episodes 50 `
  --lr 0.00002 --max-updates 32
```

`[EP]` の `teacher` は学習中の教師使用確率です。教師を含む訓練勝率・設置率だけでは採用判断しません。
`[FINAL-STRENGTH EVAL]` は教師なしの評価です。往復率 `carry_reversal_tick_rate` と非戦闘停止率
`carry_quiet_stall_tick_rate` も記録します。未エントリー率25%・時間切れ率10%を満たさない候補同士では、
その超過量が小さいものを優先して保存します。基準を満たした候補同士では最悪seed勝率・平均勝率を優先します。
設置が完了したラウンドは必ずエントリー済みとして数えます。旧v4では途中で設置目標が変わる等の理由で、
設置済みでも未エントリーと記録する場合がありました。したがって未エントリー率の旧ログとの比較には注意してください。
holdoutはベストの3モデルをまとめて読み直した後、別のseedで教師なしで行います。
この変更だけで実戦の改善を確認したわけではありません。短期検証用モデルを本番へコピーしないでください。

## スポーン付近のCarry往復を修正する再学習（2026-09-18）

最新の `series_Ghost_Champions_vs_Touyama_Gaming_0-2_20260918_155404.json` をリプレイから解析しました。
GCの攻撃24ラウンドのうち勝利5、設置11、設置後勝利5、時間切れ3でした。
Carryの設置前サンプル1846件中525件は初期スポーンからBFS距離6以内で、
直前2tickの位置へ戻る往復は681件です。分類ではsplitが12、fakeが11、defaultが1で、
マクロの戦術が存在しないことだけが原因ではありません。
詳細は `gc_v1/reports/carry_stall_20260918.json` にあります。

以下は旧v4の修正経緯です。現在の再学習には先頭のv5コマンドを使ってください。
この時点の修正版はCarry v4（39入力）・Escort v2（49入力）・Guard v2（34入力）です。
Carry/Escortには自分のMacro経由地点への距離・方向、Split/Fake/Rotate、独立役フラグ、
射撃可能な相手がいるフラグを追加しました。自チームの作戦のみを参照し、隠れた敵座標を
入力しません。Macroの重み・戦略選択は既存のものを使い、低レベルの移動はDQNが選びます。
新EscortではMacroによる1マス移動上書きをしないので、学習した行動と実行した行動が一致します。
旧Escort v1は互換性のため従来のMacro上書きを維持します。

前進報酬は同じ目標について最短距離を更新したときだけ与え、往復で繰り返し稼げません。
非戦闘のまま8tick以上進まなければ減点します。撃ち合い中の停止は壁・スモーク・身体の遮蔽・
向きを考慮して射撃できる場合だけ有利にします。停止報酬を時間コストより小さくし、
止まり続けるだけの正の報酬を廃止しました。Guardの滞在にも毎tickの正の報酬は与えません。
情報待ちの猶予20tickを過ぎて非戦闘のままスポーン距離6以内にいるCarryにも減点します。
学習と評価で同じ通常のMacro目標判断を使い、学習だけ設置目標を別の登録地点へランダム変更する
処理も廃止しました。

最初は2000ラウンドで挙動を確認し、改善を確認してから長期学習してください:

```powershell
D:\git\python\python.exe -X utf8 gc_v1/train_attacker_gc_real_curriculum.py `
  --init-carry gc_v1/data/attacker_gc_curriculum_v3/dqn_attacker_carry_gc_best_by_eval.pt `
  --init-escort gc_v1/data/attacker_gc_curriculum_v3/dqn_attacker_escort_gc_best_by_eval.pt `
  --init-guard gc_v1/data/attacker_gc_curriculum_v3/dqn_attacker_guard_gc_best_by_eval.pt `
  --output-dir gc_v1/data/attacker_gc_curriculum_v4 `
  --episodes 2000 --curriculum-episodes 1000 --final-mix 0.20 `
  --start-stats 50 10 0 --final-stats 100 60 40 `
  --eval-seeds 3026091700 5026091700 6026091700 `
  --holdout-seeds 4026091700 7026091700 8026091700 `
  --eval-interval 250 --eval-episodes 30 --holdout-episodes 50 `
  --lr 0.00002 --max-updates 32
```

評価に `carry_no_entry_rate`（設置目標からBFS距離8以内へ未到達）、
`carry_spawn_tick_rate`（自分の初期スポーンからBFS距離6以内のCarry滞在率）、
`timeout_rate`、`combat_stop_rate`、`postplant_win_rate` を追加しました。
モデル選択は未エントリー率25%以下・時間切れ率10%以下を満たす組を優先し、
その中で最低seed勝率、平均勝率、スポーン滞在率を比較します。
未達なら `entry_quality_passed: false` と記録します。最終の勝率目標は平均だけでなく
最低seedで判定し、挙動条件も満たした場合にのみ `ready_for_match_validation: true` です。
それでも攻守交代・通常の相手ステータス・累積疲労を含むシリーズ確認は別途必要です。

相手配置についての以前の説明を訂正します。Touyama Gaming v2は選手ごとのSetup/Search担当位置を
固定割当する実装です。seedは試合状況・射撃・調子等を変えますが、担当配置をシャッフルしません。
現時点の評価相手はこの通常の固定割当AIです。複数seed評価を配置ランダム化とは扱わないでください。

## 実戦の相手を段階的に強くする再学習

この節の古い学習経緯に対する変更点と、現在の推奨コマンドは上のv4節を参照してください。

`gc_v1/train_attacker_gc_real_curriculum.py` を追加しました。既存のCarry・Escort・Guardを
初期モデルとして、実ゲームの5v5をスポーンからラウンド終了まで動かします。
相手AIはTouyama Gaming v2です。各フェーズは別々のモデル・リプレイ・optimizerを持ち、
同じチームで学習します。Retrieve、Macroの設置目標判断、防御側モデルは既存のものを使います。
ゲームcoreは変更していません。

プロジェクトのルートでPowerShellから実行:

```powershell
D:\git\python\python.exe -X utf8 gc_v1/train_attacker_gc_real_curriculum.py `
  --output-dir gc_v1/data/attacker_gc_curriculum_10000 `
  --episodes 10000 `
  --curriculum-episodes 6000 `
  --start-stats 50 10 0 `
  --final-stats 100 60 40 `
  --eval-seeds 3026091700 5026091700 6026091700 `
  --holdout-seeds 4026091700 7026091700 8026091700 `
  --eval-interval 250 `
  --eval-episodes 100 `
  --holdout-episodes 200 `
  --target-win-rate 0.50
```

- 1～6000ラウンドで相手5人のHit/HS/Dodgeを50/10/0%から100/60/40%まで線形に上げます。
  6001～10000ラウンドは最終条件で学習します。数値は各ラウンド開始時の実効値です。
  カリキュラムの半分を過ぎてからは、既定で20%のラウンドを最終条件にします。
  IQ・Reaction・能力・射撃・移動・衝突・覚醒は実ゲームで処理し、開始後の覚醒等による
  ステータス補正も通常どおり反映します。GC側のステータスは変更しません。
- Carryは設置、Escortは護衛、Guardは登録位置への移動に報酬を与えます。設置で学習全体を
  終えず、各参加者のそのフェーズ最後の行動にも実際のラウンド勝敗を反映します。
  登録位置での設置と通常位置での設置の報酬差を大きくし、相手が強くなったときに
  通常位置へ逃げる方策を抑えます。
  敵が視認できる実際の撃ち合い中は、STAYを選んで射撃を継続する行動に報酬を与え、
  移動し続ける行動には小さな負の報酬を与えます。推論側で強制停止はしません。
  設置を成功させても解除されて負ければ勝利報酬を与えません。
- 学習で作成するEscort v1はDQNで移動・能力使用を選択します。旧モデルの接近時停止・
  Macroによる移動上書きはこの新モデルには適用しません。現在の本番Escortは旧版なので
  今回のコード追加だけで本番のEscort行動は変わりません。
- 評価は学習開始前と250ラウンドごとに行い、毎回最終条件で指定した複数seedを各100ラウンド評価します。
  ラウンド番号・スコア・精神疲労を変え、通常の設置目標判断を使います。
  モデル選択は複数seedの最低ラウンド勝率、平均ラウンド勝率、最低登録設置率の順です。
- 最後に、選択した3モデルを別seedの200ラウンドで確認し、`holdout_evaluation.json` に
  `round_win_rate` と `meets_win_rate_target` を記録します。目標は攻撃ラウンド勝率50%以上です。
  未達なら未達と記録します。10000回で勝てる保証や、攻守交代する試合全体の勝率保証ではありません。

保存先にCarry・Escort・Guardそれぞれの `dqn_attacker_<phase>_gc_best_by_eval.pt` と
`latest.pt`、同時点の3モデルを示す `best_by_eval_bundle.json`、学習・評価履歴を保存します。
初期モデルが最も良ければepisode 0が選ばれます。
本番モデルへの自動上書きはありません。3モデルは同時点の組として評価・採用してください。
`evaluate_real_series_gc.py` の `--carry-model` / `--escort-model` / `--guard-model` で
通常の相手ステータス・攻守交代・シリーズ疲労込みの試合も確認できます。
出力ディレクトリは空である必要があります。再実行時は別名を指定してください。

## 実ゲームによるCarry再学習と検証

設置率13.3%だった `series_Ghost_Champions_vs_Touyama_Gaming_0-2_20260917_171242.json`
を確認し、Carryの簡易環境と実戦の違いを解消する学習方法に変更しました。
新しい `train_attacker_carry_gc_real.py` は実ゲームの5v5・Touyama Gaming v2・Macro護衛・
IQ知覚・設置前の射撃とアビリティ・スパイク回収をそのまま使います。ゲームcoreは編集していません。
学習時はCarryのDQN選択に探索を入れ、敵や護衛の行動、物理判定を学習用の簡略処理に置き換えません。
Carry学習の1エピソードは設置またはラウンド終了までです。設置後の勝敗は下記のシリーズ評価で確認します。

Carry v3の観測は31次元です。

- 0～28は既存の観測です。3番は実ゲームが判断直前に保存した `moved_last_tick` を使います。
  v0～v2の入力は互換性を維持します。単純なフラグ切り替えだけの比較では改善しませんでした。
- 29番に3マス以内の生存護衛人数、30番に本人の設置進捗を追加しました。
- ウォームスタート時は追加入力と旧フラグの重みを0にし、旧モデルが実戦で受け取っていた
  入力0の判断を保ってから、実際の遷移で新特徴量を学習します。
- 設置完了、登録位置での設置、設置進捗、被弾・死亡・時間切れ・設置中断・孤立した露出を
  報酬に反映します。キャリア死亡時はその個人の遷移を終了し、回収した仲間の判断も別の遷移として収集します。
- 学習ではラウンド番号・スコア・精神疲労・初期登録目標を変えます。評価では通常の初期目標と
  実際のMacro判断を使い、スポーンから開始します。

再学習は200ラウンド行い、40・80・120・160・200ラウンド時点で別seedの実ゲーム12ラウンドを評価しました。
80ラウンド時点の候補を固定し、直近2シリーズに加えて未使用の過去試合と別の相手で評価しました。
40ラウンド時点の候補は直近2シリーズで設置率26.8%に留まり、採用していません。
200ラウンドのlatestも登録位置での設置が減ったため採用していません。

`evaluate_real_series_gc.py` は保存された各マップのseed、両チームの編成・AI、シリーズスコア、
累積疲労、攻守交代を引き継いで実ゲームを再実行します。描画なしで同条件の旧モデルと候補を比較します。
GUIの保存済み試合をそのまま再生するものではなく、保存済み試合のスコアと完全一致する保証はありません。
保存されているマップ数を評価するため、2マップが1–1になっても未保存の第3マップは追加しません。

4シリーズ・8マップの比較（Touyama Gaming v2の6マップ、Toru AI v3.1のCarnal Lust Syndicateの2マップ）:

| 指標 | 現行v2の再評価 | 採用したv3 |
| --- | ---: | ---: |
| 設置成功率 | 41.9%（31/74攻撃ラウンド） | 61.9%（52/84） |
| 攻撃側勝率 | 16.2%（12/74） | 27.4%（23/84） |
| 全ラウンド勝率 | 26.2%（37/141） | 40.0%（66/165） |
| 登録地点での設置数 | 27 | 32 |

最新試合のseedでは、旧モデルの再評価はGC視点で0–13、1–13、採用候補は13–7、8–13でした。
最新2マップの候補の設置率は66.7%（14/21）です。前の16:54の2マップはスコアが下がっており、
全seedでの改善を確認した結果ではありません。また、別の相手では通常位置への早い設置が増え、
成功設置に占める登録地点の割合は下がりました。登録位置が必ず強いと実証できたわけではありません。
今回は実戦の設置失敗と攻撃成績を優先して採用しています。

詳細な全マップ比較は `gc_v1/reports/real_carry_series_validation.json`、
本番反映のパスとSHA256は `gc_v1/reports/real_carry_promotion.json` に保存しました。
本番bestを採用候補と同一のファイルに差し替え、実際のGCファクトリがv3 / episode 80 / 31次元を
読み込むことを確認しました。旧v2は同じディレクトリの
`dqn_attacker_carry_gc_positioning_v2_before_real_training.pt` に残しています。
旧v0バックアップ、Escort・Retrieve・Guard・Macroモデルは今回差し替えていません。

```powershell
python -X utf8 gc_v1/train_attacker_carry_gc_real.py --init-model gc_v1/data/attacker_carry_gc_data/dqn_attacker_carry_gc_positioning_v2_before_real_training.pt --output-dir gc_v1/data/attacker_carry_gc_real_new_data --episodes 200 --eval-episodes 12 --eval-interval 40
python -X utf8 gc_v1/evaluate_real_series_gc.py --series competition_results/series_Ghost_Champions_vs_Touyama_Gaming_0-2_20260917_171242.json competition_results/series_Ghost_Champions_vs_Touyama_Gaming_0-2_20260917_165457.json --carry-model gc_v1/data/attacker_carry_gc_real_data/dqn_attacker_carry_gc_ep80.pt --json-output gc_v1/reports/series_latest_two_real_ep80.json
```

v3は簡易環境のCarry評価に渡せません。Guardと旧Carryの簡易環境評価は引き続き利用できます。

`gc_v1/map_data_guard_plant_gc.py` の5～9を優先プラント位置とし、
`gc_v1/map_data_guard_gc.py` の同じ数字を対応するGuard位置として使います。
Carryの6/7は引き続き進入経路の目印です。Carry側の5だけを変更しても、
今回の優先プラント位置は変わりません。設置・Guard位置を変更したら両モデルを再学習してください。

## 改善した学習

- Carryは登録パターンを均等抽選して目標を決めます。登録位置での設置を通常設置より高く評価し、選んだ目標そのものへの設置に追加報酬を与えます。
- 通常の設置可能マスを通過して強い設置位置へ進む行動に、設置遅延ペナルティを与えません。時間切れ間際の設置遅延には従来のペナルティを残します。通常位置への緊急設置も合法で、成功報酬があります。
- Guardは65%が設置地点周辺、25%が分散した位置、10%がGuard候補から開始します。担当位置への移動と、到着後の保持を両方学習します。15%では未登録位置への緊急設置も学習します。
- 敵を目撃してもGuard位置への接近報酬を敵への追跡報酬に置き換えません。射線内の敵がいても移動を禁止せず、移動と静止の価値を学習します。解除妨害、キル、死亡、勝敗の評価は残ります。
- 設置地点が見えるだけの途中のマスで止まり続ける報酬を下げ、指定位置での保持を高く評価します。到着ボーナスは各キャラ・各エピソード1回だけです。
- 学習と推論で、名前順・最寄りの未割当位置という担当位置の割当方法を揃えます。位置が人数より少ない場合は近傍の床を補完し、目標の重複を避けます。

指定位置は戦術上の事前知識として報酬に反映します。敵や解除に対応して位置を離れることも可能です。
配置の有利さ自体は、指定位置への到達率だけでなく勝率と解除失敗数で確認してください。

## 推論との接続

新しいチェックポイントには `positioning_version` を保存します(Guardは2、Carryは3)。Guardは34次元を維持し、
予備の33番に設置パターンを追加します。到着判定は指定座標そのものです。
旧モデルでは予備次元は0で、従来の到着半径1を使います。
バージョン2では、解除中の観測17を「解除者本人が見えるか」に切り替えます。
隣の空きマスが見えるだけでは、解除を妨害できると評価しません。
解除が進んでいるのに解除者を撃てない状態へのペナルティを加え、
勝利報酬と解除敗北ペナルティを1ラウンド分の位置保持報酬より大きくします。
Carryの25～27は、ラウンドで選んだ登録位置そのものへの距離・方向にします。
経路目印へ移動している間も最終的な設置位置を観測でき、登録位置の一部だけに
設置が偏る状態を改善します。選んだ位置への設置には追加報酬3を与えます。
バージョン1の再学習候補も読み込めますが、25～27と解除中の17はその版の観測を維持します。

新モデルではGuardの経路・静止・援護移動による上書きを外します。
CarryではMacroが選ぶ設置対象サイトを使い、移動と設置はCarryの出力に任せます。
Macroの「最初に踏んだ設置可能マスで設置を強制する」処理は適用しません。
旧チェックポイントは従来の処理で読み込めます。別フェーズのtrainファイルへのimportは追加していません。

## 再学習

以下は旧v2の簡易環境用です。本番Carry v3の継続学習と評価には、先頭の実ゲーム用コマンドを使ってください。

以下はプロジェクトルートで実行します。別ディレクトリへ保存するため、現在使っているモデルを上書きしません。
`--init-model` を省くと初期重みから学習します。指定すると既存モデルから報酬の変更に合わせて再学習します。

```powershell
python -X utf8 -u gc_v1/train_attacker_carry_gc.py --episodes 3000 --eval-episodes 120 --init-model gc_v1/data/attacker_carry_gc_data/dqn_attacker_carry_gc_positioning_v2_before_real_training.pt --output-dir gc_v1/data/attacker_carry_gc_positioning_data
python -X utf8 -u gc_v1/train_attacker_guard_gc.py --episodes 3000 --eval-episodes 120 --init-model gc_v1/data/attacker_guard_gc_data/dqn_attacker_guard_gc_best_by_eval.pt --output-dir gc_v1/data/attacker_guard_gc_positioning_data
```

`--episodes` に合わせて探索率を減衰させ、100エピソードごとと最後にgreedy評価します。
Carryは通常強度の敵に対する設置成功率、登録位置での設置率、目標そのものへの設置率、報酬の順に良いモデルを保存します。
Guardは勝率、指定位置への到達率、報酬の順です。探索中の平均報酬だけでbestを選びません。
最後のモデルは短い実行でも必ずlatestに保存します。中断からの完全な再開ではなく、重みを使った再学習です。

## 同条件での評価

```powershell
python -X utf8 gc_v1/evaluate_positioning_gc.py carry --episodes 120 --model gc_v1/data/attacker_carry_gc_data/dqn_attacker_carry_gc_best_by_eval_1.pt --json-output gc_v1/reports/positioning_carry_baseline.json
python -X utf8 gc_v1/evaluate_positioning_gc.py carry --episodes 120 --model gc_v1/data/attacker_carry_gc_positioning_data/dqn_attacker_carry_gc_best_by_eval.pt --json-output gc_v1/reports/positioning_carry_retrained.json
python -X utf8 gc_v1/evaluate_positioning_gc.py guard --episodes 120 --json-output gc_v1/reports/positioning_guard_baseline.json
python -X utf8 gc_v1/evaluate_positioning_gc.py guard --episodes 120 --model gc_v1/data/attacker_guard_gc_positioning_data/dqn_attacker_guard_gc_best_by_eval.pt --json-output gc_v1/reports/positioning_guard_retrained.json
```

seedは既定で20260917です。Guardは4パターン×3開始条件を順番に評価します。
`--start-mode transition` を指定すると、設置直後から配置につく場面だけを評価できます。
到達率の分母は開始時に担当位置にいなかったキャラです。最初から配置済みのキャラは到達率を押し上げません。
`position_tick_rate` は生存中のtickのうち担当位置にいた割合です。
Carryの `preferred_plant_rate` / `target_plant_rate` の分母は全エピソードです。
パターン別の数値も出力します。評価は学習用の簡易環境でDQNの行動だけを測るもので、実ゲームの試合勝率ではありません。

比較結果は `gc_v1/reports/positioning_*_baseline.json` と
`gc_v1/reports/positioning_*_retrained.json` に保存します。

2026-09-17にはCarryを300+300、Guardを300+500エピソード再学習しました。
現行の観測・報酬での120エピソード評価は以下です。Carryの候補は最後の段階の200エピソード目、
Guardの候補は最後の段階の500エピソード目のbestです。

| 指標 | 旧モデル | 再学習候補 |
| --- | ---: | ---: |
| Carry: 設置成功率 | 100.0% | 96.7% |
| Carry: 登録位置での設置率 | 57.5% | 93.3% |
| Carry: 選んだ目標そのものへの設置率 | 29.2% | 93.3% |
| Guard: 担当位置への到達率 | 16.5% | 72.3% |
| Guard: 担当位置にいるtickの割合 | 12.2% | 73.0% |
| Guard: 勝率 | 89.2% | 85.8% |

指定位置を使う行動は改善しましたが、この短い再学習では設置成功率と勝率が旧モデルに届きませんでした。
この初期比較時点では本番の既存チェックポイントを上書きせず、候補は上記の別ディレクトリに保存しました。
継続学習する場合は `--init-model` にそのディレクトリのbestを指定できます。
初期段階の候補は `dqn_attacker_*_gc_trial300.pt`、比較値は `positioning_*_trial300.json` に残しています。
初期Guard候補の勝率低下を受け、解除者本人のLOS観測と勝敗・解除妨害報酬を追加しました。

本番への配置前に、以下の実ゲーム評価も行ってください。簡易環境で登録位置の利用率が
高くなっても、実戦の設置成功率を保証しません。旧ファイルは別名で保存してください。

## 実戦の設置率25%への対応

最新試合 `series_Ghost_Champions_vs_Touyama_Gaming_0-2_20260917_165457.json` は、
GCの攻撃20ラウンド中5設置（25%）でした。当時の本番bestはCarryがversion 2 / episode 200、
Guardがversion 2 / episode 500です。

実戦ではIQAwareControllerがキャラ・tickごとにPerceivedGameViewを作ります。
この視界には `target_plant_pos` のIQ誤差があり、その属性への書き込みは一時視界内だけに保存されます。
そのため、Carry/Macroが登録設置位置を決めても実際の試合目標は変わらず、次のtickで
違う座標が提示されていました。学習環境は固定目標なので、この不一致を再現できていませんでした。

GCの新モデルでは、自チームが決めた設置目標を実試合と現在の視界の両方に保存し、
Carry/Macro/Escortが同じ目標を受け取るようにしました。学習済みの移動・設置行動を維持したまま、
戦術目標の寿命と共有を修正しています。敵の観測やゲームcoreは変更していません。
実行中のアプリはPythonコードとモデルを保持しているため、修正反映時にはアプリを再起動してください。

実ゲームを描画なしで動かす比較ツールを追加しました。IQ知覚、実際の5人編成、
Touyama Gaming v2、射撃・設置・解除、攻守交代を含みます。
候補チェックポイントを直接指定でき、本番ファイルを差し替える必要はありません。

```powershell
python -X utf8 gc_v1/evaluate_real_match_gc.py --seed 1562627217 --json-output gc_v1/reports/real_match_candidate.json
python -X utf8 gc_v1/evaluate_real_match_gc.py --seed 678513998 --map-number 2 --json-output gc_v1/reports/real_match_candidate_map2.json
python -X utf8 gc_v1/evaluate_real_match_gc.py --seed 20260918 --carry-model gc_v1/data/attacker_carry_gc_positioning_data/dqn_attacker_carry_gc_best_by_eval.pt --guard-model gc_v1/data/attacker_guard_gc_positioning_data/dqn_attacker_guard_gc_best_by_eval.pt --json-output gc_v1/reports/real_match_candidate_holdout.json
```

出力には全攻撃ラウンドを分母にした設置率、登録地点での設置数、スコア、ラウンド記録、
読み込んだモデルの版・学習エピソード・SHA256を保存します。`--save-replay` を付けると
全tickのリプレイも保存します。各マップは独立実行です。シリーズ内の累積疲労やシリーズスコアは
引き継がないため、保存済み試合そのもののスコアを完全に再現する評価ではありません。

同じ独立マップ条件・seed 1562627217・同じversion 2モデルでの修正前後比較:

| 指標 | 修正前 | 修正後 |
| --- | ---: | ---: |
| 攻撃ラウンド数 | 12 | 12 |
| 設置成功率 | 25.0%（3/12） | 50.0%（6/12） |
| 登録地点での設置数 | 2 | 6 |
| GCのスコア | 4–13 | 6–13 |

第2マップのseed 678513998では修正後2/8設置（25%）、いずれも登録地点でした。
両マップの再実行を合わせると8/20設置（40%）です。少数試合の結果であり、
勝率改善や全対戦相手での設置率を保証するものではありません。
別seed 20260918では、旧目標の読み書きを評価プロセス内で復元すると0/12設置、
修正後は4/12設置（33.3%、登録地点3回）でした。両者のスコアは1–13で、
この比較でも勝率の改善は確認できていません。
詳細は `gc_v1/reports/real_match_before.json`、`real_match_shared_target.json`、
`real_match_shared_target_map2.json`、`real_match_before_holdout.json`、
`real_match_shared_target_holdout.json` に保存しています。概要比較は
`gc_v1/reports/real_match_positioning_comparison.json` です。この目標共有の修正だけの段階ではモデルファイルの
差し替えや追加学習は行いませんでした。その後の13.3%の試合を受け、先頭に記載したv3の再学習と反映を行いました。

## 検証

```powershell
python -X utf8 -m unittest test_gc_positioning_learning test_guard_postplant_priority test_gc_macro_coordination test_gc_macro_training_timing
```

報酬の逆転、通常設置マスの通過、最初の移動tickの報酬、目撃による担当位置の消失、
不足時の目標重複、学習・推論の観測一致、新モデルの行動の上書きを検証します。
