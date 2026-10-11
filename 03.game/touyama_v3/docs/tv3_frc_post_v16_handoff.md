# frc_v1：v16より後の調査・施策の引き継ぎ

最新評価：[revision3 set5](tv3_frc_right_timeout_revision3_set5_review.md)は学習前比でtimeout12→5、右timeout12→3、設置60/72→67/72。時間切れ最優先の方針で改善と評価。新専用bestはprofile `28f115895deba593`、共通best v16は維持。残りは左2・右3で、左の入口占有連鎖と右のスパイク回収停滞を非学習トレースで確認。全timeout解消とは扱わない。

最新：ユーザーの最優先事項を時間切れ解消へ変更し、[右専用revision3の期限教師・設置位置・待機罰](tv3_frc_right_timeout_revision3.md)を実装した。通常設定はrevision3登録・旧専用revision2 set5から重みのみ初期化・新規optimizer/replay・5セット上限。教師診断で右round11/23のtimeoutを設置へ変更、通常モデルの改善にはユーザーによる学習が必要。未学習で全時間切れ解決とは扱わない。

追加評価：[右専用revision2のset5レビュー](tv3_frc_right_utility_revision2_set5_review.md)。ユーザー学習後、同じ専用構成のset0との72ラウンド比較で右設置25/38→27/39、右人数優位14→13、右timeout11→12。共通best v16は維持。追加学習・通常対戦採用は行っていない。

2026-10-12追記：この引き継ぎ後、ユーザー依頼で[右専用の能力教師revision 2](tv3_frc_right_utility_revision2.md)を実装・登録し、通常学習を5セット上限へ変更した。本文の70セット・無変更テンプレート・登録無効は引き継ぎ作成時の状態。既存bestは維持、追加学習は未実行、右性能改善は未確認。

2026-10-12作成。対象はTouyama Gaming v3 attacker plant対frc_v1。**右サイト問題は未解決。保存bestはv16 set10を維持している。** 本文は既存の調査記録を整理したもので、引き継ぎ作成時に追加学習は実行していない。

## 1. 次の担当者が最初に確認すること

- `../AGENTS.md`と共通知見の[MODEL_BUILD_BASE_POLICY.md](../../docs/MODEL_BUILD_BASE_POLICY.md)を読む。学習・optimizer更新はユーザーが行う。診断のための非学習rolloutと学習を混同しない。
- 良かった重み、実行処理の版、analysis依存、対象ラウンド開始前の履歴を分けて比較する。単に同じseed・同じset番号であるだけでは対照実験にならない。
- 設置だけ増えても人数不利なら採用しない。優先するのは人数優位の設置、次に同人数以上の設置、次に相対的な能力優位。全設置に正の完了報酬は付ける。
- 単一場面の成功、教師の成功、修正実装、学習済みモデルの改善を区別する。「原因解明・解決済み」と扱わない。
- 現在の`tv3_train_attacker_plant.py`は**TRAINING_SETS=70 / TOTAL_TRAINING_SET_LIMIT=70**。過去のv18/v19検証は各5セットだった。現在の設定を過去の5セット自動停止と説明しない。今回この定数は変更していない。

## 2. 重み・依存モデルの識別

パスは、特記しない限り`03.game/touyama_v3/`からの相対パス。

| 用途 | ファイル | 確認済み版・セット |
| --- | --- | --- |
| 比較基準・保存best | `data/best/frc_v1/attacker_plant_best.pt` | v16 set10 |
| 最新の確認済み学習候補 | `data/attacker_plant/frc_v1/evaluation_checkpoints/set_000005.pt` | v19 set5 |
| 固定analysis | `data/best/frc_v1/attacker_analysis_best.pt` | v4 set50 |
| analysis収集時のplant | `data/attacker_analysis/collection_sources/frc_v1/attacker_plant_best.pt` | v10 set30 |

2026-10-12にbestとset5候補のSHAを再確認した。

```text
plant best v16 set10:
cd9079b8813a14e05edd4751bef61751e48e3aea0a9db688aecab401d369d453
plant evaluation v19 set5:
13da973093e4dba35a0ff4372daa125adf97e21cb60c48c739e9dbe510fc65cc
analysis best v4 set50:
dae22732e282113222b69f2bcff32d39066eeea1220b623479554f73791d7680
analysis収集元 plant v10 set30:
d1812e666afac4f992ed8102e1fe7c6f1433ebcffb16cb0254c8ce0a0fc5333c
```

評価checkpointの`set_000005.pt`は実行ごとに同じパスを使う。以前のv18 set5はv19で上書きされたため、そのパスからv18を再現できるとは限らない。set10/15/20も存在するが、ファイル名だけでv19の追加学習と判断しない。必ずschema、completed_sets、warm_start_hash、analysis_hash、実ファイルSHAを確認する。v19の過去set1〜4は保存されておらず復元できない。

## 3. 同条件の72ラウンド評価

教師・epsilonとも0。編成Touyama Gaming、seed=`804000042,804000142,804000242`、各seedでstart_round=1と13から12ラウンドずつ。analysisは同じ固定モデル。start_round=13のブロックは、前半12ラウンドを実際にdefenderで戦った後半評価ではない。

| 指標 | 基準v16 set10 | 学習v18 set5 | 学習v19 set5 |
| --- | ---: | ---: | ---: |
| 設置 | 59/72 | 63/72 | 60/72 |
| 人数優位設置（味方>敵） | 45 | 37 | 37 |
| 同人数以上設置（味方>=敵） | 51 | 50 | 48 |
| 人数不利設置 | 8 | 13 | 12 |
| timeout | 11 | 8 | 9 |
| 左設置 | 33/34 | 33/34 | 34/34 |
| 左人数優位設置 | 32 | 25 | 28 |
| 左人数不利設置 | 0 | 5 | 1 |
| 右設置 | 26/38 | 30/38 | 26/38 |
| 右人数優位設置 | 13 | 12 | 9 |
| 右同人数以上設置 | 18 | 22 | 15 |
| 右人数不利設置 | 8 | 8 | 11 |
| 右timeout | 10 | 7 | 9 |

v18は設置数が増えたが左の人数条件が悪化した。v19では一部左事例が回復したが右の人数条件はさらに悪化。どちらもbest更新に必要な全体改善ではない。v16はこの左悪化の前の比較基準であり、右問題を解決した版ではない。

結果資料：[v18失敗調査](tv3_frc_v18_set5_failure_investigation.md)、[v19 set5評価](tv3_frc_v19_set5_review.md)。機械可読の要約は`logs/attacker_route_risk_review/v18_set5_review.json`と`v19_set5_review.json`。

## 4. v17：入口を塞ぐ味方の目標変更

実装：`tv3_attacker_entry_yield.py`の`yielding_entry_goal`。

非保持者が保持者の入口目標を占有し、自身の目標へ交通を考慮した経路が到達不能または大きく迂回する場合、近傍の到達可能な空きサイト床へ目標を変更する。行動はモデルに選ばせる。

seed142・round8の旧v16では、味方が保持者の目標10,40を塞ぎ、保持者11,40からの静的距離2に対し交通考慮距離16。教師は迂回を避け、塞がれた静的経路に戻ってSTAYした。5v2・HP460のまま100tick timeout。

同じ旧重み・同じ先行履歴/RNGでv17処理を使うと、63tickに補助者の目標が9,40へ変わり、69tickに5v2・HP460で設置した。ただし旧重みの72ラウンド比較では設置58、人数優位46、timeout12、右25/38。基準59、45、11、右26/38から全体改善していない。

## 5. v18：終盤に全経路候補が消える問題

実装：plannerの`allow_deadline_route_reserve`をencoder版>=18で有効化。

通常の移動+設置4tick+余裕5tickを要求すると、実際には設置可能でも候補が消える。通常候補と残り経路が両方なくなった場合だけ余裕条件を緩和する。実移動+設置4tickが実際の残り時間内に収まることは引き続き要求する。通常候補やPLANT行動を強制変更しない。

seed142・round23の旧v16は94tickに設置可能な10,40でEを選び、95tickの10,41以降候補消失、4v2・HP360でtimeout。同じ旧重み・同じ履歴/RNGでv18処理を使うと96〜99tickにPLANTし、100tickに4v3・HP360で設置した。

旧重みのv18処理による72ラウンド評価は設置58、人数優位47、同人数以上51、timeout12、右25/38。局所修正の効果は確認したが設置・timeout全体は改善していない。その後のユーザー実行v18学習5セットは上表の結果。

v17/v18資料：[入口・終盤候補の調査](tv3_frc_set10_entry_deadline_review.md)。ログは`set10_r8_helper_goal_detour_integrated.json`、`set10_r23_deadline_integrated.json`、`set10_entry_yield_72round_evaluation.json`、`set10_deadline_goal_72round_evaluation.json`（いずれも`logs/attacker_route_risk_review/`）。

## 6. v18学習後に悪化した行動の診断

左の人数不利設置5件：seed042 round13/15、seed142 round17、seed242 round3/15。右の重点timeoutはseed142 round8。

左round13は同じ対象開始RNGで44tickまで行動一致。45tickのRECON役slot3、位置8,3・HP60で**観測・最終maskも同一**なのに、旧重みはS/NW、新重みはE/NWを選んだ。新重みは翌tickに危険な7,4へ進み死亡。新重みのこの一手だけを合法な旧選択へ変える対照では、設置3v4から4v3へ回復した。

左round17は59tickに低HPのHUNT役slot2がSへ退避する教師判断だったが、保持者が先に10,3を予約してSがmaskされ、代替教師が退避意図を失ってNへ進ませた。他にも味方予約が退避を塞ぐ事例はあるが、左5件をすべて同一原因とは断定していない。

右round8では入口付近の前進→単独接敵→退避→recent_edgeによる待機が繰り返された。単に後退・待機を進捗から除く診断、支援役一手変更では3v4 timeoutのまま。こうした診断だけで学習や全体問題が直ったとはいえない。

## 7. v19：教師の退避意図・正しい行動variantへの勾配

### 退避教師の代替選択

実装：`tv3_learn_attacker_masked_teacher.py`。cover_retreatが調整maskで消えた際、既知の被射線、一方的な射線、元の退避先との距離、向きを考慮して合法な代替を選ぶ。v19以上で有効。

同じv18重みの左round17対照では、59tickの代替教師がNからSTAYに変わったがモデルの実選択は変わらず最終2v3。教師修正と重みの行動改善は別。

### 種類別教師損失

実装：`demonstration_kind_loss(... teacher_variant=True)`、設定`DEMONSTRATION_TEACHER_VARIANT=True`。

旧損失は種類ごとの最大Qを用い、教師と異なる向き・能力対象のvariantへ正の勾配を入れる場合があった。右round8・38tickでは教師S/Eに対してモデルS/NWが高Qであり、旧損失はS/NWを強化していた。新損失は教師種類のスコアに正しい教師indexのQを使い、S/Eへ勾配を与える。能力のvariantは投擲対象の違いも含む。

FRCの種類教師重み1、全index教師損失0.15。対象教師は最終maskで合法なもの。報酬・analysisはこの修正で変更していない。ユーザーがv16 best重みから新規optimizer/replayでv19を5セット学習し、上表の結果となった。

## 8. 原因を切り分けた対照と不採用施策

以下はv18対象の単一場面の診断であり、学習済み改善率ではない。

| 対照 | 結果・判断 |
| --- | --- |
| 右round8を最初から全教師で実行、先行7ラウンドはv18固定 | 67tick、5v3・HP420で設置。全教師でも必ず失敗するという説明は誤り |
| 右round8・38tickのFLASH役の向きだけS/Eへ変更 | 65tick、4v3・HP400で設置 |
| 右round8・47tickのFLASH役がFLASH使用 | 61tick、4v4・HP320で設置 |
| 左round17・59tickだけ退避役を先に計画 | 当座は退避できたが後に死亡、最終2v3。採用しない |
| 左round15・61tickだけ退避役を先に計画 | 最終3v4のまま設置66→81tickと遅延。採用しない |
| 右round8で後退・待機を進捗時計から除外 | timeoutのまま。解決策として採用しない |
| 右round8で支援役の一手を変更 | timeoutのまま。解決策として採用しない |

向き変更とFLASH使用はそれぞれ独立した反実仮想。組み合わせた効果や全評価での効果は未確認。移動を常に支援待ちにする、全edge制約解除、左固定などへ短絡しない。

対応ログ：`v18_r8_full_teacher_control.json`、`v18_r8_facing38_single_probe.json`、`v18_r8_flash47_single_probe.json`、`v18_left17_retreat_reservation_probe.json`、`v18_left15_retreat_reservation_probe.json`。

## 9. v19の実際の残存問題

v18で悪化した左5件はv19で4v3/4v4/5v3/4v4/4v4へ回復した。しかし別の左seed142 round19が88tick・2v4設置となった。

v19で先行7ラウンドも動かした右seed142 round8は90tick・2v3・HP200で設置。38tickのFLASH役は教師ABILITYに対しE/Eを選び、以後も移動・退避・待機を続け59tickに能力未使用で死亡した。保持者は14,39付近から14,31→8,32→7,34→7,39と迂回して右へ入った。向きがEになっただけでは、アビリティ判断・連携が学習できたことにならない。

v18の38tickとは位置・観測が異なるため直接の同観測比較ではない。右の人数不利設置、合法能力の見送り、支援役喪失、迂回による遅延が継続している。v19の右timeoutは9件。

## 10. 良い旧重みと新実行処理を分けた比較

seed142 round8の**先行7ラウンドを両方v16に固定**し、対象開始RNGを一致させ、対象の実行処理もv19で揃えた。

| 対象の重み / 実行処理 | 結果 |
| --- | --- |
| 旧v16重み / 旧v16処理 | timeout100、5v2・HP460 |
| 旧v16重みをメモリ内コピー / v19処理 | 設置69、5v2・HP460 |
| 学習済みv19 set5重み / 同じv19処理 | 設置59、4v2・HP320 |

最初の重み判断差は16tickのRECON役slot3。観測・最終mask一致で旧重みはABILITY、新重みはN。旧は16tick RECON、30tick FLASHで敵4へ作用。新は17tickに別対象へRECONし、FLASHは使用しなかった。

新重みの4v2は履歴固定の対照値であり、通常v19評価の2v3を訂正する値ではない。この一例は旧重みを残して新処理を検証する価値を示すが、72ラウンドの改善を証明していない。bestのschema書換えや上書きは行っていない。

資料：[v16へ戻す比較](tv3_frc_v16_backtrack_review.md)。ログ：`v16_backtrack_r8_review.json`、`v16_weights_v19_execution_r8_control.json`、`v19_weights_v19_execution_r8_matched_prefix.json`。

## 11. 追加された診断・保存手段

`tv3_review_attacker_timeout.py`は非学習rolloutを行い、観測/maskのhash、Q上位、教師判断、射線・予約・目標、先行履歴とRNGを記録する。

通常作業ディレクトリ：`03.game/touyama_v3/`。スクリプト冒頭の定数を確認・変更してから実行する。

```powershell
python tv3_review_attacker_timeout.py
```

- 既定はbest v16 set10、seed804000142、START_ROUND=1、TARGET_ROUND=8。`OUTPUT_PATH`は既存ログを上書きしない新しい名前にする。
- 実学習checkpointは`SOURCE_PLANT_CHECKPOINT`と`EXPECTED_PLANT_SET`で明示する。
- 対象のみ候補へ切替は`CANDIDATE_TARGET_ONLY=True`。`CANDIDATE_PLANT_CHECKPOINT`は実重み、`CANDIDATE_VERSION`は旧重みの実行版変更診断で、両者を同時指定しない。
- `FORCE_TEACHER_TARGET`、`DIAGNOSTIC_CHOICE_OVERRIDES`、`DIAGNOSTIC_PLAN_ORDER_OVERRIDES`などは反実仮想用。通常評価では無効にする。
- `tv3_review_attacker_set5_failures.py`の既定期待版はv18であり、現在のset5ファイルv19とは合わない。古いスクリプト既定値や出力名をそのまま最新評価と扱わない。

今後のユーザー学習では`SAVE_REVIEW_WEIGHT_CHECKPOINTS=True`により、実行UUID別の`data/attacker_plant/<相手>/review_checkpoints/v<版>_<UUID>/set_000000.pt`と各セット重みを残す。schema・入力hash・warm_start由来・未評価状態を保存する。これは重みだけでありoptimizer/replayを含む完全再開checkpointではない。未評価重みをbestへ昇格させない。

## 12. 条件専用のコードと重みを分離する仕組み

この節は後から作った基盤であり、FRC右問題への新対策を学習・採用した結果ではない。

資料：[条件別拡張とモデル分離](tv3_scoped_model_extensions.md)。`tv3_model_extensions.py`でrole・opponent・L/Rを指定する拡張を読み込む。`tv3_custom_train_attacker_plant_frc_v1_R.py`は入出力を変えないテンプレート。**現在PLANT_CUSTOM_MODULES=()で無効**。

有効化した片側専用profileでは、共通重みを固定・保存し、対象サイト専用重みだけを専用optimizer/replayで学習する。対象外は固定共通モデルとそのencoderを使う。共通重みは専用optimizerへ含めず、対象外transitionを専用replayへ入れない。対象外へ切り替わった時点で専用系列を閉じ、対象外の後続成功を専用系列へ誤帰属しない。

専用モデル・データ・ログはprofile digest別に保存する。モデルに拡張source hash、revision、routing、共通checkpoint本体とSHAを保持し、共通bestを上書きしない。コードを分けるだけで重みを共有し続ける構成ではない。

現在の配線対象はattacker plant。search/retake/guardの専用学習分離は未実装。左右両方の複数専用モデルを合成するrouterも未実装。通常対戦への自動切替はなく、guard依存も整合確認が必要。固定重みでも行動履歴が変われば左成績の完全不変は保証できない。

検証は251件の関連テストと、seed804000042の12ラウンドで同じ版・同じ重みのidentity拡張比較。867tickの観測/mask/行動hash・ラウンド結果が一致、10設置。共通transition3910に対し専用収集1951はすべてR。optimizer更新はしていない。ログ：`logs/custom_extensions/identity_routing_parity.json`、`unit_tests.txt`。これは分離機構の検証であり右性能改善の証拠ではない。

## 13. 次の調査の優先順位

1. v16重み+v19処理を同じ固定analysisで72ラウンド比較し、入口・終盤修正だけを残したときの左/右と人数条件を確認する。単一round8成功だけで採用しない。
2. 同じ先行履歴・開始RNG・観測/maskで、旧重みの能力使用と新重みの見送りを比較する。まず16tick RECON分岐とFLASH判断を調べ、教師target・Q差・合法性・支援射線・損失を分ける。
3. v19右timeoutと人数不利設置、新しい左round19を調べる。複数症状を一つの説明へまとめず、全体回帰を測る。
4. 固定analysisはv10 plantで集めた成功予測であり、v19実行成功率との整合は未解決。最新plantと収集条件を揃えた予測/実行照合を行う。右高予測を疲労・履歴だけで説明しない。
5. 対策を共通バグ修正かFRC右専用か分類する。専用profileの利用は新しい実験として扱い、既存v19のFRC関連処理が自動で専用ファイルへ移されたとは考えない。
6. 原因・修正・非学習評価が揃い、再学習が必要になった段階でユーザーへ説明する。現時点で追加学習を自動実行しない。

右へ行くこと自体が目的ではない。公開情報で危険と判断できるときの別入口・左への変更、被リコン時の退避、適切な投擲と支援による有利な設置が目的。未確認の成果や本番勝率を断定しない。
