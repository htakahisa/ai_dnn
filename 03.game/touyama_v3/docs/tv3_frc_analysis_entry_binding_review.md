# 同一状態の入口評価・目標解除の検証（2026-10-11、未解決あり）

設置先変更と右3人の仮想公開情報への反応は [tv3_frc_site_switch_review.md](tv3_frc_site_switch_review.md) を参照。変更周期・制限の緩和は回帰があり、不採用。

## 同じ公開入力でも、latestは外周Cの評価を下げる

seed804000042第6ラウンド33tickの公開観測と経路候補を採用best/latestへ渡した。latestはユーザーが作成した旧v3・set4であり、こちらで学習更新はしていない。敵配置推定と全経路入力も採用bestの値へ固定して比較：

| 候補 | 採用best成功予測 | latest成功予測 | latest score |
| --- | --- | --- | --- |
| C・supported | 99.95% | 51.86% | 0.1829 |
| C・preserve | 99.96% | 65.47% | 0.3404 |
| 安全側b→a→f→k→l・preserve | 約99.8% | 92.16% | 0.5798 |

配置推定を固定してもCより安全側の右経路が高評価になった。latest自身の配置推定を使う場合も同じ順位。左候補の評価はさらに高いが、通常のサイト変更marginにより、このtickでは右の安全側が選ばれる。数値は経路指定後の適応を含むv3の成功予測であり、指定入口だけを突破する確率ではない。未学習seed全体の予測誤差改善は確認されていない。

`frozen_804000042_r6_True.jsonl` の `comparison_analysis` に記録した。ログ有無の対戦結果と乱数は一致し、保存済み旧v3/v10の対象ラウンド記録・採用重みhashとも一致した。

## 同一開始状態の実動作：目標が経路変更を妨げた

採用bestで同じ33tickまで進め、物理・公開状態・乱数hashを一致させてから介入した。plantは採用v10を固定。経路や行動を強制せず、通常のplannerで再評価を続ける。

| 介入 | 33tickの保持者目標・行動 | 結果 |
| --- | --- | --- |
| 対照 | C (11,40)、E | 時間切れ100tick、2人生存、HP160 |
| analysisをlatestへ変更 | C (11,40)、E | 時間切れ100tick、4人生存、HP360 |
| analysis変更＋入口追従 | (22,31)、W | **左**に96tick設置、2人生存、HP200 |

analysisだけ変更した場合も安全側の右経路は選ばれたが、既存C目標が保持者を東へ向かわせた。入口追従で西へ進み、その後の通常再計画で左へ変更して設置した。右突破の成功ではない。`analysis_swap_summary.json` に記録し、対照は変更前の保存ラウンド記録と一致した。モデル差し替えは診断だけで、productionの依存hashは変更していない。

## 全体回帰で範囲を絞った

latest＋採用plantの同じ36ラウンドで検証した。最初のseedの無介入12ラウンドは、保存済み基準の設置・終了tick・生存・HP・理由・設置先と一致することを確認した。

| 設定 | 設置 | 生存条件付き設置 | 平均生存 | 時間切れ |
| --- | --- | --- | --- | --- |
| 基準（旧入口処理） | 26/36 | 25/36 | 3.44人 | 6 |
| 入口追従＋初回から接近残距離を使用 | 20/36 | 19/36 | 3.28人 | 14 |
| **入口変更時だけ目標解除・接近残距離を使用** | **31/36** | **27/36** | **3.75人** | **3** |

広い変更は不採用。`bind_changed_entry_only=True` を追加し、初回準備の開始条件は従来どおり、入口が変わった後だけ古い割当を消して新経路の接近残距離で準備を判断する。待機中も変更状態を保持し、翌tickに古い条件で再割当しない。

全体改善の多くは左への設置。右設置数は基準4回、限定修正3回であり、右攻略全体が改善したとは言わない。36ラウンド・1組の旧モデルによる診断で、全相手・未学習モデルへの一般化は未検証。

新規analysis v4のcontrollerはこの限定修正を使う。schemaの `site_entry_route_binding` を `changed_entry_outside_remaining_path_v3` に更新した。旧v3は従来条件のまま。広い変更を新規v4の既定として残さない。今回の入口切替修正だけを理由に旧モデルのschema/重みを流用しない。

## 再現・未解決点

作業ディレクトリ `03.game/touyama_v3/`。設定は各スクリプト冒頭の定数。

```powershell
python tv3_review_attacker_analysis_swap.py
python tv3_review_attacker_entry_binding.py
```

共有入力のモデル比較は `tv3_review_attacker_route_risk.py` の `COMPARISON_ANALYSIS_PATH` に同一入力契約のモデルを指定して通常実行する。一時指定は `python tv3_review_attacker_route_risk.py --compare-analysis data/attacker_analysis/frc_v1/latest.pt`。

結果は `logs/attacker_route_risk_review/`。広い変更の結果は `entry_binding_latest.json`、限定修正は `entry_binding_latest_changes_only.json`。広い変更を再検証する場合は `BIND_CHANGED_ENTRY_ONLY=False` で `_full.json` に出力する。

未解決なのは、未学習seedでの高信頼予測の誤り、初回から長い安全経路を選んだ場合の準備・実行との整合、右入口での能力使用と突破。入口切替を自律行動へ反映する一つの不整合は修正したが、右サイト解決や再学習準備完了とは扱わない。学習更新0回、採用重み不変。

最終確認：非学習テスト194件通過。限定修正で初回準備が従来と一致し、入口切替時に目標が消え、翌tickにも解除状態を保つことを検証した。同一状態のmodel swapも限定修正で再実行し、`analysis_swap_changes_only_summary.json` に保存した。採用bestとユーザーのlatestの重みhashは候補比較時と一致した。
