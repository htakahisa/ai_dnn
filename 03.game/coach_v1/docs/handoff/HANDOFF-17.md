# HANDOFF-17: 比較評価と ablation

## 実装

- `evaluate_task17.py` に、同一 seed・開始サイド・相手で checkpoint を固定した比較評価器を追加した。相手は `omoko_v1`、`touyama_v2`、`gc_v1`。各相手の開始サイドを入れ替えて試合を実行する。
- Task 16 で採用した警戒ポイント追加後の coach 2 本とキャラクター 6 本を使用する。レポートに checkpoint と警戒ポイント設定の SHA-256 を記録する。core、推論側の戦術処理、学習コードは変更していない。
- 対戦後の referee/replay から、勝率、round 勝率、plant 率、retake 率、平均生存人数、単独突入率、trade 率、ability 有効率、複数角度形成率、移動衝突率、警戒ポイント確認経過時間を集計する。相手別の成績と各試合の分子・分母を JSON に残す。
- `no_belief_memory`、`no_watch_point_input`、`no_tactical_intent`、`no_character_models`、`no_recurrent_state`、`no_team_shared_vision` の評価専用診断試験を追加した。各キャラクター視認分離は合法な個人視認センサーと個別 belief を使用する。coach は固定の slot 0 の視点を使う。
- `no_random_placement` は学習時の敵配置分布を変える条件である。既存 checkpoint の入力操作では比較できないため、レポートで再学習が必要と明示する。`no_recurrent_state` は既存再帰モデルの隠れ状態を毎 tick 消す診断であり、ConvGRU を外して再学習した構造比較ではない。これらの推論診断から構成要素の因果的寄与を断定しない。

## 指標の定義と情報境界

- ability 有効率は成功・解決済みの SMOKE/FLASH/RECON だけを分母とする。FLASH/RECON は敵への状態効果、SMOKE は敵射線遮断と味方射線遮断の差で判定する。
- 単独突入率は referee 側の近似指標で、敵の Manhattan 距離 2 以内への attacker 移動を分母とし、味方の生存者が距離 3 以内にいない移動を分子とする。敵の実座標を用いるのは試合終了後の評価だけで、actor には渡さない。
- 移動衝突率は同一 tick に複数の coach 移動指示が同じ到着地点を指定した割合。未確認エリアの経過時間は合法 belief audit に記録された警戒ポイントの確認経過 tick の平均であり、未確認地点には round 経過 tick を使用する。
- `TeamPerceptionBuilder` から actor へ渡す敵情報は合法な視認報告と公開生死情報に限る。新しい評価専用センサーで未視認敵の実位置を変えても snapshot が同じことを自動テストで確認する。replay と referee の実座標は試合後の集計関数だけが読む。

## 実行方法

```text
python -X utf8 -m coach_v1.evaluate_task17 --seed 1700 --matches-per-opponent 2 --output coach_v1/reports/task17_comparison.json
```

## 結果と残課題

`seed=1700,1701`、3 相手 × 両開始サイド × 7 条件の **42 試合**が完走した。全条件で match 勝率は **0/6**。相手別の基準条件は omoko_v1 で 0/26 round、touyama_v2 で 3/29 round、gc_v1 で 4/30 round の勝利だった。したがって現 checkpoint はこの3相手に対して十分な競争力がなく、2 seed だけで構成要素の優劣を確定できない。

| 条件 | round 勝利 | plant | retake | ability 有効 | 単独突入率 | 移動衝突率 |
|---|---:|---:|---:|---:|---:|---:|
| 基準 | 7/85 | 9/42 | 2/41 | 52/108 | 0.648 | 0.010 |
| belief memory なし | 9/87 | 11/42 | 2/44 | 52/105 | 0.790 | 0.009 |
| 警戒ポイント入力なし | 8/86 | 12/43 | 1/38 | 67/121 | 0.558 | 0.016 |
| 戦術意図なし | 8/86 | 9/42 | 1/42 | 35/74 | 0.789 | 0.018 |
| キャラクターモデルなし | 6/84 | 5/42 | 1/40 | 対象 cast なし | 0.855 | 0.110 |
| 再帰状態なし | 6/84 | 6/42 | 0/39 | 50/128 | 0.694 | 0.035 |
| team 共有視認なし | 11/89 | 10/45 | 5/43 | 60/77 | 0.732 | 0.011 |

キャラクターモデルなしでは plant が 9→5、複数角度形成率が 0.761→0.064、移動衝突率が 0.010→0.110 に変化した。北向き固定・ability 無効という強い置換なので、モデルの寄与を定量的に推定した結果ではない。再帰状態なしでは plant が 9→6、retake が 2→0 に変化したが、学習済み再帰重みを使ったままの介入であり、ConvGRU なしで再学習した比較ではない。belief、警戒ポイント、戦術意図、共有視認を外した条件の round 勝利は基準より増えたものもあり、この小標本から有効性を説明できない。特に共有視認なしの改善は、slot 0 だけを coach 視点とする評価条件や相手 AI の行動変化を含むため、共有視認が有害という根拠にはならない。

評価結果の全指標、相手別成績、各試合の分子・分母、モデルと設定の SHA-256 は `../../reports/task17_comparison.json` に保存した。使用した coach checkpoint は `../../checkpoints/experiments/task16_watch_added/coach/{attacker,defender}/latest.pt`。SHA-256 は attacker `3ae623afda2a8a96e042d1b12560abea77c014ce68028a5d2ea2def0e25b0ab7`、defender `a175cfdaf08a3c6c3c0bb01b06ef85ce8b225d8ef0adc783897e2aa896c70cdc`。キャラクターと警戒ポイント設定の hash はレポート参照。これが Task 16 で採用された再現可能な現候補であり、この評価では新しい best checkpoint へ昇格させていない。

### テスト

- `python -X utf8 -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q`: **205 件成功**。
- `python -X utf8 -m compileall -q coach_v1`: 成功。
- `git diff --check`: 成功。
- 評価専用の個人視認センサーで、slot 1 が視認した敵が slot 0 に共有されないこと、未視認敵の実座標を変えても snapshot が変わらないことを確認した。既存の encoder と本番統合の未視認敵テストも全テスト内で成功した。

### 残課題と次の推奨作業

`no_random_placement` は 70/20/10 の学習分布を変えて同じ手順で再学習する必要がある。ConvGRU を除いた構造比較も再学習が必要。いずれも今回の凍結 checkpoint 診断には含まれない。次は seed を増やした独立評価と、これら2条件の再学習済み checkpoint を同一 seed・同一相手に対して比較する。現状は全相手に match 勝利がないため、先に Task 16 現候補の対 omoko/touyama/gc 改善を独立した訓練・検証 seed で確認することを推奨する。
