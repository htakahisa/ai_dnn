# HANDOFF-16: 苦手地点の自動収集

## 実装範囲

2026-09-30。`01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、Task 14/15 の引継ぎ、
`full_match.py`、replay 形式、`TeamExecutionCoordinator` と belief memory を確認した。
固定マップの実試合後に候補を集計し、警戒ポイント設定、checkpoint、推論時の行動は変更していない。
core ファイルも変更していない。既存の headless match と replay だけで実装できたため、
core 変更の必要性はなかった。

## 変更内容

- `task16_hard_points.py`: 試合後の replay、round record、coach action/audit を照合する。
  同じ敵付近での複数死亡、未視認の近接、単独死亡、複数人の向き重複、長時間未視認後の死亡、
  合法 belief で長時間未確認の警戒ポイント付近での死亡、植設後の敗北、FLASH/RECON 後に
  敵 status 変化を確認できなかった場合を候補として出す。座標、向き、phase、件数、相手、
  損失内容、replay の round/tick/frame、既存警戒ポイント ID、判定根拠を含む。
- `collect_task16.py`: Task 15 の opponent pool を使って実試合を実行し、候補 JSON と
  参照先の replay/round report を保存する。seed、相手、score、checkpoint SHA-256 も保存する。
- `full_match.py`: 試合結果へ coach action と decision audit を追加する。
- `coordinator.py`: 既存 belief snapshot にある警戒ポイントの確認経過 tick を、
  計算済みの team decision audit に記録する。actor 入力には追加しない。
- `test_coach_v1_task16_hard_points.py`: 候補抽出、件数の重複排除、replay 参照、
  可視敵を未視認として扱わないこと、効果が観測された ability を候補にしないこと、
  入力 replay を変更しないこと、round 不整合の拒否を確認する。

## 実試合での収集

以下は初回の難易度1・2試合による動作確認結果。後段の10試合集計が最新版である。

実行コマンド:

```text
python -X utf8 -m coach_v1.collect_task16 --seed 1600 \
  --matches-per-opponent 1 --max-difficulty 1 \
  --output coach_v1/reports/task16_candidates.json
```

`omoko_v1` と `simple_random` に対し各1試合。両試合とも side swap を含む。
coach の score は順に 1-13、11-13。134候補を
`reports/task16_candidates.json` に保存した。内訳は向き重複85、単独死亡17、
敵付近の繰り返し死亡14、FLASH/RECON効果未確認9、植設後の敗北4、
未視認近接3、長時間未確認の警戒ポイント2。
うち118候補は既存警戒ポイントと座標が完全一致しない。
上位の繰り返し死亡候補は `[10,23]` が11件、`[7,22]` が6件、
`[6,23]` が5件。ただし近くの敵をキラーと断定していない。
replay と round report は `reports/task16_candidates_replays/` に保存した。

## 情報境界と判定上の注意

replay には審判視点の敵実座標が入るが、抽出は試合終了後だけに実行する。
`TeamExecutionCoordinator` はこれを actor 観測に渡さない。
追加 audit は合法な belief のポイント確認経過 tick と public clock だけを保持し、
敵の実座標、game object、actor observation を保持しない。
既存 Task 14 の未視認敵位置変更時の coach/character 観測不変テストも全体テストで再実行した。

候補は人手レビュー用であり、教師ラベルではない。replay にはキラー ID がないため、
死亡近傍の敵位置は推定候補にとどめた。FLASH/RECON は action の要求と3tick以内の
敵 status を照合しており、cast 成功や本当の無効性を証明しない。
SMOKE/HUNT の有効性はこの replay 形式から確定できないため候補化しない。
「未視認」は「未クリア」と同義ではなく、後者は belief の確認経過 tick を使用する。
上位件数は2試合に基づくため、警戒ポイントへの追加前に相手と seed を増やして確認する。

## テスト結果

- Task 16 テスト: 5件成功。
- `python -X utf8 -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q`:
  183件成功。
- `python -X utf8 -m compileall -q coach_v1`: 成功。
- `git diff --check`: 成功。

## 残課題と次の推奨タスク

初回確認時点では、実試合は難易度1の相手2件のみ。候補の多くは重複する近隣セルで、キラー推定と能力効果は
replay の記録範囲に制限される。次は Task 17 の比較評価・ablation と合わせ、
難易度2/3の相手と未使用 seed でも Task 16 収集を行い、複数相手で繰り返す地点だけを
人が警戒ポイント追加候補として審査する。追加を決めた場合は固定マップ設定と学習データを
更新し、attacker/defender の該当モデルを再学習する。

## 2026-09-30 追加確認：Task 17 へ進む前の依存関係

初回2試合だけでは相手分布が狭く、近隣セルを別候補として重複計上し、キラー位置も推定だった。
これを Task 16 内で修正した。`full_match.py` の Task 16 専用オプションで既存 combat tracker の
kill 通知を試合後記録へ写し、実際のキラー座標を取得する。core と actor 観測は変更しない。
同名選手同士の historical coach 戦では round 統計の名前衝突があるため、replay に coach side を
記録して side を判定する。半径2マス以内で同一種類・side・phase・facing の候補をまとめ、
`member_positions` に元の各座標を保持する。

`--seed 1600 --matches-per-opponent 2 --max-difficulty 3` で opponent pool 全5相手を実行し、
各相手について attacker/defender の両開始 side を収集した。10試合すべて完走し、
`reports/task16_candidates.json` を更新した。**228候補**、うち**52候補が複数相手で発生**。
内訳は向き重複80、実キラー位置での複数死亡52、単独死亡36、FLASH/RECON効果未確認31、
長時間未確認の警戒ポイント10、未視認近接9、植設後の敗北6、長時間未視認後の死亡4。
全候補の replay 参照先が存在することを確認した。

座標の完全一致だけでは既存の学習分布を判定できないため、各警戒ポイントの
`random_radius` を含む範囲も照合し、レポートに `covering_watch_point_ids` と
`uncovered_positions` を追加した。レビュー優先の範囲外候補は次のとおり。
件数は近隣セルをまとめたもの。実キラー位置に基づく。

| 代表座標 | side / facing | 死亡件数 | 試合・round | 複数相手での再発 |
|---|---|---:|---:|---|
| `[12,40]` | defender / S | 15 | 4試合・8round | gc_v1、simple_random |
| `[11,31]` | attacker / E | 13 | 3試合・8round | omoko_v1、touyama_v2 |
| `[7,34]` | defender / E | 6 | 3試合・6round | omoko_v1、simple_random |
| `[13,31]` | attacker / S | 5 | 2試合・4round | gc_v1、touyama_v2 |

最多の `[6,23]`（25件、3相手）は既存 `watch_r06_c25` と `watch_r08_c23` の
周辺ランダム化半径に含まれる。従って新ポイント追加の根拠にはせず、既存分布で学習しても
失敗する地点として診断する。`[11,30]` も既存 `center_mid_angle` の半径内。

これらは**警戒ポイントへまだ追加していない**。追加は replay で死因と向きを人が確認してから
決める。追加を選ぶと設定 hash と学習分布が変わるため、該当 checkpoint の再学習・再評価が必要。
その判断をせずに「Task 16 で発見した地点を反映済み」とみなして Task 17 の最終比較を行うと、
後で比較をやり直すことになる。

能力候補31件は requested FLASH/RECON 後の敵 status 変化が見えなかった例であり、
cast 成功や能力有効率の分母を保証しない。Task 17 は主要評価に **ability 有効率**を要求するため、
本比較を開始する前に使用成立・効果・分母の計測を実装してテストする必要がある。
この計測は Task 17 の評価基盤として実装すべきであり、Task 16 の候補を有効率として流用しない。
SMOKE/HUNT も含む効果定義は ability ごとに決める。

従って効率的な順序は、(1) 上記候補の人手レビューと追加／見送りの決定、
(2) 追加する場合の再学習・独立評価、(3) Task 17 冒頭の能力有効率計測の検証、
(4) 固定した設定・checkpoint で比較と ablation、となる。
Task 17 の比較本体を (1)～(3) より先に始めない。

追加変更後の Task 16 テスト9件、全体テスト187件が成功した。
警戒ポイント半径のテストも含む。`compileall` と `git diff --check` も成功。
GC の実試合で更新された既存デバッグログは元へ戻し、Task 16 専用ファイルと
`coordinator.py`・`full_match.py` 以外の既存変更を残していない。
