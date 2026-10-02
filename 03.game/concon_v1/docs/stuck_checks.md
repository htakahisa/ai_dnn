# 移動・設置の詰まり診断

`concon_v1` ディレクトリで実行します。学習やモデルの保存は行いません。

```powershell
py .\check_co1_stuck.py -map A1
py .\check_co1_stuck.py -map A1 A2 A3
```

標準は敵なしの移動環境で、モデル不要の移動優先（`move-first`）を30回試します。
1回の上限は300 tickです。通常の100 tick以内の設置回数も別に表示し、
時間不足と、その後も続く停止を確認できます。通常と同じ制限なら
`--max-ticks 100`、短い確認なら `--rounds 5` を指定します。
座標は0始まりの `[行, 列]` です。
時間切れでも、到達した段階・未到達の地点・運搬者を含む全員の位置と目的地を表示します。
`stops=0` は同じ停止条件が10 tick以上続かなかったことを表します。
設置成功や、すべての経路に詰まりがないことを保証する数値ではありません。

## 行動の選び方

```powershell
# 動けるときは待機を避ける対照実験。実際の学習モデルの性能評価ではありません。
py .\check_co1_stuck.py -map A1 A2 --policy move-first

# ランダム行動でも確認する場合は明示的に指定します。
py .\check_co1_stuck.py -map A3 --policy random

# 保存済みモデルで確認。各マップの latest を優先し、なければ best を読みます。
py .\check_co1_stuck.py -map A2 --policy model

# episode 60 / 全2000 episode の探索率に近い設定
py .\check_co1_stuck.py -map A3 --policy model --epsilon 0.959

# 指定したチェックポイントを確認
py .\check_co1_stuck.py -map A2 --policy model --model .\data\attacker_A2_data\co1_attacker_A2_best.pt
```

`random` は待機も含めた有効行動から一様に選びます。後退・迂回も選ぶため、
目的地に向かわず歩き回ることがあり、時間切れだけでは詰まりと判定できません。
標準の経路確認用である `move-first` は、
設置可能なら設置し、それ以外は目的地への距離が最も短くなる有効な移動を選びます。
同距離なら UP / DOWN / LEFT / RIGHT の順を使います。地点に到達した後は待機しますが、
譲るための退避が指定されている場合は退避します。同じ経路に偏るため、
ランダム試行と合わせて確認してください。
マップ変更とモデルの設定が一致しない場合はエラーにします。
変更直後のマップは標準の `--policy move-first` で確認できます。

## 実戦環境

```powershell
py .\check_co1_stuck.py -map A2 --mode battle --policy model --max-ticks 100 --rounds 3 --opponents gc_v1 fnatic_v3
```

`battle` は学習で使う実戦環境とIQ知覚、戦闘・アビリティ・スパイク回収を使います。
対戦相手は標準で `gc_v1`。複数相手を指定すると相手ごとに `--rounds` 回試します。
`battle` でも `random` / `move-first` を選べますが、接敵やアビリティによる
行動の上書きは本番と同じです。上限を100より大きくすると診断のために
ラウンド時間を延ばします。全滅などによる終了は延長されません。

## 出力の読み方

`STOP` は座標・目的地・制約などが `--stuck-ticks`（標準10）以上変わらない状態です。
停止中でも戦闘が続くことがあるため、すべてが永久的な詰まりとは限りません。

- `ally_blocked`: 目的地へ距離を縮める隣接マスが味方に塞がれ、移動・設置を選べない。
- `movement_restricted`: 移動・設置の有効行動がない。味方以外の制約を確認する。
- `unreachable`: 地形上、目的地に到達できない。
- `policy_wait`: 移動や設置を選べるが待機を選び続けている。
- `controller_hold`: 実戦で接敵・アビリティ等の処理により経路ポリシーが呼ばれていない。
- `movement_not_applied`: 移動を選んでも位置が変わらない。実戦の効果や競合などを確認する。
- `sync_wait`: 最初の `a` で他グループを待っている。長期化した場合は他の味方も確認する。

通常の護衛の待機、死亡、回収フェーズ、進行中の設置は詰まりとして数えません。
`ended_by=moved` は位置が変わったこと、`stage_changed` は位置が同じまま段階が
変わったこと、`constraints_changed` はその他の制約変更を表します。
`still_stopped` は停止条件が試行終了まで続いたことです。
JSONの `resolved` は実際に位置が変わった場合だけ `true` にします。
詰まったキャラの座標、目的地、距離、許可行動、塞いでいる味方も表示します。
`EXCLUDED` はその出発地点から距離上限を超えるか到達できない候補です。
他の出発地点から選べる場合もあるため、候補全体が未使用とは限りません。

```powershell
py .\check_co1_stuck.py -map A1 A2 A3 --output .\reports\stuck_check.json
```

JSONには全試行の終了状態、停止時の5人の状態、隣接マスの地形・味方・
目的地方向、候補間のBFS距離を保存します。停止ログにはその段階の到達済み地点と
必要な全地点も表示します。`SUMMARY` の
`plants_within_100_ticks` が通常制限内の設置数、`plants` が診断上限までの設置数です。
