# 戻り切れる距離での撤退練習（v5 run）

完了したv4はST174991です。保存履歴611件の撤退開始距離は中央値23マス、520件が12マス超でした。旧制限は12tickで、スポーン到達まで練習を続けられないケースが多い状態でした。

通常設定は `gc3_train_attacker_micro_directives.py` 冒頭です。

- 戦闘開始後4tickから、スポーンからの歩行距離が6〜10マスのプレイヤーを練習対象にします。
- 撤退は本人のスポーンへ、到達するまで最大20tick続けます。一度に一人、各プレイヤーはラウンド中1回までです。
- それ以上遠いプレイヤーには、この段階では撤退練習を出しません。開始距離を `retreat_practice_events.start_distance` に保存します。
- 撤退後は2tick待機し、元の設置計画へ再進軍します。設置・オーブ回収への保護、A/Bの選択、ユーザーの5の候補、ミクロの確率的な行動選択は継続します。
- 前回のv4 `latest.pt` を引き継ぎます。基礎ミクロ・設置後ガード・相手の重みは固定です。報酬は変更しません。

追加20,000判断で、ST174991→194991です。モデルは `data/attacker_micro_directives_v5/`、ログは `logs/attacker_micro_directives_v5/` に保存します。旧runは保持します。初回 `RESUME=False`、途中再開は同じrunで `RESUME=True` です。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3\macro_v2
python gc3_train_attacker_micro_directives.py
```

グラフは同じ作業ディレクトリの別PowerShellから実行します。既定はv5で、5秒ごとに自動更新します。

```powershell
python tools/gc3_watch_attacker_directive_metrics.py
```

撤退の接近率だけでなく、スポーン到達と時間切れ件数を比較してください。近距離での到達が増えてから、開始距離を広げます。この設定を用意しただけでは学習は始まりません。
