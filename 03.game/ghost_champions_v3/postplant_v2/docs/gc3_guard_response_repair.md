# 解除通知後の向きと追加学習の修正（2026-10-15）

ST16384 の固定評価は、開始時より悪化していました。ST17920 で停止した旧 run は保存したまま、
最新の修正版の学習は別 run `attacker_postplant_response_v3` です。
追加確認で隣接スモークの判定不一致が見つかったため、起動されていた `attacker_postplant_response_v2` は
ST0の初期評価中に停止し、保存状態を残してv3の本学習を起動し直しました。

## 確認した問題

- 旧 `public_coverage` は、敵を含む既知プレイヤーのいるマスを終点候補から除外していました。
  解除者本人のマスも除外されるため、公開入力が実際の射撃機会を正しく表現していませんでした。
- 解除可能範囲の最大値を使っていたため、解除者と別方向の空きマス一つを向いても高いカバー値になり得ました。
- 実データでは解除通知のある判断が少なく、例として収集923判断中43判断でした。
  その後さらに512判断へ無作為抽出するので、解除対応の学習機会が薄くなっていました。

通知そのものは診断した試合で欠落していませんでした。指定された失敗試合の seed は不明なので、
同一リプレイの原因を断定したものではありません。隣の敵が解除しており、自分が逆向きの条件を回帰テストで再現しました。

### FNC相手に頻発するという追加報告

保存済みの旧runのFNC戦35ラウンドを、切り捨て前の全rolloutから集計しました。
解除通知がある判断は277件。そのうち認識している公開スパイク位置から2マス以内で、
スパイクと同じマスを除いた判断47件中21件が、スパイクに対して90度を超える外向きの行動を選んでいました。
この選択は9ラウンドにまたがっており、学習記録上も一場面だけの傾向ではありません。

これは認識上のスパイク位置と、サンプリングされた意図したコマンドの集計です。
真の解除者との距離、実行後の向き、向きの強制の有無を表す値ではありません。
また、記録には学習途中の異なるモデルが含まれ、ST17920単独の成功率を表すものではありません。
FNC戦の解除負けは35ラウンド中7ラウンドでしたが、解除負けすべての原因を向きの選択に帰属させてはいません。

集計は次のコマンドで再実行できます。設定は各スクリプト冒頭です。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3\postplant_v2
python tools/gc3_audit_attacker_guard_records.py
python tools/gc3_audit_attacker_guard_fnc.py
```

前者は保存済み35ラウンドの全判断を調べ、後者はFNC相手の複数seedで、
向き変更が可能で視界・射線が遮られていない隣接解除者への反応を実戦で追跡します。
後者は歴史的な失敗seed群と、別の固定seed群を分けて記録します。
失敗seedを当時と異なる固定モデル・ラウンド開始状態で実行しても、同じ過去試合の再現にはならないため、
当時の勝敗と同じになるとは限りません。集計自体は実行中の学習を変更しません。

記録は `logs/validation/gc3_guard_fnc_historical_choices.json` と `gc3_guard_fnc_multi_seed.json` にあります。

### 隣接スモークの実戦再現と追加修正

FNC、seed `20361036`、旧ST17920モデルのtick58で次の場面を再現しました。

- Absolは `(8,3)`、解除中のAlfajerは隣の `(8,4)`。公開通知と敵の視認は届いています。
- 敵のマスにはスモークがありますが、ゲーム本体は隣接していれば射線を通します。
- Absolは `(8,2)` へMOVEして西を向き、東にいる解除者から離れました。移動後は射線も失っています。
- 学習側はスモーク内を距離にかかわらず遮蔽としていたため、この場面への補助損失を作れていませんでした。

学習側もゲームの `_smoke_allows_line` と同じ隣接例外を使うよう修正しました。
公開されたRECONリビールによるスモーク越しの射撃も評価し、壁と途中の生存プレイヤーは引き続き遮蔽物にします。
修正後は同じ実戦場面が学習対象になり、上記の後退コマンドを適切な解除対応として評価しないことを確認しました。
同じスモーク条件の回帰テストでは、補助学習後にゲームの射撃角度・スモーク規則を満たす行動を選べることも確認しています。

旧モデルの実行時挙動を学習なしで強制的に変更したわけではありません。
記録は `gc3_guard_fnc_adjacent_example.json`（修正前）と `gc3_guard_fnc_adjacent_example_fixed.json`（修正後）、
再開処理の確認は `gc3_guard_smoke_response_v3.json` です。いずれも `logs/validation/` にあります。

## 修正内容

- 終点の解除者と、射線途中の遮蔽物を区別します。
  視認・公開済みの解除者がいればその位置を使い、見えなければ公開された解除可能範囲の平均カバーを使います。
- 通知中の判断を優先して学習バッチへ残し、他のラウンドも残します。
- 公開情報から計算した射撃可能な向きに対し、既存の MOVE / WATCH / MOVE_WATCH の確率の合計を学習する補助損失を加えます。
  実行時の行動は従来どおりサンプリングし、向きやコマンドを強制しません。
  向き変更を強制されている状態や、視界が完全に遮られている状態には無効な教師を与えません。
- 通知のない場面では開始モデルとの KL を抑え、従来の行動を忘れにくくします。
  更新が KL 上限を超えた場合は重みと optimizer 状態を戻し、小さい更新幅で再試行します。
- 固定評価で2回連続 best を更新できなければ自動停止します。`status.json` の `stopped_early` に記録します。
  これは「悪化だけ」でなく、同等の評価が続いた場合も含みます。

入力の意味は checkpoint の `feature_version` に保存します。旧モデルは従来の入力のまま読み込みます。
`feature_version=3` が隣接スモークを修正した入力です。v1・v2の入力の意味もそれぞれ維持します。
報酬・optimizer・ソースが変わるため、旧 run の manifest を書き換えて再開することはしません。

## 通常実行

設定は `gc3_train_attacker_guard_response.py` 冒頭、共通設定は同ファイルが読み込む
`gc3_train_attacker_guard.py` 冒頭です。通常の実行にオプションは不要です。

- `START_CHECKPOINT`: 旧 run の `best.pt`。成績の落ちた latest ではなく、評価で選ばれたモデルから始めます。
- `RESUME=True`: v3の初回起動は完了しており、次回は保存状態から再開します。別の新規runを作る場合だけ `False` にします。
- `TRAINING_STEPS=20000`: 修正版 run 内での累計判断数の目標です。
- `RESPONSE_LOSS_WEIGHT=1.0`, `RESPONSE_SAMPLE_FRACTION=0.5`: 解除通知への補助学習と抽出比率です。
- `ANCHOR_KL_WEIGHT=2.0`, `EVALUATION_PATIENCE=2`, `KL_BACKTRACK_STEPS=6`: 忘却抑制・停止・更新幅の設定です。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3\postplant_v2
python gc3_train_attacker_guard_response.py
```

すでに同じrunが実行中の場合は、PIDと保存済みSTを表示して、今回のコマンドだけ終了します。
バックグラウンドの学習は継続します。これは学習完了の表示ではありません。

描画も同じ作業ディレクトリから、専用ビューアーを使います。既定では `best.pt` を表示します。

```powershell
python tools/gc3_watch_attacker_guard_response.py
```

学習グラフは同じ作業ディレクトリから次のコマンドで開けます。旧runのグラフとは別です。

```powershell
python tools/gc3_watch_attacker_guard_response_metrics.py
```

旧 run のモデル・manifest と修正前のソースは、
`data/attacker_postplant_antidefuse_v1/repair_backup_20261015/` に退避しました。
旧 ST17920 のモデルの重みを修正版に置き換えたわけではないため、旧ビューアーの挙動が改善済みという意味ではありません。
ST0で止めたv2の状態・ソースは `data/attacker_postplant_response_v2/before_smoke_fix_20261015/` に残しています。

## 検証

- 回帰テスト20件：隣接解除者への向き、敵がいる終点と途中の遮蔽物、隣接スモークと公開RECON、隠れた座標の非参照、
  向きの強制、少数通知の保持、補助学習の勾配、KL 上限、旧 checkpoint の互換性、
  自動停止、optimizer・乱数を含む再開、実際の設置からの引き継ぎを確認しました。
- `validation_response_v3_verified`：FNC の実試合から64判断だけ学習し、評価・保存・厳密な manifest 検証を通した再開初期化を確認しました。
  これは学習処理の動作確認で、7ロスターに対する勝率改善を示す評価ではありません。
- 旧モデル3個と manifest のハッシュが退避コピーと一致することを確認しました。

確認記録は `logs/validation/gc3_guard_response_repair.json`、実戦の診断は
`gc3_guard_facing_audit.json` / `gc3_guard_facing_audit_latest.json` にあります。
動作確認の詳細ログは `logs/validation_response_v3_verified/` にあります。

```powershell
Set-Location D:\git\ai_dnn\03.game
python -m unittest ghost_champions_v3.postplant_v2.test.test_gc3_guard_response ghost_champions_v3.postplant_v2.test.test_gc3_attacker_guard -v
```
