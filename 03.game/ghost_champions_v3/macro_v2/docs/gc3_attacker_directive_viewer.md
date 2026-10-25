# 指示対応ミクロのbestを描画する

設定場所は `tools/gc3_watch_attacker_directives.py` 冒頭です。通常は `RUN_NAME='attacker_micro_directives_v5'`、`CHECKPOINT_NAME='best.pt'`（ST189327）、相手は `OPPONENT='FNC'` の固定ロジックです。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3\macro_v2
python tools/gc3_watch_attacker_directives.py
```

「再開」で設置前から試合を再生します。設置後は既存のガードモデルに切り替わります。画面上に現在の指示（待機・進軍・撤退）と目的地を表示します。撤退の目的地は本人のスポーンです。

「次のラウンド」で同じウィンドウのままA/Bを交互に切り替えます。相手ロスターは選択欄で変更できます。「同じseedで比較」は同じ初期条件から再生します。

マクロの練習条件は対象runの `manifest.json` から読み込みます。別runの設定を混ぜたり、bestがなければ別モデルへ切り替えたりしません。候補マップと設置後モデルも保存された学習条件と照合します。通常のGC v3実戦モデルを変更せず、学習やチェックポイント更新を行いません。

一時的な動作検証のみ、次のオプションで更新なしの1ラウンドを実行できます。

```powershell
python tools/gc3_watch_attacker_directives.py --verify-only
```
