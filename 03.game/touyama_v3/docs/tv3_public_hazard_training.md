# 設置型アビリティの観測と再学習

2026-10-10 に共通の行動観測へ設置型アビリティの情報を追加した。

- NEON: 予告と発動中の範囲を別のチャンネルで入力する。
- DESTRUCTION (ASH): 表示されている設置範囲とレベルを入力する。
- BALEMOON: 表示されている予告範囲を入力する。
- 自身の生命契約: 有効状態、残り時間、最大HPの減少量を入力する。

周囲7×7、待機・上下左右の移動先、目標地点、公開されているスパイク位置に対して範囲情報を渡す。追加は227項目で、共通観測は505から732項目になる。

敵味方、隠れたトラップ、投射物の未来の着弾位置、設置物の内部残り時間は参照しない。公開エフェクトには所有者の情報がないため、範囲は「潜在的な危険」として観測する。味方の範囲にも一律に罰を与える実装は行わず、既存のHP減少に対する減点を使って学習する。searchとplantではHP10減少につき報酬-0.1。生命契約は踏んだ後に範囲外へ出ても継続するので、自身の状態も必要となる。

危険範囲を理由に移動・待機・プラント・解除を禁止したり、推論側で回避行動を強制したりしない。行動は学習済みQ値と合法行動のマスクから選択する。回避能力の獲得は再学習後の評価で確認する必要がある。

## 互換性

共通観測を使うdefender search/retakeとattacker plant/guardのモデル、最新チェックポイント、replayは旧版と互換性がない。識別値も更新し、旧モデルの読み込み・再開は拒否する。旧モデルのメタデータだけを書き換えて使用しない。

Scenario、盤面、配置、射線判定、分析モデルの観測は今回変更していない。現在のScenarioと一致するdefender_analysis/attacker_analysisはそのまま使用できる。以前の射線仕様の分析モデルが残っている場合は、先に分析の再学習が必要。

searchの更新はretakeの依存ハッシュを変える。plantの更新はguard収集元のハッシュを変える。新しいplantからguardケースを再収集する。retakeを保存ケースから学習する構成でも、更新後のsearchを使ってケースを再収集する。

## 通常の実行

学習はユーザーが実行する。実行中の旧コードの学習を止めてから新しい実行を開始する。

各スクリプト冒頭で対象AIと回数を設定する。defender search/retakeは `TRAINING_MODE = "fresh"`、attacker plant/guardは `RESUME_TRAINING = False`、`EVALUATION_ONLY = False` にする。収集スクリプトも冒頭の対象AI・保存先・ケース数を確認する。既存モデルを残す場合は、実行前にdata配下へバックアップするか、冒頭の保存先定数を変更する。

作業ディレクトリは `03.game/touyama_v3/`。`03.game` から移動した後、通常はオプションなしで実行する。

```powershell
Set-Location touyama_v3
python tv3_train_defender_search.py
python tv3_train_defender_retake.py
python tv3_train_attacker_plant.py
python tv3_collect_attacker_guard.py
python tv3_train_attacker_guard.py
```

searchはプラントまで、plantもプラントまでの学習。プラント後の回避はretakeとguardで学習する。
