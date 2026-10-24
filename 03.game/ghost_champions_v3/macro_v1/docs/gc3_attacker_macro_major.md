# マクロ：メジャーチームのメインAIとの継続学習

固定ロジック相手のrunはST20,621・175ラウンドで完了。
固定評価28試合の初期は5勝・13設置、保存best（ST11,519）は11勝・15設置。
latest（ST20,621）は8勝・13設置だったため、次段階は保存bestから引き継ぐ。

次段階もマクロだけを学習する。設置前・設置後モデルは凍結し、指定の報酬を維持する。
既存の教師データの再収集、BC、下位モデルの再学習は実行しない。

## 対戦相手

| チーム | AIキー |
|---|---|
| FNC | `fnatic_v3` |
| FRC | `frc_v1` |
| TYG | `touyama_gaming_v2` |
| OMG | `omoko_gaming_v1` |
| SPS | `toru_ai_v3.1` |
| GRG | `concon_v1` |
| GC | `ghost_champions_v2` |

ロスターとAIを組で扱い、7チームをシャッフルした一巡ごとに全チームと対戦する。
GCはParty Presetの既定AIのGC v2。SPSは従来の対戦設定で使用するToru AI v3.1。
相手はゲームに登録されている既存のコントローラーを使う。相手の戦術はそのまま、モデルの重みを凍結する。
設置時などに遅れて読み込まれるモデルも最初のforward前に凍結し、ラウンド終了時に重みを検証する。
実際に読み込んだチェックポイントのパスとハッシュを各ラウンドに記録する。

## 通常実行

作業ディレクトリは `D:\git\ai_dnn\03.game\ghost_champions_v3\macro_v1`。
通常設定は `gc3_train_attacker_macro_major.py` 冒頭の名前付き定数。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3\macro_v1
python gc3_train_attacker_macro_major.py
```

- `RUN_NAME="attacker_macro_major_v1"`：前段とは別run。
- `INITIAL_CHECKPOINT`：前段のマクロbest（ST11,519）。重みを引き継ぎ、optimizerは新規。
- `TRAINING_STEPS=30000`：ST11,519を含む累計目標。7ラウンド単位で収集するため超過する場合がある。
- `RESUME=False`：初回。2回目以降は`True`にして累計目標を変更し、同じコマンドを実行する。
- `OPPONENT_MODE="major_main"`、`OPPONENT_AIS`：上記AI対応。
- `PREPLANT_CHECKPOINT` / `POSTPLANT_CHECKPOINT`：前段と同じ凍結した下位モデル。
- `OPPONENT_VALIDATION`：学習なしの対戦相手検証結果。

保存先は `data/attacker_macro_major_v1/` と `logs/attacker_macro_major_v1/`（macro_v1内）。
前段のモデル・ログ・manifestを上書きしない。
相手が変わるため、前段の評価値と新段階の評価値を同じグラフに混ぜない。
前段は当時のソース署名で保持する。今回の変更後に前段をそのまま再開するのではなく、新しい上記runへ引き継ぐ。

## グラフと実戦

同じ作業ディレクトリで別のPowerShellから実行する。

```powershell
python tools/gc3_watch_attacker_macro_major_metrics.py
```

新段階の実reward・return・エントロピー・実勝率・設置率・追加+10獲得率を5秒ごとに更新する。
開始前は記録待ち。前段のグラフは従来の `gc3_watch_attacker_macro_metrics.py` で見られる。

実戦描画はGCv3ディレクトリから実行する。

```powershell
Set-Location D:\git\ai_dnn\03.game\ghost_champions_v3
python tools/gc3_watch_attacker_preplant.py
```

「主要AIマクロbest」「主要AIマクロ最新」を選ぶと、描画でも各チームのメインAIと戦う。
旧「マクロ best」「マクロ最新」は固定ロジックとの比較用。
通常ゲームも新段階のbestができれば優先する。

## 学習なしの検証

macro_v1を作業ディレクトリにして実行する。

```powershell
python tools/gc3_verify_attacker_macro_major.py
```

7チームで実ラウンドを凍結再生し、接続・確率再計算・重みとモデルファイルが変わらないことを検証する。
ConConの遅延読み込みされるリテイクも、両サイトの本番モデルを先に検証する。
記録は `logs/validation/gc3_macro_major_opponents.json`。
これは学習開始ではなく、教師データも作成しない。
学習開始・再開時に現在のソース、AI対応、相手モデルのハッシュとこの検証結果を照合する。
