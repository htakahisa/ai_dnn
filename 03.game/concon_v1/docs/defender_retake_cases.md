# リテイク開始データの収集

`co1_collect_defender_retake.py` は既存の本番 search 重みを固定し、通常の setup/search から各AIと対戦します。スクリプトは設置が完了した tick の終了時点でゲームを保存します。設置完了と同じ tick の残りの defender 行動は待機にし、リテイクモデルの判断はまだ実行しません。収集スクリプトは学習・重みの更新を行いません。

既定の対象は omoko_v1、touyama_v2、fnatic_v3、gc_v1、toru_ai_v3.1 です。各チーム・各サイト50件、合計500件を収集します。スクリプトは設置位置や人数を作り替えません。設置前に終了した試合、設置時点でラウンドが終了した試合、defender が全滅した試合、収集済みサイトの追加試合は保存件数に含みません。

収集数は `co1_collect_defender_retake.py` 冒頭の `DEFAULT_CASES_PER_SITE = 50` で設定します。合計件数は「この定数 × 対象AI数 × 対象サイト数」です。定数を100に変更すると、既定の5チーム・左右2サイトでは合計1000件になります。`--cases-per-site` を指定した場合は引数を優先します。試行上限は同じ場所の `DEFAULT_MAX_ATTEMPTS_PER_TEAM`、1試合のtick上限は `DEFAULT_MAX_TICKS` です。

## 実行

以下のコマンドは `concon_v1` ディレクトリから実行します。

```powershell
# 既定: 各AI・各サイト50件、合計500件
python co1_collect_defender_retake.py

# 少数で収集・容量を確認する場合: 各AI・各サイト2件、合計20件
python co1_collect_defender_retake.py --cases-per-site 2 --output-dir data/retake_cases_small

# search 重みを明示する場合
python co1_collect_defender_retake.py --search-model data/defender_search_data/co1_defender_search_best.pt

# 中断した既定の収集を再開する場合
python co1_collect_defender_retake.py --resume

# 試合数上限で不足した場合: 上限を増やして再開
python co1_collect_defender_retake.py --resume --max-attempts-per-team 3000

# 特定のAI・サイトのみ収集する場合
python co1_collect_defender_retake.py --opponents omoko_v1 gc_v1 --sites L --cases-per-site 50 --output-dir data/retake_cases_left
```

既定の保存先は `data/defender_retake_cases` です。既存データへの上書きは拒否し、`--resume` で件数を引き継ぎます。再開時は対象AI・対象サイト・seed・search 重みの内容・マップが一致する必要があります。`--cases-per-site` を増やして同じデータセットに追加収集できます。

`--max-attempts-per-team` の既定値は各AI1000試合で、再開前の試合も数えます。`--max-ticks` の既定値は実際の試合設定の `DEFENDER_SETUP_TICKS + ROUND_DURATION_TICKS` です。現在は setup 20 tick と設置前ラウンド100 tickの合計120 tickで、ゲーム側の定数変更にも連動します。上限で未収集のサイトが残った場合、スクリプトは不足件数を表示し終了コード2で停止します。片方のサイトに攻めないAIから両サイト分を必ず収集できるわけではありません。

## 保存内容

- `collection.json`: search 重みのパス・ハッシュ、マップハッシュ、seed、対象AI・サイト、目標件数。
- `cases.jsonl`: 各保存データのファイル名、対戦相手、設置サイト、設置位置、残り時間、双方の人数、キャラクター概要。
- `rounds.jsonl`: 除外試合も含めた試行記録。再開時に試行番号を引き継ぎます。
- `*.case.gz`: 全キャラクター、ゲーム内の効果・タイマー、相手AIの進行状態、コントローラーの記憶、IQ知覚の状態、乱数状態を含む圧縮スナップショット。
- `tensors/*.pt`: モデルの重みなどのテンソルを内容ごとに一度だけ保存した共通データ。各スナップショットはこれを参照します。

モデルの重みを500件分複製しないため、容量は共通テンソルと各場面の状態の合計になります。データを移動する場合は、`tensors` を含むデータセットのディレクトリ全体を移動してください。

スナップショットは同じプロジェクトのコードとPython環境で再開するための形式です。コード・依存環境を大きく変更した場合は、復元確認または再収集が必要です。Pythonオブジェクトを復元するため、自分で収集した信頼できるファイルのみ読み込んでください。

## 復元

```python
from concon_v1.co1_retake_cases import load_case

game, metadata = load_case("data/defender_retake_cases/omoko_v1_L_000001.case.gz")
```

ファイル名の末尾はそのAIとの試行番号です。実在する名前は `cases.jsonl` で確認します。`load_case` は `init_round` やコントローラーの `reset_round` を呼ばず、相手AI・キャラクター・ゲームの参照関係を復元します。IQ知覚の一時キャッシュは再計算し、解除位置の記憶と知覚ビューの参照は復元後のキャラクターIDに対応させます。Fnatic v3 のスモークへのリコン使用記録・対戦履歴も、復元後のオブジェクトIDに対応させます。

`co1_train_defender_retake.py --cases-dir data/defender_retake_cases --case-epochs 10` で、復元したゲームのdefenderコントローラーを学習用のコントローラーへ接続し、保存状態から学習できます。学習側は乱数系列を管理するため `load_case(..., restore_rng=False)` を使います。重みテンソルの読込結果はキャッシュし、各試合では独立したコピーを使用します。実行例と周回数については [defender_retake_training.md](defender_retake_training.md) を参照してください。

## テスト

```powershell
python -m unittest discover -s test -p test_co1_retake_cases.py
```

テストは制御した設置状態で5種類すべての相手AIの保存・復元・次tickの継続、テンソルの共有保存、復元した場面の独立性、乱数状態、件数上限・再開を検証します。実際の500件収集や学習は実行しません。
