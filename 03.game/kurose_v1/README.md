# KUROSE v1

Qwen2.5-0.5B-Instructをベースに、QLoRAで会話応答を追加学習する構成です。ベースモデルの重みは更新せず、KUROSEのLoRAアダプターだけを保存します。

## 環境準備

学習にはCUDA対応GPUが必要です。Windowsでは、利用するPython環境にCUDA対応PyTorchを入れてから、このフォルダーでKUROSE用ライブラリを入れます。RTX 3060で動作確認した構成はPython 3.13、PyTorch 2.8.0 + CUDA 12.8です。以下はCUDA 12.8版を使う場合の例です。

```powershell
python -m pip install --force-reinstall --no-deps torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

CUDAが見えていることを確認してから、KUROSE用依存を導入します。

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install -r requirements.txt
```

CUDA確認の最後が`True`になっている必要があります。コマンドの`python`は、Torchをインストールしたものと同じインタープリターを指すようにしてください。

## 教師データ

初回は `training_kurose.py` が、`../tactical_llm_v1/data/teacher.jsonl` を自動でKUROSE用に変換します。状況と質問をuser発話、回答をassistant発話として使い、`group`単位で学習用と評価用に分割します。`reason`は内部の説明用データなので学習には使いません。元ファイルは変更しません。

変換結果は `data/train.jsonl` と `data/eval.jsonl` に保存されます。これらが既にある場合は前作データから再生成しません。前作データを使わず独自データから始める場合は、`train.example.jsonl` と `eval.example.jsonl` をそれぞれ `train.jsonl`・`eval.jsonl` にコピーして編集してください。

JSONLは1行につき1会話です。`user`と`assistant`の1往復に、任意で先頭の`system`メッセージを付けられます。各行は次の形式です。

```json
{"messages":[{"role":"user","content":"質問"},{"role":"assistant","content":"望ましい回答"}]}
```

例のデータは形式確認用です。実データはKUROSEに覚えさせたい品質の回答へ整えてください。評価用データは学習用に混ぜず、学習のたびに同じ質問で応答の変化を確認できるよう維持します。

## 初回学習と追加学習

作業ディレクトリを`kurose_v1/`にして、通常は設定定数を確認したうえでオプションなしで実行します。

```powershell
python training_kurose.py
```

初回は新しいアダプターを作り、`data/kurose_lora/`へ保存します。次回以降は同じ場所のアダプターを読み込み、追加学習します。学習設定は`training_kurose.py`冒頭の定数で調整できます。

追加学習では、`data/train.jsonl`をその回の新しい会話例と、以前の代表的な会話例を含む小さなデータセットに更新します。全履歴を毎回学習し直す必要はありません。ただし新しい例だけに偏ると以前の応答を忘れることがあるため、古い例も一部残してください。`data/eval.jsonl`は固定し、学習に使わないでください。

既存アダプターを読み込めば追加学習になります。最初からやり直したい場合は、`training_kurose.py`の`OUTPUT_DIR`を別の保存先に変更してください。既存のアダプターと同じ保存先へ新規学習すると、既存の学習結果を上書きすることがあります。

## 対話実行

学習済みアダプターを使って対話します。

```powershell
python chat_kurose.py
```

`/reset`で会話履歴を消去し、`/exit`で終了します。ベースモデルは初回にHugging Face Hubからダウンロードされます。

## 主な設定

- ベースモデル: `Qwen/Qwen2.5-0.5B-Instruct`
- 初回学習率: `2e-4`
- 追加学習率: `5e-5`
- 最大トークン長: `512`
- 保存対象: LoRAアダプター

モデル本体の再配布や利用条件、教師データの利用権は、それぞれのライセンスと利用規約を確認してください。