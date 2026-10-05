# defender search の基本モデル

concon_v1 ディレクトリから実行します。

```powershell
python co1_train_defender_search_position.py
```

学習スクリプトは guard と同じ BFS 教師による位置取りの学習方式で、
5 人共通の移動・8 方向 facing の行動価値を学習します。
小文字 a〜e は配置地点、大文字 A〜E は対応する視線先です。
5 人の担当はラウンド開始時に重複せず割り当て、setup から開始後まで維持します。
推論はモデルの行動価値と合法手マスクで決定し、BFS による経路誘導は行いません。

現在の setup 制限では初期位置から a〜e に到達できません。
学習側は setup 中に到達できるセルの中から各配置地点に近い待機地点を選び、
setup 用と開始後用の行動価値を別々に学習します。
開始後は a〜e へ移動し、各 A〜E へ facing して待機します。
setup の制限や配置指定を変更した場合は基本モデルを再学習してください。

スクリプトは本番エンジンで setup の移動と開始後の IQ 知覚付き移動を検証します。
検証対象は敵なしの配置到達・連続待機・facing・到着後の離脱です。
基本モデルの学習は戦闘の学習ではないため、敵との勝率は評価しません。
マップ、味方位置、知覚で開示された敵位置、facing は観測に含めています。
今回の基本モデルが行動選択に使うのは配置・phase・自分の位置・味方の占有情報です。
敵情報への反応、交戦、退避、加勢、アビリティは次の search 学習の対象です。

全員の配置到達・待機・facing の検証に合格すると、スクリプトは以下を保存します。

```text
data/defender_search_data/co1_defender_search_positioning.pt
```

ConconDefenderController は search の best があれば追加学習モデルを使用し、
best がなければ setup と未設置時に保存済み基本モデルを使用します。
スパイク設置後は既存の DefaultDefenderController を使用します。
基本モデルがない場合は既存の defender を使用します。

## 通常ゲームでの確認

concon_v1 ディレクトリから起動できます。

```powershell
python ../run_game.py
```

編成画面で defender 側のチームを `Gorigons`、AI を `ConCon v1` にして開始します。
Gorigons を選択すると AI の初期値も ConCon v1 になります。
起動ログの `[ConCon defender search] model=...` に読み込んだモデルの絶対パスが表示されます。
defender は setup と未設置時に今回の基本モデルで移動・待機・facing を行います。
スパイク設置後は既存の defender 処理に切り替わります。

更新回数の指定と再検証の例:

```powershell
python co1_train_defender_search_position.py --positioning-steps 500
python co1_train_defender_search_position.py --resume data/defender_search_data/co1_defender_search_positioning.pt --positioning-steps 0
python -m unittest discover -s test -p test_co1_defender_positioning.py
```

検証に不合格の場合、スクリプトはモデルを保存せず終了します。

追加の search 戦闘学習は [defender_search_training.md](defender_search_training.md) を参照してください。
`co1_train_defender_search.py` の引数なし実行は戦闘の追加学習です。
基本モデルを作り直す場合は `co1_train_defender_search_position.py` を実行します。
このスクリプトは内部で `co1_train_defender_search.py` を positioning モードで呼び出し、その他のオプションをそのまま渡します。
