# attacker のマップ別学習・評価

`concon_v1` ディレクトリで実行します。`-map` で攻撃パターンを選択します。
省略時は A1 です。`--map` も使用できます。

```powershell
python co1_train_attacker.py -map A1
python co1_train_attacker.py -map A2
python evaluate_co1_attacker.py -map A1
python evaluate_co1_attacker.py -map A2
```

既存の `--episodes`、`--seed`、`--mode`、`--opponents`、`--eval-rounds` などは
引き続き指定できます。通常は実戦と同じ IQ 知覚を使う `--mode battle` です。

| 指定 | 経由地点 | 設置先 | モデル保存先 |
| --- | --- | --- | --- |
| A1 | `co1_map_attacker_A1.py` | 左側 | `data/attacker_A1_data/` |
| A2 | `co1_map_attacker_A2.py` | 右側 | `data/attacker_A2_data/` |

ファイル名は `co1_attacker_A1_best.pt` / `co1_attacker_A1_latest.pt` など、
指定したマップ名になります。評価はそのマップの `best` を読み込みます。
`best` は従来どおり探索率が下限に達してから評価・選出されます。
A2 の評価には、先に A2 の学習でモデルを作成する必要があります。

保存先を変える場合は学習に `--save-dir` を指定し、評価にはそのモデルを
`--model` で指定してください。

```powershell
python co1_train_attacker.py -map A2 --save-dir data/experiment_A2
python evaluate_co1_attacker.py -map A2 --model data/experiment_A2/co1_attacker_A2_best.pt
```

モデルにマップ名・設定の識別情報を記録し、指定マップと異なるモデルはエラーにします。
マップ名が記録されていない既存モデルは A1 として扱います。
既存の A1 モデルは、共通の学習・評価・推論ファイルで引き続き利用できます。

## 共通処理と推論

- `co1_attacker_common.py`: 経由地点の進行、観測、行動マスク、DQN 定義。
- `co1_attacker_scenarios.py`: マップ設定の読み込み・検証と保存先の決定。
- `co1_train_attacker.py`: 共通の学習・チェックポイント評価。
- `evaluate_co1_attacker.py`: 共通の実戦評価。
- `co1_learn_attacker.py`: 共通の推論。

学習環境と評価環境は同じマップ設定を推論コントローラーに渡します。
アプリケーション側で直接生成する場合も指定できます。

```python
from concon_v1.co1_attacker_controller import ConconAttackerController

controller = ConconAttackerController(map_name="A2")
```

通常のゲームでの `concon_v1` の選択は従来どおり A1 を使います。

観測サイズは経由地点の段階数に合わせて `24 + len(waypoint_order)` で決まります。
`abc` は 27 要素、`abcd` は 28 要素、`abcde` は 29 要素です。
A1 / A2 の `abcd` は既存モデルと同じ観測形式を維持します。
経由順を変更したマップは、新しい設定で学習し直してください。
候補番号の one-hot は 5 枠で、設置先候補の番号が 4 を超える場合は最後の枠にまとめます。
目標の座標は別の要素で保持するため、A2 の各設置位置も区別できます。

## A3 以降を追加する場合

1. `co1_map_attacker_A3.py` を作成し、同じ地形上に経由地点を配置します。
2. `co1_attacker_scenarios.py` の `SCENARIOS` に、例えば以下を追加します。

   ```python
   "A3": ScenarioSettings(
       map_module="co1_map_attacker_A3",
       plant_side="left",
       waypoint_order="abcde",
       max_candidate_bfs_distance=12,
   ),
   ```

3. `python co1_train_attacker.py -map A3` で学習します。

経由順は `waypoint_order` に指定します。`abc` なら `a → b → c → 設置`、
`abcde` なら `a → b → c → d → e → 設置` です。
指定する文字は重複のない英小文字で、最初は分隊分割用の `a` にします。
`a` は 1〜2 個、それ以外の各文字は 1〜5 個をマップ内に配置してください。
`a` が 1 個なら全員がその地点へ向かいます。経由地点が 1 個の段階から進む場合、
距離上限内で到達できる次の候補へ、人数をできるだけ均等に分けます。
5 人で候補が 2 個なら 3 人・2 人、3 個なら 2 人・2 人・1 人です。
分割済みの複数地点から進む場合は、それぞれの地点から最も近い次の候補を選びます。
指定していない経由地点の文字や壁上の経由地点はエラーになります。
`max_candidate_bfs_distance` は経由地点候補を選択する BFS 距離上限です。
A1 は従来の 12、A2 は既存の `b → c` の最短距離が最大 16 のため 16 を使用します。
次の候補へこの上限内で到達できる配置にしてください。
設置先の選択にはこの距離上限を適用しません。

学習スクリプトの実行は手動で行ってください。
