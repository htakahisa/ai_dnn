# 設置後の guard モデル

実行ディレクトリは `concon_v1` です。学習・評価・推論は同じマップ設定、観測、行動を使います。
学習スクリプトは自動実行していません。

## 学習と評価

```powershell
python co1_train_guard.py -map L
python co1_train_guard.py -map R

python evaluate_co1_guard.py -map L --rounds 36 --output data/guard_L_eval.json
python evaluate_co1_guard.py -map R --rounds 36 --output data/guard_R_eval.json
```

既定の対戦相手は attacker A1/A2/A3 と同じ5チームです。

| 指定名 | 対戦チーム |
| --- | --- |
| `omoko_v1` | Omoko Gaming |
| `touyama_v2` | Touyama Gaming |
| `fnatic_v3` | Fnatic2023 |
| `gc_v1` | Ghost Champions |
| `toru_ai_v3.1` | Team Elites |

学習は既定で3000エピソード、50エピソードごとにチェックポイントを保存します。
探索率が下限0.05に達してから、探索なしで各チーム36試合を評価します。
`best` の更新には、5チームの平均勝率と最も低いチーム別勝率が両方とも以前以上であることを要求します。
既存の `best` がある場合は、同じ評価条件で再評価して比較します。

```powershell
python co1_train_guard.py -map L --episodes 5000 --eval-rounds 60
python co1_train_guard.py -map R --save-dir data/guard_R_trial
python evaluate_co1_guard.py -map R --model data/guard_R_trial/co1_guard_R_best.pt
```

`--opponents` にチーム指定を並べると対象を絞れます。指定しなければ5チームすべてを使います。
`--seed`、`--checkpoint-interval`、`--device cpu|cuda` も指定できます。
`--resume` は重みを引き継ぐ追加学習です。optimizer、リプレイ、探索率のスケジュールは新しく始まります。

## 学習の開始状態

学習環境は通常ゲームの設置後状態を合成し、通常のtick・射撃・アビリティ・IQ知覚・敵AIで進めます。
carry モデルによる設置前の戦闘は実行しません。生存人数は各チーム1〜5人、HPと使用済みアビリティも変えます。
敵AIは開始状態の設定後、各チームの通常のリテイク処理を使います。

| 開始状態 | 内容 |
| --- | --- |
| `hold` | attacker は割り当てられた防御点に配置され、対応する大文字の方向を向く。defender はサイトからBFS距離8〜20マスで開始。 |
| `transition` | attacker はスパイクからBFS距離2〜6マスに配置され、防御点への移動も学ぶ。defender は距離8〜20マスで開始。 |
| `smoke` | attacker は距離2〜6マスで開始。スパイク周囲のスモークと、解除を1tick進めたdefenderを設定し、解除阻止を学ぶ。 |

既定では前半20%が `hold`、次の20%が `hold` / `transition`、残り60%が3状態の混合です。
`smoke` の開始スモークは開始前に使用されたものを表す合成状態です。敵AIに毎回スモーク使用や解除継続を強制しません。
評価では3状態を順番に使い、チーム別・開始状態別に勝率、解除率、終了理由を出力します。
`--rounds 36` は各チーム合計36試合（各開始状態12試合）です。

```powershell
# スモーク内解除の場面を重点的に追加学習する例
python co1_train_guard.py -map L --resume data/guard_L_data/co1_guard_L_best.pt --start-modes smoke --save-dir data/guard_L_smoke_trial
python evaluate_co1_guard.py -map L --model data/guard_L_smoke_trial/co1_guard_L_best.pt --start-modes smoke
```

学習モードを絞っても、`best` の選出は3状態すべての評価で行います。
評価結果は設置後状態に対する成績です。設置前からのラウンド全体の勝率とは別です。

## 配置と行動

- `a〜e` はキャラの配置先、`A〜E` は対応するキャラの facing 目標座標です。大文字は移動先ではありません。
- マップは各小文字・大文字を1点ずつ含めます。数字部分は通常ゲームの地形と一致する必要があります。
- 設置時の生存者に重複しない配置先をランダムに割り当てます。5人未満なら、どの点が空いても構いません。
- 一度決めた割り当てはラウンド中変更しません。死亡した味方の配置先へ再割り当てしません。
- モデルは上下左右への移動、停止、8方向の facing、SMOKE / FLASH / RECON の使用を選びます。
- アビリティの対象候補は設置位置、割り当てられた facing 目標点、知覚できた敵の位置です。アビリティtickではゲーム仕様に従い現在の向きを維持します。
- 推論側は解除時の接近や2tick停止を固定制御しません。モデルは解除通知、射撃可能性、スパイクへの射線、停止tick数、アビリティ残数などから判断します。
- 敵位置は本番と同じIQ知覚と12tickの味方共有記憶を使います。未発見の敵の実座標はモデルへ渡しません。
- 移動と射撃、スモーク隣接時の視認、Reconによるスモーク越し射撃は通常ゲームの処理を使います。射撃は自動です。

報酬は起爆・敵全滅による勝利を +10、解除などの敗北を -10 とし、死亡、敵へのダメージ、配置への移動、facing、射撃可能な位置での連続停止を補助します。
解除通知があり射撃できない場合は、モデルがスパイクへ接近するよう距離の報酬を切り替えます。
地形の通行不可・占有・アビリティ残数は行動マスクで扱います。解除通知によって移動やReconを強制するマスクは使いません。
現在の行動にはウルト使用を含めていません。

## 保存先とパターン追加

| マップ | 配置定義 | 保存先 |
| --- | --- | --- |
| `L` | `co1_map_guard_L.py` | `data/guard_L_data/` |
| `R` | `co1_map_guard_R.py` | `data/guard_R_data/` |

各保存先には `co1_guard_L_best.pt` / `co1_guard_L_latest.pt` のようなファイル、エピソード別チェックポイント、`training_log.jsonl` を作成します。
モデルにはマップ名、配置・facing・地形の署名、観測・行動のサイズを保存します。別マップや変更前のマップのモデルは読み込みエラーになります。
マップ変更後に旧 `best` が残る場合は、別の `--save-dir` で再学習してください。

パターンを増やす場合は、たとえば `co1_map_guard_L2.py` を作り、`co1_guard_scenarios.py` の `SCENARIOS` に追加します。

```python
"L2": GuardSettings("co1_map_guard_L2", "left"),
```

共通ファイルを複製せず、次のコマンドで学習・評価できます。

```powershell
python co1_train_guard.py -map L2
python evaluate_co1_guard.py -map L2
```

## 通常対戦への接続

通常対戦・大会では、`co1_attacker_scenarios.py` の `CONCON_ATTACKER_POSTPLANT_MODELS` に左右の学習済みモデルを登録しています。
攻撃側はスパイク設置後、実際の設置サイトに応じて左なら L、右なら R の `best` を読み込んで使用します。
このモデルは攻撃側の guard 用です。防御側のリテイクは `ConconDefenderController` の標準動作です。
設定は以下の形です。
循環importを避けるため、推論クラスのimportはfactoryの中で行います。

```python
def left_guard():
    from concon_v1.co1_learn_guard import ConconGuardController
    return ConconGuardController(map_name="L")


def right_guard():
    from concon_v1.co1_learn_guard import ConconGuardController
    return ConconGuardController(map_name="R")


CONCON_ATTACKER_POSTPLANT_MODELS = {
    "left": (left_guard,),
    "right": (right_guard,),
}
```

同じサイトに複数factoryを設定すると、既存のラウンドコントローラが設置時に1つ選びます。
推論は既定で `best` を読み込みます。`latest` を試す場合はfactoryから `model_path` を指定してください。

## 確認

```powershell
python -m unittest discover -s test -p test_co1_guard.py -q
```

配置の維持、敵情報の非開示、スモーク／Recon射線、停止tick、モデル互換性、5チーム×左右サイトの通常ゲーム処理、評価時のRNG保持を確認します。
