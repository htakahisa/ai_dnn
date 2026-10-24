# 条件別拡張と、共通・専用重みの分離

最新の通常設定は[時間切れ対策revision3](tv3_frc_right_timeout_revision3.md)。revision2のソースとモデルは保持し、revision3は別ファイル・別profileへ保存する。ExtensionContextに公開threatsとcarrier_goalを追加した。新しい専用重みの初期値は今回学習したrevision2 set5、適用外は引き続き固定共通v16を使う。

2026-10-12追記：ユーザー依頼でFRC右のテンプレートをrevision 2の能力教師へ変更し、通常設定で登録、5セット上限にした。[右専用の変更・実行手順](tv3_frc_right_utility_revision2.md)を参照。以下の「無変更テンプレート」「登録は空」は基盤追加時の記録であり、現在の設定ではない。学習は未実行。

2026-10-12。仕組みと無変更テンプレートを追加した。拡張の登録は既定で空、既存モデルの更新・専用モデルの学習は行っていない。今回のFRC右への具体的な修正を、テンプレートへ移したわけではない。

## 1. 処理だけでなく重みも分ける

右でだけ処理を呼んでも、同じニューラルネットを右のデータで再学習すれば左の出力も変わり得る。左右専用拡張では、次の構成を使う。

| 条件 | 使用モデル | 学習時 |
| --- | --- | --- |
| 専用条件：例として相手frc_v1・目標サイトR・attacker plant | 共通から初期化した別の専用重みと、個別拡張 | この区間のデータだけで専用重みを更新 |
| 適用外：同じ相手の目標サイトL | 固定した共通重みと、その保存版のencoder | 共通重みは更新せず、局所epsilon・教師選択も使わない |
| 別の相手・段階 | 登録条件に合う拡張がなければ元の共通処理・モデル | この拡張は適用しない |

共通モデルは専用optimizerのparametersから除外し、requires_grad=Falseにする。単なる別ファイル保存や同じtensorへの別名ではない。専用側のパラメータを変えても、共通側が変わらないことをテストした。

専用checkpointへ共通checkpointの実バイト列を固定して保存し、schemaにsite・共通SHA・区間境界の扱いを記録する。後で通常の共通bestが更新されても、保存済み専用モデルの適用外側が黙って入れ替わらない。共通の相手・analysis依存・旧schema・SHAも読み込み時に検証する。

**重みの非更新は保証するが、実際の左勝率が絶対に同じという保証ではない。** 右の行動が変われば、その後のゲーム状態・敵味方の履歴も変わる。左右・別相手の回帰評価は引き続き必要。

## 2. ファイルと拡張窓口

共通dispatcherは[tv3_model_extensions.py](../tv3_model_extensions.py)。最初のテンプレートは[tv3_custom_train_attacker_plant_frc_v1_R.py](../tv3_custom_train_attacker_plant_frc_v1_R.py)。役割・相手・サイト・識別子・revisionを宣言する。

```python
EXTENSION_SPEC = dict(
    id='attacker_plant_frc_v1_R', revision=1,
    role='attacker_plant', opponent='frc_v1', site='R',
)
```

siteは**選択経路の目標サイト**を意味する。現在位置が右半面か、設置床内かという判定とは異なる。接近中も、目標がRなら右用拡張の対象になる。別サイトへ変更すると使用モデルも切り替わる。

| 関数 | 挿入場所と用途 |
| --- | --- |
| adjust_goal(context, goal) | 共通の目標割当後、encoder前。専用の接近・支援・投擲位置目標を返す |
| adjust_inputs(context, inputs) | 共通encoder後、共通の予約・辺通過・通路制限前。特徴量、教師、候補制限、同じ枠数での能力対象候補を調整 |
| adjust_reward(context, reward) | 既存の途中報酬計算後。条件に合う役のステップ報酬を調整 |

今回のテンプレートはすべて入力値をそのまま返す。学習loopを持たない共有の拡張ファイルであり、通常推論でも同じ目標・入力hookを使う。trainer本体を推論へimportしたり、別モデルの学習loopを実行したりしない。

contextはrole・opponent・site・公開snapshot・公開味方・目標・選択行動だけ。game本体や敵の非公開座標・blind等は渡さない。hookは状態をグローバル変数へ蓄積しない純粋な処理にする。報酬ラベルを後の観測へ混ぜない。

共通処理は元の共通ファイルに残す。専用の差分だけを拡張ファイルへ書く。専用goalでも合法な床が必要で、予約等の共通制限は後から適用される。特徴量の次元は変えない。移動・terminalの禁止は解除できない。ABILITY/ULTIMATEの対象候補を同じ種類・枠数で置換する場合、合法にした候補は公開maskと実エンジンのvalidate_actionで検証する。候補変更と特徴量の意味の整合は拡張側で保つ。

推論の最終選択を直接強制するhookは用意していない。観測次元・行動種類の追加や、任意の損失関数・終端報酬の差し替えは今回の窓口の範囲外で、別途schemaと呼出箇所の設計が必要。

## 3. 共通と専用の登録方法

[tv3_train_attacker_plant.py](../tv3_train_attacker_plant.py)冒頭で設定する。

```python
PLANT_CUSTOM_MODULES = ()  # 現在の既定値。共通処理のみ。
PLANT_COMMON_DIRECTORY = HERE / 'data' / 'best'
```

将来FRC右の拡張を使うときは、ファイルを編集してrevisionを管理し、明示的に登録する。

```python
PLANT_CUSTOM_MODULES = ('tv3_custom_train_attacker_plant_frc_v1_R',)
TARGET_OPPONENTS = ('frc_v1',)
```

相手が合わない登録は当該相手のprofileへ含めない。複数hookは登録順に適用する。一つの相手の学習profileには、重みの適用サイトL/R/*を一つに限定する。L専用とR専用を同時登録して一つの重みへ混ぜない。それぞれ別の登録・学習profileで作成する。

roleやopponentの`*`は共通対象のhook、siteの`*`は両サイト対象のモデルprofileを表す。site=*では左右別の重み分岐を作らない。サイト限定にしたい処理を*で登録しない。

左右専用の学習には、適合する共通plantが先に必要。初期値として許可したschema差分に限り、共通の重みを専用ネットへコピーし、専用optimizer/replayは新規にする。共通がない・依存が違う・入力の初期化互換性がない場合、黙って別モデルへ切り替えず停止する。

右専用の経験は右目標の区間だけをreplayへ返す。共通側へ切り替わった境界では専用の行動列を閉じ、共通側の観測を専用Qでbootstrapしない。閉じた専用行動へ、後の共通側の設置成功をそのまま後付けしない。区間終了を設置成功と呼ぶものではない。通常評価は左右とも実際に動かし、全ラウンドの人数・設置・時間切れを確認する。

## 4. 保存先と、学習・評価・推論の一致

共通モデルの保存先は従来どおり。専用モデルは拡張contractと固定共通hashから求めたprofile IDで分ける。

```text
data/attacker_plant/custom_profiles/<profile-id>/frc_v1/latest.pt
data/attacker_plant/custom_profiles/<profile-id>/frc_v1/evaluation_checkpoints/...
data/best/custom_profiles/<profile-id>/frc_v1/attacker_plant_best.pt
logs/attacker_plant/custom_profiles/<実行対象構成のid>/...
```

途中重み保存も専用データディレクトリ内。共通best/latestやログへ専用の結果を上書きしない。専用bestは専用profile内の比較であり、共通bestを上回ったと自動的に判定するものではない。採用前に共通対照との回帰評価を行う。

checkpointのschemaには、拡張API版・適用role/opponent/site・順序・id・revision・拡張ソースSHA・重みroutingを保存する。共通モデルの実バイト列も保存する。ファイルを変更してrevisionを上げ忘れてもSHA不一致で読み込みを拒否する。読み込んだbyte列そのものを実行し、古いimport/pyc cacheと新しいhashの組合せを避ける。

古い専用モデルも継続利用する場合は、その拡張ソース版も保持する。例えば変更後を`tv3_custom_train_attacker_plant_frc_v1_R_v2.py`として別ファイルにし、学習時の登録を切り替える。旧モデルのSHAを現在のファイルへ書き換えて読み込ませない。

推論では現在のPLANT_CUSTOM_MODULESを使わず、checkpointに固定したprofileを復元する。設定を空へ戻しただけで専用モデルが共通モデルに変わることはない。共通運用へ戻す場合は共通checkpointを使う。

plant単体の読み込み例は次の形になる。

```python
TouyamaV3AttackerPlantController.from_best(
    'frc_v1',
    best_dir=HERE / 'data' / 'best' / 'custom_profiles' / profile_id,
    analysis_dir=HERE / 'data' / 'best',
)
```

通常対戦の採用先はこの作業では変更していない。専用plantを試合全体へ採用するには、後続guardのケース・モデル・依存hashも合わせる必要がある。既存guardの依存検証を無視して専用plantだけ差し替えない。

## 5. 対象範囲と検証

dispatcherのrole/opponent/site照合は他段階にも使えるが、実際の学習・重みroutingを接続したのは今回attacker plant。search / retake / guardへの接続は、各モデルのschema・ケース・状態遷移に合わせた追加実装が必要。接続済みと扱わない。

確認した内容：

- 無登録では旧schema・共通保存先・共通モデルの実行を保持する。
- 相手・段階・左右の照合、適用順、重複、非法goal・mask・候補・報酬を検証。
- 拡張revision・scope・ファイルSHA変更、固定共通SHAの不一致を拒否。
- 共通パラメータが専用optimizerの対象外であること、専用パラメータ変更で共通が変わらないこと、左右のモデル・encoder・教師／探索の切り替えを検証。
- 設定表示の共通モードと専用モードを、学習せず確認。
- 実モデルの同一seed・12ラウンドで、同じ版・同じ重みの無変更右専用分岐と共通を比較。867tickの観測・mask・選択行動、およびラウンド結果が一致。共通の3910遷移に対し、専用へ返された1951遷移はすべて右目標の区間。共通checkpointのSHAは不変。

実対戦の比較ログは`logs/custom_extensions/identity_routing_parity.json`。これは機構の無変更確認であり、専用学習後の右性能改善を証明するものではない。今回、学習・optimizer更新・既存モデルへの書き込みは行っていない。

通常の作業ディレクトリは`03.game/touyama_v3/`。将来の学習実行は`python tv3_train_attacker_plant.py`で、登録・対象・回数・共通入力元は冒頭定数を確認する。現時点ではPLANT_CUSTOM_MODULES=()を維持し、追加学習の実行を依頼していない。
