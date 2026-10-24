# FRC右専用：射撃につながる能力の教師例（revision 2）

追記：ユーザーによる5セット学習が完了し、[set0とset5の72ラウンド比較](tv3_frc_right_utility_revision2_set5_review.md)を行った。設置58→60、同人数以上設置51→54、右同人数以上18→21、右timeout11→12。ユーザー方針により、人数優位47→43の減少だけで評価を覆さず、部分改善と扱い、次は時間切れを調べる。以下の学習未実行という記載は実装時点の記録。

2026-10-12。ユーザー依頼で右専用修正を実装した。学習・optimizer更新は行っていない。性能改善は未確認。

## 変更点と根拠

v18/v19の調査では、合法な能力の見送り、支援役の喪失、人数不利設置が残った。単一場面のFLASH使用で人数条件が改善した対照はあるが、能力を無条件に使うことの有効性は確認されていない。今回の変更は新しい短期実験であり、過去の失敗全体を解決した修正ではない。

`tv3_custom_train_attacker_plant_frc_v1_R.py`のrevisionを2にし、既存の合法候補から教師の投擲先を選ぶ処理を追加した。

- FLASH：実エンジンと同じ投射経路・壁・現在の煙から公開敵への作用を見積もる。対象に射線と向きを持つ別の味方が1人以上いる候補を優先し、その対象数、全作用対象数、到着の早さで比較する。
- RECON：公開敵が既にrevealされておらず、味方の壁・身体遮蔽を除いた射線が煙で遮られている場合、投射終点からRECON_RADIUS内の敵を検知できる候補を教師にする。投擲先座標を着弾点として扱わない。
- 自分がblind・低HP・スパイク保持者、退避教師、設置等の任務教師、支援移動中、同種能力の飛行／有効効果中、期限不足なら追加教師へ変更しない。条件が合わなければ元の教師を維持する。

観測・候補・mask・報酬は変更しない。能力使用だけの追加報酬は付けず、既存の実作用監査による報酬と無効投擲の費用を使う。通常推論はQから行動を選び、教師による強制は行わない。

公開位置・現在の煙・味方射線からの予測であり、投射中の移動や味方の次の行動、非公開の敵状態まで保証するものではない。

## 学習設定と実行

`tv3_train_attacker_plant.py`冒頭を次の設定にした。

```python
TARGET_OPPONENTS = ('frc_v1',)
PLANT_CUSTOM_MODULES = ('tv3_custom_train_attacker_plant_frc_v1_R',)
TRAINING_SETS = 5
TOTAL_TRAINING_SET_LIMIT = 5
RESUME_TRAINING = False
```

共通入力は`PLANT_COMMON_DIRECTORY = HERE/'data'/'best'`。現在の共通best v16 set10から専用の重みだけを初期化し、optimizer/replayは新規。左は固定した共通v16モデルとそのencoderを使う。analysisは固定v4 set50を使用する。専用側はv19処理を使用する。

作業ディレクトリ：`03.game/touyama_v3/`。**学習はユーザーが実行する。**

```powershell
Set-Location C:\Users\ronet\MyProject\git\AI_dnn\03.game\touyama_v3
python tv3_train_attacker_plant.py
```

初期評価とset5の教師・探索なし評価を比較する。全体・左右別の人数優位設置、同人数以上設置、人数不利設置、timeoutを確認し、設置数の増加だけで採用しない。基準best v16の72ラウンド結果も併せて見る。

出力は`data/attacker_plant/custom_profiles/<profile-id>/frc_v1/`、専用bestは`data/best/custom_profiles/<profile-id>/frc_v1/`、ログは`logs/attacker_plant/custom_profiles/`。専用best選定はこのprofile内の比較であり、通常対戦への採用とは別。

## 互換性と再学習範囲

LOS・scenario・配置・入力次元・analysisは変更しない。既存共通bestはその保存版のまま読み込む。専用ソースrevision/SHAと固定共通SHAを専用checkpointに記録する。旧revision 1の専用checkpointは現ソースと不一致になるため、SHAを書き換えて再開しない。

必要な学習は今回のFRC右専用plantのみ。通常対戦への採用後は、専用plantに合うguardケースの再収集・guard再学習が必要。今回の変更によるanalysis・defenderの再学習は不要。右の行動で後の状態が変わるため、共通重みの凍結だけで左成績の不変は保証しない。

## 検証

関連57テスト通過。対象外のscope、mask不変、退避・低HP・blind・保持者・支援移動・設置優先、煙・向き・既検知・期限・飛行中の境界、実エンジンと同じ投射／壁判定を確認した。`python tv3_train_attacker_plant.py --describe`で固定共通bestのSHAと専用登録、analysis依存の整合を確認した。このオプションは設定確認用で学習を行わない。

重み更新なし、教師・探索0、seed804000142の12ラウンドも完走。右hookの2170呼出し中、FLASH教師を2件変更、RECON変更は0件。このseedでRECON修正が発火した証拠はなく、単体テストで条件を検証した。8設置は未学習の旧重みによる動作確認値で、修正による改善率ではない。固定共通bestのSHA不変を確認。記録は`logs/custom_extensions/frc_right_utility_revision2_smoke.json`。
