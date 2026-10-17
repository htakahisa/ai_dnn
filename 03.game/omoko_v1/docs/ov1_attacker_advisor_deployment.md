# FLASH・SMOKEの組合せを通常対戦へ採用（2026-10-17）

私は、ユーザーの採用指示に従い、omoko_v1 の通常対戦に、別seed比較で確認した組合せを実装した。

| アビリティ | 通常対戦の判定モデル |
| --- | --- |
| FLASH | 再学習候補から採用したモデル |
| SMOKE | 再学習候補から採用したモデル |
| RECON | 従来のモデル |

通常対戦の `Ov1AttackerController` は、明示的なアドバイザー指定がない場合、`DeployedAbilityAdvisor` を使用する。このアドバイザーは相手AIを問わず、上表の能力別の切替を行う。今回の別seedでの勝率比較の確認範囲はfrc_v1である。

## 採用済みモデル

- FLASH・SMOKE：`ability_advisor_v1/data/ov1_attacker_ability_advisor_flash_smoke.pt`
- RECON：`ability_advisor_v1/data/ov1_attacker_ability_advisor.pt`
- 採用記録：`ability_advisor_v1/data/ov1_attacker_ability_advisor_deployment.json`

私は、FLASH・SMOKE用の採用済みファイルが、別seed評価で使った候補と同じSHA-256を持つことを確認した。私は、RECON用の従来モデルも同評価の参照モデルと同じSHA-256を持つことを確認した。

学習処理は、引き続き次の候補を `_candidate.pt` に保存する。通常対戦は採用済みの `_flash_smoke.pt` を使用するため、候補ファイルの再学習だけでは採用済みのFLASH・SMOKEの重みは変わらない。

## 起動・履歴・互換性

起動処理は、attacker開始とdefender開始の両方で、2モデルの読込とcarry/escort・左右サイトの全4区分の互換性を検査する。起動処理は、一方のモデルの欠落・破損・互換性不一致でもエラーで停止する。

`DeployedAbilityAdvisor` は、2モデルで同じ試合履歴、観測履歴、実使用履歴、助言回数、使用待ち状態を共有する。アドバイザーは、FLASH・SMOKEに対して採用済みモデルの重み・検証済み区分・閾値を使用し、RECONに対して従来モデルの重み・検証済み区分・閾値を使用する。公開敵への関与と未検証区分の扱いは、各モデルの従来の判定処理に従う。

今回の実装は、射線仕様、配置、観測次元、carry・escort・retrieve・guardの重みを変更していない。ユーザーは、今回の組込みのために再学習する必要はない。ユーザーがゲームを再起動すると、通常対戦はこの組合せを使用する。

ユーザーは、`ov1_learn_attacker_ability_advisor.py` 冒頭の `DEFAULT_MODEL_PATH`、`DEFAULT_FLASH_SMOKE_MODEL_PATH`、`DEPLOYED_CANDIDATE_ABILITIES` で採用設定を確認できる。

## 検証結果

私は、以下の検証を実行した。

- 関連テスト36件が成功した。テストは、能力別の切替、両方向のモデル選択、両モデルの必須読込、全4区分の検査、共有履歴、推論例外後の状態復元、通常コントローラーの既定接続、評価処理の条件分離を確認した。
- 通常コントローラーを実際に初期化し、既定のアドバイザーが `DeployedAbilityAdvisor` になることを確認した。
- 実際の採用済みモデルと、評価に使用した実験用の `flash_smoke` 条件を、保存済み特徴量150件（FLASH・SMOKE・RECON各50件）で比較した。使用・温存、判定理由、スコアの不一致は0件だった。この比較は入力を固定した推論の一致確認であり、追加の勝率実験ではない。
- 両モデルの全4区分の互換性と、評価済みモデルとのハッシュ一致を確認した。

推論の一致確認は `ability_advisor_v1/logs/ov1_attacker_advisor_deployment_check.json` に保存している。採用の根拠となった別seed評価では、frc_v1 相手の240ラウンドで勝利数が133回から146回へ増えた。ユーザーは、[別seed比較](ov1_attacker_flash_smoke_holdout.md) で試合ごとのばらつきと評価範囲を確認できる。

## 今後の比較評価

通常の評価スクリプトは、次の4条件を同じseedで比較する。

| 結果のキー | 評価条件 |
| --- | --- |
| `baseline` | 助言なし |
| `advised` | 従来モデルを全種類に使用 |
| `candidate` | 次の再学習候補を全種類に使用 |
| `deployed` | 通常対戦と同じ、採用済みFLASH・SMOKEと従来RECONの組合せ |

評価処理は、`deployed_vs_advised` に採用済み組合せと従来モデルの差、`candidate_vs_deployed` に次の全種類候補と採用済み組合せの差を保存する。ユーザーは、評価スクリプト冒頭の `DEFAULT_MODEL`、`DEFAULT_CANDIDATE_MODEL`、`DEFAULT_DEPLOYED_FLASH_SMOKE_MODEL` で比較対象を確認できる。

作業ディレクトリ：`C:\Users\ronet\MyProject\git\AI_dnn\03.game\omoko_v1`（ユーザーが移動済みの状態）。ユーザーが比較を実行する場合の通常コマンドは以下である。

```powershell
python ability_advisor_v1/ov1_evaluate_attacker_ability_advisor.py
```

私は、今回の組込み時に学習と対戦評価全体を再実行していない。
