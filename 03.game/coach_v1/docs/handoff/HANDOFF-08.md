# HANDOFF-08: キャラクター固有モデルと trainer

## 2026-09-24 Task 08 完了記録

### 事前確認と範囲

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、`03.DESIGN.md`、Task 00～07 の引き継ぎ、既存のキャラクター観測・行動環境、checkpoint metadata を確認した。
- Task 08 は共有可能なネットワーク・trainer構造と代表キャラクター「ごりまる」の train/learning 対に限定した。残り4人の学習、coach、ゲーム接続coordinator、core変更は行っていない。
- 作業前からの `coach_v1/10.task_template.md` と `omoko_v1` 側の変更は触れていない。

### 実装

- `CharacterModel` を追加。安全な `[29,26,44]` grid と `[106]` vector だけを入力し、8方向 facing、通常abilityの使用有無、対象マスのlogitを出す。移動headはない。対象headは空間特徴とvector由来の文脈を組み合わせる。
- `CharacterTrainer` を追加。`CharacterEnvironment.prepare` が返す安全な観測と、学習側から与えられた合法な教師ラベルだけで更新する。maskを学習損失と選択へ適用する。対象lossはabilityを使うラベルのみに計算する。
- validation loss、facing正解率、ability使用判断正解率、不使用判断正解率、対象マス正解率、結果ラベルのある使用例の有効率を履歴へ保存する。最後の有効率は提供されたrolloutの実績で、現在policyの反実仮想評価ではない。
- seed固定、epochごとの決定的な並べ替え、optimizer状態を含む`latest.pt`、validation loss更新時の`best.pt`、`latest.pt`からの再開を実装した。map・警戒ポイント・観測/行動version・ロスター・character・network構成の不一致をload時に拒否する。
- `train_character_gorimaru.py`と`learning_character_gorimaru.py`を追加。learningは対応するtrainファイルも他モデルのtrainファイルもimportしない。actorは既存`CharacterAction`のみを返し、移動を決めない。
- `CharacterObservationEncoder`にhashの読み取り専用propertyを追加した。観測shape・version・意味は変えていない。

### 未視認敵情報の監査

- model/trainer/learningの入力APIは`CharacterObservation`だけを受ける。ゲーム実体、Scenario、EnemyPlacement、criticを受け取らない。教師ラベルと結果ラベルは学習側でのみ使い、network入力へ連結しない。
- 固定マップで未視認敵の実位置だけが異なる2局面を作り、`TeamPerceptionBuilder` → `BeliefMemory` → `CharacterObservationEncoder` → policyを通した。合法な目撃は両方空で、grid、vector、target mask、選択行動が一致した。
- 味方の合法な共有視認と経過tick付きclear履歴は既存のsnapshot/belief経路を利用する。

### 学習確認と制約

- 固定seed `17` の安全な合成観測4種に教師ラベルを付け、ごりまるモデルを20 epoch学習した。validation 12例でlossは `6.5405 → 0.5215`、facing/ability使用/対象マス正解率はそれぞれ `0%/50%/0% → 100%/100%/100%`。これは学習経路の動作確認であり、実戦性能ではない。
- 実ゲームrolloutからの教師ラベルとability結果の収集、1v1/2v1局面の進行、実戦validationはまだない。Task 09の固有シナリオ作成とTask 10のゲーム接続で必要になる。`ScenarioGenerator`の私有敵配置をactorへ直接渡してはならない。
- ability行動payloadのfacingが既存game側で適用されない問題はTask 07から継続。Task 10で非core接続の可否を確認する。

### core変更判断

- core変更は不要。既存の観測DTO、行動tuple、checkpoint契約を利用した。

### 変更ファイル

- `coach_v1/models/__init__.py`、`models/character_model.py`（新規）
- `coach_v1/training/character_trainer.py`（新規）
- `coach_v1/train_character_gorimaru.py`、`learning_character_gorimaru.py`（新規）
- `coach_v1/observation/character_encoder.py`（hash property）
- `coach_v1/training/__init__.py`、`training/README.md`（公開APIと使い方）
- `coach_v1/test_coach_v1_task08_character_trainer.py`（新規、自動テスト3件）
- `coach_v1/HANDOFF-08.md`（本記録）

### テスト結果

- `python -m unittest -q coach_v1.test_coach_v1_task08_character_trainer ... coach_v1.test_coach_v1_task00_design`: 78件成功（Task 08は3件）。
- `python -m compileall -q coach_v1`: 成功。
- `git diff --check -- coach_v1`: whitespace errorなし。GitのLF/CRLF警告のみ。

### 次の推奨タスク

**Task 09-A: ごりまる固有シナリオと実rollout評価**。合法な知覚境界で生成した観測と実結果から教師/評価サンプルを収集し、警戒ポイント中心と合法ランダム位置の両方で学習・検証する。その後、Task 09-B～Eで残り4人の独立checkpointを作る。
