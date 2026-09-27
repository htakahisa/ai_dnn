# HANDOFF-11: coach 学習環境とモデル

## 2026-09-28 Task 11 実装記録

### 事前確認と範囲

- `01.MODEL_DETAIL.md`、`02.ALL_TASKS.md`、`HANDOFF-10.md`、coach/character 観測、belief、シナリオ生成、character checkpoint、coordinator、実ゲームの tick・setup・round 処理を確認した。作業開始時の `git status` は空だった。
- core ファイルは変更していない。推論時の戦術規則・経路探索・警戒ポイントへの強制誘導は追加していない。

### 実装

- `models/coach_model.py`: 固定27チャネル・84特徴の観測を CNN→GRU に入力し、1 forward で固定5 slot の movement（5択）、intent（9択）、objective（3択）を出す actor。別クラスの centralized critic だけが敵実位置 tensor を受ける。action mask は壁・範囲外・死亡 slot・設置/解除不能・defender setup の進入禁止を除外する。
- `training/coach_environment.py`: headless 実ゲーム、凍結した正式 character checkpoint、既存の安全な sensor/belief/coordinator、70/20/10 の既存シナリオ生成器を接続した。`2v1`、`2v2`、`3v3`、`5v5` の限定局面を選べる。限定局面は経過45 tick としてゲーム tick/残時間も合わせる。報酬はラウンド勝敗を主に、初回クリアマス、無効移動、味方死亡、plant/defuse に小さな補助値を加える。移動距離や警戒ポイント到達そのものは報酬にしない。
- rollout の actor 観測と実際の coordinator 入力を tick ごとに照合する。defender setup 後の belief/ログをクリアし、学習 tick では対象チームを先に処理する。ゲームの `move_character` が初回 actor 呼び出し前に適用する強制 facing を観測にも反映する。headless の自動 round 移行も episode 終了として扱う。
- `training/coach_trainer.py` / `train_coach.py`: on-policy rollout、PPO 更新、rollout 保存、side 別 actor/training checkpoint、optimizer と乱数状態を含む再開を実装した。`learning_coach.py` は同名 train ファイルに依存せず actor checkpoint のみを読み、side・固定マップ・観測/行動 version・roster・モデル設定を検証する。coordinator は round リセット時に recurrent actor の状態も消去する。
- actor checkpoint `latest.pt` には critic、optimizer、敵実位置を保存しない。critic と optimizer は `training_latest.pt`、学習専用 truth を含む rollout は `rollouts/` に分離する。

### 学習と評価

- 初回実装時は attacker / defender の `2v1` を各30 episode、最大25 live tick 学習した。後述の追補で Task 11 に指定された全カリキュラムを実施した。
- 未使用 seed 100～109 の10 episode で、合法 action mask を使うランダム coach と比較した。値は1 episode あたりの平均。現行の stage 別 checkpoint を使った最終結果は追補の表を正とする。

| side | policy | 報酬 | 新規クリアマス | 無効移動 | ラウンド勝利 |
|---|---|---:|---:|---:|---:|
| attacker | 学習済み | 0.0484 | 24.2 | 0.0 | 0.0 |
| attacker | ランダム | 0.0246 | 24.3 | 2.4 | 0.0 |
| defender | 学習済み | 0.1000 | 54.0 | 0.0 | 0.0 |
| defender | ランダム | 0.0276 | 33.7 | 3.8 | 0.0 |

- この短い限定局面では報酬と無効移動が改善し、defender の新規クリアマスも増えた。attacker の新規クリアマスはランダムと同程度。双方とも評価10件でラウンド勝利は0件のため、戦術能力全般の改善やフルラウンドの性能を意味しない。

### 自動テストと情報境界

- 新規 Task 11 テスト10件。5人同時出力、critic 専用入力、action mask、4段階の実ゲーム環境、凍結 character、rollout/PPO、保存・再開、段階間の再開、side 誤読込拒否、headless round 移行を確認した。
- fake game と実学習ゲームの両方で、未視認敵を合法な2地点へ動かしても coach grid/vector が同一で、critic truth だけが変化することを確認した。学習 tick で actor が実際に受け取った観測と rollout の記録も一致する。actor の `act` には `CoachObservation` しか渡さない。
- `python -X utf8 -m unittest discover -s coach_v1 -p 'test_coach_v1_task*.py' -q`: **131件成功**。`python -X utf8 -m compileall -q coach_v1`、`git diff --check -- coach_v1` 成功。

### 変更ファイル

- 新規: `models/coach_model.py`、`training/coach_environment.py`、`training/coach_trainer.py`、`learning_coach.py`、`train_coach.py`、`evaluate_task11.py`、`test_coach_v1_task11_coach.py`、本ファイル。
- 新規生成: `checkpoints/coach/{attacker,defender}/` 以下の side 別 checkpoint と保存 rollout、`reports/task11_{attacker,defender}_{2v1,2v2,3v3,5v5}.json`。
- 更新: `coordinator.py`（round 変更時の recurrent actor reset）。

### 残課題と次の推奨タスク

- 2v1・2v2・3v3・5v5 の限定局面は下記追補で学習と評価を実施した。plant/defuse、自然な1ラウンドと opponent pool での評価は Task 12 以降。setup 自体は学習せず固定 STAY で通過した。実ゲームの全モードへ正式登録する段階ではない。
- 学習環境は事前観測と actor 入力の一致のため自チームを先に動かす。実戦の通常 `_move_order()` との差は、後続タスクの長い rollout 評価で確認する。
- Task 11 の4段階の実施結果を確認してから **Task 12: attacker の目的行動・1ラウンド全体**へ進む。defender の目的行動・1ラウンド全体は Task 13 に引き継ぐ。

## 2026-09-28 追補: Task 11 カリキュラム全段階の学習と評価

ユーザーの指摘どおり、`02.ALL_TASKS.md` の `2v1→2v2→3v3→5v5限定局面` は Task 11 の範囲である。初回の「2v2以降は後続タスク」という扱いを訂正し、両陣営で段階的に学習・評価した。

- attacker は `2v1:30 → 2v2:20 → 3v3:50 → 5v5:90` episode、defender は `2v1:30 → 2v2:50 → 3v3:30 → 5v5:90` episode。各段階は最大25 live tick の限定局面。段階変更時は同一 side の actor/critic/optimizer checkpoint から再開した。2v1 は seed を固定して別 directory に再生成し、対応 checkpoint を保持した。
- 各 side・stage の checkpoint を固定し、未使用 seed 100～109 で合法ランダム coach と比較した。`evaluate_task11.py` に `--checkpoint` を追加し、レポートに checkpoint の SHA-256 と training step を保存した。値は1 episode あたりの平均。`報酬`、`新規クリア`、`無効移動` の順に学習済み / ランダムを示す。

| side | stage | 報酬 | 新規クリア | 無効移動 |
|---|---|---:|---:|---:|
| attacker | 2v1 | 0.0484 / 0.0246 | 24.2 / 24.3 | 0.0 / 2.4 |
| attacker | 2v2 | 0.1276 / 0.0246 | 64.3 / 24.3 | 0.1 / 2.4 |
| attacker | 3v3 | 0.1486 / -0.0204 | 100.0 / 23.3 | 3.8 / 6.7 |
| attacker | 5v5 | 0.0764 / -0.1162 | 38.2 / 19.9 | 0.0 / 15.6 |
| defender | 2v1 | 0.1000 / 0.0276 | 54.0 / 33.7 | 0.0 / 3.8 |
| defender | 2v2 | 0.0458 / 0.1276 | 114.2 / 33.7 | 14.4 / 3.8 |
| defender | 3v3 | 0.0616 / -0.0196 | 120.8 / 31.7 | 13.7 / 8.3 |
| defender | 5v5 | 0.0260 / -0.1812 | 18.0 / 29.4 | 0.0 / 21.5 |

- attacker は4段階すべてで平均報酬と無効移動がランダムより良い。defender の2v2はクリア範囲を増やしたが報酬と衝突が悪化し、3v3も衝突が残る。5v5 defender は無効移動を避ける一方で探索が弱い。3v3 checkpoint を5v5で使う比較も行ったが、クリアは104.7へ増えた一方で無効移動26.4、報酬-0.2212、ラウンド勝利-0.1に悪化したため昇格しない。これらを Task 13 の改善対象として残す。
- 評価10件・25 tick ではラウンド勝敗の判断材料が乏しい。`2v2` defender のランダムは10件中1勝し、報酬比較に影響した。限定局面の補助指標改善と、plant/defuse・1ラウンド・フルマッチの性能は分けて解釈する。
- 追加テストで、checkpoint 再開後に `2v1` から `2v2` へ学習を継続できることを確認した。Task 11 のカリキュラム全4段階は実施済み。次は Task 12 の attacker 目的行動と1ラウンド学習へ進める。defender 側の衝突・探索は Task 13 で重点的に扱う。

## 2026-09-28 追補: 残課題の切り分けと Task 11 内の修正

上記「次は Task 12」は更新する。現行の `02.ALL_TASKS.md` に Task 11.5 が追加されており、次は Task 11.5 の残課題監査を行う。

### 原因と修正

- defender 2v2 の行動ログを tick ごとに追跡した。seed 101 では2人が互いの占有マスに向かって反対方向の移動を8 tick 以上繰り返し、双方が動けなかった。無効移動はすべて味方占有によるものだった。従来の合法 action mask は壁と範囲外を除外する一方、観測済みの味方占有マスを許可していた。`models/coach_model.py` の mask に味方占有判定を追加した。可視の味方位置だけを使う移動可否の判定で、経路探索や戦術規則は追加していない。
- `training/coach_environment.py` は最終 tick のクリア報酬が欠落し、headless 自動ラウンド更新時には死亡・解除報酬の判定が新ラウンドのキャラクターを見ていた。終了前の合法な視認と旧ラウンドのキャラクター参照を保持して修正した。移動・衝突・報酬内訳の診断値も追加した。
- `training/coach_environment.py` の凍結 character 読込で trainer の Torch 乱数を消費し、エピソード間のシナリオ乱数も再開時にずれていた。乱数状態を保存してモデルを読み、シナリオは episode ごとの seed で生成するよう修正した。連続2 episode と1 episode 後の再開で actor/critic の全 tensor が完全一致する自動テストを追加した。

### 同条件での再評価

旧 stage checkpoint をそのまま使用し、未使用 seed 100～109、最大25 tick で評価した。報酬・新規クリア・無効移動は各 episode 平均。初回レポートは修正前の履歴として保持し、再評価レポートを正とする。

| side / stage | mask | 学習済み報酬 | 学習済み新規クリア | 学習済み無効移動 | ランダム報酬 / 無効移動 |
|---|---|---:|---:|---:|---:|
| defender 2v2 | 修正前 | 0.0458 | 114.2 | 14.4 | 0.1326 / 3.8 |
| defender 2v2 | 修正後 | 0.1918 | 125.6 | 1.0 | 0.0574 / 0.7 |
| attacker 2v2 | 修正後 | 0.1430 | 72.0 | 0.1 | 0.0494 / 0.3 |
| defender 5v5 | 修正後 | 0.0260 | 18.0 | 0.0 | 0.0086 / 4.6 |

2v2 の修正前後では環境の最終 tick 報酬も修正されており、報酬差のすべてを mask の効果とは断定しない。無効移動は14.4→1.0に減り、反復衝突は解消した。残る少数の衝突原因は未確認。attacker 2v2 では修正後もランダムより報酬・新規クリア・無効移動が良好。defender 5v5 の学習済み行動は25 tick 中の移動指示が平均6回で、探索不足は残る。

直前の移動指示と自己位置の変化だけを入力する `action_feedback` もモデル・学習・推論に実験用オプションとして実装した。未視認敵の実位置は含まない。既定値は **無効** で、既存 checkpoint のモデル構造と動作を保つ。`train_coach.py --action-feedback` で別 checkpoint に学習できる。defender 2v1:30→2v2:40 episode、未使用 seed 100～119 では罰則0.01と0の両方で平均移動3回・無効移動0回・新規クリア29回となり、停止中心の方策へ収束した。罰則だけを原因とする仮説は棄却し、この実験 checkpoint は正式 checkpoint に昇格しない。罰則0.01→0.02の旧モデル継続学習も貪欲行動が変わらなかった。

### 検証と残課題

- 自動テストは140件成功。味方占有 mask、学習側での mask 強制、報酬の終了境界、再開再現性、実ゲームの未視認敵位置を変えた actor 観測の不変性、実験入力の学習経路と旧 checkpoint 読込を含む。`compileall` と `git diff --check` も成功。core ファイル変更なし。
- `diagnose_task11.py` と `reports/task11_diag_*` / `reports/task11_trace_*` に診断を保存した。モデル実験 checkpoint は `checkpoints/experiments/` に分離し、両 side の正式 checkpoint は変更していない。
- 現時点で特定した残課題の切り分けは以下。Task 11.5 ではこれに加え、HANDOFF-00～10 の全残課題を監査し、Task 12 前の項目を完了させる。Task 12 にはまだ進んでいない。

| 項目 | 状態 | 実施タスク |
|---|---|---|
| 味方占有 mask、終了 tick の報酬、再開再現性 | 完了・自動テスト済み | Task 11 |
| 学習環境の自チーム先行処理と実戦の移動順の差、過去 HANDOFF の残課題監査 | 未完了 | Task 11.5（Task 12 前） |
| attacker の plant・護衛・自然な1ラウンド、残る移動衝突の評価 | 未完了 | Task 12 |
| defender 5v5 の探索不足、retake・defuse・自然な1ラウンド、残る移動衝突の評価 | 未完了 | Task 13 |
| フルマッチ・陣営交代・setup・round 越え | 未完了 | Task 14 |
| 複数対戦相手と opponent pool | 未完了 | Task 15 |
