【前提】
・機械学習については、基本的に行動は学習によって対応し、推論側にロジックを入れて行動を制御することは可能な限りせずに、学習させた結果として行動させるようにする。
・ゲームのcore となるファイルはabilities_los.py, battle_logic.py, game_core.py, map_data.py, map_data_defender_setup.py, run_game.py
あたりです。
・基本的には、coreのファイルの修正は不要な認識で、明確な仕様変更、バグがあるときのみ修正対象とします。
・学習用の train ファイルと、推論用の learning ファイルについては、それぞれ対になっているため、例えば、 train_carry.py は learning_carry.py に import してOKです。
ただし、train_carry.py を 別の推論の learning_guard.py 等への import は禁止します。
・学習の実行は絶対に勝手にしないでください。こちらでします。

【重要：2026-10-10 の射線判定変更とモデル互換性】
・壁の射線判定を `grid_lines.py` の `wall_line_cells()` による対称な判定へ変更した。角に接する隣接セルも遮蔽判定に含める。現在の識別子は `WALL_LOS_VERSION = 'symmetric_supercover_v1'`。
・`tv3_scenario.py` の `Scenario.metadata()` は `wall_los` を含む。さらに、`default_posts()` が射線判定を使って退避位置（`retreat`）を決めるため、今回の変更では posts 自体も変わった。単なる識別子追加ではない。
・`Scenario.signature` は、盤面、分岐、posts、sensor、runtime、wall_los、Toru 基準モデルのハッシュから計算される。学習直後でも、その後にこれらが変われば保存モデルと不一致になる。
・実際に gc_v1 の defender_analysis_best.pt は 2026-10-10 09:58:11 に保存され、分析学習は 10:03:54 に終了していた。その後、grid_lines.py は 15:11:34、tv3_scenario.py は 15:13:56 に更新され、リテイク学習開始時に `Site analysis checkpoint mismatch` が発生した。
・確認時、モデルの version=3、opponent=gc_v1、入力特徴量401項目は一致し、scenario のみ不一致だった。保存値は `0e0ce1ce9622a8480a383afe19061110a2565fe6cd4c909c47d8b61a03da55bc`、変更後の値は `56a32296b2b4b10fd07a8fdf248e687824952376a3b88c95cdb5d01227ca9da6`。旧射線判定で posts を生成し、metadata から wall_los を除くと保存値と完全に一致した。これらは当時の確認値であり、固定の期待値としてコードに埋め込まない。
・不一致時は保存モデルと現在の設定を比較して原因を特定する。チェックを無効化したり、保存済み scenario や依存モデルのハッシュだけを書き換えたりして互換性があるように見せない。
・現在の射線仕様を使う場合、影響を受けた守備モデルは defender_analysis → defender_search → defender_retake の順で再学習する。search は分析モデルのハッシュに、retake は search モデルのハッシュに依存するため、後段だけの再学習では解決しない。旧モデルを使う場合は、そのモデルを生成した設定との整合性を確認する。
・再学習はユーザーが実行する。各スクリプト冒頭の名前付き定数で対象AI・学習回数・新規学習／再開などを確認・変更し、作業ディレクトリを `03.game/touyama_v3/` として、通常はオプションなしで次の順に実行する。

```powershell
Set-Location touyama_v3  # 03.game から移動
py tv3_train_defender_analysis.py
py tv3_train_defender_search.py
py tv3_train_defender_retake.py
```


【説明】
このプロジェクトでは、valorant を上から見たようなゲームを作成しています。
5v5で、それぞれグリッドのマスを1tick に一度移動したり、アビリティ(smoke, recon, flash)を使用できます。
ルールはvalorant と同じで、attacker, defender がいて、スパイクをプラントしたり、解除したりというものです。キャラは8方向を向くことができます。

このゲームをAIに学習させて移動させたり、アビリティを使用したりさせたいです。
射撃は自動で行われるので、AIがするのは、移動とアビリティの使用のみです。

マップのデータは map_data.py にあり、 0が床、1が壁(壁は移動不可能)、2がスパイク配置可能な床、3がattacker の配置位置、4がdefender の配置位置です。

# Touyama Gaming v3 新モデル設計方針

## 状態と確定事項

- 2026-10-10：学習・推論・収集コードとゲーム入口の移植を実施。4種類のマップをv3内で使用する。旧15コードファイルと旧6モデルディレクトリは削除済み。新モデル生成・データ収集・学習実行は行っていない。
- `touyama_v2` からコピーした `touyama_v3` を改修する。ゲームでの選択名 `Touyama Gaming v3` と既存の入口は維持する。
- 味方は **Touyama Gaming の5人専用**。Toru v4 の任意編成向け学習をそのままコピーしない。
- 学習・収集・独立評価の対象は `gc_v1`、`concon_v1`、`omoko_v1`、`fnatic_v3`、`frc_v1`、`toru_ai_v4` の6種類。ConCon v1は `Gorigons` 編成を使用する。`touyama_v2`、自分自身、`toru_ai_v3` / `toru_ai_v3.1` は対象に含めない。
- 学習はユーザーが実行する。少量の学習を動作確認と呼んで自動実行することも禁止する。
- 上位 `../AGENTS.md` と本書の配置・設定・学習実行制限を守る。本書にTouyama v3の作業指示と設計方針をまとめる。

## 調査した実装と知見

Toru v4 の説明書だけでなく、関連するモデル・コントローラー・学習設定も確認した。以下を移植時の参照元とする。

| 用途 | 主な参照元（`../toruAI_v4/` 以下） |
| --- | --- |
| 攻撃側の設置先予測 | `docs/training.md`、`tv4_model.py`、`tv4_observer.py`、`tv4_train_defender_analysis.py` |
| 設置前の寄り | `docs/defender_training.md`、`tv4_defender_policy.py`、`tv4_train_defender_search.py`、`tv4_map_retake_L.py` / `R.py` |
| リテイク学習 | `docs/retake_training.md`、`docs/retake_deadline_review.md`、`docs/retake_opponent_review.md`、`tv4_train_retake.py`、`tv4_retake_combat.py`、`tv4_retake_coordination.py` |
| 攻撃経路と突入 | `docs/attacker_analysis.md`、`docs/attacker_plant.md`、`tv4_learn_attacker_analysis.py`、`tv4_attacker_route_planner.py`、`tv4_attacker_entry_utility.py`、`tv4_attacker_plant_controller.py` |
| 設置後の守備 | `docs/attacker_guard.md`、`tv4_learn_attacker_guard.py`、`tv4_attacker_guard_controller.py`、`tv4_guard_runtime.py`、`tv4_collect_attacker_guard.py` |
| 保存・実ゲーム接続 | `AGENT.md`、`docs/game_best_models.md`、`tv4_game_controller.py`、`tv4_scenario.py` |

注意点：

- 設置先予測は **defender_analysis**。`attacker_analysis` は攻撃側が相手防衛配置・経路結果を予測する別モデルであり、混同しない。
- Toru v4 のリテイクは停止・到達不足に加え、死亡した解除担当の進捗が後任の合法DEFUSEを塞ぐ不具合があった。現行 `../frc_v1/actions.py` は生存者だけを判定している。Touyama v3 でもこの合法性を検証する。
- 最新リテイク方式は `living_defuser_opponent_learning_v4`、入力528要素。searchの505要素とは異なる。旧519要素のリテイクやTouyama v2のDQNを同じ形式として読み込まない。
- Toru v4 の資料には旧設定も残る。例えば並列数は説明書で3、確認時のsearch/retakeソースでは6。設定は参照時のソース・保存メタデータを確認し、Touyama v3の冒頭定数で明示する。
- Toru v4 の調査では修正後も勝率50%未満の相手・サイトが残る。バグ修正比較、教師付き結果、再学習後の教師なし性能は別物として扱う。

## モデル構成

全モデルを `touyama_v3/data/` に保存し、Toru v4 のモデルやデータに書き込まない。相手別モデルの内部では味方5人が重みを共有し、位置・能力・アビリティ・役割・HP等で区別する。

| モデル | 分割 | 目的 |
| --- | --- | --- |
| defender_analysis | 相手attacker AI別、6モデル | 公開観測から最終設置サイトL/Rの確率を予測 |
| defender_search | 相手attacker AI別、6モデル | 生存・能力温存・交戦と早めの合流を両立 |
| defender_retake | 相手attacker AI × L/R、12モデル | 実際の設置サイトを取り返し、期限内に解除 |
| attacker_analysis | 相手defender AI別、6モデル | 防衛配置・経路の設置成功見込みを推定 |
| attacker_plant | 相手defender AI別、6モデル | 集団移動、突入前の分裂、能力連携、回収、設置 |
| attacker_guard | 相手defender AI別、6モデル | 配置と能力使用を学習し、解除阻止・遅延・設置後勝利 |

plant/guardは左右別モデルにせず、サイトを入力して共通重みを使う。左右選択率を強制的に均等化しない。選ばれないサイトの独立評価は「対象なし」と表示する。

### 味方と相手の扱い

- 通常の味方編成は `tv3_character_stats_touyama.py` と既存の `Touyama Gaming` プリセットの5人を用いる。両者の能力値・順序・エンジンへの反映が一致するか実装前に照合する。
- 汎用AI向けの学習外編成評価は採用しない。同じ5人で学習用seedと独立評価seedを分ける。汎用編成での性能を主張しない。
- 相手AIと相手選手編成は別設定にする。`toru_ai_v4` は相手の攻撃時にanalysis/plant/guard、防衛時にanalysis/search/retakeの採用bestを固定して使う。
- Toru v4 は任意編成に対応するため、初期の相手選手は既存の `Team Elites` プリセットを暫定候補とする。採用時に全フェーズのbest・対象相手キー・互換性を点検し、ログに実際の相手編成を記録する。
- Toru v4 の既存対象には `touyama_v2` があり、新しい `touyama_v3` はない。既存bestの評価には `touyama_v2` 向けの重みを**固定した対戦基準**として指定する接続案を検証する。v3向けに学習済みとは扱わず、読込条件を無言で緩めない。接続できない場合は理由を明示する。
- 同名選手の識別には既存のチーム別キーとslotを使う。Touyama v2との対戦は対象から削除済みで、旧v2向けリテイク学習の個別調整も外す。ConCon向けには既定の学習設定を使い、必要な調整はConConとの独立評価から判断する。

## 全フェーズ共通の交戦方針（重要）

1. 公開観測で射撃可能な敵が見えたら、停止して敵の方向へ正しくfacingし、自動射撃を行う。敵が見えたという理由だけで能力を連打したり、意味なく移動し続けたりしない。射撃アクションを新設せず、エンジンの自動射撃を利用する。
2. 数マスの合法な移動で味方と複数方向からの射線を通せる場合は積極的に移動し、移動先から敵の方向へfacingして停止射撃する。壁・スモーク・味方の身体による遮断、被弾リスク、援護人数を考慮する。射線数のためだけに孤立・渋滞しない。移動方向とfacingは独立して扱い、前進先を向くだけにしない。
3. 敵defenderがスモーク内で解除している場合、**attacker guard** は公開された解除通知・設置地点から素早くスパイク位置へ接近し、隣接射撃が通る位置と適切なfacingで解除を阻止する。設置マスが占有・到達不能なら、到達可能な解除範囲周辺から妨害する。非公開の解除者の実座標は使わない。「defender時」という表現は、スパイクを守る設置後guardを指すものとして整理する。defender retakeは味方の解除を援護し、敵attackerを排除する。

この3点は学習入力・教師例・報酬・独立評価に反映する。初期配置の明示指定を除き、推論では可能な限り学習した行動を使う。期限の迫った解除担当や失明・移動不能なども考慮し、全員に同じ停止・移動を一律強制しない。

## ローカルマップと初期配置

必要な地形注釈はv3内にコピーし、v3の実装からToru v4のマップを直接importしない。

| コピー元 | v3側のファイル | 用途 |
| --- | --- | --- |
| `../toruAI_v4/tv4_map_attacker_branch.py` | `tv3_map_attacker_branch.py` | 攻撃経路・観測履歴の分岐領域 |
| `../toruAI_v4/tv4_map_defender_init.py` | `tv3_map_defender_init.py` | search初期監視位置とfacing目標 |
| `../toruAI_v4/tv4_map_retake_L.py` | `tv3_map_retake_L.py` | 左サイトの合流位置と突入口 |
| `../toruAI_v4/tv4_map_retake_R.py` | `tv3_map_retake_R.py` | 右サイトの合流位置と突入口 |

4ファイルはコピー済み。マップ文字列はコピー元と同じで、初期配置ファイルの説明だけ用途に合わせて修正した。branch先頭の `=` は見出しとして除外する。元地形・オーブは `map_data.NEW_MAZE_STR` から保持し、注釈文字や初期配置マップの数字で地形を作り直さない。初期配置マップには元地形との数字の差があるため、a-e/A-Eの注釈だけを読み込む。サイズ、文字の一意性、小文字の歩行可能性・setup時の到達可能性を検証する。

小文字と大文字の対応を使い、初期到着時に小文字位置から対応する大文字位置への8方向facingを設定する。大文字マスへ移動しない。担当の順序は既存 `TOUYAMA_ROSTER_ORDER` を使用し、現時点では次のとおり。座標は0始まりの（行, 列）。位置・向きの変更は `tv3_map_defender_init.py` のマーカーを編集して行い、別コードに座標を二重管理しない。

| 担当 | マーカー | 初期位置 | facing目標 | 初期方向 |
| --- | --- | --- | --- | --- |
| 夢の街 | a → A | (7, 3) | (15, 3) | S |
| いぐるん | b → B | (11, 16) | (14, 19) | SE |
| ろびぃな | c → C | (11, 21) | (13, 23) | SE |
| Tortlilyan | d → D | (8, 31) | (14, 31) | S |
| えんぺん | e → E | (2, 40) | (14, 40) | S |

これは**初期配置だけの指定**。その後に同じ場所・方向へ固定し続けるものではない。敵の公開観測に合わせたfacing、短距離の射線展開、退避、予測サイトへの寄りを認める。配置フェーズでは既存の移動可能マスクと味方占有を守り、テレポートで配置しない。

到達可能性の確認結果：現在のsetup進入制限ではa/b/c/dはスポーンから到達不能、eは最寄りスポーンから最短29歩でsetupの20tickを超える。通常ラウンドの地形では全点に到達可能で、最寄りスポーンからa=21、b=12、c=15、d=16、e=19歩（味方占有・交戦を除いた下限）。したがって初期監視位置は「setup終了時に必ず配置完了する地点」ではなく「序盤に移動して確保する地点」とする。setup中は許可範囲内で目標へ近づき、開始後に移動を続けて到着時に指定facingを設定する。安全な経路・敵接触・早期寄りを優先できるようにし、初期配置が終わるまで敵観測や交戦を無効にしない。ゲーム共通のsetup制限はこの指定だけを理由に変更しない。

## defender：予測、search、retake

### 設置先予測

- Toru v4 のMLPによる教師あり学習、観測履歴、相手別モデルを移植する。未学習時は左右50%とする。
- 公開視認、最終目撃と情報の古さ、監視済み領域、味方状態、過去ラウンド、公開されたスパイク落下位置を入力する。非公開の所持者・敵位置・内部目標・次の行動は読まない。
- 教師ラベルは実際に設置したサイト。未設置ラウンドに左右ラベルを付けない。長いラウンドに偏らないようラウンド→tickの順に抽出する。
- Toru v4の採用bestは互換性を確認して読めるようにするが、Touyamaの観測分布で再評価する。通常保存先へ参照元を上書きしない。必要な再学習は専用の5人で行う。
- 予測器の安定判定（Toru v4では80%以上が3tick継続）とsearchの早期移動報酬（65%以上）は別設定。正解率、判断率、全設置での正解割合、正解時の猶予tick、判断変更数を記録する。

### search

- 初期位置と向きは上記 `tv3_map_defender_init.py` で指定する。旧searchの「大文字地点へ移動し、コード内の別座標を監視する」処理は新searchで廃止する。
- 序盤は判定機への安全な情報収集を重視する。初期監視位置と隣接する合法な遮蔽マスを行き来するpeek/hideを許可する。隠れた敵位置を使わず、公開の目撃・被弾・失明・地形から危険を判断する。往復回数や移動そのものを目的化せず、新しい観測、情報の鮮度、生存で評価する。退避位置がなければ無理な往復を強制しない。
- 攻撃できるタイミングでは停止射撃し、数マスの移動で味方の複数射線を作れるなら展開する。FLASH・SMOKE・RECON等の合法能力と連携して敵の進行を妨害してよい。情報収集のために交戦を全面禁止せず、温存と交戦の結果を学習する。
- 予測確率・不確かさと左右の合流地点までの通路距離をモデルへ渡す。65%以上を早期寄りの初期基準にし、遠い生存者の前進を加点、足踏み・後退を減点する。最終行動はQ値で選ぶ。
- 公開の本人交戦、被ダメージ、撃破時には早期寄りの追加報酬を外し、通常の戦闘学習を続ける。味方の別地点の交戦だけでは除外しない。
- 小文字a/b/cの合流位置と大文字A/B/Cの突入口を区別する。担当位置を分散し、同じマスの占有・味方の移動予約で渋滞しないようにする。
- 設置時の生存者全員について実設置側の合流地点までの距離と解除余裕を評価する。Toru v4の初期目安は残り55 − 解除6 − 交戦20 − 余裕5 = 移動24tick。実際のエンジン定数・残りタイマーから算出し、結果を保証する値にはしない。
- 設置そのものに罰を付けず、生存・人数差・資源温存・到達余裕を準備スコアにする。最後の能力使用はコストとして学習させ、使用は禁止しない。
- 学習区間は設置完了tickまで。設置時は `PLANTED` とし、未実行のretakeの勝敗を付けない。bestは準備スコアを基準にする。

### retake

- 採用searchとdefender_analysisを固定し、通常setupから得た実設置状態を保存する。初期収集目安は相手ごと左右各50件。不足・収集上限は明示し、架空の配置で補完しない。
- ケースはHP・位置・能力残数・効果・タイマー・相手の内部状態・履歴を含む実ゲーム状態を復元する。シミュレーションに必要な非公開状態を方策入力へ渡さない。
- 合流、複数入口からの突入、援護、解除担当の継続と死亡時の引継ぎを扱う。目標はエンジンが解除を受け付ける設置地点・周囲8マスの範囲。時間不足時に合流待ちを続けない教師例を作る。
- 勝敗を主報酬にし、担当の解除範囲への進行・解除進捗の正負差分・実際の能力効果・援護を補助評価する。開始→中断の反復で報酬を稼げないようにする。
- Double DQN、target network、教師模倣、行動種類の模倣、解除担当サンプルの優先抽出を移植する。教師・探索は減衰させ、独立評価・本番では0にする。
- 担当抽出50%、種類模倣0.50を初期候補にする。Toru v4のfrc左75%/1.0などの相手・左右別調整は参照値として残し、Touyamaの診断に基づいて調整する。元AIの成績だけで有効と断定しない。
- 初期目標は相手・左右別の独立評価勝率50%。目標未満モデルだけ更新回数を増やす方式を採用候補にする。評価6seed、12設置未満は評価数不足というToru v4の基準を引き継ぐ。
- bestはリテイク勝率→解除回数→平均報酬。停止、未到達、未解除開始、解除中断、担当死亡・引継ぎ、合法DEFUSE有無を診断ログに残す。実設置サイトでモデルを選び、予測が外れた場合も切り替える。

## attacker：集団移動、分裂突入、guard

### analysis / plant

- Toru v4の攻撃側analysisで公開観測から配置分布・経路結果を推定し、固定したanalysisを使ってplantを学習する。候補経路の結果予測を最善経路の保証と呼ばない。
- 基本は生存者全員が所持者と同じ進行経路で移動し、所持者の進路を塞がない別マスを目標にする。回収・公開交戦・失明時の安全な退避は例外として扱う。
- 突入前に到達すると、同じサイトの別入口へ一部を分け、複数方向から射線を通す。初期候補は3人の主隊＋2人の別入口隊。人数・分裂開始距離・迂回上限は名前付き定数にする。生存人数、能力、経路距離、残り時間から役割を割り当て、特定の選手名で固定しない。
- Toru v4のplannerは通常移動中にも別方向の偵察役・別入口担当を作るため、そのまま移植しない。Touyamaでは突入前まで集団を維持する条件を追加し、同条件の実行器でanalysisの教師データも集める。
- 射線の数だけでなく、味方による射線遮断、入口占有、所持者の到達可否、突入時刻の差を入力・教師例・報酬で扱う。移動・facing・能力・回収・設置は学習したQ値で選ぶ。
- FLASHの着弾時間、SMOKEの射線遮断、RECONの情報取得を連携に使う。能力を使うだけでは加点せず、発生した効果を投擲元transitionへ対応付ける。無駄な重複使用を診断する。
- 公開観測で定期的に経路を更新し、被害・死亡・停滞で再検討する。設置開始後は経路・別入口担当を安定させる。予測された領域を使う場合も隠れた敵の実座標を標的にしない。
- plantの学習・評価区間は設置完了まで。bestは少被害設置率（初期設定：5人生存かつ開始時チーム最大HPの90%以上）→従来の3人生存・HP30%以上の設置率→通常設置率→生存・被害・能力温存・消費・時間の順に比較する。設置後勝率はguard評価で測る。
- FRCのようなサイト内守備には、能力を全消費することではなく、HPと人数を残して設置することを主目的にする。設置時の残HPと少被害設置に追加報酬を付け、能力を使わず安全に設置できる行動も認める。通常設定は `tv3_train_attacker_plant.py` 冒頭の `LOW_DAMAGE_MIN_SURVIVORS`、`LOW_DAMAGE_HP_FRACTION`、`LOW_DAMAGE_PLANT_REWARD`、`PLANT_HP_RETENTION_REWARD`。
- plant入力に12個の公開連携特徴を追加する。フラッシュ飛行中の露出、入口ごとの準備人数、合流待ちの経過時間、同じ公開敵を射撃できる味方人数と射線の角度を渡す。位置・facing・失明・スモーク・味方の身体を確認し、2対1・3対1の射線を教える。
- 教師は主隊が入口に着いたら、別入口隊が到着可能な場合に最大6tickだけ準備を待つ。残り時間不足や公開交戦時に待ち続けず、フラッシュ飛行中は遮蔽からの飛び出しを避ける。推論の候補を待機・能力だけに絞らず、学習したQ値で選ぶ。
- `preserve` / `supported` は参考情報。`preserve` でも有効な突入支援能力を使える教師例に修正した。射線改善は公開情報のポテンシャル差で評価し、実際に味方複数人が同じ敵へ射撃した際の命中を補助加点する。能力使用回数には加点しない。
- plantの入力schemaはversion 5。attacker analysisの入力・実行器は変更していないので、そのbestは再利用できる。plantは新規学習し、guardは新plantのbest確定後に再収集・学習する。修正前plantのreplay・optimizer・評価値を新条件へ混在させない。

### guard

- plant学習完了後にanalysis/plantの採用bestを固定し、相手ごと左右合計50件の実設置状態を収集する。plant学習中の同時収集・guard学習はしない。
- 味方を遮蔽物・侵入口・設置地点周辺へ分散し、複数の射線、失明時の退避、能力による遅延・妨害を学習する。主目的は解除阻止・爆発までの遅延・設置後勝利。撃破・能力消費は補助指標。
- FLASH、SMOKE、RECONと編成が持つその他の合法能力・ULTを扱う。自動効果を架空の手動アクションとして追加しない。
- 公開の解除通知と設置地点周辺の解除可能範囲を使う。スモーク内の隠れた解除者の実位置は使わない。隣接射撃・接近・妨害を教師例と報酬で教え、解除者を隠すスパイク直上のSMOKEを無条件で推奨しない。
- **Toru v4の例外をそのまま移植しない。** `tv4_attacker_guard_controller.py` にはスモーク内の解除通知時、接近候補だけを残し、到達後は待機facing候補まで絞る処理がある。これは推論時の戦術的強制。Touyamaでは合法性と味方占有の処理を維持し、この対策は観測入力・教師例・進行報酬・停滞減点で実現する。失明・移動不能・強制facingの合法性は尊重する。
- 1セットはその相手の全保存ケースを各1回学習する単位。独立評価はケース再生ではなく、別seedの通常setupから固定plantと候補guardを実行する。
- bestはguard勝率を最優先し、全体勝率・生存・被害・資源を併記する。設置0件ならguardのbestを生成しない。

## 保存・依存関係・実ゲームへの接続

- `latest` は毎セット保存する再開用。重み・target・optimizer・replay・乱数状態・完了セット・設定を保持する。`best` は教師・探索なしの固定評価条件で改善した候補だけを採用する。
- 定期評価と最終評価を行い、最終セットが評価周期に重なる場合は二重実行しない。同点・悪化でbestを更新しない。採用セットと学習完了セットを区別する。
- schema、マップ、編成、相手AIと相手の固定best、観測・合法性・報酬・実行器、前段bestのhash、収集manifestを記録する。同条件の再開で不一致を拒否する。新条件では旧評価値を比較基準にせず再評価・再選定する。
- defender_analysis変更→search再評価・必要な新規学習→retakeケース再収集→retake新規学習。search変更時も後段のケース・retakeを更新する。
- attacker_analysis変更→plant新規学習→guardケース再収集→guard新規学習。plant変更時もguardケース・guardを更新する。
- 設置完了tickの状態を保ってsearch→retake、plant→guardを切り替える。試合・ラウンドのリセット、サイド交代、相手変更に対応する。
- 未作成・不一致のモデルを別相手・別サイトの重みで無言補完しない。学習・収集では停止して理由を表示。本番の標準controllerへの切替はフェーズ単位で明示し、fallbackを対象AIの学習済み性能として集計しない。

予定する配置：

```text
touyama_v3/
  AGENTS.md
  tv3_train_defender_analysis.py
  tv3_train_defender_search.py
  tv3_collect_defender_retake.py
  tv3_train_defender_retake.py
  tv3_train_attacker_analysis.py
  tv3_train_attacker_plant.py
  tv3_collect_attacker_guard.py
  tv3_train_attacker_guard.py
  tv3_map_attacker_branch.py
  tv3_map_defender_init.py
  tv3_map_retake_L.py
  tv3_map_retake_R.py
  tv3_learn_*.py / tv3_*_controller.py / tv3_map_*.py
  test/
  docs/
  data/
    best/<相手AI>/
      defender_analysis_best.pt
      search_best.pt
      retake_L_best.pt
      retake_R_best.pt
      attacker_analysis_best.pt
      attacker_plant_best.pt
      attacker_guard_best.pt
    defender_analysis/<相手AI>/latest.pt
    defender/search/<相手AI>/latest.pt
    defender/retake/<相手AI>/L_latest.pt / R_latest.pt
    attacker_analysis/<相手AI>/latest.pt
    attacker_plant/<相手AI>/latest.pt
    attacker_guard/<相手AI>/latest.pt
    retake_cases/
    attacker_guard_cases/
  logs/<フェーズ>/<相手AI>/
```

新規ファイルには `tv3_` を付け、フェーズ名を省略しない。既存の入口・ファイル名は配置ルール適合だけを理由に変更しない。共通モデル・観測・学習更新は中立の共通モジュールに置き、別フェーズのtrainを推論側へimportしない。

## 移行順序と削除条件

1. 固定5人の能力、6対象AIのキー・相手編成、Toru v4の固定best接続、同名選手の扱いを確認する。
2. 公開観測・保存契約・予測器を移植し、defender search、retake収集と学習コードを実装する。
3. 集団移動と分裂時期を反映したattacker analysis実行器、plant、guard収集と学習コードを実装する。
4. 既存v3のゲーム入口へ接続し、非学習のテストで状態引継ぎ・合法性・読込条件を検証する。
5. 移行後に参照を調べ、不要になったcarry/escort/retrieveの学習・推論、旧専用マップ・共通コード・コピー済みモデルをv3内だけで整理する。回収機能はplant側で置き換えてから旧retrieveを削除する。能力表・必要な入口・参照中のファイルは保持する。

ユーザーは不要ファイル削除を許可済み。旧carry/escort/retrieve、旧learning、旧common、旧マップとコピー済みモデルは、新ゲーム入口への置換と参照確認後に削除済み。`touyama_v2` と `toruAI_v4` は復元元・参照元として保ち、既存の他作業の変更を巻き戻さない。

## 通常実行手順

設定は各対象スクリプト冒頭の名前付き定数で変更する。対象AI、固定味方・相手編成、セット数、評価頻度・seed、モデルとケース入力、出力先、新規/再開、教師・探索、報酬をまとめる。CLIは検証時の一時上書きだけにする。

作業ディレクトリは `03.game/touyama_v3/`。以下はユーザーが順に実行する新実装のコマンド。学習は自動実行しない。

```powershell
Set-Location C:\Users\ronet\MyProject\git\AI_dnn\03.game\touyama_v3

# defender：各前段の採用bestが確定してから次へ
python tv3_train_defender_analysis.py
python tv3_train_defender_search.py
python tv3_collect_defender_retake.py
python tv3_train_defender_retake.py

# attacker：plant完了後にguardを収集・学習
python tv3_train_attacker_analysis.py
python tv3_train_attacker_plant.py
python tv3_collect_attacker_guard.py
python tv3_train_attacker_guard.py
```

既存ゲームの作業ディレクトリは `03.game`、通常起動は `python run_game.py`。v3は新bestを読む構成へ置換済み。best未作成のフェーズは理由を表示し標準controllerへ切り替える。新モデル性能は未学習のため未確認。

## 実装時の検証項目

- 非公開敵位置を変更しても同じ公開snapshotの入力・合法候補・教師判断が変わらないこと。
- 5人移動中と分裂開始後を分け、左右サイトで所持者の経路塞ぎ・移動予約衝突・無限停止を検証すること。
- a-eへの到達とA-Eへの初期facing、初期配置後の固定解除、安全な隣接peek/hide、公開敵への停止射撃と短距離クロス射線展開を検証すること。
- 実エンジンの解除範囲、解除担当死亡後の合法DEFUSEと引継ぎ、残り時間不足、解除中断の報酬を検証すること。
- guardのスモーク解除・失明・移動不能・射線遮断を検証し、戦術的な候補強制に頼らず学習できる入力・教師例を用意すること。
- モデルの相手・サイト・hash・schema不一致、ケース不足、設置0件、latest/best、再開乱数、サイド交代を検証すること。
- 単体テストは `test/` に置く。重み更新を伴う学習スクリプトは実行しない。学習後の実力はユーザーが実行した教師・探索なしの独立評価で判断すること。

## 残る設計上の確認

ユーザーが答えた味方専用編成・対象相手・今回の範囲は確定済み。実装を妨げる追加回答は現時点で不要。

Toru v4の相手編成は `Team Elites` を初期設定にした。v2向け固定bestの接続は `tv3_opponents.py` に実装し、schema・hashを検証する。v2との同名選手対戦は既存の `TeamPlayerKey` で区別し、実エンジンの非学習テストで確認済み。これは学習の自動実行や既存Toruモデルの更新を許可するものではない。

## 実装検証記録

- v3内の72件の非学習テストが通過。対象AIディレクトリからテストを実行して確認した。
- 共通の推論デバイス・既存ゲームファクトリーの6件のテストも通過。
- 初期位置/facingとラウンドリセット、公開情報制約、集団移動中の偵察分離抑制、短距離クロス射線教師、スモーク解除接近の教師と合法候補の保持、設置完了tick後のguard切替、死亡後の合法DEFUSEを確認した。
- Toru v4の既存v2向けbestから攻撃・防衛コントローラーを読み込めることを確認した。学習や性能評価の対戦は行っていない。
- plantの通常移動では所持者から離れすぎる行動に小さな費用を付ける。突入・分裂・被弾・失明・交戦時には外す。設定は `tv3_train_attacker_plant.py` 冒頭の `GROUP_DISTANCE_LIMIT` / `GROUP_SEPARATION_PENALTY`。
- plantの集団維持報酬は行動実行前に固定した経路計画で判定する。経路計画がない場合はこの費用を付けない。次tickの再計画で `attack_plan=None` となる場合の回帰テストを追加した。保存設定の `formation_context` で修正前latestとの再開混在を拒否する。
- 保存ケースのディレクトリ名は実装に合わせて `data/retake_cases/`、`data/attacker_guard_cases/` とする。
- テストは学習済み性能・勝率を保証しない。新モデル生成、収集、学習はユーザーが通常手順で実行する。

## 学習コマンドメモ

### 2026-10-10：FRC交戦調査後のplant互換性

- 最新のplant新規学習はversion 10・入力892。詳細は `docs/tv3_frc_combat_review.md`。以下の古いversion 5/6の記載よりこの追記を優先する。
- 同じ公開敵への援護、公開敵の身体による射線遮断、移動先の援護なし射線、最近の公開目撃を使った能力支援、射線上のSMOKE候補をplant専用に修正した。推論で退避・停止を強制しない。
- plant version 9・入力872のbestは、専用の旧実行経路でschemaを厳密に確認して読み込む。旧重みをv10入力へ流用したり、schemaだけ書き換えたりしない。新規v10学習へv9 latestのreplay/optimizerは混在させない。
- 今回は共有combat、attacker analysis、defender retake、core、scenarioを変更していない。既存retakeの進行を妨げない。新plantを採用した後のattacker guardは、そのplantでケースを再収集して学習する。
- 学習は実行していない。6相手の旧best読込、FRC旧bestの同一seedでの行動・評価値の完全一致、非学習テスト136件通過を確認。v10の学習済み性能は未確認。

### FRC調査後の方針

- 詳細は `docs/frc_plant_review.md`。旧best再現は13/36設置、失敗23件は全て時間切れ。旧重みに修正を適用しただけでの改善は確認できていない。新学習後の独立評価で確認する。
- plantはversion 6・入力628。合法設置/前進の入力、行動種類ごとの教師損失、設置役の経験混合、不要停止の学習費用を使う。FRCは種類損失1.0・設置役65%・通常の2倍更新。Toru v4のリテイク担当者学習を参照した。
- 辺の往復制限は直近8tickだけ。交戦・失明・退避を妨げない。別入口はサイト外と内の異なる隣接マスとして検証する。前進・能力・設置を推論側で強制しない。
- 壁の角を取りこぼす実エンジンの不具合を修正した。壁に触れる線は双方向とも遮断し、高速公開視界と合わせる。青い点線の丸はリビールであり、自キャラからの射撃可能性とは分ける。
- LOS版をscenario署名に含めるため、旧analysisも含めたv3モデル・ケースは新条件で更新する。attacker analysis→plant→guard収集/学習、defender analysis→search→retake収集/学習の順。既存best/latest/ケースを勝手に更新しない。Toru固定相手の保存初期配置とモデルhashは保持する。
- 学習実行は禁止のまま。重み更新なしの診断、単体テストのみを行う。評価各ラウンドをset・seed付きで記録する。

- Defender
py tv3_train_defender_analysis.py
py tv3_train_defender_search.py
py tv3_collect_defender_retake.py
py tv3_train_defender_retake.py

- Attacker
py tv3_train_attacker_analysis.py
py tv3_train_attacker_plant.py
py tv3_collect_attacker_guard.py
py tv3_train_attacker_guard.py
