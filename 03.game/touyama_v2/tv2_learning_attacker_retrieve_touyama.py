"""touyama_v2/learning_attacker_retrieve_touyama.py

固定チーム(Tortlilyan/いぐるん/ろびぃな/夢の街/えんぺん)専用の
Attacker「retrieve phase」推論コントローラー(落下スパイク回収)。

touyama_v2/train_attacker_retrieve.py で学習した Dueling DQN を読み込み、
run_game.py / battle_logic.py の decide_move(char, game_state) 呼び出し
規約にそのまま乗せられる形で返す。

完全に自己完結。run_game.py / controllers.py / battle_logic.py /
abilities_los.py への依存はなく、必要なLOS計算・BFS距離マップ・行動マスク・
観測構築はこのファイル内に複製する。game_core / map_data系の import も
不要(gridはgame_state["grid"]から都度取得するため)。

--------------------------------------------------------------------------
【重要】設計方針の変更について:

以前の版は「リトリーバー役1人だけがこのフェーズの対象、残り4人は
list(char.pos)で待機」という設計だった。しかし狭い通路で非リトリーバーが
リトリーバーの経路を塞ぐと、待機ロジックには「どく」動作が存在しないため
永久にデッドロックしてタイムアウトする問題があった。

train_attacker_retrieve.py 側をリトリーバー+ブロッキング役(味方1体)の
2エージェント・重み共有構成に変更し、「塞いでいたらどく」ことを報酬で
学習させたことに合わせ、この推論コントローラーも
「呼ばれた全キャラがこのモデルの判断に従う」設計に変更した。
待機のみで固定する非リトリーバー分岐は廃止した。

【リトリーバー選出】
学習環境は「retriever 1人 + blocker 最大1人」という前提だったが、実戦は
チーム最大5人なので、複数の非リトリーバーが同時に存在しうる
(学習分布よりエージェント数が多いケースがある点は留意)。リトリーバー
選出自体は従来通り、「スパイク未保持の生存アタッカーのうちBFS距離最短の
1人」をstickyに選ぶ(learning_attacker_guard_touyama.pyのチーム内選出と
同種の考え方)。選ばれなかった全員は is_retriever=0 の観測を与えられ、
同じ重み共有ネットワークで「自分がリトリーバーの次の一歩マスを
塞いでいないか」を踏まえて行動する。
--------------------------------------------------------------------------

観測ベクトル(OBS_DIM=24)は train_attacker_retrieve.py の
RetrieveEnv._build_obs() と要素・並び順を完全一致させている:
    [0]    自己座標r
    [1]    自己座標c
    [2-5]  隣接4マスの壁フラグ(up, down, left, right)
    [6-9]  隣接4マスのBFS距離(壁・占有マスは1.0でブロック扱い)
    [10]   自己座標のBFS距離(スパイクまで)
    [11-14] ロールone-hot(FLASH, SMOKE, RECON, HUNT)
    [15]   自己アビリティ残チャージ(0/1)
    [16]   視認中敵の有無フラグ
    [17-18] 視認中敵への相対方向(dr, dc)
    [19]   視認中敵のブラインド状態フラグ
    [20]   視認中敵のリビール状態フラグ
    [21]   is_retriever(自分がリトリーバー役か)
    [22]   blocking_flag(自分がリトリーバーの次の一歩マスを塞いでいるか)
    [23]   retriever_dist_norm(リトリーバーの目標までの正規化BFS距離)

行動空間(N_ACTIONS=6)も train_attacker_retrieve.py と完全一致:
    0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY, 5=ABILITY

行動マスクは、学習側が2エージェント構成になったことで占有マスも
明示的に禁止する仕様に変わったため、推論側もチーム問わず生存キャラが
占有する全マスを禁止する(実ゲームの衝突判定と一致)。

チェックポイントは train_attacker_retrieve.py の _save_checkpoint() が
保存する dict 形式
({"model_state_dict","obs_dim","n_actions","episode","success_rate",
"roster_order"}) を読み込む。後方互換のため、素の state_dict のみの
ファイルが渡された場合もフォールバックで対応する。

このチーム(5人)で1つのコントローラーインスタンスを共有する想定
(重み共有Dueling DQN)。
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from tv2_character_stats_touyama import (
    CHARACTER_TABLE as TOUYAMA_STATS_TABLE,
    TOUYAMA_ROSTER_ORDER,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right (行動ID 0-3と対応)

OBS_DIM = 33
N_ACTIONS = 6
ACTION_ABILITY = 5
ACTION_STAY = 4
# 向き(facing)は移動・アビリティとは独立に常に自由選択できる。
# action_idx = base_idx(0-5)*8 + facing_idx(0-7)。
# tv2_train_attacker_retrieve.pyのdecode_actionと同一規約。
FACING_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ACTION_DIM = N_ACTIONS * len(FACING_DIRS)  # 48
SIGHTING_STALENESS_CAP = 20  # チーム共有の目撃情報を保持する最大tick数(train側と同一)

ROLES = ["FLASH", "SMOKE", "RECON", "HUNT"]

DEFAULT_MODEL_PATH = (
    "touyama_v2/data/attacker_retrieve_touyama_data/"
    "dqn_attacker_retrieve_touyama_best_by_eval.pt"
)

VERBOSE_LOG = False

# ---------------------------------------------------------------------------
# Dueling DQN (touyama_v2/train_attacker_retrieve.py と同一構造。
# 属性名 self.value / self.advantage も state_dict 互換のため一致させる)
# ---------------------------------------------------------------------------
class DuelingQNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=ACTION_DIM, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, n_actions)
        )

    def forward(self, x):
        feat = self.feature(x)
        v = self.value_head(feat)
        a = self.advantage_head(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


# ---------------------------------------------------------------------------
# 補助関数(LOS / BFS)。abilities_los.py / train_attacker_retrieve.py の複製実装。
#
# 注意: game_state には smokes(煙リスト)が含まれないため、この推論用LOSは
# 壁のみを考慮する(スモークによる遮蔽は考慮しない。他のtouyama_v2推論
# コントローラーと同じ簡略化方針)。
# ---------------------------------------------------------------------------
def _line_cells(p1, p2):
    y0, x0 = int(p1[0]), int(p1[1])
    y1, x1 = int(p2[0]), int(p2[1])
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    cells = []
    while True:
        cells.append((y0, x0))
        if x0 == x1 and y0 == y1:
            return cells
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _has_los(grid, p1, p2):
    for r, c in _line_cells(p1, p2):
        if grid[r, c] == 1:
            return False
    return True


def _bfs_distance_map(grid, goal):
    """goal(落下スパイク位置)から各床マスへの最短距離マップ(壁越え不可)。
    到達不能マスは-1。train_attacker_retrieve.py の bfs_distance_map と同一ロジック。"""
    height, width = grid.shape
    dist = np.full((height, width), -1, dtype=np.int32)
    gr, gc = int(goal[0]), int(goal[1])
    if grid[gr, gc] == 1:
        return dist
    dist[gr, gc] = 0
    queue = deque([(gr, gc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1 and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist

def _ability_charge(char):
    """ロールに対応する残チャージ数を取得する。HUNT(アビリティ無し)は常に0。"""
    return {
        "FLASH": getattr(char, "flash_charges", 0),
        "SMOKE": getattr(char, "smoke_charges", 0),
        "RECON": getattr(char, "recon_charges", 0),
    }.get(char.ability_name, 0)


def _decode_action(action_idx):
    """action_idx = base_idx(0-5) * 4 + facing_idx(0-3)。
    tv2_train_attacker_retrieve.pyのdecode_actionと同一規約。
    戻り値は (base_action_idx, facing)。"""
    idx = int(action_idx)
    base_idx, facing_idx = divmod(idx, len(FACING_DIRS))
    return base_idx, FACING_DIRS[facing_idx]


# ---------------------------------------------------------------------------
# 推論コントローラー
# ---------------------------------------------------------------------------
class LearningAttackerRetrieveTouyamaController:
    """touyama_v2固定チーム専用、落下スパイク回収フェーズ(retrieve)のAttacker AI。

    Attackerチーム全員(5人)がこの1インスタンスを共有して呼び出される。
    リトリーバー役・非リトリーバー役の両方が同じ重み共有ネットワークの
    判断に従う(train_attacker_retrieve.pyのマルチエージェント学習と対応)。

    ステータス(コンボ・タイガーパッシブ込みの確定値)は run_game.py の
    既存エンジンが character_stats_touyama.py を経由して自動適用済みの
    char オブジェクトをそのまま利用する(本ファイル側では再計算しない)。
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, greedy=True, verbose=VERBOSE_LOG):
        self.greedy = greedy
        self.verbose = verbose
        self.model = DuelingQNet(OBS_DIM, ACTION_DIM).to(DEVICE)

        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                if verbose:
                    ep = checkpoint.get("episode")
                    sr = checkpoint.get("success_rate")
                    print(
                        f"[LearningAttackerRetrieveTouyamaController] loaded: {model_path} "
                        f"(episode={ep}, success_rate={sr})"
                    )
            else:
                # 後方互換: 素のstate_dictのみが渡された場合
                state_dict = checkpoint
                if verbose:
                    print(f"[LearningAttackerRetrieveTouyamaController] loaded (raw state_dict): {model_path}")
            self.model.load_state_dict(state_dict)
        except Exception as exc:
            print(f"[LOAD ERROR] attacker retrieve(touyama) model '{model_path}' の読込に失敗: {exc}")
        self.model.eval()

        # 落下スパイク位置(spike_pos)はキャラが拾うまで不変のため、
        # BFS距離マップはラウンド中、同じspike_posであればキャッシュを使い回す。
        self._dist_map = None
        self._dist_map_source = None  # 再計算要否判定用: 直前のspike_pos

        # チーム共有の目撃情報(train_attacker_retrieve.pyのRetrieveEnv.team_sightingと同一方針)。
        # decide_move()は1tickにつきチーム全員ぶん呼ばれるため、_team_sighting_seenで
        # 同じキャラ名が再度来たタイミングを「新しいtickの開始」とみなして更新する。
        self._team_sighting = None
        self._team_sighting_seen = set()

        self._debug_log_path = "attacker_retrieve_touyama_debug.log"

    # -- ラウンド開始時のリセット -----------------------------------------
    def reset_round(self):
        self._dist_map = None
        self._dist_map_source = None
        self._team_sighting = None
        self._team_sighting_seen = set()

    def _ensure_dist_map(self, grid, spike_pos):
        spike_pos = (int(spike_pos[0]), int(spike_pos[1]))
        if spike_pos == self._dist_map_source and self._dist_map is not None:
            return
        self._dist_map = _bfs_distance_map(grid, spike_pos)
        self._dist_map_source = spike_pos

    def _update_team_sighting(self, char, grid, chars):
        """チーム共有の目撃情報を更新する(1tickにつき1回のみ再計算)。
        decide_move()はチーム全員ぶん毎tick呼ばれるため、同じキャラ名が
        再度来た時点で新しいtickが始まったと判定してstalenessを進める
        (train_attacker_retrieve.pyのRetrieveEnv._update_team_sightingと同一方針)。"""
        if char.name in self._team_sighting_seen:
            self._team_sighting_seen = set()
            if self._team_sighting is not None:
                self._team_sighting["tick_ago"] += 1
                if self._team_sighting["tick_ago"] > SIGHTING_STALENESS_CAP:
                    self._team_sighting = None
        self._team_sighting_seen.add(char.name)

        teammates = [o for o in chars if o.team == char.team and getattr(o, "is_alive", True)]
        enemies = [o for o in chars if o.team != char.team and getattr(o, "is_alive", True)]

        visible = [
            e for mate in teammates for e in enemies
            if _has_los(grid, tuple(mate.pos), tuple(e.pos))
        ]
        if visible:
            nearest = min(
                visible,
                key=lambda e: max(abs(e.pos[0] - char.pos[0]), abs(e.pos[1] - char.pos[1])),
            )
            self._team_sighting = {
                "pos": (int(nearest.pos[0]), int(nearest.pos[1])),
                "tick_ago": 0,
            }

    # -- 観測構築 ----------------------------------------------------------
    # train_attacker_retrieve.py の RetrieveEnv._build_obs() と要素・並び順を
    # 完全一致させること。
    def _build_observation(self, char, grid, chars, visible_enemies, team_sighting):
        height, width = grid.shape
        r, c = int(char.pos[0]), int(char.pos[1])

        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = r / height
        obs[1] = c / width

        occupied = {
            tuple(o.pos) for o in chars if o is not char and getattr(o, "is_alive", True)
        }

        max_dist_scale = float(height + width)
        wall_flags = []
        neighbor_dists = []
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            in_bounds = 0 <= nr < height and 0 <= nc < width
            is_wall = (not in_bounds) or grid[nr, nc] == 1
            wall_flags.append(1.0 if is_wall else 0.0)

            blocked = is_wall or (in_bounds and (nr, nc) in occupied)
            if blocked:
                neighbor_dists.append(1.0)
            else:
                nd = self._dist_map[nr, nc] if self._dist_map is not None else -1
                neighbor_dists.append(min(1.0, nd / max_dist_scale) if nd >= 0 else 1.0)

        obs[2], obs[3], obs[4], obs[5] = wall_flags
        obs[6], obs[7], obs[8], obs[9] = neighbor_dists

        raw_dist = self._dist_map[r, c] if self._dist_map is not None else -1
        obs[10] = min(1.0, raw_dist / max_dist_scale) if raw_dist >= 0 else 1.0

        role_index = {"FLASH": 11, "SMOKE": 12, "RECON": 13, "HUNT": 14}.get(char.ability_name, 11)
        obs[role_index] = 1.0

        obs[15] = float(1 if _ability_charge(char) > 0 else 0)

        if visible_enemies:
            nearest = min(
                visible_enemies,
                key=lambda e: max(abs(e.pos[0] - r), abs(e.pos[1] - c)),
            )
            edr = float(np.clip((nearest.pos[0] - r) / height, -1.0, 1.0))
            edc = float(np.clip((nearest.pos[1] - c) / width, -1.0, 1.0))
            obs[16] = 1.0
            obs[17] = edr
            obs[18] = edc
            obs[19] = 1.0 if getattr(nearest, "blind_remaining", 0) > 0 else 0.0
            obs[20] = 1.0 if (
                getattr(nearest, "reveal_remaining", 0) > 0 or getattr(nearest, "los_revealed", False)
            ) else 0.0

        char_facing = getattr(char, "facing", None)
        if char_facing in FACING_DIRS:
            obs[21 + FACING_DIRS.index(char_facing)] = 1.0

        if team_sighting is not None:
            tr, tc = team_sighting["pos"]
            obs[29] = 1.0
            obs[30] = float(np.clip((tr - r) / height, -1.0, 1.0))
            obs[31] = float(np.clip((tc - c) / width, -1.0, 1.0))
            obs[32] = min(team_sighting["tick_ago"], SIGHTING_STALENESS_CAP) / SIGHTING_STALENESS_CAP

        return obs

    # -- 行動マスク ---------------------------------------------------------
    def _action_mask(self, char, grid, chars):
        """壁・占有マスへの移動 / チャージ0でのABILITYは禁止する。
        train_attacker_retrieve.py が2エージェント構成になり占有マスを
        明示的にマスクする仕様へ変わったことに合わせる。"""
        mask = np.zeros(N_ACTIONS, dtype=bool)
        height, width = grid.shape
        r, c = int(char.pos[0]), int(char.pos[1])
        occupied = {
            tuple(o.pos) for o in chars if o is not char and getattr(o, "is_alive", True)
        }

        for a in range(4):
            dr, dc = CARDINAL[a]
            nr, nc = r + dr, c + dc
            walkable = (
                0 <= nr < height and 0 <= nc < width
                and grid[nr, nc] != 1
                and (nr, nc) not in occupied
            )
            mask[a] = walkable
        mask[4] = True  # STAY は常に許可
        mask[ACTION_ABILITY] = _ability_charge(char) > 0
        return np.repeat(mask, len(FACING_DIRS))

    # -- メイン ----------------------------------------------------------
    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        chars = game_state.get("chars", [])
        # チーム共有目撃情報はフェーズ適用対象外のキャラも含め、呼ばれた時点で
        # 毎tick必ず更新する(train側のenv.step()が全員ぶん1回で更新するのと対応)。
        self._update_team_sighting(char, grid, chars)

        if not char.is_alive:
            return list(char.pos)

        spike_pos = game_state.get("spike_pos")

        # このコントローラーはretrieve phase専用(落下スパイク回収)。
        # スパイクが未落下(誰かが保持中 or 未使用)、またはこのキャラが
        # 既に保持している場合は対象外。安全側としてその場に留まる
        # (上位のフェーズ切替側でcarry/escort/guardフェーズの
        # コントローラーへ委譲する想定)。
        if spike_pos is None or getattr(char, "has_spike", False):
            return list(char.pos)

        self._ensure_dist_map(grid, spike_pos)

        enemies = [e for e in chars if e.team != char.team]
        visible_enemies = [
            e for e in enemies if e.is_alive and _has_los(grid, tuple(char.pos), tuple(e.pos))
        ]

        # 全員が対称に「スパイクへの最短距離を縮める」ことを学習したモデル
        # なので、呼ばれたキャラは役割区分なくそのままモデルの判断に従う。
        obs = self._build_observation(char, grid, chars, visible_enemies, self._team_sighting)
        mask = self._action_mask(char, grid, chars)

        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        mask_t = torch.from_numpy(mask).to(DEVICE)

        with torch.no_grad():
            q_values = self.model(obs_t).squeeze(0).clone()
            q_values[~mask_t] = -1e9
            action_idx = int(torch.argmax(q_values).item())

        base_idx, facing = _decode_action(action_idx)

        if self.verbose:
            with open(self._debug_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{char.name},{tuple(char.pos)},spike={tuple(spike_pos)},"
                    f"action={action_idx},base={base_idx},facing={facing},"
                    f"Qvals={np.round(q_values.cpu().numpy(), 4).tolist()}\n"
                )

        if base_idx < 4:
            dr, dc = CARDINAL[base_idx]
            next_pos = [char.pos[0] + dr, char.pos[1] + dc]
            # 向き(facing)は移動方向とは無関係にモデルが選択する。battle_logic.pyが
            # 移動と同時にこのfacing指定を適用する(被弾直後の強制向きが最優先)。
            return next_pos, {"facing": facing}

        if base_idx == ACTION_STAY:
            return list(char.pos), {"facing": facing}

        # ACTION_ABILITY: アビリティ使用Tickはbattle_logic側で移動・向き変更が
        # 行われないため、facingは付与しない(carry推論コントローラーと同一方針)。
        # 視認中の敵がいればそれを最優先の狙点とし、いなければ回収目標である
        # spike_pos自体を予防的な狙点とする(learning_attacker_guard_touyama.py
        # の fallback方針と同様)。
        if visible_enemies:
            nearest = min(
                visible_enemies,
                key=lambda e: max(abs(e.pos[0] - char.pos[0]), abs(e.pos[1] - char.pos[1])),
            )
            target_pos = (int(nearest.pos[0]), int(nearest.pos[1]))
        else:
            target_pos = (int(spike_pos[0]), int(spike_pos[1]))

        return list(char.pos), {"ability": char.ability_name, "target": target_pos}