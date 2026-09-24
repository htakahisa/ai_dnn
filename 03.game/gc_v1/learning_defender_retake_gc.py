"""gc_v1/learning_defender_retake_gc.py

固定チーム(Xdll/SyouTa/Absol/eKo/SugarZ3ro)専用の
Defender「retake phase」推論コントローラー(プラント後限定)。

gc_v1/train_defender_retake.py で学習した Dueling DQN を読み込み、
run_game.py / battle_logic.py の decide_move(char, game_state) 呼び出し
規約にそのまま乗せられる形で返す。

完全に自己完結。run_game.py / controllers.py / battle_logic.py /
abilities_los.py への依存はなく、必要なLOS計算・BFS距離マップ・行動マスク・
観測構築はこのファイル内に複製する。game_core からは定数のみを参照する
(ロジックは参照しない)。map_data系のimportも不要(gridはgame_state["grid"]
から都度取得するため、search phaseのような専用マップファイルへの依存がない)。

ステータス(GCステータス・タイガーパッシブ込みの確定値)は
character_stats_gc.py 側の定義に基づき、run_game.py の既存エンジン
(combo_awakening.py / game_core.Character)が実戦時に自動適用するため、
本ファイルではステータスの再計算は行わない。char.accuracy等の実値を
そのまま利用する。

行動空間は学習側と完全に同一(N_ACTIONS=9):
    0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY, 5=DEFUSE, 6=ABILITY,
    7=ULTIMATE, 8=COLLECT_ORB
観測ベクトルも同ファイルの build_observation() と要素・並び順を完全一致
させている(OBS_DIM=53)。旧37次元チェックポイントにも読込互換性を持たせる。
インデックスコメントを明示して対応関係を追跡できるようにしている。

アビリティ使用時のターゲットは、学習環境(RetakeEnv.apply_ability)が
プラント地点(planted_pos)を中心に効果を計算する設計だったことに合わせ、
常に planted_pos を狙点として実ゲーム側の execute_ai_ability() に渡す。

このチーム(5人)で1つのコントローラーインスタンスを共有する想定
(重み共有Dueling DQN)。
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from game_core import (
    BLIND_DURATION_TICKS,
    DEFUSE_REQUIRED_TICKS,
    SPIKE_DETONATION_TICKS,
)
try:
    from .character_stats_gc import (
        CHARACTER_TABLE as GC_STATS_TABLE,
        GC_ROSTER_ORDER,
    )
    from .gc_facing import FACING_DIRS, append_facing_onehot, facing_towards
    from .defender_objectives_gc import (
        RETAKE_COORDINATION_DIM,
        retake_coordination_features,
        retake_coordination_state,
    )
    from .ultimate_tactics_gc import (
        ORB_CONTEXT_DIM,
        build_ultimate_action,
        can_collect_orb,
        orb_context_features,
        ultimate_context_features,
    )
except ImportError:
    from character_stats_gc import (
        CHARACTER_TABLE as GC_STATS_TABLE,
        GC_ROSTER_ORDER,
    )
    from gc_facing import FACING_DIRS, append_facing_onehot, facing_towards
    from defender_objectives_gc import (
        RETAKE_COORDINATION_DIM,
        retake_coordination_features,
        retake_coordination_state,
    )
    from ultimate_tactics_gc import (
        ORB_CONTEXT_DIM,
        build_ultimate_action,
        can_collect_orb,
        orb_context_features,
        ultimate_context_features,
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CARDINAL_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
MOVE_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}

BASE_OBS_DIM = 37
ULTIMATE_CONTEXT_OBS_DIM = BASE_OBS_DIM + 4
ORB_CONTEXT_OBS_DIM = ULTIMATE_CONTEXT_OBS_DIM + ORB_CONTEXT_DIM
LEGACY_OBS_DIM = ORB_CONTEXT_OBS_DIM + len(FACING_DIRS)
OBS_DIM = LEGACY_OBS_DIM + RETAKE_COORDINATION_DIM
LEGACY_N_ACTIONS = 7
N_ACTIONS = 9

SITE_ZONE_RADIUS = 6
ENTRY_READY_RADIUS = 3
DEFUSE_SAFETY_MARGIN_TICKS = 4
ENTRY_SAFETY_MARGIN_TICKS = DEFUSE_SAFETY_MARGIN_TICKS + ENTRY_READY_RADIUS

ACTION_DEFUSE = 5
ACTION_ABILITY = 6
ACTION_ULTIMATE = 7
ACTION_COLLECT_ORB = 8
FACING_HEAD_VERSION = 2

ROLE_INDEX = {"フラッシュ": 0, "スモーカー": 1, "シーカー": 2, "タイガー": 3}

DEFAULT_MODEL_PATH = (
    "data/defender_retake_gc_data/" "dqn_defender_retake_gc_best_by_eval.pt"
)


# ---------------------------------------------------------------------------
# Dueling DQN (gc_v1/train_defender_retake.py と同一構造)
# ---------------------------------------------------------------------------
class DefenderRetakeDuelingDQN(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, n_actions)
        )
        facing_hidden = hidden // 2
        self.facing_feature = nn.Sequential(
            nn.Linear(obs_dim, facing_hidden),
            nn.ReLU(),
            nn.Linear(facing_hidden, facing_hidden),
            nn.ReLU(),
        )
        self.facing_output = nn.Linear(
            facing_hidden + n_actions, len(FACING_DIRS)
        )
        self.action_dim = n_actions
        self.facing_head_version = FACING_HEAD_VERSION

    def forward(self, x):
        feat = self.feature(x)
        value = self.value_head(feat)
        adv = self.adv_head(feat)
        return value + adv - adv.mean(dim=1, keepdim=True)

    def facing_values(self, x, actions):
        features = self.facing_feature(x)
        actions = torch.as_tensor(actions, dtype=torch.long, device=x.device).view(-1)
        action_onehot = torch.nn.functional.one_hot(
            actions, num_classes=self.action_dim
        ).to(dtype=features.dtype)
        return self.facing_output(torch.cat((features, action_onehot), dim=1))

    def facing_parameters(self):
        return [
            parameter
            for module in (self.facing_feature, self.facing_output)
            for parameter in module.parameters()
        ]


# ---------------------------------------------------------------------------
# 補助関数(LOS / BFS)。abilities_los.py / train_defender_retake.py の複製実装。
#
# 実戦ではgame_stateのsmoke_cellsを使い、ゲーム本体と同じ遮蔽を適用する。
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


def _has_los(grid, p1, p2, smoke_cells=()):
    cells = _line_cells(p1, p2)
    for r, c in cells:
        if grid[r, c] == 1:
            return False
    return len(cells) <= 2 or not any(cell in smoke_cells for cell in cells)


def _bfs_distance_map(grid, goal):
    """goalから各床マスへの最短距離マップ(壁越え不可)。到達不能マスは-1。
    train_defender_retake.py の bfs_distance_map と同一ロジック。"""
    height, width = grid.shape
    dist = np.full((height, width), -1, dtype=np.int32)
    gr, gc = int(goal[0]), int(goal[1])
    if grid[gr, gc] == 1:
        return dist
    dist[gr, gc] = 0
    queue = deque([(gr, gc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in CARDINAL_MOVES:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < height
                and 0 <= nc < width
                and grid[nr, nc] != 1
                and dist[nr, nc] == -1
            ):
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def _good_directions(dist_map, grid, r, c):
    """train_defender_retake.py の good_directions と同一ロジック。
    BFS距離マップ上で実際に距離を縮められる方向(up/down/left/right)を
    1.0、それ以外を0.0とする4次元フラグ。"""
    good = [0.0, 0.0, 0.0, 0.0]
    height, width = grid.shape
    raw = dist_map[r, c]
    if raw < 0:
        return good
    for i, (dr, dc) in enumerate(CARDINAL_MOVES):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
            nd = dist_map[nr, nc]
            if nd != -1 and nd < raw:
                good[i] = 1.0
    return good


def _bfs_next_step_avoiding_occupied(dist_map, grid, char, chars):
    """Spikeへ向かうBFS最短の合法1手を返す。"""
    r0, c0 = int(char.pos[0]), int(char.pos[1])
    if dist_map is None:
        return [r0, c0]

    cur = int(dist_map[r0, c0])
    if cur <= 0:
        return [r0, c0]

    occupied = {
        (int(o.pos[0]), int(o.pos[1]))
        for o in chars
        if o is not char and getattr(o, "is_alive", True)
    }

    best = None
    for dr, dc in CARDINAL_MOVES:
        nr, nc = r0 + dr, c0 + dc
        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
            continue
        if grid[nr, nc] == 1 or (nr, nc) in occupied:
            continue
        nd = int(dist_map[nr, nc])
        if nd < 0 or nd >= cur:
            continue
        cand = (nd, nr, nc)
        if best is None or cand < best:
            best = cand

    if best is None:
        return [r0, c0]
    return [best[1], best[2]]


def _ability_charge(char):
    """ロールに対応する残チャージ数を取得する。HUNT(アビリティ無し)は常に0。"""
    return {
        "FLASH": getattr(char, "flash_charges", 0),
        "SMOKE": getattr(char, "smoke_charges", 0),
        "RECON": getattr(char, "recon_charges", 0),
    }.get(char.ability_name, 0)


# ---------------------------------------------------------------------------
# 推論コントローラー
# ---------------------------------------------------------------------------
class LearningDefenderRetakeGCController:
    """gc_v1固定チーム専用、プラント後フェーズ(リテイク)のDefender AI。

    Defenderチーム全員(5人)がこの1インスタンスを共有して呼び出される。

    ステータス(コンボ・タイガーパッシブ込みの確定値)は run_game.py の
    既存エンジンが character_stats_gc.py を経由して自動適用済みの
    char オブジェクトをそのまま利用する(本ファイル側では再計算しない)。
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, greedy=True, verbose=False):
        self.greedy = greedy
        self.verbose = verbose
        self.model_obs_dim = OBS_DIM
        self.model_action_dim = N_ACTIONS
        self.facing_head_enabled = False
        self.model_episode = None
        self.model = DefenderRetakeDuelingDQN().to(DEVICE)
        try:
            checkpoint = torch.load(
                model_path, map_location=DEVICE, weights_only=False
            )
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model_obs_dim = int(state_dict["feature.0.weight"].shape[1])
            self.model_action_dim = int(state_dict["adv_head.2.weight"].shape[0])
            if self.model_action_dim not in (LEGACY_N_ACTIONS, N_ACTIONS):
                raise ValueError(
                    f"unsupported defender-retake action dimension: "
                    f"{self.model_action_dim}"
                )
            self.model = DefenderRetakeDuelingDQN(
                obs_dim=self.model_obs_dim, n_actions=self.model_action_dim
            ).to(DEVICE)
            incompatible = self.model.load_state_dict(state_dict, strict=False)
            missing = [
                key for key in incompatible.missing_keys
                if not key.startswith(("facing_feature.", "facing_output."))
            ]
            if missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"defender-retake checkpoint keys mismatch: "
                    f"missing={missing} unexpected={list(incompatible.unexpected_keys)}"
                )
            facing_version = int(
                checkpoint.get("facing_head_version", 0)
                if isinstance(checkpoint, dict) else 0
            )
            self.model.facing_head_version = facing_version
            self.facing_head_enabled = (
                self.model_obs_dim == OBS_DIM
                and self.model_action_dim == N_ACTIONS
                and facing_version >= 1
                and not any(
                    key.startswith(("facing_feature.", "facing_output."))
                    for key in incompatible.missing_keys
                )
            )
            self.model_episode = (
                checkpoint.get("episode") if isinstance(checkpoint, dict) else None
            )
            if verbose:
                print(f"[LearningDefenderRetakeGCController] loaded: {model_path}")
        except Exception as exc:
            print(
                f"[LOAD ERROR] defender retake(gc) model '{model_path}' の読込に失敗: {exc}"
            )
        self.model.eval()

        # プラント地点(planted_pos)は1ラウンド中不変のため、
        # BFS距離マップ・site_zoneはラウンド開始後の初回呼び出し時に
        # 1度だけ計算してキャッシュする。
        self._dist_map = None
        self._site_zone = None
        self._dist_map_source = None  # 再計算要否判定用: 直前のplanted_pos

        self._debug_log_path = "defender_retake_gc_debug.log"

    # -- ラウンド開始時のリセット -----------------------------------------
    def reset_round(self):
        self._dist_map = None
        self._site_zone = None
        self._dist_map_source = None

    def _ensure_dist_map(self, grid, planted_pos):
        planted_pos = (int(planted_pos[0]), int(planted_pos[1]))
        if planted_pos == self._dist_map_source and self._dist_map is not None:
            return
        self._dist_map = _bfs_distance_map(grid, planted_pos)
        height, width = grid.shape
        self._site_zone = {
            (r, c)
            for r in range(height)
            for c in range(width)
            if 0 <= self._dist_map[r, c] <= SITE_ZONE_RADIUS
        }
        self._dist_map_source = planted_pos

    # -- 観測構築 ----------------------------------------------------------
    # train_defender_retake.py の build_observation() と要素・並び順を
    # 完全一致させること。インデックスはコメントで明示する。
    def _build_observation(
        self, char, game_state, chars, enemies, visible_enemies, detonate_timer
    ):
        grid = game_state["grid"]
        height, width = grid.shape
        planted_pos = game_state["planted_pos"]
        pr, pc = int(planted_pos[0]), int(planted_pos[1])
        r, c = int(char.pos[0]), int(char.pos[1])

        obs = np.zeros(OBS_DIM, dtype=np.float32)

        obs[0] = r / height  # [0] 自己座標r
        obs[1] = c / width  # [1] 自己座標c
        obs[2] = char.hp / char.max_hp if char.max_hp else 0.0  # [2] 自己HP割合

        good_dir = _good_directions(self._dist_map, grid, r, c)
        obs[3], obs[4], obs[5], obs[6] = (
            good_dir  # [3-6] BFS推奨方向(up,down,left,right)
        )

        raw_dist = self._dist_map[r, c]
        dist_to_plant = min(1.0, raw_dist / (height + width)) if raw_dist >= 0 else 1.0
        obs[7] = dist_to_plant  # [7] BFSプラント距離

        obs[8] = 1.0 if (r, c) in self._site_zone else 0.0  # [8] サイトゾーン内フラグ
        obs[9] = (
            1.0 if max(abs(pr - r), abs(pc - c)) <= 1 else 0.0
        )  # [9] プラント隣接フラグ

        obs[10] = (
            1.0 if _ability_charge(char) > 0 else 0.0
        )  # [10] 自己アビリティ残チャージ
        obs[11] = (
            char.blind_remaining / BLIND_DURATION_TICKS if BLIND_DURATION_TICKS else 0.0
        )  # [11]
        obs[12] = char.defuse_timer / DEFUSE_REQUIRED_TICKS  # [12] 解除進捗
        obs[13] = detonate_timer / SPIKE_DETONATION_TICKS  # [13] 起爆タイマー割合

        allies = [a for a in chars if a.team == char.team and a.is_alive]
        obs[14] = len(allies) / 5.0  # [14] 生存味方数
        obs[15] = (
            sum(1 for a in allies if tuple(a.pos) in self._site_zone) / 5.0
        )  # [15] サイトゾーン内味方数
        allies_near_entry = sum(
            1
            for a in allies
            if max(abs(pr - a.pos[0]), abs(pc - a.pos[1])) <= ENTRY_READY_RADIUS
        )
        obs[16] = allies_near_entry / 5.0  # [16] エントリー圏内味方数

        others = [a for a in allies if a is not char]
        if others:
            nearest_ally_dist = min(
                max(abs(a.pos[0] - r), abs(a.pos[1] - c)) for a in others
            ) / max(height, width)
        else:
            nearest_ally_dist = 1.0
        obs[17] = nearest_ally_dist  # [17] 最近接味方距離

        # 味方アビリティ発動中フラグ。smoke情報がgame_stateに無いため、
        # 敵側のデバフ状態(blind/reveal)のみで近似する。
        obs[18] = (
            1.0
            if any(
                e.is_alive
                and (
                    getattr(e, "blind_remaining", 0) > 0
                    or getattr(e, "reveal_remaining", 0) > 0
                )
                for e in enemies
            )
            else 0.0
        )  # [18] 味方アビリティ発動中(近似)

        obs[19] = len(visible_enemies) / 5.0  # [19] 視認中敵数
        obs[20] = len([e for e in enemies if e.is_alive]) / 5.0  # [20] 生存敵総数

        # [21-32] 視認中の近い敵、最大2体分(6次元 x 2)
        sorted_enemies = sorted(
            visible_enemies,
            key=lambda e: max(abs(e.pos[0] - r), abs(e.pos[1] - c)),
        )
        idx = 21
        for e in sorted_enemies[:2]:
            edx = (e.pos[1] - c) / width
            edy = (e.pos[0] - r) / height
            edist = max(abs(e.pos[0] - r), abs(e.pos[1] - c)) / max(height, width)
            ehp = e.hp / e.max_hp if e.max_hp else 0.0
            eblind = 1.0 if getattr(e, "blind_remaining", 0) > 0 else 0.0
            erevealed = (
                1.0
                if (
                    getattr(e, "reveal_remaining", 0) > 0
                    or getattr(e, "los_revealed", False)
                )
                else 0.0
            )
            obs[idx : idx + 6] = [edx, edy, edist, ehp, eblind, erevealed]
            idx += 6
        # 視認中敵が2体未満の残り枠は0.0のまま(np.zerosで初期化済み)

        # [33-36] ロールone-hot(フラッシュ/スモーカー/シーカー/タイガー)
        role_idx = 33 + ROLE_INDEX.get(char.role, 0)
        obs[role_idx] = 1.0

        obs[BASE_OBS_DIM:ULTIMATE_CONTEXT_OBS_DIM] = ultimate_context_features(
            char,
            engaged=bool(visible_enemies),
            objective_window=True,
            urgent=(
                detonate_timer <= 20
                or (char.max_hp > 0 and char.hp / char.max_hp <= 0.45)
            ),
        )
        obs[ULTIMATE_CONTEXT_OBS_DIM:ORB_CONTEXT_OBS_DIM] = orb_context_features(
            char, game_state.get("available_orbs", ())
        )
        append_facing_onehot(
            obs[ORB_CONTEXT_OBS_DIM:LEGACY_OBS_DIM],
            getattr(char, "facing", "S"),
        )
        coordination = retake_coordination_state(
            char,
            allies,
            self._dist_map,
            detonate_timer,
            entry_radius=ENTRY_READY_RADIUS,
            defuse_ticks=DEFUSE_REQUIRED_TICKS,
            safety_margin=DEFUSE_SAFETY_MARGIN_TICKS,
        )
        obs[LEGACY_OBS_DIM:OBS_DIM] = retake_coordination_features(
            coordination, SPIKE_DETONATION_TICKS
        )

        return obs

    # -- 行動マスク ---------------------------------------------------------
    # train_defender_retake.py の action_mask() と同一ロジック。
    def _action_mask(
        self, char, grid, chars, lock_movement, available_orbs=()
    ):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        r, c = int(char.pos[0]), int(char.pos[1])
        occupied = {
            tuple(o.pos)
            for o in chars
            if o is not char and getattr(o, "is_alive", True)
        }

        for a in range(4):
            dr, dc = MOVE_DELTAS[a]
            nr, nc = r + dr, c + dc
            walkable = (
                0 <= nr < grid.shape[0]
                and 0 <= nc < grid.shape[1]
                and grid[nr, nc] != 1
                and (nr, nc) not in occupied
            )
            mask[a] = walkable and not lock_movement
        mask[4] = True  # stay は常に許可

        pr = self._dist_map_source[0] if self._dist_map_source else r
        pc = self._dist_map_source[1] if self._dist_map_source else c
        dist_to_plant = max(abs(pr - r), abs(pc - c))
        mask[ACTION_DEFUSE] = dist_to_plant <= 1

        mask[ACTION_ABILITY] = _ability_charge(char) > 0
        mask[ACTION_ULTIMATE] = build_ultimate_action(
            grid, char, chars, destination=self._dist_map_source
        ) is not None
        mask[ACTION_COLLECT_ORB] = can_collect_orb(char, available_orbs)

        return mask

    def _select_facing(self, obs, action_idx):
        if not self.facing_head_enabled:
            return None
        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)
            values = self.model.facing_values(
                obs_t, torch.tensor([action_idx], device=DEVICE)
            ).squeeze(0)
        return FACING_DIRS[int(values.argmax().item())]

    # -- メイン ----------------------------------------------------------
    def decide_move(self, char, game_state):
        if not char.is_alive:
            return list(char.pos)

        is_planted = bool(game_state.get("is_planted", False))
        planted_pos = game_state.get("planted_pos")

        # このコントローラーはプラント後フェーズ(リテイク)専用。
        # 万一プラント前に呼ばれた場合は安全側としてその場に留まる
        # (上位のフェーズ切替側で search フェーズのコントローラーへ
        # 委譲する想定)。
        if not is_planted or planted_pos is None:
            return list(char.pos)

        grid = game_state["grid"]
        chars = game_state.get("chars", [])
        detonate_timer = float(game_state.get("detonate_timer", 0.0))
        smoke_cells = {
            tuple(map(int, cell))
            for cell in game_state.get("smoke_cells", ())
        }

        self._ensure_dist_map(grid, planted_pos)

        enemies = [e for e in chars if e.team != char.team]
        visible_enemies = [
            e
            for e in enemies
            if e.is_alive and _has_los(grid, tuple(char.pos), tuple(e.pos), smoke_cells)
        ]

        # ------------------------------------------------------------------
        # Retake Commitment
        # ------------------------------------------------------------------
        r0, c0 = int(char.pos[0]), int(char.pos[1])
        pr, pc = int(planted_pos[0]), int(planted_pos[1])
        raw_dist = int(self._dist_map[r0, c0])
        cheb_dist = max(abs(pr - r0), abs(pc - c0))

        alive_defenders = [
            d for d in chars if d.team == char.team and getattr(d, "is_alive", True)
        ]
        alive_enemies = [
            e for e in enemies if getattr(e, "is_alive", True)
        ]

        def _retake_dist(unit):
            ur, uc = int(unit.pos[0]), int(unit.pos[1])
            d = int(self._dist_map[ur, uc])
            return d if d >= 0 else 10**9

        defusing_defenders = [
            d for d in alive_defenders if int(getattr(d, "defuse_timer", 0)) > 0
        ]
        designated_defuser = (
            min(defusing_defenders or alive_defenders, key=lambda d: (_retake_dist(d), d.name))
            if alive_defenders else None
        )
        is_designated = designated_defuser is char

        # A defuse is consecutive: turning to fight or selecting an ability
        # resets its progress. Keep the same defuser committed once started.
        if (
            not getattr(self, "facing_head_enabled", False)
            and is_designated
            and cheb_dist <= 1
            and int(getattr(char, "defuse_timer", 0)) > 0
        ):
            return list(char.pos), "DEFUSE"

        # Smoke the spike before the first defuse tick.  Entry range is used
        # here rather than only the 1-cell defuse range: waiting until the
        # defuser is already exposed at the spike was too late in practice.
        near_defuser = any(
            max(abs(int(d.pos[0]) - pr), abs(int(d.pos[1]) - pc)) <= ENTRY_READY_RADIUS
            for d in alive_defenders
        )
        active_defuser = any(
            int(getattr(d, "defuse_timer", 0)) > 0 for d in alive_defenders
        )
        spike_area = {
            (rr, cc)
            for rr in range(pr - 1, pr + 2)
            for cc in range(pc - 1, pc + 2)
            if 0 <= rr < grid.shape[0]
            and 0 <= cc < grid.shape[1]
            and int(grid[rr, cc]) != 1
        }
        smoke_covers_spike = bool(spike_area & smoke_cells)
        if (
            not getattr(self, "facing_head_enabled", False)
            and
            str(getattr(char, "ability_name", "")).upper() == "SMOKE"
            and int(getattr(char, "smoke_charges", 0)) > 0
            and (near_defuser or active_defuser)
            and not smoke_covers_spike
        ):
            if self.verbose:
                print(
                    f"[GC RETAKE SMOKE] {char.name} cover spike="
                    f"{tuple(planted_pos)}"
                )
            return list(char.pos), {
                "ability": "SMOKE",
                "target": (pr, pc),
            }

        move_ticks_needed = max(0, raw_dist - 1) if raw_dist >= 0 else 10**9
        latest_safe_total = (
            move_ticks_needed
            + DEFUSE_REQUIRED_TICKS
            + DEFUSE_SAFETY_MARGIN_TICKS
        )
        must_commit_now = detonate_timer <= latest_safe_total

        # Attacker全滅後は戦術判断を終了し、「解除だけ」を最優先する。
        # 既に誰かが隣接していれば、その1人を即解除担当に固定。
        # まだ誰も隣接していなければ、生存Defender全員を最短でSpikeへ寄せる。
        if not getattr(self, "facing_head_enabled", False) and not alive_enemies:
            adjacent = [
                d for d in alive_defenders
                if max(
                    abs(int(d.pos[0]) - pr),
                    abs(int(d.pos[1]) - pc),
                ) <= 1
            ]
            if adjacent:
                emergency_defuser = min(adjacent, key=lambda d: d.name)
                if emergency_defuser is char:
                    if self.verbose:
                        print(
                            f"[GC RETAKE COMMIT] {char.name} ALL_ENEMIES_DEAD "
                            f"-> DEFUSE detonate={detonate_timer:.1f}"
                        )
                    return list(char.pos), "DEFUSE"
                # 解除担当の周囲で味方が動き回って詰まらせない。
                return list(char.pos)

            forced_next = _bfs_next_step_avoiding_occupied(
                self._dist_map, grid, char, chars
            )
            if self.verbose and forced_next != [r0, c0]:
                print(
                    f"[GC RETAKE COMMIT] {char.name} ALL_ENEMIES_DEAD_FAST "
                    f"from={tuple(char.pos)} to={tuple(forced_next)} "
                    f"detonate={detonate_timer:.1f}"
                )
            return forced_next

        # 解除デッドライン: 指定解除担当は戦闘判断よりSpikeを優先。
        if (
            not getattr(self, "facing_head_enabled", False)
            and is_designated
            and must_commit_now
        ):
            if cheb_dist <= 1:
                if self.verbose:
                    print(
                        f"[GC RETAKE COMMIT] {char.name} DEADLINE -> DEFUSE "
                        f"detonate={detonate_timer:.1f} need={latest_safe_total}"
                    )
                return list(char.pos), "DEFUSE"
            forced_next = _bfs_next_step_avoiding_occupied(
                self._dist_map, grid, char, chars
            )
            if self.verbose:
                print(
                    f"[GC RETAKE COMMIT] {char.name} DEADLINE "
                    f"from={tuple(char.pos)} to={tuple(forced_next)} "
                    f"detonate={detonate_timer:.1f} need={latest_safe_total}"
                )
            return forced_next

        # Inside smoke, start the protected defuse without waiting for the
        # network/deadline. Adjacent enemies remain visible by game rules.
        if (
            not getattr(self, "facing_head_enabled", False)
            and is_designated
            and cheb_dist <= 1
            and (r0, c0) in smoke_cells
            and not visible_enemies
        ):
            return list(char.pos), "DEFUSE"

        # 遠距離からは全員BFS最短。サイト近辺(3マス以内)に入ってからDQNへ戻す。
        if (
            not getattr(self, "facing_head_enabled", False)
            and raw_dist > ENTRY_READY_RADIUS
        ):
            forced_next = _bfs_next_step_avoiding_occupied(
                self._dist_map, grid, char, chars
            )
            if forced_next != [r0, c0]:
                if self.verbose:
                    print(
                        f"[GC RETAKE COMMIT] {char.name} FAST_ROTATE "
                        f"dist={raw_dist} from={tuple(char.pos)} "
                        f"to={tuple(forced_next)}"
                    )
                return forced_next

        time_critical = detonate_timer <= ENTRY_SAFETY_MARGIN_TICKS

        # A defender who sees an enemy while isolated should not take the
        # first unsupported duel.  Move toward the spike/retake group until
        # another defender is within the entry radius.  Once supported, the
        # established rule of holding still during a visible gunfight applies.
        nearby_allies = [
            d for d in alive_defenders
            if d is not char
            and max(abs(int(d.pos[0]) - r0), abs(int(d.pos[1]) - c0)) <= ENTRY_READY_RADIUS
        ]
        if (
            not getattr(self, "facing_head_enabled", False)
            and
            visible_enemies
            and not nearby_allies
            and getattr(char, "blind_remaining", 0) <= 0
            and not time_critical
            and raw_dist > 1
        ):
            regroup_next = _bfs_next_step_avoiding_occupied(
                self._dist_map, grid, char, chars
            )
            if regroup_next != [r0, c0]:
                if self.verbose:
                    print(
                        f"[GC RETAKE REGROUP] {char.name} isolated="
                        f"{tuple(char.pos)} -> {tuple(regroup_next)}"
                    )
                return regroup_next

        # サイト近辺では従来DQNを使う。
        lock_movement = (
            not getattr(self, "facing_head_enabled", False)
            and not time_critical
            and bool(visible_enemies)
            and bool(nearby_allies)
        )

        # Hold the angle during a real gunfight.  Keep the deadline exception:
        # a nearly detonated spike still has to be defused even if an enemy is
        # visible.  A blinded defender is also allowed to reposition.
        if (
            not getattr(self, "facing_head_enabled", False)
            and
            visible_enemies
            and nearby_allies
            and getattr(char, "blind_remaining", 0) <= 0
            and not time_critical
        ):
            nearest = min(
                visible_enemies,
                key=lambda e: max(
                    abs(int(e.pos[0]) - r0), abs(int(e.pos[1]) - c0)
                ),
            )
            facing = facing_towards(char.pos, nearest.pos)
            if facing is not None:
                char.facing = facing
                return list(char.pos), {"facing": facing}
            return list(char.pos)

        obs = self._build_observation(
            char, game_state, chars, enemies, visible_enemies, detonate_timer
        )
        mask = self._action_mask(
            char,
            grid,
            chars,
            lock_movement,
            available_orbs=game_state.get("available_orbs", ()),
        )

        model_obs = obs[:self.model_obs_dim]
        model_mask = mask[:self.model_action_dim]
        obs_t = torch.from_numpy(model_obs).float().unsqueeze(0).to(DEVICE)
        mask_t = torch.from_numpy(model_mask).to(DEVICE)

        with torch.no_grad():
            q_values = self.model(obs_t).squeeze(0).clone()
            q_values[~mask_t] = -1e9
            action_idx = int(torch.argmax(q_values).item())

        facing = self._select_facing(obs, action_idx)
        if facing is not None:
            char.facing = facing

        if self.verbose:
            with open(self._debug_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{char.name},{tuple(char.pos)},planted={tuple(planted_pos)},"
                    f"detonate={detonate_timer:.1f},action={action_idx},"
                    f"Qvals={np.round(q_values.cpu().numpy(), 4).tolist()}\n"
                )

        if action_idx <= 4:
            dr, dc = MOVE_DELTAS[action_idx]
            next_pos = [char.pos[0] + dr, char.pos[1] + dc]
            return next_pos if facing is None else (next_pos, {"facing": facing})

        if action_idx == ACTION_DEFUSE:
            return list(char.pos), "DEFUSE"

        if action_idx == ACTION_ULTIMATE:
            ultimate = build_ultimate_action(
                grid, char, chars, destination=planted_pos
            )
            if ultimate is not None:
                if facing is not None:
                    ultimate = dict(ultimate, facing=facing)
                return list(char.pos), ultimate
            return list(char.pos)

        if action_idx == ACTION_COLLECT_ORB:
            return list(char.pos), "COLLECT_ORB"

        # ACTION_ABILITY: 学習環境(RetakeEnv.apply_ability)がプラント地点
        # 中心に効果を計算する設計だったため、狙点は常にplanted_posとする。
        target_pos = (int(planted_pos[0]), int(planted_pos[1]))
        payload = {"ability": char.ability_name, "target": target_pos}
        if facing is not None:
            payload["facing"] = facing
        return list(char.pos), payload
