"""gc_v1/learning_attacker_guard_gc.py

固定チーム(Xdll/SyouTa/Absol/eKo/SugarZ3ro)専用の
Attacker「guard phase」推論コントローラー(プラント後、解除阻止に特化)。

gc_v1/train_attacker_guard.py で学習した Dueling DQN を読み込み、
run_game.py / battle_logic.py の decide_move(char, game_state) 呼び出し
規約にそのまま乗せられる形で返す。

完全に自己完結。run_game.py / controllers.py / battle_logic.py /
abilities_los.py への依存はなく、必要なLOS計算・BFS距離マップ・行動マスク・
観測構築はこのファイル内に複製する。game_core / map_data_guard_gc からは
定数・マップ文字列のみを参照する(ロジックは参照しない)。

ステータス(GCステータス・タイガーパッシブ込みの確定値)は
character_stats_gc.py 側の定義に基づき、run_game.py の既存エンジン
(combo_awakening.py / game_core.Character)が実戦時に自動適用するため、
本ファイルではステータスの再計算は行わない。char.accuracy等の実値を
そのまま利用する。

--------------------------------------------------------------------------
ガードポジションの割り当てについて(train_attacker_guard.py との相違点):

train_attacker_guard.py はロースター順インデックスで guard_positions[i]
を固定的に割り当てていた(学習環境ではスポーン=ガードポジションだった
ため)。実ゲームではAttackerのスポーン地点(area_3)はガードポジション
とは無関係なので、本ファイルでは learning_defender_search_gc.py の
_ensure_defense_assignment と同じ方式(名前順で決定的に、各キャラの
現在地から最も近い未割当ガード地点を貪欲割り当て)を採用する。観測は
「自分の担当地点までのBFS距離・方向・到着フラグ」という相対情報のみで
構成されているため、この適応でも学習済み重みとの整合性は保たれる設計。

map_data_guard_plant_gc.py の5～9を設置パターン、
map_data_guard_gc.py の同じ数字5～9を対応Guard候補として使う。
実際の設置位置が登録地点と一致しない場合は、BFS歩行距離で最も近い
登録設置パターンへフォールバックする。
--------------------------------------------------------------------------

旧モデルの優先順位ツリー(_build_observation / decide_move):
    1. 解除進行中(game_state["defender_defuse_info"]、LOS不要)
    2. 敵目撃情報(sighting)
    3. どちらも無い場合、担当ガードポジションへ向かい到着後は静止
       (ただし到着後も敵未視認時は多少の索敵行動を許容する設計。
       これは学習側の報酬設計で反映済みで、本ファイル側は行動マスクを
       敵視認時のみ固定するだけで足りる)
--------------------------------------------------------------------------

OBS_DIM=34: train_attacker_guard.py と完全に一致させること。
再学習版(positioning_version=1)は34次元を維持し、予備次元33に設置パターンを
追加する。到着判定は登録座標そのもの。移動・静止はDQNの選択をそのまま使う。

このチーム(5人)で1つのコントローラーインスタンスを共有する想定
(重み共有Dueling DQN)。
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from game_core import (
    BLIND_DURATION_TICKS,
    REVEAL_DURATION_TICKS,
    SPIKE_DETONATION_TICKS,
    DEFUSE_REQUIRED_TICKS,
)
from character_stats_gc import (
    CHARACTER_TABLE as GC_STATS_TABLE,
    GC_ROSTER_ORDER,
)
from map_data_guard_gc import NEW_MAZE_STR as GUARD_MAZE_STR
from map_data_guard_plant_gc import NEW_MAZE_STR as GUARD_PLANT_MAZE_STR
try:
    from .positioning_gc import guard_candidates
except ImportError:
    from positioning_gc import guard_candidates
try:
    from .postplant_utils import postplant_watch_cells
    from .ultimate_tactics_gc import build_ultimate_action, ultimate_context_features, orb_context_features, can_collect_orb
except ImportError:
    from postplant_utils import postplant_watch_cells
    from ultimate_tactics_gc import build_ultimate_action, ultimate_context_features, orb_context_features, can_collect_orb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOVES = [(0, 0)] + CARDINAL  # stay, up, down, left, right
OBS_DIM = 34
ULTIMATE_CONTEXT_OBS_DIM = 38  # v4: ready/combat/objective/urgency cast context.
ORB_OBS_DIM = ULTIMATE_CONTEXT_OBS_DIM + 4
LEGACY_ACTION_DIM = 10  # move_idx(0-4) * 2 + use_ability_flag(0/1)
ACTION_DIM = 11
COLLECT_ORB_ACTION_INDEX = 11
ACTION_DIM = 12
ULTIMATE_ACTION_INDEX = 10

ABILITY_RANGE = 8
GUARD_POS_REACH_RADIUS = 1
SIGHTING_STALENESS_CAP = 20
MAX_TICKS = SPIKE_DETONATION_TICKS  # 55: プラント後の起爆までの時間と一致させる

DEFAULT_MODEL_PATH = (
    "data/attacker_guard_gc_data/" "dqn_attacker_guard_gc_best_by_eval.pt"
)


# ---------------------------------------------------------------------------
# Dueling DQN (gc_v1/train_attacker_guard.py と同一構造)
# ---------------------------------------------------------------------------
class AttackerGuardDuelingDQN(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, action_dim)
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value_head(f)
        a = self.advantage_head(f)
        return v + (a - a.mean(dim=1, keepdim=True))


# ---------------------------------------------------------------------------
# 補助関数(LOS / BFS)。abilities_los.py / train_attacker_guard.py の複製実装。
#
# 注意: game_state には smokes(煙リスト)が含まれないため、この推論用LOSは
# 壁のみを考慮する(スモークによる遮蔽は考慮しない)。同様に own_smoke_active
# も常に0として扱う(learning_defender_search_gc.py と同じ簡略化方針)。
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
    """goalから各床マスへの最短距離マップ(壁越え不可)。
    train_attacker_guard.py の bfs_distance_map と同一ロジック。"""
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
            if (
                0 <= nr < height
                and 0 <= nc < width
                and grid[nr, nc] != 1
                and dist[nr, nc] == -1
            ):
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def _bfs_best_direction(dist_map, grid, r0, c0):
    """dist_map上で、(r0,c0)から見て最も距離が縮む隣接方向(dr,dc)を返す。
    到達不能・移動不要・dist_map未確定なら(0,0)を返す。
    train_attacker_guard.py の bfs_best_direction と同一ロジック。"""
    if dist_map is None:
        return 0, 0
    height, width = grid.shape
    cur = dist_map[r0, c0]
    if cur < 0:
        return 0, 0
    best_dr, best_dc, best_d = 0, 0, cur
    for dr, dc in CARDINAL:
        nr, nc = r0 + dr, c0 + dc
        if 0 <= nr < height and 0 <= nc < width and dist_map[nr, nc] >= 0:
            if dist_map[nr, nc] < best_d:
                best_d = dist_map[nr, nc]
                best_dr, best_dc = dr, dc
    return best_dr, best_dc


def _parse_grid(maze_str):
    lines = [l.strip() for l in maze_str.strip("\n").split("\n") if l.strip()]
    return np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)


GUARD_PATTERN_MARKERS = tuple(range(5, 10))

_GUARD_PATTERN_GRID = _parse_grid(GUARD_MAZE_STR)
_GUARD_PLANT_PATTERN_GRID = _parse_grid(GUARD_PLANT_MAZE_STR)

if _GUARD_PATTERN_GRID.shape != _GUARD_PLANT_PATTERN_GRID.shape:
    raise RuntimeError(
        "map_data_guard_gc.py と map_data_guard_plant_gc.py の"
        "マップサイズが一致しません。"
    )

_PLANT_PATTERN_CELLS = {
    marker: [
        (r, c)
        for r in range(_GUARD_PLANT_PATTERN_GRID.shape[0])
        for c in range(_GUARD_PLANT_PATTERN_GRID.shape[1])
        if int(_GUARD_PLANT_PATTERN_GRID[r, c]) == marker
    ]
    for marker in GUARD_PATTERN_MARKERS
}

_GUARD_PATTERN_CELLS = {
    marker: [
        (r, c)
        for r in range(_GUARD_PATTERN_GRID.shape[0])
        for c in range(_GUARD_PATTERN_GRID.shape[1])
        if int(_GUARD_PATTERN_GRID[r, c]) == marker
    ]
    for marker in GUARD_PATTERN_MARKERS
}

_ACTIVE_GUARD_PATTERNS = [
    marker for marker in GUARD_PATTERN_MARKERS if _PLANT_PATTERN_CELLS[marker]
]

if not _ACTIVE_GUARD_PATTERNS:
    raise RuntimeError("map_data_guard_plant_gc.py に設置パターン5～9がありません。")

for _marker in _ACTIVE_GUARD_PATTERNS:
    if not _GUARD_PATTERN_CELLS[_marker]:
        raise RuntimeError(
            f"設置パターン{_marker}に対応するGuard位置{_marker}が"
            "map_data_guard_gc.pyにありません。"
        )


def _resolve_guard_pattern(grid, planted_pos):
    """実際のplanted_posから対応Guardパターンを解決する。

    1) 登録設置位置と完全一致
    2) 不一致ならBFS歩行距離が最も近い登録設置位置
    3) BFSで到達不能ならChebyshev距離
    """
    planted_pos = (int(planted_pos[0]), int(planted_pos[1]))

    for marker in _ACTIVE_GUARD_PATTERNS:
        if planted_pos in _PLANT_PATTERN_CELLS[marker]:
            return marker

    dist_from_actual = _bfs_distance_map(grid, planted_pos)
    best = None
    for marker in _ACTIVE_GUARD_PATTERNS:
        for ref_pos in _PLANT_PATTERN_CELLS[marker]:
            d = int(dist_from_actual[ref_pos[0], ref_pos[1]])
            if d < 0:
                continue
            key = (d, marker, ref_pos[0], ref_pos[1])
            if best is None or key < best[0]:
                best = (key, marker)

    if best is not None:
        return best[1]

    return min(
        _ACTIVE_GUARD_PATTERNS,
        key=lambda marker: min(
            max(abs(ref[0] - planted_pos[0]), abs(ref[1] - planted_pos[1]))
            for ref in _PLANT_PATTERN_CELLS[marker]
        ),
    )


def _ability_charge(char):
    """ロールに対応する残チャージ数を取得する。HUNT(アビリティ無し)は常に0。"""
    return {
        "FLASH": getattr(char, "flash_charges", 0),
        "SMOKE": getattr(char, "smoke_charges", 0),
        "RECON": getattr(char, "recon_charges", 0),
    }.get(char.ability_name, 0)


# ---------------------------------------------------------------------------
# チーム共有メモリ(敵目撃情報のみ管理。train_attacker_guard.pyのGuardMemoryと
# 同一ロジック。解除中フラグはgame_state["defender_defuse_info"]から
# LOS不要で直接取得できるため、ここでは追跡しない)
# ---------------------------------------------------------------------------
class _TeamMemory:
    def __init__(self):
        self.last_seen_enemy = None  # {"pos": (r, c), "name": str, "tick_ago": int}

    def reset(self):
        self.last_seen_enemy = None

    def update(self, grid, my_team, chars, smoke_cells=()):
        allies = [c for c in chars if c.team == my_team and c.is_alive]
        enemies = [c for c in chars if c.team != my_team]

        visible_enemies = []
        for a in allies:
            for e in enemies:
                if not e.is_alive:
                    continue
                if (
                    _has_los(grid, tuple(a.pos), tuple(e.pos), smoke_cells)
                    and e not in visible_enemies
                ):
                    visible_enemies.append(e)

        if visible_enemies:
            tracked = None
            if self.last_seen_enemy is not None:
                tracked_name = self.last_seen_enemy.get("name")
                tracked = next(
                    (e for e in visible_enemies if e.name == tracked_name), None
                )
            if tracked is None:
                # 解除中の敵がいれば最優先、いなければ最も近い敵を追跡する
                defusing = [
                    e for e in visible_enemies if getattr(e, "defuse_timer", 0) > 0
                ]
                pool = defusing if defusing else visible_enemies
                tracked = min(
                    pool,
                    key=lambda e: (
                        min(
                            max(abs(e.pos[0] - a.pos[0]), abs(e.pos[1] - a.pos[1]))
                            for a in allies
                        )
                        if allies
                        else 0
                    ),
                )
            self.last_seen_enemy = {
                "pos": tuple(tracked.pos),
                "name": tracked.name,
                "tick_ago": 0,
            }
        elif self.last_seen_enemy is not None:
            self.last_seen_enemy["tick_ago"] += 1
            if self.last_seen_enemy["tick_ago"] > SIGHTING_STALENESS_CAP:
                self.last_seen_enemy = None


# ---------------------------------------------------------------------------
# 推論コントローラー
# ---------------------------------------------------------------------------
class LearningAttackerGuardGCController:
    """gc_v1固定チーム専用、プラント後フェーズ(ガード)のAttacker AI。

    Attackerチーム全員(5人)がこの1インスタンスを共有して呼び出される。

    ステータス(コンボ・タイガーパッシブ込みの確定値)は run_game.py の
    既存エンジンが character_stats_gc.py を経由して自動適用済みの
    char オブジェクトをそのまま利用する(本ファイル側では再計算しない)。
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, greedy=True, verbose=False):
        self.greedy = greedy
        self.verbose = verbose
        self.model = AttackerGuardDuelingDQN().to(DEVICE)
        self.positioning_version = 0
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            self.positioning_version = int(checkpoint.get("positioning_version", 0))
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            action_dim = int(checkpoint.get("n_actions", LEGACY_ACTION_DIM))
            obs_dim = int(checkpoint.get("obs_dim", OBS_DIM))
            expected_actions = ACTION_DIM if self.positioning_version >= 5 else (11 if self.positioning_version >= 3 else LEGACY_ACTION_DIM)
            expected_obs_dim = (ORB_OBS_DIM if self.positioning_version >= 5 else
                                ULTIMATE_CONTEXT_OBS_DIM
                                if self.positioning_version >= 4 else OBS_DIM)
            if action_dim != expected_actions:
                raise ValueError(
                    f"guard action count mismatch: {action_dim} != {expected_actions}"
                )
            if obs_dim != expected_obs_dim:
                raise ValueError(
                    f"guard observation count mismatch: {obs_dim} != {expected_obs_dim}"
                )
            self.model = AttackerGuardDuelingDQN(
                obs_dim=obs_dim, action_dim=action_dim
            ).to(DEVICE)
            self.model.load_state_dict(state_dict)
            if verbose:
                print(f"[LearningAttackerGuardGCController] loaded: {model_path}")
        except Exception as exc:
            print(
                f"[LOAD ERROR] attacker guard(gc) model '{model_path}' の読込に失敗: {exc}"
            )
        self.model.eval()

        self.team_memory = _TeamMemory()

        # プラント地点(planted_pos)は1ラウンド中不変のため、
        # BFS距離マップはラウンド開始後の初回呼び出し時に1度だけ計算する。
        self.spike_dist_map = None
        self._planted_pos_cache = None

        # ガードポジションの貪欲割り当て(名前順・現在地から最も近い未割当地点)。
        # プラント地点が確定した最初のtickで1度だけ行う。
        self._assignment_done = False
        self._active_guard_pattern = None
        self._assigned_guard_positions = {}  # char_name -> (r,c)
        self._assigned_dist_maps = {}  # char_name -> np.ndarray(BFS距離マップ)

        self.sighting_dist_map = None
        self._sighting_dist_map_source = (
            None  # 再計算要否判定用: 直前のlast_seen_enemy["pos"]
        )

        self._processed_this_tick = set()
        self._debug_log_path = "attacker_guard_gc_debug.log"

    # -- ラウンド開始時のリセット -----------------------------------------
    def reset_round(self):
        self.team_memory.reset()
        self.spike_dist_map = None
        self._planted_pos_cache = None
        self._assignment_done = False
        self._active_guard_pattern = None
        self._assigned_guard_positions.clear()
        self._assigned_dist_maps.clear()
        self.sighting_dist_map = None
        self._sighting_dist_map_source = None
        self._processed_this_tick.clear()

    def _ensure_spike_dist_map(self, grid, planted_pos):
        planted_pos = (int(planted_pos[0]), int(planted_pos[1]))
        if planted_pos == self._planted_pos_cache and self.spike_dist_map is not None:
            return
        self.spike_dist_map = _bfs_distance_map(grid, planted_pos)
        self._planted_pos_cache = planted_pos

    def _ensure_guard_assignment(self, char, grid, chars, planted_pos):
        """実際の設置位置に対応するGuardパターンを選び、味方へ割り当てる。

        登録設置位置と完全一致しない緊急設置でも、
        BFS歩行距離で最も近い登録設置パターンへフォールバックする。
        対応Guard候補が5人より多ければ、各キャラの現在地から近い未割当地点を
        名前順で貪欲に割り当てる。少ない場合は再利用する。
        """
        if self._assignment_done:
            return

        marker = _resolve_guard_pattern(grid, planted_pos)
        self._active_guard_pattern = marker
        candidates = list(_GUARD_PATTERN_CELLS[marker])

        teammates = [c for c in chars if c.team == char.team and c.is_alive]
        teammates = sorted(teammates, key=lambda c: c.name)
        if getattr(self, "positioning_version", 0) >= 1:
            candidates = guard_candidates(grid, marker, len(teammates))

        # 各候補へのBFS距離マップを一度だけ作り、実際の歩行距離で割当する。
        candidate_maps = {pos: _bfs_distance_map(grid, pos) for pos in candidates}

        pool = list(candidates)
        for teammate in teammates:
            usable = pool if pool else candidates

            def _distance_to_candidate(pos):
                d = int(candidate_maps[pos][int(teammate.pos[0]), int(teammate.pos[1])])
                if d < 0:
                    return 10**9
                return d

            chosen = min(
                usable,
                key=lambda pos: (
                    _distance_to_candidate(pos),
                    pos[0],
                    pos[1],
                ),
            )

            if chosen in pool:
                pool.remove(chosen)

            self._assigned_guard_positions[teammate.name] = chosen
            self._assigned_dist_maps[teammate.name] = candidate_maps[chosen]

        self._assignment_done = True

        if self.verbose:
            print(
                f"[GC GUARD] planted={tuple(planted_pos)} "
                f"pattern={marker} assignments={self._assigned_guard_positions}"
            )

    def _maybe_advance_tick(self, char, grid, chars, smoke_cells=()):
        """同じキャラクターが再び呼ばれたら新しいtickに入ったとみなし、
        チーム共有メモリを1回だけ更新する。"""
        if not self._processed_this_tick or char.name in self._processed_this_tick:
            self._processed_this_tick.clear()
            self.team_memory.update(grid, char.team, chars, smoke_cells)
        self._processed_this_tick.add(char.name)

    def _update_sighting_dist_map(self, grid):
        sighting_pos = (
            self.team_memory.last_seen_enemy["pos"]
            if self.team_memory.last_seen_enemy is not None
            else None
        )
        if sighting_pos != self._sighting_dist_map_source:
            self.sighting_dist_map = (
                _bfs_distance_map(grid, sighting_pos)
                if sighting_pos is not None
                else None
            )
            self._sighting_dist_map_source = sighting_pos

    def _guard_position_step(self, char, grid, chars):
        """Return one step toward the assigned post-plant guard position.

        This is deliberately used only before arrival.  It prevents the
        spike-watch rule from stopping a player in an intermediate position.
        """
        target = self._assigned_guard_positions.get(char.name)
        dist_map = self._assigned_dist_maps.get(char.name)
        if target is None or dist_map is None:
            return None
        r, c = int(char.pos[0]), int(char.pos[1])
        if int(dist_map[r, c]) <= GUARD_POS_REACH_RADIUS:
            return None

        occupied = {
            tuple(map(int, other.pos))
            for other in chars
            if other is not char and getattr(other, "is_alive", True)
        }
        current = int(dist_map[r, c])
        candidates = []
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                continue
            if grid[nr, nc] == 1 or (nr, nc) in occupied:
                continue
            distance = int(dist_map[nr, nc])
            if 0 <= distance < current:
                candidates.append((distance, (nr, nc)))
        if not candidates:
            return None
        return list(min(candidates, key=lambda item: (item[0], item[1]))[1])

    def _active_defuse_info(self, game_state):
        """game_state["defender_defuse_info"](LOS不要、battle_logic.py提供)
        から、現在解除中の敵がいれば進捗率を返す。"""
        info = game_state.get("defender_defuse_info") or {}
        for _name, (timer, required) in info.items():
            if timer > 0:
                return {
                    "name": _name,
                    "progress_ratio": min(timer / required if required else 0.0, 1.0)
                }
        return None

    # -- 観測構築 ----------------------------------------------------------
    # train_attacker_guard.py の build_observation() と要素・並び順を
    # 完全一致させること。
    def _build_observation(
        self, char, game_state, unit_has_spike_los, active_defuse_info, detonate_timer
    ):
        grid = game_state["grid"]
        chars = game_state.get("chars", [])
        height, width = grid.shape
        r0, c0 = int(char.pos[0]), int(char.pos[1])

        obs_dim = (ORB_OBS_DIM if getattr(self, "positioning_version", 0) >= 5 else
                   ULTIMATE_CONTEXT_OBS_DIM
                   if getattr(self, "positioning_version", 0) >= 4 else OBS_DIM)
        obs = np.zeros(obs_dim, dtype=np.float32)

        obs[0] = r0 / height
        obs[1] = c0 / width
        obs[2] = char.hp / char.max_hp if char.max_hp else 0.0
        obs[3] = 1.0 if getattr(char, "moved_this_tick", False) else 0.0

        ability_index = {"SMOKE": 4, "FLASH": 5, "RECON": 6, "HUNT": 7}.get(
            char.ability_name, 7
        )
        obs[ability_index] = 1.0
        obs[8] = 1.0 if _ability_charge(char) > 0 else 0.0

        enemies = [e for e in chars if e.team != char.team]
        visible_enemies = [
            e for e in enemies if e.is_alive and _has_los(
                grid, char.pos, e.pos, game_state.get("smoke_cells", ())
            )
        ]
        obs[9] = 1.0 if visible_enemies else 0.0

        teammates = [
            c for c in chars if c.team == char.team and c is not char and c.is_alive
        ]
        obs[10] = len(teammates) / 4.0
        if teammates:
            nearest_d = min(
                max(abs(t.pos[0] - r0), abs(t.pos[1] - c0)) for t in teammates
            )
            obs[11] = min(nearest_d, height) / height

        obs[12] = (
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
        )
        # 再学習版は学習時と同じく、smoke_cellsによる展開中フラグを使う。
        obs[13] = (float(bool(game_state.get("smoke_cells", ())))
                   if getattr(self, "positioning_version", 0) >= 1 else 0.0)

        dist_here = (
            self.spike_dist_map[r0, c0] if self.spike_dist_map is not None else -1
        )
        obs[14] = min(
            dist_here if dist_here >= 0 else height + width, height + width
        ) / (height + width)
        best_dr, best_dc = _bfs_best_direction(self.spike_dist_map, grid, r0, c0)
        obs[15] = float(best_dr)
        obs[16] = float(best_dc)
        if active_defuse_info is not None and getattr(self, "positioning_version", 0) >= 2:
            obs[17] = float(any(e.name == active_defuse_info["name"] for e in visible_enemies))
        else:
            obs[17] = 1.0 if unit_has_spike_los else 0.0

        if self.team_memory.last_seen_enemy is not None:
            ls = self.team_memory.last_seen_enemy
            obs[18] = 1.0
            best_dr, best_dc = _bfs_best_direction(self.sighting_dist_map, grid, r0, c0)
            obs[19] = float(best_dr)
            obs[20] = float(best_dc)
            obs[21] = (
                min(ls["tick_ago"], SIGHTING_STALENESS_CAP) / SIGHTING_STALENESS_CAP
            )

        obs[22] = len(visible_enemies) / 5.0
        if visible_enemies:
            nearest_enemy = min(
                visible_enemies,
                key=lambda e: max(abs(e.pos[0] - r0), abs(e.pos[1] - c0)),
            )
            obs[23] = (nearest_enemy.pos[0] - r0) / height
            obs[24] = (nearest_enemy.pos[1] - c0) / width
            dist = max(abs(nearest_enemy.pos[0] - r0), abs(nearest_enemy.pos[1] - c0))
            obs[25] = min(dist, height) / height

        if active_defuse_info is not None:
            obs[26] = 1.0
            obs[27] = active_defuse_info["progress_ratio"]
        obs[28] = min(detonate_timer, MAX_TICKS) / MAX_TICKS

        dist_map = self._assigned_dist_maps.get(char.name)
        bfs_dist = dist_map[r0, c0] if dist_map is not None else -1
        if bfs_dist < 0:
            bfs_dist = height + width
        obs[29] = min(bfs_dist, height + width) / (height + width)
        best_dr, best_dc = _bfs_best_direction(dist_map, grid, r0, c0)
        obs[30] = float(best_dr)
        obs[31] = float(best_dc)
        radius = 0 if getattr(self, "positioning_version", 0) >= 1 else GUARD_POS_REACH_RADIUS
        obs[32] = 1.0 if bfs_dist <= radius else 0.0

        obs[33] = ((self._active_guard_pattern - 4) / 5.0
                   if getattr(self, "positioning_version", 0) >= 1 else 0.0)

        if getattr(self, "positioning_version", 0) >= 4:
            progress = (active_defuse_info.get("progress_ratio", 0.0)
                        if active_defuse_info is not None else 0.0)
            obs[OBS_DIM:ULTIMATE_CONTEXT_OBS_DIM] = ultimate_context_features(
                char,
                engaged=bool(visible_enemies),
                objective_window=active_defuse_info is not None,
                urgent=(progress >= 0.5 or detonate_timer <= 12),
            )
        if getattr(self, "positioning_version", 0) >= 5:
            obs[ULTIMATE_CONTEXT_OBS_DIM:ORB_OBS_DIM] = orb_context_features(
                char, game_state.get("available_orbs", ())
            )

        return obs, visible_enemies

    # -- 行動マスク ---------------------------------------------------------
    # train_attacker_guard.py の build_action_mask() と同一ロジック。
    def _action_mask(self, char, grid, chars, lock_movement=False, ultimate_target=None, available_orbs=()):
        """lock_movement=True の場合、stay以外の移動を禁止する。
        敵を視認している間は静止させ、射撃の当たりやすさを優先する
        (「多少の索敵は許容するが強く抑制」は学習側の報酬設計で反映済み。
        本ファイル側のマスクは敵視認時のみの固定で足りる)。"""
        action_count = ACTION_DIM if self.positioning_version >= 5 else (11 if self.positioning_version >= 3 else LEGACY_ACTION_DIM)
        mask = np.ones(action_count, dtype=bool)
        r, c = int(char.pos[0]), int(char.pos[1])
        occupied = {
            tuple(o.pos)
            for o in chars
            if o is not char and getattr(o, "is_alive", True)
        }

        for move_idx, (dr, dc) in enumerate(MOVES):
            if lock_movement and move_idx != 0:
                mask[move_idx * 2] = False
                mask[move_idx * 2 + 1] = False
                continue
            nr, nc = r + dr, c + dc
            walkable = (
                0 <= nr < grid.shape[0]
                and 0 <= nc < grid.shape[1]
                and grid[nr, nc] != 1
                and (nr, nc) not in occupied
            )
            if not walkable:
                mask[move_idx * 2] = False
                mask[move_idx * 2 + 1] = False

        if _ability_charge(char) <= 0 or char.ability_name == "HUNT":
            for move_idx in range(5):
                mask[move_idx * 2 + 1] = False

        if self.positioning_version >= 3:
            mask[ULTIMATE_ACTION_INDEX] = build_ultimate_action(
                grid, char, chars, destination=ultimate_target
            ) is not None
        if self.positioning_version >= 5:
            mask[COLLECT_ORB_ACTION_INDEX] = can_collect_orb(
                char, available_orbs
            )

        return mask

    # -- メイン ----------------------------------------------------------
    def _select_action(self, obs, mask):
        device = DEVICE
        with torch.no_grad():
            q = self.model(torch.from_numpy(obs).float().unsqueeze(0).to(device)).squeeze(0)
            q = q.masked_fill(~torch.from_numpy(mask).to(device), -1e9)
        return int(torch.argmax(q).item())

    def decide_move(self, char, game_state):
        if not char.is_alive:
            return list(char.pos)

        is_planted = bool(game_state.get("is_planted", False))
        planted_pos = game_state.get("planted_pos")

        # このコントローラーはプラント後フェーズ(ガード)専用。
        # 万一プラント前に呼ばれた場合は安全側としてその場に留まる
        # (上位のフェーズ切替側で carry/escort/retrieve フェーズの
        # コントローラーへ委譲する想定)。
        if not is_planted or planted_pos is None:
            return list(char.pos)

        grid = game_state["grid"]
        chars = game_state.get("chars", [])
        detonate_timer = float(game_state.get("detonate_timer", 0.0))

        self._ensure_spike_dist_map(grid, planted_pos)
        self._ensure_guard_assignment(char, grid, chars, planted_pos)
        self._maybe_advance_tick(char, grid, chars, game_state.get("smoke_cells", ()))
        self._update_sighting_dist_map(grid)

        watch_cells = postplant_watch_cells(grid, planted_pos)
        unit_has_spike_los = any(
            _has_los(grid, tuple(char.pos), cell, game_state.get("smoke_cells", ()))
            for cell in watch_cells
        )
        active_defuse_info = self._active_defuse_info(game_state)

        obs, visible_enemies = self._build_observation(
            char, game_state, unit_has_spike_los, active_defuse_info, detonate_timer
        )

        # Reach the learned strong position first.  The old spike-watch
        # override below is intentionally not allowed to interrupt this route.
        # A visible enemy or active defuse remains higher priority so that a
        # player does not walk into an active duel/defuse situation blindly.
        learned_positioning = getattr(self, "positioning_version", 0) >= 1
        if not learned_positioning and not visible_enemies and active_defuse_info is None:
            guard_step = self._guard_position_step(char, grid, chars)
            if guard_step is not None:
                if self.verbose:
                    print(
                        f"[ATTACKER GUARD GC] {char.name} "
                        f"moving_to_guard={tuple(self._assigned_guard_positions[char.name])} "
                        f"step={tuple(guard_step)} spike_los={unit_has_spike_los}"
                    )
                return guard_step

        assigned_guard_positions = getattr(self, "_assigned_guard_positions", {})
        ultimate_target = assigned_guard_positions.get(char.name, planted_pos)
        mask = self._action_mask(
            char, grid, chars,
            lock_movement=bool(visible_enemies) and not learned_positioning,
            ultimate_target=ultimate_target,
            available_orbs=game_state.get("available_orbs", ()),
        )

        action_idx = self._select_action(obs, mask)

        if action_idx == COLLECT_ORB_ACTION_INDEX:
            return list(char.pos), "COLLECT_ORB"

        if getattr(self, "positioning_version", 0) >= 3 and action_idx == ULTIMATE_ACTION_INDEX:
            ultimate = build_ultimate_action(
                grid, char, chars, destination=ultimate_target
            )
            if ultimate is not None:
                return list(char.pos), ultimate

        if self.verbose:
            device = DEVICE
            with torch.no_grad():
                q_values = self.model(torch.from_numpy(obs).float().unsqueeze(0).to(device)).squeeze(0)
                q_values = q_values.masked_fill(~torch.from_numpy(mask).to(device), -1e9)
            with open(self._debug_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{char.name},{tuple(char.pos)},planted={tuple(planted_pos)},"
                    f"detonate={detonate_timer:.1f},action={action_idx},"
                    f"Qvals={np.round(q_values.cpu().numpy(), 4).tolist()}\n"
                )

        move_idx, use_ability_int = divmod(action_idx, 2)
        use_ability = bool(use_ability_int)
        move_offset = MOVES[move_idx]
        # 設置マスまたは周囲8マスへ射線が通る場所を確保できたら、
        # 解除に来る敵を待ち構える。能力使用はその場で実行する。
        if (not learned_positioning and unit_has_spike_los
                and not use_ability and active_defuse_info is None):
            move_offset = MOVES[0]
        next_pos = [char.pos[0] + move_offset[0], char.pos[1] + move_offset[1]]

        if self.verbose:
            print(
                f"[ATTACKER GUARD GC] {char.name} pos={tuple(char.pos)} "
                f"action={action_idx} move={move_offset} ability={use_ability}"
            )

        if not use_ability:
            return next_pos

        # アビリティ使用: 狙点は「射線内の最近接の敵」を最優先、
        # いなければチーム共有メモリの直近目撃座標、それも無ければ
        # プラント地点そのものを予防的な狙点とする
        # (train_attacker_guard.py の ability_requests 組み立てロジックに準拠)。
        target_pos = None
        if visible_enemies:
            nearest = min(
                visible_enemies,
                key=lambda e: max(
                    abs(e.pos[0] - char.pos[0]), abs(e.pos[1] - char.pos[1])
                ),
            )
            dist = max(
                abs(nearest.pos[0] - char.pos[0]), abs(nearest.pos[1] - char.pos[1])
            )
            if dist <= ABILITY_RANGE:
                target_pos = (int(nearest.pos[0]), int(nearest.pos[1]))
        elif self.team_memory.last_seen_enemy is not None:
            pos = self.team_memory.last_seen_enemy["pos"]
            target_pos = (int(pos[0]), int(pos[1]))
        else:
            target_pos = (int(planted_pos[0]), int(planted_pos[1]))

        if target_pos is None:
            return next_pos

        return next_pos, {"ability": char.ability_name, "target": target_pos}
