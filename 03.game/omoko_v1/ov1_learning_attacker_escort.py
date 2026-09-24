"""omoko_v1/learning_attacker_escort.py

固定チーム専用の
Attacker Carry Phase「escort(護衛)」推論コントローラー。

train_attacker_escort.py(omoko_v1版)で学習した、Escort役4体用の
Dueling DQNモデル(dict形式チェックポイント)を使い、各escortキャラクターの
移動・アビリティ判断を行う。

【4体は重み共有】
train_attacker_escort.py側は4体のescortが同一ネットワークの重みを
共有して学習している(パラメータ共有方式)。そのため、run_game.py側でも
carry以外の4キャラクターすべてに「同じOv1LearningAttackerEscortController
インスタンス」を割り当てる想定である(インスタンスは1つ、decide_moveは
キャラクターごとに呼ばれる)。

completely self-contained: run_game.py / controllers.py / battle_logic.py /
abilities_los.py は一切importしない。必要なロジックはすべてこのファイル内に
複製する。run_game.py / controllers.py は変更しない。

【観測ベクトルはtrain_attacker_escort.py(omoko_v1版)のEscortEnv._get_obs()と
完全に一致させる必要がある(全52次元)。ここがズレると学習結果が正しく
反映されない。汎用版(旧learning_attacker_escort.py)からの変更点は、
ABILITY_TYPESに"HUNT"が加わったアビリティonehotの3種→4種化(OBS_DIM: 36→41)、
「サイト-エスコート-キャリアー」順判定用3次元追加(41→44)、
facing onehotの4→8方向化(44→52)。】

【本番環境との差異・既知の制約】
1. キャリアーの「進むべき方向」予測
   学習環境では、キャラクター同士の衝突を考慮しない固定BFS経路
   (壁のみを障害物とした経路)をキャリアーの行動基準にしていた。
   推論側もこれに合わせ、味方・敵の位置を無視した「壁のみのBFS勾配」で
   キャリアーの理想進行方向を毎tick再計算する。これにより
   「自分がその理想進行方向のマスに立っているかどうか」を
   ブロック中フラグとして使える。

2. スモークによる射線遮蔽は考慮できない
   battle_logic.pyのmove_characterが渡すgame_stateには、現在有効な
   スモークの情報が含まれていないため、視界判定は壁のみを考慮した
   Bresenham判定になる(学習環境よりやや楽観的)。

3. アビリティの発動判定
   char.ability_name / char.flash_charges / char.smoke_charges /
   char.recon_chargesなど、実際のCharacterオブジェクトが持つ値を
   そのまま使う。HUNT(タイガー)役はgame_core.pyの仕様上これらが
   全て0で初期化されるため、total_charges<=0判定で自動的に
   アビリティ行動がマスクされる(追加分岐は不要)。射程内に有効な
   標的がいない場合は、実際のアビリティチャージを無駄撃ちしないよう
   STAYにフォールバックする。

run_game.pyからは他のlearning_attacker_*.py系コントローラーと同様の
インターフェース(decide_move(char, game_state) -> next_pos または
(next_pos, {"ability": ..., "target": (r, c)}))で呼び出される想定。
"""

import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from character_stats import CHARACTER_TABLE as STATS_TABLE
from ov1_roster import ROSTER_ORDER
from ov1_map_data_escort import NEW_MAZE_STR as ESCORT_MAZE_STR
from ov1_map_data_defender_simulate import NEW_MAZE_STR as DEFENDER_SIM_MAZE_STR
from ov1_train_attacker_escort import (
    LINEUP_ABILITY_MARKERS,
    LINEUP_TRIGGER_RADIUS,
    FLASH_RANGE,
    DEFENDER_SPAWN_VALUE,
)
from ov1_ultimate_training import (
    ESCORT_ORB_MAX_PATH_DISTANCE,
    ESCORT_ORB_MIN_REMAINING_TICKS,
    orb_context,
    orb_priority,
    ultimate_context,
    ultimate_ready,
)

# ---------------------------------------------------------------------------
# 行動定義(train_attacker_escort.py の EscortEnv と同一でなければならない)
# ---------------------------------------------------------------------------
(
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY,
    ACTION_ABILITY, ACTION_ULTIMATE, ACTION_ORB,
) = range(8)
BASE_N_ACTIONS = 8
FACING_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
N_ACTIONS = BASE_N_ACTIONS * len(FACING_DIRS)
_MOVE_DELTA = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
    ACTION_STAY: (0, 0),
}


def decode_action(action_idx):
    """action_idx = base_idx(0-5) * 4 + facing_idx(0-3)。
    train_attacker_escort.py(omoko_v1版) EscortEnv.decode_action() と
    同一規約。戻り値は (base_action, facing)。"""
    idx = int(action_idx)
    base_idx, facing_idx = divmod(idx, len(FACING_DIRS))
    return base_idx, FACING_DIRS[facing_idx]


def _facing_towards(from_pos, to_pos):
    """Return the eight-way facing direction from ``from_pos`` to ``to_pos``."""
    dr = int(to_pos[0]) - int(from_pos[0])
    dc = int(to_pos[1]) - int(from_pos[1])
    vertical = "N" if dr < 0 else "S" if dr > 0 else ""
    horizontal = "E" if dc > 0 else "W" if dc < 0 else ""
    return vertical + horizontal if vertical and horizontal else vertical or horizontal or "N"

BLIND_DURATION_TICKS = 3
REVEAL_DURATION_TICKS = 5
# HUNT(タイガー/Tortlilyan)を含む4種。HUNTはアビリティ行動を持たないため
# total_charges<=0判定で自動的にマスクされる(game_core.pyの仕様上、
# タイガー役はflash/smoke/recon_chargesが全て0で初期化されるため)。
ABILITY_TYPES = ("FLASH", "RECON", "SMOKE", "HUNT")
# RECON/SMOKEの射程。FLASHはFLASH_RANGE(=飛翔速度×最大飛翔Tick)を別途使う。
ABILITY_RANGE = 6


DIST_BAND_MIN = 2
DIST_BAND_MAX = 7
DIST_NORM_MAX = 15.0

OBS_DIM = 73  # train_attacker_escort.py(omoko_v1版) EscortEnv._obs_dim() と一致
              # (facing onehotが4→8方向化により44→52、チーム共有目撃情報4次元追加により52→56、
              #  警戒点の推奨立ち位置(大文字ヒント)方向3次元追加により59→62)

SIGHTING_STALENESS_CAP = 20  # train_attacker_escort.pyのSIGHTING_STALENESS_CAPと同一値

PATH_EQUALITY_TOL = 0.5  # train_attacker_escort.pyと同一の許容誤差

ESCORT_HOLD_TICKS_DEFAULT = 3  # train_attacker_escort.pyのESCORT_HOLD_TICKSと同値。
                                 # チェックポイントにhold_ticksが保存されていればそちらを優先する。


def _parse_defender_watch_points(defender_maze_str, escort_maze_str):
    """守備配置想定マップ(ov1_map_data_defender_simulate.py)上のDefenderスポーン
    候補(DEFENDER_SPAWN_VALUE)のうち、escort map側で壁でないセルを、
    escortが警戒してfacingする対象点として返す。"""
    defender_lines = [line.strip() for line in defender_maze_str.strip("\n").split("\n") if line.strip()]
    escort_lines = [line.strip() for line in escort_maze_str.strip("\n").split("\n") if line.strip()]
    if len(defender_lines) != len(escort_lines):
        raise ValueError("defender simulate mapとescort mapの行数が一致していません")
    points = []
    for r, (d_line, e_line) in enumerate(zip(defender_lines, escort_lines)):
        if len(d_line) != len(e_line):
            raise ValueError("defender simulate mapとescort mapの行長が一致していません")
        for c, marker in enumerate(d_line):
            if marker == str(DEFENDER_SPAWN_VALUE) and e_line[c] != "1":
                points.append((r, c))
    return points


def _parse_ability_lineup_points(maze_str):
    """マップ上のS/R/Fマーカーから、アビリティ種別ごとの定点座標一覧を作る
    (train_attacker_escort.py _parse_escort_map と同一の対応表を使う)。"""
    lines = [line.strip() for line in maze_str.strip("\n").split("\n") if line.strip()]
    if not lines or len({len(line) for line in lines}) != 1:
        raise ValueError("escort mapの行長が一致していません")
    points = {ability: [] for ability in LINEUP_ABILITY_MARKERS.values()}
    for r, line in enumerate(lines):
        for c, marker in enumerate(line):
            if marker in LINEUP_ABILITY_MARKERS:
                points[LINEUP_ABILITY_MARKERS[marker]].append((r, c))
    return points


WATCH_POINTS = tuple(sorted(_parse_defender_watch_points(DEFENDER_SIM_MAZE_STR, ESCORT_MAZE_STR)))
ABILITY_LINEUP_POINTS = _parse_ability_lineup_points(ESCORT_MAZE_STR)
WATCH_POINT_HINTS = {}
POSITION_HINT_DECAY = (1.0, 0.9, 0.75)  # train_attacker_escort.pyのPOSITION_HINT_DECAYと同一値


# ---------------------------------------------------------------------------
# 汎用ヘルパー(train_attacker_escort.py と同一ロジック)
# ---------------------------------------------------------------------------
def _chebyshev(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


def _roster_rank(name):
    """train側(EscortEnv)のescort index順と一致させるための順位付け。

    train側のalive escortランクは、
    escort_names = [name for name in ROSTER_ORDER if name != carrier_name]
    の並び(=ROSTER_ORDER上の並び順)そのままである。名前のアルファベット順
    ではないため、推論側も同じ基準で順位付けする必要がある。
    """
    try:
        return ROSTER_ORDER.index(str(name))
    except ValueError:
        return len(ROSTER_ORDER)


def _line_cells(p1, p2):
    """Bresenham法で2点間のセル列を返す(abilities_los.py と同一ロジック)。"""
    y0, x0 = int(p1[0]), int(p1[1])
    y1, x1 = int(p2[0]), int(p2[1])
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
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


def _has_los_walls_only(grid, p1, p2):
    """壁のみを考慮した射線判定。スモークはgame_stateから参照できないため
    考慮しない(既知の制約。学習環境よりやや楽観的な視界判定になる)。"""
    for r, c in _line_cells(p1, p2):
        if grid[r, c] == 1:
            return False
    return True


def _build_distance_map_walls_only(grid, source_cells):
    """指定座標群を始点とした、壁のみを障害物としたマルチソースBFS距離マップ。
    キャラクター同士の占有は考慮しない(学習環境の固定経路と同じ前提)。
    """
    height, width = grid.shape
    dist = np.full((height, width), np.inf, dtype=np.float32)
    q = deque()
    for r, c in source_cells:
        if 0 <= r < height and 0 <= c < width and grid[r, c] != 1:
            dist[r, c] = 0.0
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
                if dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
    return dist


def _bfs_next_step(grid, start, goal, blocked=()):
    """Return one walkable step from start toward goal, if one exists."""
    start = tuple(map(int, start))
    goal = tuple(map(int, goal))
    if start == goal:
        return start
    blocked = {tuple(map(int, cell)) for cell in blocked}
    dist_map = _build_distance_map_walls_only(grid, [goal])
    r, c = start
    best = start
    best_dist = dist_map[r, c] if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] else np.inf
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
            continue
        if grid[nr, nc] == 1 or (nr, nc) in blocked:
            continue
        distance = dist_map[nr, nc]
        if np.isfinite(distance) and distance < best_dist:
            best = (nr, nc)
            best_dist = distance
    return best


# ---------------------------------------------------------------------------
# チーム共有目撃情報(carrier+escort全体。train_attacker_escort.pyの
# EscortEnv._update_team_sightingと同一方針)。escortコントローラーは
# carrierを直接操作しないが、game_state["chars"]にはcarrierのCharacter
# オブジェクトも含まれるため、その現在位置からのLOSも判定に含める。
# ---------------------------------------------------------------------------
class _TeamSightingMemory:
    def __init__(self):
        self.last_seen_enemy = None  # {"pos": (r, c), "name": str, "tick_ago": int}

    def reset(self):
        self.last_seen_enemy = None

    def update(self, grid, my_team, chars):
        allies = [c for c in chars if getattr(c, "team", None) == my_team and getattr(c, "is_alive", True)]
        enemies = [c for c in chars if getattr(c, "team", None) != my_team and getattr(c, "is_alive", True)]

        visible_enemies = []
        for a in allies:
            for e in enemies:
                if _has_los_walls_only(grid, tuple(a.pos), tuple(e.pos)) and e not in visible_enemies:
                    visible_enemies.append(e)

        if visible_enemies:
            tracked = None
            if self.last_seen_enemy is not None:
                tracked_name = self.last_seen_enemy.get("name")
                tracked = next((e for e in visible_enemies if e.name == tracked_name), None)
            if tracked is None:
                tracked = min(
                    visible_enemies,
                    key=lambda e: min(
                        _chebyshev(a.pos, e.pos) for a in allies
                    ) if allies else 0,
                )
            self.last_seen_enemy = {"pos": tuple(map(int, tracked.pos)), "name": tracked.name, "tick_ago": 0}
        elif self.last_seen_enemy is not None:
            self.last_seen_enemy["tick_ago"] += 1
            if self.last_seen_enemy["tick_ago"] > SIGHTING_STALENESS_CAP:
                self.last_seen_enemy = None


# ---------------------------------------------------------------------------
# Dueling DQN(train_attacker_escort.py と同一アーキテクチャ)
# ---------------------------------------------------------------------------
class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x):
        feat = self.feature(x)
        value = self.value_head(feat)
        advantage = self.advantage_head(feat)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class Ov1LearningAttackerEscortController:
    """Escort Phase(omoko_v1固定チーム版)の学習済みモデルで、護衛
    キャラクター4体の移動・アビリティ使用を決定する。4体は同一インスタンス
    (同一ネットワーク)を共有する想定。
    """

    def __init__(
        self,
        model_path,
        device=None,
        greedy=True,
        epsilon=0.0,
        max_ticks=100,
        verbose=False,
    ):
        self.device = device or torch.device("cpu")
        self.greedy = greedy
        self.epsilon = epsilon
        self.max_ticks = max_ticks
        self.verbose = verbose

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Escortモデルが見つかりません: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        obs_dim = int(checkpoint.get("obs_dim", OBS_DIM))
        n_actions = int(checkpoint.get("n_actions", N_ACTIONS))

        if obs_dim != OBS_DIM or n_actions != N_ACTIONS:
            raise ValueError(
                f"チェックポイントの観測/行動空間がこのコントローラーと不一致です: "
                f"obs_dim={obs_dim}(期待値{OBS_DIM}) n_actions={n_actions}(期待値{N_ACTIONS})。"
                f"train_attacker_escort.pyのバージョンが古い可能性があります。"
            )

        self.policy_net = DuelingQNetwork(obs_dim, n_actions).to(self.device)
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.policy_net.eval()

        self.hold_ticks = int(checkpoint.get("hold_ticks", ESCORT_HOLD_TICKS_DEFAULT))

        if self.verbose:
            print(
                f"[Ov1LearningAttackerEscortController] モデル読込完了: {model_path} "
                f"(obs_dim={obs_dim}, episode={checkpoint.get('episode')}, "
                f"success_rate={checkpoint.get('success_rate')}, "
                f"roster_order={checkpoint.get('roster_order')}, "
                f"spike_holder_default={checkpoint.get('spike_holder_default')})"
            )

        # goal座標(r, c) -> 壁のみBFS距離マップ のキャッシュ
        self._goal_dist_cache = {}
        # carry_pos(r, c) -> 壁のみBFS距離マップ のキャッシュ
        # (escort自身からキャリアーまでの距離・方向をBFSベースで測るため)
        self._carry_dist_cache = {}
        # キャラクター名ごとのラウンド内状態(tick数・移動履歴・停滞カウント)
        self._char_state = {}

        # チーム共有の目撃情報(carrier+escort全体)。4体で1インスタンスを
        # 共有するため、tick先頭(同じキャラが再度呼ばれた時点)で1回だけ更新する。
        self.team_sighting = _TeamSightingMemory()
        self._processed_this_tick = set()
        self._watch_assignments = {}

        # ov1_train_attacker_escort.pyのlineup_used_cellsと同一方針。
        # 各定点は1ラウンドにつき1回だけ(4体全体で共有)。
        self._lineup_used_cells = {ability: set() for ability in ABILITY_LINEUP_POINTS}

    # ------------------------------------------------------------------
    # ラウンド開始時にrun_game.pyから呼ばれる(hasattr判定で自動検出される)
    # ------------------------------------------------------------------
    def reset_round(self):
        self._char_state.clear()
        self.team_sighting.reset()
        self._processed_this_tick.clear()
        self._watch_assignments.clear()
        for used in self._lineup_used_cells.values():
            used.clear()

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------
    def _get_char_state(self, char):
        return self._char_state.setdefault(
            char.name,
            {"tick": 0, "last_delta": (0.0, 0.0), "stuck": 0},
        )

    def _maybe_advance_tick(self, char, grid, chars):
        """同じキャラクターが再び呼ばれたら新しいtickに入ったとみなし、
        チーム共有目撃情報を1回だけ更新する
        (ov1_learning_attacker_guard.pyの_maybe_advance_tickと同一方針)。"""
        if char.name in self._processed_this_tick:
            self._processed_this_tick.clear()
            self.team_sighting.update(grid, char.team, chars)
        self._processed_this_tick.add(char.name)

    def _refresh_watch_assignments(self, grid, chars, my_team):
        """現在LOSが通る警戒点を、生存escortへ均等に割り当てる。

        ランクはtrain側(EscortEnv._refresh_watch_assignments)と同じく
        ROSTER_ORDER上の並び順を使う。名前のアルファベット順ソートのままだと、
        ROSTER_ORDERがアルファベット順でない場合に学習時とrankが食い違い、
        担当警戒点の割り当てが学習時と一致しなくなる。
        """
        escorts = [
            c for c in chars
            if getattr(c, "team", None) == my_team
            and getattr(c, "is_alive", True)
            and not getattr(c, "has_spike", False)
        ]
        escorts.sort(key=lambda c: _roster_rank(c.name))
        if not escorts or not WATCH_POINTS:
            self._watch_assignments = {}
            return

        usable = [
            point for point in WATCH_POINTS
            if any(_has_los_walls_only(grid, tuple(c.pos), point) for c in escorts)
        ]
        if not usable:
            self._watch_assignments = {c.name: None for c in escorts}
            return

        self._watch_assignments = {
            c.name: usable[index % len(usable)]
            for index, c in enumerate(escorts)
        }

    def _watch_point_info(self, char, grid):
        point = self._watch_assignments.get(char.name)
        if point is None:
            return 0.0, 0.0, 0.0
        if not _has_los_walls_only(grid, tuple(char.pos), point):
            return 0.0, 0.0, 0.0
        r, c = int(char.pos[0]), int(char.pos[1])
        return 1.0, (point[0] - r) / max(1, grid.shape[0]), (point[1] - c) / max(1, grid.shape[1])

    def _position_hint_info(self, char, grid):
        """割り当てられた警戒点に対応する推奨立ち位置(大文字ヒント)への方向情報。
        train_attacker_escort.py EscortEnv._position_hint_info() と同一ロジック。
        LOSの有無は問わない(まずそこへ移動することを促すための情報のため)。"""
        point = self._watch_assignments.get(char.name)
        if point is None:
            return 0.0, 0.0, 0.0
        hints = WATCH_POINT_HINTS.get(point) or []
        if not hints:
            return 0.0, 0.0, 0.0
        r, c = int(char.pos[0]), int(char.pos[1])
        nearest = min(hints, key=lambda cell: _chebyshev((r, c), cell))
        return 1.0, (nearest[0] - r) / max(1, grid.shape[0]), (nearest[1] - c) / max(1, grid.shape[1])

    @staticmethod
    def _is_wall(grid, r, c):
        height, width = grid.shape
        if not (0 <= r < height and 0 <= c < width):
            return True
        return grid[r, c] == 1

    def _get_goal_dist_map(self, grid, goal):
        key = (grid.tobytes(), tuple(goal))
        cached = self._goal_dist_cache.get(key)
        if cached is None:
            cached = _build_distance_map_walls_only(grid, [tuple(goal)])
            self._goal_dist_cache[key] = cached
        return cached

    def _get_carry_dist_map(self, grid, carry_pos):
        """carry_posを起点とした壁のみBFS距離マップ(キャッシュ付き)。
        escort自身からキャリアーまでの距離・方向はチェビシェフ距離ではなく
        こちらを使う。"""
        key = (grid.tobytes(), tuple(carry_pos))
        cached = self._carry_dist_cache.get(key)
        if cached is None:
            cached = _build_distance_map_walls_only(grid, [tuple(carry_pos)])
            self._carry_dist_cache[key] = cached
        return cached

    def _path_status(self, grid, escort_pos, carry_pos, carry_dist_map, goal_dist_map):
        """train_attacker_escort.py EscortEnv._path_status() と同一ロジック。
        escort_posがキャリアーの最短経路上か、退避先(経路上でない隣接マス)が
        あるかを判定する。"""
        r, c = escort_pos
        cr, cc = carry_pos
        total = goal_dist_map[cr, cc]
        if not np.isfinite(total):
            return False, False

        def _on_path(rr, cc_):
            d1 = carry_dist_map[rr, cc_]
            d2 = goal_dist_map[rr, cc_]
            if not (np.isfinite(d1) and np.isfinite(d2)):
                return False
            return abs(d1 + d2 - total) < PATH_EQUALITY_TOL

        on_path = _on_path(r, c)
        escape_available = False
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if self._is_wall(grid, nr, nc):
                continue
            if not _on_path(nr, nc):
                escape_available = True
                break
        return on_path, escape_available

    def _resolve_carry_and_goal(self, char, game_state):
        """護衛対象(キャリアー)の位置と、その目的地(goal)を決める。

        優先順位:
          1. 生存している味方でスパイクを持っている者 -> その位置。
             goalはis_plantedならplanted_pos、そうでなければtarget_plant_pos。
          2. 誰もスパイクを持っていない場合(設置済み) -> planted_posを
             疑似的なキャリアー位置として扱う(サイト周辺の護衛に切り替わる)。
          3. スパイクが地面に落ちている場合 -> spike_posを疑似的な
             キャリアー位置として扱う(回収を待つ形で近くに集まる)。
          4. どれも取得できない場合 -> target_plant_pos、それも無ければ
             自分自身の位置(実質、何もしない)。
        """
        chars = game_state.get("chars", [])
        is_planted = bool(game_state.get("is_planted", False))
        planted_pos = game_state.get("planted_pos")
        target_plant_pos = game_state.get("target_plant_pos")
        spike_pos = game_state.get("spike_pos")

        carrier = next(
            (
                c for c in chars
                if getattr(c, "is_alive", True)
                and getattr(c, "team", None) == char.team
                and getattr(c, "has_spike", False)
            ),
            None,
        )

        if carrier is not None:
            carry_pos = tuple(int(v) for v in carrier.pos)
            goal = tuple(planted_pos) if is_planted and planted_pos else (
                tuple(target_plant_pos) if target_plant_pos else carry_pos
            )
            return carry_pos, goal

        if is_planted and planted_pos:
            pos = tuple(int(v) for v in planted_pos)
            return pos, pos

        if spike_pos:
            pos = tuple(int(v) for v in spike_pos)
            return pos, pos

        if target_plant_pos:
            pos = tuple(int(v) for v in target_plant_pos)
            return pos, pos

        pos = tuple(int(v) for v in char.pos)
        return pos, pos

    def _predict_carry_next_step(self, grid, carry_pos, goal):
        """キャリアーの理想進行方向(他キャラクターの占有を無視した、
        壁のみのBFS勾配)を予測する。他エージェントを避けないため、
        「今このマスに立っていたらキャリアーの進路を塞いでいる」
        という判定にそのまま使える。
        """
        if carry_pos == goal:
            return carry_pos

        dist_map = self._get_goal_dist_map(grid, goal)
        height, width = grid.shape
        r, c = carry_pos
        best_cell = carry_pos
        best_dist = dist_map[r, c] if 0 <= r < height and 0 <= c < width else np.inf

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if grid[nr, nc] == 1:
                continue
            d = dist_map[nr, nc]
            if np.isfinite(d) and d < best_dist:
                best_dist = d
                best_cell = (nr, nc)

        return best_cell

    def _nearest_visible_enemy(self, grid, chars, my_team, from_pos, max_range=None):
        best_char, best_dist = None, None
        for c in chars:
            if not getattr(c, "is_alive", True) or getattr(c, "team", None) == my_team:
                continue
            enemy_pos = (int(c.pos[0]), int(c.pos[1]))
            dist = _chebyshev(from_pos, enemy_pos)
            if max_range is not None and dist > max_range:
                continue
            if not _has_los_walls_only(grid, from_pos, enemy_pos):
                continue
            if best_dist is None or dist < best_dist:
                best_char, best_dist = c, dist
        return best_char, best_dist

    # ------------------------------------------------------------------
    # おもこ専用: 「にげるっすー!」覚醒中は、視認可能な敵がいる限り
    # 壁のみ考慮したLOSが通らないマスへの移動を最優先させる。
    # ------------------------------------------------------------------
    _OMOKO_ESCAPE_AWAKENING_NAME = "にげるっすー!"

    def _omoko_escape_active(self, char):
        return (
            getattr(char, "base_name", char.name) == "おもこ"
            and self._OMOKO_ESCAPE_AWAKENING_NAME in getattr(char, "active_awakenings", {})
        )

    def _visible_enemy_positions(self, grid, chars, my_team, pos):
        return [
            (int(c.pos[0]), int(c.pos[1]))
            for c in chars
            if getattr(c, "is_alive", True)
            and getattr(c, "team", None) != my_team
            and _has_los_walls_only(grid, pos, (int(c.pos[0]), int(c.pos[1])))
        ]

    def _find_escape_cell(self, grid, pos, chars, my_team, search_limit=80):
        visited = {pos}
        queue = deque([pos])
        checked = 0
        while queue and checked < search_limit:
            cur = queue.popleft()
            checked += 1
            if cur != pos and not self._visible_enemy_positions(grid, chars, my_team, cur):
                return cur
            r, c = cur
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (r + dr, c + dc)
                if nxt in visited or self._is_wall(grid, *nxt):
                    continue
                visited.add(nxt)
                queue.append(nxt)
        return None

    def _omoko_escape_step(self, char, grid, chars):
        if not self._omoko_escape_active(char):
            return None
        pos = (int(char.pos[0]), int(char.pos[1]))
        if not self._visible_enemy_positions(grid, chars, char.team, pos):
            return None
        safe_cell = self._find_escape_cell(grid, pos, chars, char.team)
        if safe_cell is None or safe_cell == pos:
            return None
        dist_map = _build_distance_map_walls_only(grid, [safe_cell])
        r, c = pos
        best_cell, best_dist = pos, dist_map[r, c]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if self._is_wall(grid, nr, nc):
                continue
            d = dist_map[nr, nc]
            if np.isfinite(d) and d < best_dist:
                best_dist = d
                best_cell = (nr, nc)
        return [best_cell[0], best_cell[1]]

    def _available_lineup_cell(self, ability, grid, pos, carry_pos):
        """ov1_train_attacker_escort.pyの_available_lineup_cell()・
        _apply_ability()内フォールバックと同一ロジック。「キャリアーが
        定点からLINEUP_TRIGGER_RADIUS以内に接近している」または
        「escort自身から射程内・射線が通っている」のいずれかを満たす
        未使用の定点のうち最も近いものを返す。DQNがACTION_ABILITYを選び、
        かつ視認可能な敵がいない場合にのみ呼ばれる想定
        (_action_mask()・decide_move()側で判定順序を揃える)。
        """
        max_range = FLASH_RANGE if ability == "FLASH" else ABILITY_RANGE
        used = self._lineup_used_cells.get(ability, set())
        candidates = []
        for cell in ABILITY_LINEUP_POINTS.get(ability, []):
            if cell in used:
                continue
            carry_near = carry_pos is not None and _chebyshev(carry_pos, cell) <= LINEUP_TRIGGER_RADIUS
            los_ready = _chebyshev(pos, cell) <= max_range and _has_los_walls_only(grid, pos, cell)
            if carry_near and los_ready:
                candidates.append(cell)
        if not candidates:
            return None
        return min(candidates, key=lambda cell: _chebyshev(pos, cell))

    def _team_effect_active(self, chars, my_team):
        """味方の誰かが敵にかけたblind/revealが現在有効かどうか。
        スモークの有無はgame_stateから取得できないため考慮しない
        (既知の制約)。
        """
        for c in chars:
            if getattr(c, "team", None) == my_team or not getattr(c, "is_alive", True):
                continue
            if getattr(c, "blind_remaining", 0) > 0 or getattr(c, "reveal_remaining", 0) > 0:
                return True
        return False

    # ------------------------------------------------------------------
    # 観測構築(train_attacker_escort.py(omoko_v1版)の
    # EscortEnv._get_obs()と要素の順序・個数を完全一致させること。全41次元。)
    # ------------------------------------------------------------------
    def _build_obs(self, char, game_state, st):
        grid = game_state["grid"]
        height, width = grid.shape
        chars = game_state.get("chars", [])
        r, c = int(char.pos[0]), int(char.pos[1])

        carry_pos, goal = self._resolve_carry_and_goal(char, game_state)
        cr, cc = carry_pos
        next_step = self._predict_carry_next_step(grid, carry_pos, goal)
        carry_dist_map = self._get_carry_dist_map(grid, carry_pos)

        obs = []
        obs.append(r / max(1, height - 1))
        obs.append(c / max(1, width - 1))

        # escort自身からキャリアーまでの距離・方向はBFS実距離ベース
        # (チェビシェフ距離は壁を無視するため、曲がった通路で
        # 実際の経路と逆方向を指してしまうことがある)
        raw_dist = carry_dist_map[r, c]
        dist_to_carry = raw_dist if np.isfinite(raw_dist) else DIST_NORM_MAX
        obs.append(min(1.0, dist_to_carry / DIST_NORM_MAX))

        # 方向成分も座標の単純差分ではなく、BFS距離を最も縮める方向を使う
        best_dr, best_dc, best_d = 0, 0, dist_to_carry
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not self._is_wall(grid, nr, nc):
                nd = carry_dist_map[nr, nc]
                if np.isfinite(nd) and nd < best_d:
                    best_d = nd
                    best_dr, best_dc = dr, dc
        obs.append(float(best_dr))
        obs.append(float(best_dc))

        # キャリアーの進行方向(次の理想セルへの差分)
        obs.append(float(np.sign(next_step[0] - cr)))
        obs.append(float(np.sign(next_step[1] - cc)))

        # 壁フラグ(4方向)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            obs.append(1.0 if self._is_wall(grid, r + dr, c + dc) else 0.0)

        # 各方向に動いた場合のキャリアーまでのBFS距離勾配
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            wall = self._is_wall(grid, nr, nc)
            if wall:
                obs.append(1.0)
            else:
                gdist = carry_dist_map[nr, nc]
                gdist = gdist if np.isfinite(gdist) else DIST_NORM_MAX
                obs.append(min(1.0, gdist / DIST_NORM_MAX))

        # 斜め壁フラグ
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            obs.append(1.0 if self._is_wall(grid, r + dr, c + dc) else 0.0)

        # 隣接4方向に生存中の味方escortがいるか(train_attacker_escort.pyと一致させる)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            occupied_by_ally = any(
                getattr(oc, "is_alive", True)
                and getattr(oc, "team", None) == char.team
                and oc is not char
                and not getattr(oc, "has_spike", False)
                and (int(oc.pos[0]), int(oc.pos[1])) == (nr, nc)
                for oc in chars
            )
            obs.append(1.0 if occupied_by_ally else 0.0)

        # 距離帯の逸脱量
        if dist_to_carry < DIST_BAND_MIN:
            band_dev = (dist_to_carry - DIST_BAND_MIN) / DIST_NORM_MAX
        elif dist_to_carry > DIST_BAND_MAX:
            band_dev = (dist_to_carry - DIST_BAND_MAX) / DIST_NORM_MAX
        else:
            band_dev = 0.0
        obs.append(band_dev)

        # 最寄りの視認可能な敵
        enemy_char, enemy_dist = self._nearest_visible_enemy(grid, chars, char.team, (r, c))
        if enemy_char is not None:
            er, ec = int(enemy_char.pos[0]), int(enemy_char.pos[1])
            obs.append(1.0)
            obs.append(max(-1.0, min(1.0, (er - r) / DIST_NORM_MAX)))
            obs.append(max(-1.0, min(1.0, (ec - c) / DIST_NORM_MAX)))
            obs.append(min(1.0, enemy_dist / DIST_NORM_MAX))
            obs.append(min(1.0, getattr(enemy_char, "blind_remaining", 0) / max(1, BLIND_DURATION_TICKS)))
            obs.append(min(1.0, getattr(enemy_char, "reveal_remaining", 0) / max(1, REVEAL_DURATION_TICKS)))
        else:
            obs.extend([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

        # 自分のアビリティ状態(実際のCharacterオブジェクトの値をそのまま使う)。
        # HUNT(タイガー)役はflash/smoke/recon_chargesが常に0なので、
        # total_charges<=0となり自然に「未使用フラグ=0」相当の扱いになる
        # (train_attacker_escort.py側でescort_ability_used初期値をTrue相当に
        # している挙動と一致する)。
        total_charges = (
            getattr(char, "flash_charges", 0)
            + getattr(char, "smoke_charges", 0)
            + getattr(char, "recon_charges", 0)
        )
        obs.append(1.0 if total_charges > 0 else 0.0)
        for ability in ABILITY_TYPES:
            obs.append(1.0 if char.ability_name == ability else 0.0)

        # チーム状況：誰かの効果(blind/reveal)が現在有効か
        obs.append(1.0 if self._team_effect_active(chars, char.team) else 0.0)

        obs.append(st["last_delta"][0])
        obs.append(st["last_delta"][1])
        obs.append(min(1.0, st["stuck"] / 10.0))
        obs.append(1.0 - min(1.0, st["tick"] / max(1, self.max_ticks)))
        # 自分が現在、キャリアーの理想進行先セルに立っているか(＝塞いでいるか)
        obs.append(1.0 if (r, c) == next_step and (r, c) != (cr, cc) else 0.0)

        # 自分の現在facing(8方向)のonehot(train_attacker_escort.pyのEscortEnv._get_obs()と一致)
        char_facing = getattr(char, "facing", None)
        for d in FACING_DIRS:
            obs.append(1.0 if char_facing == d else 0.0)

        # 「サイト-エスコート-キャリアー」順(自分がキャリアーより前に出ているか)判定用。
        goal_dist_map = self._get_goal_dist_map(grid, goal)
        escort_goal_d = goal_dist_map[r, c]
        carry_goal_d = goal_dist_map[cr, cc]
        ahead = (
            np.isfinite(escort_goal_d) and np.isfinite(carry_goal_d)
            and escort_goal_d < carry_goal_d
        )
        on_path, escape_available = self._path_status(
            grid, (r, c), carry_pos, carry_dist_map, goal_dist_map
        )
        obs.append(1.0 if ahead else 0.0)
        obs.append(1.0 if on_path else 0.0)
        obs.append(1.0 if escape_available else 0.0)

        # チーム共有の目撃情報(carrier+escort全体。自分が直接視認していなくても
        # 誰かが見ていれば共有される)。train_attacker_escort.pyのEscortEnv._get_obs()
        # と同一の埋め方。
        last_seen = self.team_sighting.last_seen_enemy
        if last_seen is not None:
            tr, tc = last_seen["pos"]
            obs.append(1.0)
            obs.append(max(-1.0, min(1.0, (tr - r) / DIST_NORM_MAX)))
            obs.append(max(-1.0, min(1.0, (tc - c) / DIST_NORM_MAX)))
            obs.append(min(last_seen["tick_ago"], SIGHTING_STALENESS_CAP) / SIGHTING_STALENESS_CAP)
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0])

        watch_visible, watch_dr, watch_dc = self._watch_point_info(char, grid)
        obs.extend([watch_visible, watch_dr, watch_dc])

        hint_available, hint_dr, hint_dc = self._position_hint_info(char, grid)
        obs.extend([hint_available, hint_dr, hint_dc])

        available_orbs = {
            tuple(map(int, cell)) for cell in game_state.get("available_orbs", ())
        }
        allies = [
            other for other in chars
            if getattr(other, "is_alive", True) and other.team == char.team
        ]
        team_visible = self.team_sighting.last_seen_enemy is not None
        goal_distance = self._get_goal_dist_map(grid, goal)[r, c]
        site_entry = np.isfinite(goal_distance) and goal_distance <= 12
        obs.extend(ultimate_context(
            char,
            self_sighting=enemy_char is not None,
            team_sighting=team_visible,
            tactical=site_entry and (enemy_char is not None or team_visible or bool(watch_visible)),
        ))
        obs.extend(orb_context(char, available_orbs, grid, allies))

        obs_arr = np.array(obs, dtype=np.float32)
        assert obs_arr.shape[0] == OBS_DIM, (
            f"観測次元がOBS_DIM({OBS_DIM})と不一致: {obs_arr.shape[0]}。"
            f"train_attacker_escort.pyとのズレを確認してください。"
        )
        return obs_arr

    def _opportunistic_orb_target(self, char, grid, chars, available_orbs, game_state):
        """Return a nearby orb only when the escort can safely detour for it.

        The target is exposed through the existing ACTION_ORB action so the
        decision remains part of the learned policy.  This gate only removes
        the action when an enemy is known/visible or the round is too late.
        """
        if not available_orbs:
            return None

        allies = [
            other for other in chars
            if getattr(other, "is_alive", True) and other.team == char.team
        ]
        if not orb_priority(char, allies):
            return None
        if self._get_char_state(char)["tick"] <= self.hold_ticks:
            return None

        # A shared sighting is still actionable enemy information, even when
        # this escort currently has no direct LOS.
        if self.team_sighting.last_seen_enemy is not None:
            return None
        pos = (int(char.pos[0]), int(char.pos[1]))
        if self._nearest_visible_enemy(grid, chars, char.team, pos)[0] is not None:
            return None

        remaining = int(game_state.get("round_timer", self.max_ticks - self._get_char_state(char)["tick"]))
        if remaining < ESCORT_ORB_MIN_REMAINING_TICKS:
            return None

        candidates = []
        for orb in sorted(available_orbs):
            dist_map = _build_distance_map_walls_only(grid, [orb])
            distance = dist_map[pos[0], pos[1]]
            if np.isfinite(distance) and distance <= ESCORT_ORB_MAX_PATH_DISTANCE:
                candidates.append((int(distance), tuple(orb)))
        return min(candidates)[1] if candidates else None

    def _action_mask(self, char, grid, chars, carry_pos, available_orbs=(), orb_target=None):
        r, c = int(char.pos[0]), int(char.pos[1])
        base_mask = np.ones(BASE_N_ACTIONS, dtype=bool)
        for a, (dr, dc) in _MOVE_DELTA.items():
            if a == ACTION_STAY:
                continue
            if self._is_wall(grid, r + dr, c + dc):
                base_mask[a] = False

        # 開始直後hold_ticks分は移動を禁止し、キャリアーが先に動き出すまで待つ
        # (train_attacker_escort.pyのget_action_mask()と同一ロジック。
        #  向き(facing)は移動・アビリティとは無関係に常に自由選択できる)。
        st = self._get_char_state(char)
        if st["tick"] <= self.hold_ticks:
            for a in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
                base_mask[a] = False

        total_charges = (
            getattr(char, "flash_charges", 0)
            + getattr(char, "smoke_charges", 0)
            + getattr(char, "recon_charges", 0)
        )
        if total_charges <= 0:
            # HUNT(タイガー)役もここで自動的にマスクされる(常にtotal_charges==0のため)。
            base_mask[ACTION_ABILITY] = False
        else:
            # チャージがあっても、射程内に有効な標的がいなければABILITYは
            # マスクする。学習済みネットワークがABILITYを選び続けて
            # 実質STAYのままブロックし続けるのを防ぐ。
            # FLASHのみ、視認可能な敵がいなくても未使用の定点セルが
            # 射程内・射線内にあればアンマスクする(ov1_train_attacker_escort.py
            # のget_action_mask()と同一方針。以前存在した「無条件自動投擲」は
            # action-reward対応を壊すため廃止し、ACTION_ABILITY選択時にのみ
            # 発動するよう学習側・推論側を揃えた)。
            enemy_range = FLASH_RANGE if char.ability_name == "FLASH" else ABILITY_RANGE
            enemy_char, _ = self._nearest_visible_enemy(
                grid, chars, char.team, (r, c), max_range=enemy_range
            )
            has_target = enemy_char is not None
            if not has_target:
                has_target = self._available_lineup_cell(
                    char.ability_name, grid, (r, c), carry_pos
                ) is not None
            if not has_target:
                base_mask[ACTION_ABILITY] = False

        allies = [
            other for other in chars
            if getattr(other, "is_alive", True) and other.team == char.team
        ]
        base_mask[ACTION_ULTIMATE] = ultimate_ready(char)
        base_mask[ACTION_ORB] = orb_target is not None

        return np.repeat(base_mask, len(FACING_DIRS))

    # ------------------------------------------------------------------
    # コントローラー本体
    # ------------------------------------------------------------------
    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        chars = game_state.get("chars", [])
        st = self._get_char_state(char)
        st["tick"] += 1

        self._maybe_advance_tick(char, grid, chars)
        self._refresh_watch_assignments(grid, chars, char.team)

        # 「ん? だれかいるワン!」の覚醒中だけ、スモークをまたいで
        # 射線が通る敵を発見したら射撃を優先する。通常の視界判定や、
        # 敵がいない時のスモーク越し定点アビリティには介入しない。
        smoke_cells = set(game_state.get("smoke_cells", set()))
        if getattr(char, "sees_through_smoke", False) and smoke_cells:
            pos = (int(char.pos[0]), int(char.pos[1]))
            smoke_targets = [
                enemy
                for enemy in chars
                if getattr(enemy, "is_alive", True)
                and getattr(enemy, "team", None) != char.team
                and _has_los_walls_only(grid, pos, (int(enemy.pos[0]), int(enemy.pos[1])))
                and any(cell in smoke_cells for cell in _line_cells(pos, tuple(enemy.pos)))
            ]
            if smoke_targets:
                target = min(
                    smoke_targets,
                    key=lambda enemy: _chebyshev(pos, (int(enemy.pos[0]), int(enemy.pos[1]))),
                )
                st["last_delta"] = (0.0, 0.0)
                st["stuck"] += 1
                return [pos[0], pos[1]], {
                    "facing": _facing_towards(pos, tuple(target.pos))
                }

        escape_step = self._omoko_escape_step(char, grid, chars)
        if escape_step is not None:
            st["last_delta"] = (0.0, 0.0)
            st["stuck"] += 1
            return escape_step

        carry_pos, _goal = self._resolve_carry_and_goal(char, game_state)
        available_orbs = {
            tuple(map(int, cell)) for cell in game_state.get("available_orbs", ())
        }
        orb_target = self._opportunistic_orb_target(
            char, grid, chars, available_orbs, game_state
        )
        obs = self._build_obs(char, game_state, st)
        mask = self._action_mask(
            char, grid, chars, carry_pos, available_orbs, orb_target=orb_target
        )

        if (not self.greedy) and np.random.random() < self.epsilon:
            action_idx = int(np.random.choice(np.flatnonzero(mask)))
        else:
            with torch.no_grad():
                state_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                q = self.policy_net(state_t).squeeze(0).cpu().numpy()
            q = np.where(mask, q, -1e9)
            action_idx = int(np.argmax(q))

        action, facing = decode_action(action_idx)
        r, c = int(char.pos[0]), int(char.pos[1])

        if not getattr(char, "facing_forced_this_tick", False):
            char.facing = facing

        if action == ACTION_ORB:
            st["last_delta"] = (0.0, 0.0)
            st["stuck"] += 1
            if orb_target is not None:
                occupied = {
                    (int(other.pos[0]), int(other.pos[1]))
                    for other in chars
                    if other is not char and getattr(other, "is_alive", True)
                }
                orb_next = _bfs_next_step(grid, (r, c), orb_target, occupied)
                if orb_next != (r, c):
                    st["last_delta"] = (
                        float(orb_next[0] - r), float(orb_next[1] - c)
                    )
                    st["stuck"] = 0
                    return [int(orb_next[0]), int(orb_next[1])], {"facing": facing}
            return [r, c], "COLLECT_ORB"

        if action == ACTION_ULTIMATE:
            payload = {"ultimate": str(getattr(char, "ultimate_name", "")).upper(), "facing": facing}
            if payload["ultimate"] == "ESCAPE":
                payload["target"] = tuple(map(int, _goal))
            st["last_delta"] = (0.0, 0.0)
            st["stuck"] += 1
            return [r, c], payload

        # Keep inference deterministic with the combat action mask used during
        # training, including pre-existing checkpoints.
        visible_enemy, _ = self._nearest_visible_enemy(grid, chars, char.team, (r, c))
        if visible_enemy is not None and action != ACTION_ABILITY:
            st["last_delta"] = (0.0, 0.0)
            st["stuck"] += 1
            return [r, c], {"facing": _facing_towards((r, c), tuple(visible_enemy.pos))}

        if action == ACTION_ABILITY:
            enemy_range = FLASH_RANGE if char.ability_name == "FLASH" else ABILITY_RANGE
            enemy_char, _ = self._nearest_visible_enemy(
                grid, chars, char.team, (r, c), max_range=enemy_range
            )
            if enemy_char is not None:
                st["last_delta"] = (0.0, 0.0)
                st["stuck"] += 1
                target = (int(enemy_char.pos[0]), int(enemy_char.pos[1]))
                return list(char.pos), {"ability": char.ability_name, "target": target}
            # 視認可能な敵がいない場合、マップ上の定点(S/R/F)へのフォールバックを
            # 許可する(_apply_ability()の学習側ロジックと同一。ACTION_ABILITY
            # をネットワークが選んだ場合にのみ発動する)。
            lineup_cell = self._available_lineup_cell(char.ability_name, grid, (r, c), carry_pos)
            if lineup_cell is not None:
                self._lineup_used_cells[char.ability_name].add(lineup_cell)
                st["last_delta"] = (0.0, 0.0)
                st["stuck"] += 1
                return list(char.pos), {"ability": char.ability_name, "target": lineup_cell}
            # 射程内に有効な標的がいない場合、実チャージを無駄撃ちしないよう
            # STAYにフォールバックする。facingだけは選択どおり反映する。
            st["last_delta"] = (0.0, 0.0)
            st["stuck"] += 1
            return [r, c], {"facing": facing}

        dr, dc = _MOVE_DELTA[action]
        nr, nc = r + dr, c + dc

        if action == ACTION_STAY or self._is_wall(grid, nr, nc):
            st["last_delta"] = (0.0, 0.0)
            st["stuck"] += 1
            # その場に留まる場合でも、選択したfacingへその場旋回させる
            # (battle_logic.move_character のケース1.5「TURN」に対応)。
            return [r, c], {"facing": facing}

        st["last_delta"] = (float(dr), float(dc))
        st["stuck"] = 0
        # 移動方向とは無関係にfacingを明示指定する
        # (前進しながら横を警戒する、等の挙動をbattle_logic側で反映させる)。
        return [nr, nc], {"facing": facing}
