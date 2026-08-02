"""train_attacker_guard.py

Attacker Guard Phase 専用の学習スクリプト（Dueling DQN、重み共有マルチエージェント）。

【目的】
スパイク設置後、生存している味方（最大5体）で設置地点周辺を防衛し、
解除(DEFUSE)を阻止しながら起爆まで持ちこたえられるようになるまで学習する。

【配置優先順位（ご要望通り）】
  1. スパイク設置位置そのもの
  2. 設置位置に隣接する8マス
  3. map_data_guard.py に記載された「6」の位置（学習専用の推奨配置点）
他の味方が既にそのマスを占有（または他の味方がより優先度の高いスロットへの
割当を済ませている）場合は、優先順位を1→2→3の順に落として次の候補へ回す
「近い順の貪欲割当」を毎tick計算し、各エージェントへの誘導目標(goal)として
観測・報酬整形の両方に使う。実際にどう動くかは強化学習が決める
（ハードコードで強制移動させるわけではない）。

【重要：6は学習専用】
map_data_guard.py の grid==6 は本番の map_data.py には存在しない。
そのため、carryモデルの優先地点(5)と同じ設計判断で、6の座標を
「gridの値」ではなく「座標のリスト」としてチェックポイントに保存し、
推論側はその座標を直接BFSの目的地として使う（gridの値は見ない）。

【停止優先（無駄な移動をしない）】
既に射線が通っている状態で「動かなかった」場合にボーナスを、
既に射線が通っている状態で「動いた」場合に軽いペナルティを与える。
これにより、良い位置を確保したら極力その場に留まる挙動を誘導する
（射撃中の移動ペナルティは実ゲーム側のMOVING_ACCURACY等で
既に不利になる設計だが、位置取りの学習自体もこれで後押しする）。

【解除阻止の緊急対応】
解除中の敵(active defuser)がいる場合、配置優先度(1〜3)よりも
「解除者への接近・射線確保」を優先目標に切り替える。この間は
停止ボーナスを一時停止し、積極的に動いて阻止に向かうよう誘導する。

【アビリティ】
train_attacker_escort.py と同じ設計を踏襲する：
- 1ラウンドに1回だけ使用可能（使用済みなら選択不可）。
- 有効な標的（視界内の敵）がいない状態での使用は罰則（無駄撃ち防止）。
- 既に同種の効果（blind/reveal）がかかっている敵への重ねがけは罰則
  （味方が既に使用済みで効果が残っている状況の重複使用を抑制）。
- 敵がblind/reveal状態のときに撃破できたら、その効果をかけた本人に
  ボーナス。
- 追加：撃破した敵が「解除中(active defuser)」だった場合、
  解除阻止ボーナスをチーム全体（生存中の全guard）に与える
  （簡略化した集団戦闘モデルでは個々の射手を正確に特定できないため、
  escortモデルのteam_progress報酬と同じ考え方でチーム共有とする）。

【設計方針（既存ルールの継承）】
- このファイルは完全に自己完結している（map_data_guard.py /
  map_data.py 以外のゲーム本体コードに依存しない）。
- run_game.py / controllers.py など既存の共有インフラは変更・複製しない。
- obs_dim は静的に定義せず env.reset().shape[0] から動的に取得する
  （train_attacker_escort.py で発生した次元不一致バグの再発防止）。

【既知の簡略化】
- 戦闘解決は escort と同じ汎用ステータス近似モデル（実キャラの
  個体差は考慮しない）。
- 敵(Defender)は簡易スクリプトAI（設置地点へBFSで直進、隣接したら解除）。
- 「解除阻止ボーナス」は正確な射手を特定せずチーム全体に付与する。
"""

import argparse
import os
import random
import sys
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

EPISODE_COUNT = 9000

# ---------------------------------------------------------------------------
# map_data_guard.py（推奨配置点=6を含む学習専用マップ）を優先ロードし、
# 無ければ map_data.py にフォールバックする（tier3は空になる）。
# ---------------------------------------------------------------------------
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = (
    os.path.dirname(_CURRENT_DIR)
    if os.path.basename(_CURRENT_DIR) == "attacker_v3"
    else _CURRENT_DIR
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from map_data_guard import NEW_MAZE_STR  # noqa: E402
    print("[MAP] map_data_guard.py を使用します（推奨配置点=6対応・学習専用）")
except ImportError:
    from map_data import NEW_MAZE_STR  # noqa: E402
    print("[MAP] map_data_guard.py が見つからないため map_data.py を使用します（tier3なし）")


SITE_CELL_VALUE = 2
ATTACKER_SPAWN_VALUE = 3
DEFENDER_SPAWN_VALUE = 4
FORMATION_CELL_VALUE = 6  # 学習専用マップにのみ存在する推奨配置点

MAX_HP = 100
HEADSHOT_DAMAGE = 160
BODY_DAMAGE = 40

BLIND_DURATION_TICKS = 3
REVEAL_DURATION_TICKS = 5
BLIND_ACCURACY_MULTIPLIER = 0.30
REVEALED_DODGE_MULTIPLIER = 0.50
SMOKE_DURATION_TICKS = 15
DEFUSE_REQUIRED_TICKS = 6
DETONATE_TICKS = 45  # game_core.py の SPIKE_DETONATION_TICKS と一致させる

GENERIC_ACCURACY = 0.55
GENERIC_DODGE = 0.18
GENERIC_HS_RATE = 0.30

ABILITY_TYPES = ("FLASH", "RECON", "SMOKE")
ABILITY_RANGE = 6

N_GUARDS = 5
N_ENEMIES = 3

DIST_NORM_MAX = 20.0


# ---------------------------------------------------------------------------
# 汎用ヘルパー
# ---------------------------------------------------------------------------
def _cells_with_value(grid, value):
    return list(zip(*np.where(grid == value)))


def _line_cells(p1, p2):
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


def _has_los(grid, smoke_cells, p1, p2):
    cells = _line_cells(p1, p2)
    for r, c in cells:
        if grid[r, c] == 1:
            return False
    if smoke_cells and len(cells) > 2:
        if any(cell in smoke_cells for cell in cells):
            return False
    return True


def _chebyshev(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


def _build_distance_map_from_coords(grid, source_cells):
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


def _neighbors8(pos, grid):
    r, c = pos
    height, width = grid.shape
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
                out.append((nr, nc))
    return out


def _build_candidate_tiers(grid, plant_pos, formation_cells):
    """優先順位付きスロット候補を返す: (tier1, tier2, tier3)。"""
    height, width = grid.shape
    r, c = plant_pos
    tier1 = [plant_pos] if grid[r, c] != 1 else []
    tier2 = _neighbors8(plant_pos, grid)
    tier3 = list(formation_cells)
    return tier1, tier2, tier3


def _assign_goals(grid, plant_pos, formation_cells, positions_ordered, cell_dist_maps):
    """近い順（BFS実距離）の貪欲割当で、各エージェントの誘導目標(goal)とtier(0/1/2)を返す。

    チェビシェフ距離（直線距離）ではなく、壁を考慮したBFS距離で「近い」を
    判定する。直線距離だと壁の向こうのセルを「近い」と誤認し、
    キャラクターが角で詰まる原因になるため。

    positions_ordered: [(agent_index, current_pos), ...] のリスト。
    cell_dist_maps: {cell(tuple): distance_map(np.ndarray)} のキャッシュ辞書。
        呼び出し元が episode/round 単位で使い回すこと（毎回作り直すと重い）。
    戻り値: {agent_index: (goal_pos, tier)}  tier: 0=設置位置, 1=隣接8マス, 2=map6
    """
    def _dist_map_for(cell):
        dm = cell_dist_maps.get(cell)
        if dm is None:
            dm = _build_distance_map_from_coords(grid, [cell])
            cell_dist_maps[cell] = dm
        return dm

    def _bfs_dist(pos, cell):
        dm = _dist_map_for(cell)
        d = dm[pos[0], pos[1]]
        return float(d) if np.isfinite(d) else float("inf")

    tier1, tier2, tier3 = _build_candidate_tiers(grid, plant_pos, formation_cells)
    tiers = [tier1, tier2, tier3]
    claimed = set()
    result = {}
    for agent_idx, pos in positions_ordered:
        chosen, chosen_tier = None, None
        for tier_idx, tier_cells in enumerate(tiers):
            avail = [s for s in tier_cells if s not in claimed]
            if avail:
                chosen = min(avail, key=lambda s: _bfs_dist(pos, s))
                chosen_tier = tier_idx
                break
        if chosen is None:
            chosen, chosen_tier = plant_pos, 0
        claimed.add(chosen)
        result[agent_idx] = (chosen, chosen_tier)
    return result


# ---------------------------------------------------------------------------
# 環境
# ---------------------------------------------------------------------------
class GuardEnv:
    """スパイク防衛（Guard）役を学習させる軽量マルチエージェント環境。"""

    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY, ACTION_ABILITY = range(6)
    N_ACTIONS = 6
    _MOVE_DELTA = {
        ACTION_UP: (-1, 0),
        ACTION_DOWN: (1, 0),
        ACTION_LEFT: (0, -1),
        ACTION_RIGHT: (0, 1),
        ACTION_STAY: (0, 0),
    }

    def __init__(
        self,
        maze_str=NEW_MAZE_STR,
        max_ticks=DETONATE_TICKS + 20,
        n_guards=N_GUARDS,
        n_enemies=N_ENEMIES,
        shaping_coef=0.15,
        urgency_shaping_coef=0.30,
        cover_stationary_bonus=0.10,
        cover_move_penalty=0.08,
        ability_success_reward=2.0,
        ability_redundant_penalty=1.0,
        ability_waste_penalty=0.3,
        kill_bonus=5.0,
        defuse_stopped_bonus=6.0,
        mission_success_reward=8.0,
        mission_fail_penalty=8.0,
        all_dead_penalty=4.0,
        seed=None,
    ):
        lines = [l.strip() for l in maze_str.strip("\n").split("\n") if l.strip()]
        self.grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
        self.height, self.width = self.grid.shape

        self.site_cells = _cells_with_value(self.grid, SITE_CELL_VALUE)
        self.attacker_spawns = _cells_with_value(self.grid, ATTACKER_SPAWN_VALUE)
        self.defender_spawns = _cells_with_value(self.grid, DEFENDER_SPAWN_VALUE)
        self.formation_cells = _cells_with_value(self.grid, FORMATION_CELL_VALUE)
        self.walkable_cells = list(zip(*np.where(self.grid != 1)))

        self.max_ticks = max_ticks
        self.n_guards = n_guards
        self.n_enemies = n_enemies
        self.shaping_coef = shaping_coef
        self.urgency_shaping_coef = urgency_shaping_coef
        self.cover_stationary_bonus = cover_stationary_bonus
        self.cover_move_penalty = cover_move_penalty
        self.ability_success_reward = ability_success_reward
        self.ability_redundant_penalty = ability_redundant_penalty
        self.ability_waste_penalty = ability_waste_penalty
        self.kill_bonus = kill_bonus
        self.defuse_stopped_bonus = defuse_stopped_bonus
        self.mission_success_reward = mission_success_reward
        self.mission_fail_penalty = mission_fail_penalty
        self.all_dead_penalty = all_dead_penalty

        self.rng = random.Random(seed)

        # ラウンド内状態
        self.tick = 0
        self.plant_pos = (0, 0)
        self.detonate_timer = DETONATE_TICKS

        self.guard_pos = []
        self.guard_hp = []
        self.guard_alive = []
        self.guard_ability_type = []
        self.guard_ability_used = []
        self.guard_last_delta = []
        self.guard_stuck = []
        self._prev_goal_dist = [0.0] * n_guards
        self._prev_had_cover = [False] * n_guards

        self.enemy_pos = []
        self.enemy_hp = []
        self.enemy_alive = []
        self.enemy_defuse_timer = []
        self.enemy_blind_remaining = []
        self.enemy_blind_source = []
        self.enemy_reveal_remaining = []
        self.enemy_reveal_source = []
        self.active_defuser_idx = None

        self.smokes = []

        self._enemy_goal_dist_map = None
        self._cell_dist_maps = {}  # ゴール候補セル -> BFS距離マップ のキャッシュ（reset()ごとにクリア）
        self._blocking_none = None  # 予約（互換のため）

    # ------------------------------------------------------------------
    def _random_walkable_near(self, center, max_dist, exclude=()):
        exclude = set(exclude)
        candidates = [
            cell for cell in self.walkable_cells
            if _chebyshev(cell, center) <= max_dist and cell not in exclude
        ]
        if not candidates:
            candidates = [c for c in self.walkable_cells if c not in exclude]
        return self.rng.choice(candidates) if candidates else center

    def reset(self):
        self.tick = 0
        self.smokes = []
        self.detonate_timer = DETONATE_TICKS

        self.plant_pos = self.rng.choice(self.site_cells) if self.site_cells else (0, 0)
        self._enemy_goal_dist_map = _build_distance_map_from_coords(self.grid, [self.plant_pos])
        self._cell_dist_maps = {}  # ラウンドが変わればplant_posも変わるためクリア

        # --- Guard：設置直後を想定し、設置地点付近にランダム配置 ---
        occupied = {self.plant_pos}
        self.guard_pos = []
        for _ in range(self.n_guards):
            pos = self._random_walkable_near(self.plant_pos, max_dist=10, exclude=occupied)
            occupied.add(pos)
            self.guard_pos.append(pos)

        self.guard_hp = [MAX_HP] * self.n_guards
        self.guard_alive = [True] * self.n_guards
        self.guard_ability_type = [self.rng.choice(ABILITY_TYPES) for _ in range(self.n_guards)]
        self.guard_ability_used = [False] * self.n_guards
        self.guard_last_delta = [(0.0, 0.0)] * self.n_guards
        self.guard_stuck = [0] * self.n_guards

        # --- 敵：守備側スポーンに配置 ---
        self.enemy_pos = []
        enemy_candidates = list(self.defender_spawns)
        self.rng.shuffle(enemy_candidates)
        for i in range(self.n_enemies):
            if i < len(enemy_candidates):
                pos = enemy_candidates[i]
            else:
                pos = self._random_walkable_near(self.plant_pos, max_dist=30, exclude=occupied)
            self.enemy_pos.append(pos)

        self.enemy_hp = [MAX_HP] * self.n_enemies
        self.enemy_alive = [True] * self.n_enemies
        self.enemy_defuse_timer = [0] * self.n_enemies
        self.enemy_blind_remaining = [0] * self.n_enemies
        self.enemy_blind_source = [None] * self.n_enemies
        self.enemy_reveal_remaining = [0] * self.n_enemies
        self.enemy_reveal_source = [None] * self.n_enemies
        self.active_defuser_idx = None

        goals = self._compute_goals()
        for i in range(self.n_guards):
            goal, _tier = goals[i]
            self._prev_goal_dist[i] = self._goal_distance(self.guard_pos[i], goal)
            self._prev_had_cover[i] = self._has_los_to_plant(self.guard_pos[i])

        return [self._get_obs(i, goals) for i in range(self.n_guards)]

    # ------------------------------------------------------------------
    def _is_wall(self, r, c):
        if not (0 <= r < self.height and 0 <= c < self.width):
            return True
        return self.grid[r, c] == 1

    def _smoke_cell_set(self):
        cells = set()
        for smoke in self.smokes:
            cells.update(smoke["cells"])
        return cells

    def _has_los_to_plant(self, pos):
        smoke_cells = self._smoke_cell_set()
        return _has_los(self.grid, smoke_cells, pos, self.plant_pos)

    def _occupied_by_other_guards(self, exclude_idx):
        return {
            self.guard_pos[i] for i in range(self.n_guards)
            if i != exclude_idx and self.guard_alive[i]
        }

    def _occupied_all(self, kind, idx):
        occ = set()
        for i in range(self.n_guards):
            if kind == "guard" and i == idx:
                continue
            if self.guard_alive[i]:
                occ.add(self.guard_pos[i])
        for i in range(self.n_enemies):
            if kind == "enemy" and i == idx:
                continue
            if self.enemy_alive[i]:
                occ.add(self.enemy_pos[i])
        return occ

    def _nearest_visible_enemy(self, from_pos, max_range=None):
        smoke_cells = self._smoke_cell_set()
        best_idx, best_dist = None, None
        for i in range(self.n_enemies):
            if not self.enemy_alive[i]:
                continue
            dist = _chebyshev(from_pos, self.enemy_pos[i])
            if max_range is not None and dist > max_range:
                continue
            if not _has_los(self.grid, smoke_cells, from_pos, self.enemy_pos[i]):
                continue
            if best_dist is None or dist < best_dist:
                best_idx, best_dist = i, dist
        return best_idx, best_dist

    def _compute_goals(self):
        """全guardの誘導目標を計算する。緊急時(誰かが解除中)は
        解除者への接近を、それ以外は配置優先順位に基づく割当を返す。
        戻り値: {agent_idx: (goal_pos, tier)}  tier: 0/1/2=配置スロット, -1=緊急対応
        """
        if self.active_defuser_idx is not None and self.enemy_alive[self.active_defuser_idx]:
            defuser_pos = self.enemy_pos[self.active_defuser_idx]
            return {i: (defuser_pos, -1) for i in range(self.n_guards)}

        order = [(i, self.guard_pos[i]) for i in range(self.n_guards) if self.guard_alive[i]]
        assigned = _assign_goals(
            self.grid, self.plant_pos, self.formation_cells, order, self._cell_dist_maps
        )
        # 死亡中のguardにもダミー値を入れておく（インデックスアクセスを安全にするため）
        for i in range(self.n_guards):
            if i not in assigned:
                assigned[i] = (self.plant_pos, 0)
        return assigned

    def _goal_distance(self, pos, goal):
        """壁を考慮したBFS距離。ゴール（配置スロットや緊急時の解除者位置）
        までの「実際に歩ける経路上の距離」を返す。チェビシェフ距離（直線）
        を使うと壁の向こうを近いと誤認し、角で詰まる原因になるため。
        """
        dm = self._cell_dist_maps.get(goal)
        if dm is None:
            dm = _build_distance_map_from_coords(self.grid, [goal])
            self._cell_dist_maps[goal] = dm
        d = dm[pos[0], pos[1]]
        return float(d) if np.isfinite(d) else DIST_NORM_MAX

    # ------------------------------------------------------------------
    def get_action_mask(self, i):
        mask = np.ones(self.N_ACTIONS, dtype=bool)
        if not self.guard_alive[i]:
            mask[:] = False
            mask[self.ACTION_STAY] = True
            return mask

        r, c = self.guard_pos[i]
        for a, (dr, dc) in self._MOVE_DELTA.items():
            if a == self.ACTION_STAY:
                continue
            if self._is_wall(r + dr, c + dc):
                mask[a] = False

        if self.guard_ability_used[i]:
            mask[self.ACTION_ABILITY] = False

        return mask

    def _get_obs(self, i, goals):
        if not self.guard_alive[i]:
            return np.zeros(45, dtype=np.float32)

        r, c = self.guard_pos[i]
        pr, pc = self.plant_pos
        goal, tier = goals[i]
        gr, gc = goal
        urgency = 1.0 if tier == -1 else 0.0

        obs = []
        obs.append(r / max(1, self.height - 1))
        obs.append(c / max(1, self.width - 1))

        dist_to_plant = _chebyshev((r, c), (pr, pc))
        obs.append(min(1.0, dist_to_plant / DIST_NORM_MAX))
        obs.append(max(-1.0, min(1.0, (pr - r) / DIST_NORM_MAX)))
        obs.append(max(-1.0, min(1.0, (pc - c) / DIST_NORM_MAX)))

        dist_to_goal = self._goal_distance((r, c), (gr, gc))
        obs.append(min(1.0, dist_to_goal / DIST_NORM_MAX))
        obs.append(max(-1.0, min(1.0, (gr - r) / DIST_NORM_MAX)))
        obs.append(max(-1.0, min(1.0, (gc - c) / DIST_NORM_MAX)))

        tier_onehot = [0.0, 0.0, 0.0]
        if tier in (0, 1, 2):
            tier_onehot[tier] = 1.0
        obs.extend(tier_onehot)

        obs.append(urgency)

        if self.active_defuser_idx is not None and self.enemy_alive[self.active_defuser_idx]:
            dpos = self.enemy_pos[self.active_defuser_idx]
            ddist = _chebyshev((r, c), dpos)
            obs.append(min(1.0, ddist / DIST_NORM_MAX))
            obs.append(max(-1.0, min(1.0, (dpos[0] - r) / DIST_NORM_MAX)))
            obs.append(max(-1.0, min(1.0, (dpos[1] - c) / DIST_NORM_MAX)))
            has_los_defuser = 1.0 if self._has_los_to_plant((r, c)) and _has_los(
                self.grid, self._smoke_cell_set(), (r, c), dpos
            ) else 0.0
            obs.append(has_los_defuser)
        else:
            obs.extend([1.0, 0.0, 0.0, 0.0])

        obs.append(1.0 if self._has_los_to_plant((r, c)) else 0.0)

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            obs.append(1.0 if self._is_wall(r + dr, c + dc) else 0.0)

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            wall = self._is_wall(nr, nc)
            gdist = self._goal_distance((nr, nc), (gr, gc))
            obs.append(1.0 if wall else min(1.0, gdist / DIST_NORM_MAX))

        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            obs.append(1.0 if self._is_wall(r + dr, c + dc) else 0.0)

        enemy_idx, enemy_dist = self._nearest_visible_enemy((r, c))
        if enemy_idx is not None:
            er, ec = self.enemy_pos[enemy_idx]
            obs.append(1.0)
            obs.append(max(-1.0, min(1.0, (er - r) / DIST_NORM_MAX)))
            obs.append(max(-1.0, min(1.0, (ec - c) / DIST_NORM_MAX)))
            obs.append(min(1.0, enemy_dist / DIST_NORM_MAX))
            obs.append(self.enemy_blind_remaining[enemy_idx] / max(1, BLIND_DURATION_TICKS))
            obs.append(self.enemy_reveal_remaining[enemy_idx] / max(1, REVEAL_DURATION_TICKS))
        else:
            obs.extend([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

        obs.append(0.0 if self.guard_ability_used[i] else 1.0)
        for ability in ABILITY_TYPES:
            obs.append(1.0 if self.guard_ability_type[i] == ability else 0.0)

        team_effect_active = any(v > 0 for v in self.enemy_blind_remaining) or any(
            v > 0 for v in self.enemy_reveal_remaining
        )
        obs.append(1.0 if team_effect_active else 0.0)

        obs.append(self.guard_last_delta[i][0])
        obs.append(self.guard_last_delta[i][1])
        obs.append(min(1.0, self.guard_stuck[i] / 10.0))
        obs.append(1.0 - min(1.0, self.detonate_timer / max(1, DETONATE_TICKS)))
        obs.append(0.0 if self.guard_last_delta[i] == (0.0, 0.0) else 1.0)  # moved_this_tick

        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    def _apply_ability(self, i):
        ability = self.guard_ability_type[i]
        pos = self.guard_pos[i]
        self.guard_ability_used[i] = True

        if ability in ("FLASH", "RECON"):
            enemy_idx, dist = self._nearest_visible_enemy(pos, max_range=ABILITY_RANGE)
            if enemy_idx is None:
                return -self.ability_waste_penalty
            if ability == "FLASH":
                already = self.enemy_blind_remaining[enemy_idx] > 0
                self.enemy_blind_remaining[enemy_idx] = BLIND_DURATION_TICKS
                self.enemy_blind_source[enemy_idx] = i
            else:
                already = self.enemy_reveal_remaining[enemy_idx] > 0
                self.enemy_reveal_remaining[enemy_idx] = REVEAL_DURATION_TICKS
                self.enemy_reveal_source[enemy_idx] = i
            return -self.ability_redundant_penalty if already else self.ability_success_reward

        enemy_idx, dist = self._nearest_visible_enemy(pos, max_range=ABILITY_RANGE)
        if enemy_idx is None:
            return -self.ability_waste_penalty

        target_pos = self.enemy_pos[enemy_idx]
        smoke_cells = self._smoke_cell_set()
        enemy_had_los_to_plant = _has_los(self.grid, smoke_cells, target_pos, self.plant_pos)

        cells = {
            (rr, cc)
            for rr in range(target_pos[0] - 1, target_pos[0] + 2)
            for cc in range(target_pos[1] - 1, target_pos[1] + 2)
            if 0 <= rr < self.height and 0 <= cc < self.width and self.grid[rr, cc] != 1
        }
        self.smokes.append({"cells": cells, "remaining": SMOKE_DURATION_TICKS})

        return self.ability_success_reward if enemy_had_los_to_plant else -self.ability_waste_penalty

    # ------------------------------------------------------------------
    def _resolve_combat(self):
        smoke_cells = self._smoke_cell_set()

        guards = [("guard", i, self.guard_pos[i]) for i in range(self.n_guards) if self.guard_alive[i]]
        enemies = [("enemy", i, self.enemy_pos[i]) for i in range(self.n_enemies) if self.enemy_alive[i]]

        shooters = []
        for kind, idx, pos in guards:
            e_idx, _ = self._nearest_visible_enemy(pos, max_range=None)
            if e_idx is not None:
                shooters.append((kind, idx, "enemy", e_idx))
        for kind, idx, pos in enemies:
            best_idx, best_dist = None, None
            for gkind, gidx, gpos in guards:
                if not _has_los(self.grid, smoke_cells, pos, gpos):
                    continue
                d = _chebyshev(pos, gpos)
                if best_dist is None or d < best_dist:
                    best_idx, best_dist = (gkind, gidx), d
            if best_idx is not None:
                shooters.append((kind, idx, best_idx[0], best_idx[1]))

        self.rng.shuffle(shooters)

        kill_bonus_targets = []
        defuse_stopped = False

        for shooter_kind, shooter_idx, target_kind, target_idx in shooters:
            shooter_alive = (
                self.guard_alive[shooter_idx] if shooter_kind == "guard"
                else self.enemy_alive[shooter_idx]
            )
            target_alive = (
                self.guard_alive[target_idx] if target_kind == "guard"
                else self.enemy_alive[target_idx]
            )
            if not shooter_alive or not target_alive:
                continue

            accuracy = GENERIC_ACCURACY
            if shooter_kind == "enemy" and self.enemy_blind_remaining[shooter_idx] > 0:
                accuracy *= BLIND_ACCURACY_MULTIPLIER

            dodge = GENERIC_DODGE
            if target_kind == "enemy" and self.enemy_reveal_remaining[target_idx] > 0:
                dodge *= REVEALED_DODGE_MULTIPLIER

            hit_chance = max(0.0, min(1.0, accuracy * (1.0 - dodge)))
            if self.rng.random() >= hit_chance:
                continue

            damage = HEADSHOT_DAMAGE if self.rng.random() < GENERIC_HS_RATE else BODY_DAMAGE

            if target_kind == "guard":
                self.guard_hp[target_idx] = max(0, self.guard_hp[target_idx] - damage)
                if self.guard_hp[target_idx] <= 0:
                    self.guard_alive[target_idx] = False
            else:
                was_blind = self.enemy_blind_remaining[target_idx] > 0
                was_revealed = self.enemy_reveal_remaining[target_idx] > 0
                blind_src = self.enemy_blind_source[target_idx]
                reveal_src = self.enemy_reveal_source[target_idx]
                was_defuser = self.active_defuser_idx == target_idx

                self.enemy_hp[target_idx] = max(0, self.enemy_hp[target_idx] - damage)
                if self.enemy_hp[target_idx] <= 0:
                    self.enemy_alive[target_idx] = False
                    if was_blind and blind_src is not None:
                        kill_bonus_targets.append(blind_src)
                    if was_revealed and reveal_src is not None:
                        kill_bonus_targets.append(reveal_src)
                    if was_defuser:
                        defuse_stopped = True
                        self.active_defuser_idx = None

        return kill_bonus_targets, defuse_stopped

    # ------------------------------------------------------------------
    def _advance_enemies(self):
        """敵の簡易スクリプトAI：設置地点へBFS直進、隣接したら解除を試みる。"""
        for i in range(self.n_enemies):
            if not self.enemy_alive[i]:
                continue
            r, c = self.enemy_pos[i]
            dist = _chebyshev((r, c), self.plant_pos)

            if dist > 1:
                best_cell, best_d = (r, c), self._enemy_goal_dist_map[r, c]
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if self._is_wall(nr, nc):
                        continue
                    occ = self._occupied_all("enemy", i)
                    if (nr, nc) in occ:
                        continue
                    d = self._enemy_goal_dist_map[nr, nc]
                    if np.isfinite(d) and d < best_d:
                        best_d, best_cell = d, (nr, nc)
                self.enemy_pos[i] = best_cell
                self.enemy_defuse_timer[i] = 0
                if self.active_defuser_idx == i:
                    self.active_defuser_idx = None
                continue

            # 隣接している：解除を試みる（既に他の敵が解除中でなければ自分が担当）
            if self.active_defuser_idx is None:
                self.active_defuser_idx = i
            if self.active_defuser_idx == i:
                self.enemy_defuse_timer[i] += 1

    # ------------------------------------------------------------------
    def step(self, actions):
        self.tick += 1
        rewards = [0.0] * self.n_guards
        info = {"success": False, "defused": False, "all_dead": False}

        for i in range(self.n_enemies):
            self.enemy_blind_remaining[i] = max(0, self.enemy_blind_remaining[i] - 1)
            self.enemy_reveal_remaining[i] = max(0, self.enemy_reveal_remaining[i] - 1)
        for smoke in self.smokes:
            smoke["remaining"] -= 1
        self.smokes = [s for s in self.smokes if s["remaining"] > 0]

        for i in range(self.n_guards):
            rewards[i] -= 0.01

        goals_before = self._compute_goals()

        used_ability_this_tick = set()
        for i in range(self.n_guards):
            if not self.guard_alive[i] or actions[i] is None:
                continue
            if actions[i] == self.ACTION_ABILITY:
                rewards[i] += self._apply_ability(i)
                used_ability_this_tick.add(i)
                self.guard_last_delta[i] = (0.0, 0.0)
                self.guard_stuck[i] += 1

        self._advance_enemies()

        move_order = [
            i for i in range(self.n_guards)
            if self.guard_alive[i] and i not in used_ability_this_tick and actions[i] is not None
        ]
        self.rng.shuffle(move_order)
        for i in move_order:
            action = actions[i]
            r, c = self.guard_pos[i]

            if action == self.ACTION_STAY or action not in self._MOVE_DELTA:
                self.guard_last_delta[i] = (0.0, 0.0)
                self.guard_stuck[i] += 1
                continue

            dr, dc = self._MOVE_DELTA[action]
            nr, nc = r + dr, c + dc

            if self._is_wall(nr, nc):
                self.guard_last_delta[i] = (0.0, 0.0)
                self.guard_stuck[i] += 1
                continue

            occ = self._occupied_all("guard", i)
            if (nr, nc) in occ:
                self.guard_last_delta[i] = (0.0, 0.0)
                self.guard_stuck[i] += 1
                continue

            self.guard_pos[i] = (nr, nc)
            self.guard_last_delta[i] = (float(dr), float(dc))
            self.guard_stuck[i] = 0

        goals_after = self._compute_goals()

        for i in range(self.n_guards):
            if not self.guard_alive[i]:
                continue
            goal, tier = goals_after[i]
            urgency = tier == -1
            new_dist = self._goal_distance(self.guard_pos[i], goal)
            coef = self.urgency_shaping_coef if urgency else self.shaping_coef
            rewards[i] += (self._prev_goal_dist[i] - new_dist) * coef
            self._prev_goal_dist[i] = new_dist

            moved = self.guard_last_delta[i] != (0.0, 0.0)
            has_cover_now = self._has_los_to_plant(self.guard_pos[i])
            if not urgency:
                if not moved and has_cover_now:
                    rewards[i] += self.cover_stationary_bonus
                elif moved and self._prev_had_cover[i]:
                    rewards[i] -= self.cover_move_penalty
            self._prev_had_cover[i] = has_cover_now

        kill_bonus_targets, defuse_stopped = self._resolve_combat()
        for guard_idx in kill_bonus_targets:
            if 0 <= guard_idx < self.n_guards:
                rewards[guard_idx] += self.kill_bonus
        if defuse_stopped:
            for i in range(self.n_guards):
                if self.guard_alive[i]:
                    rewards[i] += self.defuse_stopped_bonus

        done = False
        is_defused = any(
            self.enemy_alive[i] and self.enemy_defuse_timer[i] >= DEFUSE_REQUIRED_TICKS
            for i in range(self.n_enemies)
        )
        no_guards_alive = not any(self.guard_alive)
        no_enemies_alive = not any(self.enemy_alive)

        if is_defused:
            done = True
            info["defused"] = True
            for i in range(self.n_guards):
                if self.guard_alive[i]:
                    rewards[i] -= self.mission_fail_penalty
        elif no_guards_alive:
            done = True
            info["all_dead"] = True
            # 生存者がいないので追加ペナルティは付与できない（既に死亡報酬計算対象なし）
        elif no_enemies_alive:
            done = True
            info["success"] = True
            for i in range(self.n_guards):
                if self.guard_alive[i]:
                    rewards[i] += self.mission_success_reward
        else:
            self.detonate_timer -= 1
            if self.detonate_timer <= 0:
                done = True
                info["success"] = True
                for i in range(self.n_guards):
                    if self.guard_alive[i]:
                        rewards[i] += self.mission_success_reward
            elif self.tick >= self.max_ticks:
                done = True
                info["success"] = False
                for i in range(self.n_guards):
                    if self.guard_alive[i]:
                        rewards[i] -= self.mission_fail_penalty

        next_obs = [self._get_obs(i, goals_after) for i in range(self.n_guards)]
        return next_obs, rewards, done, info


# ---------------------------------------------------------------------------
# Dueling DQN
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


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "next_mask", "done"))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


def select_action(net, state, mask, epsilon, device):
    if random.random() < epsilon:
        valid_actions = np.flatnonzero(mask)
        return int(random.choice(valid_actions))
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = net(state_t).squeeze(0).cpu().numpy()
    q = np.where(mask, q, -1e9)
    return int(np.argmax(q))


def optimize_model(policy_net, target_net, optimizer, buffer, batch_size, gamma, device):
    if len(buffer) < batch_size:
        return None

    batch = buffer.sample(batch_size)

    states = torch.as_tensor(np.array(batch.state), dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.as_tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
    dones = torch.as_tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)
    next_masks = torch.as_tensor(np.array(batch.next_mask), dtype=torch.bool, device=device)

    q_values = policy_net(states).gather(1, actions)

    with torch.no_grad():
        next_q_policy = policy_net(next_states)
        next_q_policy = next_q_policy.masked_fill(~next_masks, -1e9)
        next_actions = next_q_policy.argmax(dim=1, keepdim=True)
        next_q_target = target_net(next_states).gather(1, next_actions)
        target = rewards + gamma * next_q_target * (1.0 - dones)

    loss = F.smooth_l1_loss(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()

    return float(loss.item())


def evaluate(env, policy_net, device, episodes=20):
    successes = 0
    total_reward = 0.0
    defused_count = 0
    all_dead_count = 0

    for _ in range(episodes):
        obs_list = env.reset()
        done = False
        episode_reward = 0.0
        info = {"success": False}

        while not done:
            actions = []
            for i in range(env.n_guards):
                if not env.guard_alive[i]:
                    actions.append(None)
                    continue
                mask = env.get_action_mask(i)
                with torch.no_grad():
                    state_t = torch.as_tensor(obs_list[i], dtype=torch.float32, device=device).unsqueeze(0)
                    q = policy_net(state_t).squeeze(0).cpu().numpy()
                q = np.where(mask, q, -1e9)
                actions.append(int(np.argmax(q)))

            next_obs_list, rewards, done, info = env.step(actions)
            episode_reward += sum(rewards)
            obs_list = next_obs_list

        total_reward += episode_reward
        if info.get("success"):
            successes += 1
        if info.get("defused"):
            defused_count += 1
        if info.get("all_dead"):
            all_dead_count += 1

    success_rate = successes / episodes
    avg_reward = total_reward / episodes
    defused_rate = defused_count / episodes
    all_dead_rate = all_dead_count / episodes
    return success_rate, avg_reward, defused_rate, all_dead_rate


def main():
    parser = argparse.ArgumentParser(description="Attacker Guard Phase 学習スクリプト")
    parser.add_argument("--episodes", type=int, default=EPISODE_COUNT)
    parser.add_argument("--max-ticks", type=int, default=DETONATE_TICKS + 20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-episodes", type=int, default=4000)
    parser.add_argument("--target-update-every", type=int, default=800)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "attacker_v3", "data", "attacker_guard_data"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    env = GuardEnv(max_ticks=args.max_ticks, seed=args.seed)
    eval_env = GuardEnv(max_ticks=args.max_ticks, seed=args.seed + 1)

    print(f"[INFO] formation_cells(tier3)={env.formation_cells}")

    # obs_dim は静的に固定せず、実際の reset() の出力形状から動的に取得する
    # （train_attacker_escort.py で発生した次元不一致バグの再発防止）。
    obs_list = env.reset()
    obs_dim = obs_list[0].shape[0]
    n_actions = env.N_ACTIONS
    print(f"[INFO] obs_dim={obs_dim} n_actions={n_actions} n_guards={env.n_guards} device={device}")

    policy_net = DuelingQNetwork(obs_dim, n_actions).to(device)
    target_net = DuelingQNetwork(obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)

    best_success_rate = -1.0
    best_eval_reward = float("-inf")
    global_step = 0

    def _make_checkpoint(episode, success_rate, avg_reward, defused_rate, all_dead_rate):
        return {
            "model_state_dict": policy_net.state_dict(),
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "episode": episode,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "defused_rate": defused_rate,
            "all_dead_rate": all_dead_rate,
            # 本番マップ(map_data.py)には grid==6 は存在しない。
            # 推論側はこの座標リストを固定の目的地候補として使う。
            "formation_cells": [(int(r), int(c)) for r, c in env.formation_cells],
        }

    for episode in range(1, args.episodes + 1):
        progress = min(1.0, episode / args.eps_decay_episodes)
        epsilon = args.eps_start + (args.eps_end - args.eps_start) * progress

        obs_list = env.reset()
        done = False
        episode_reward = 0.0
        info = {"success": False}

        while not done:
            actions = []
            masks = []
            for i in range(env.n_guards):
                if not env.guard_alive[i]:
                    actions.append(None)
                    masks.append(None)
                    continue
                mask = env.get_action_mask(i)
                action = select_action(policy_net, obs_list[i], mask, epsilon, device)
                actions.append(action)
                masks.append(mask)

            next_obs_list, rewards, done, info = env.step(actions)

            for i in range(env.n_guards):
                if masks[i] is None:
                    continue
                next_mask = env.get_action_mask(i)
                agent_done = done or not env.guard_alive[i]
                buffer.push(obs_list[i], actions[i], rewards[i], next_obs_list[i], next_mask, agent_done)

            obs_list = next_obs_list
            episode_reward += sum(rewards)
            global_step += 1

            if len(buffer) >= max(args.batch_size, args.warmup_steps):
                optimize_model(
                    policy_net, target_net, optimizer, buffer,
                    args.batch_size, args.gamma, device,
                )

            if global_step % args.target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if episode % 50 == 0:
            print(
                f"[EP {episode}/{EPISODE_COUNT}] reward={episode_reward:.2f} eps={epsilon:.3f} "
                f"success={info.get('success')} defused={info.get('defused')} "
                f"all_dead={info.get('all_dead')} ticks={env.tick}"
            )

        if episode % args.eval_every == 0:
            success_rate, avg_reward, defused_rate, all_dead_rate = evaluate(
                eval_env, policy_net, device, args.eval_episodes
            )
            print(
                f"[EVAL @ EP {episode}/{EPISODE_COUNT}] success_rate={success_rate:.2%} "
                f"avg_reward={avg_reward:.2f} defused_rate={defused_rate:.2%} "
                f"all_dead_rate={all_dead_rate:.2%}"
            )

            latest_path = os.path.join(args.save_dir, "dqn_attacker_guard_latest.pt")
            torch.save(
                _make_checkpoint(episode, success_rate, avg_reward, defused_rate, all_dead_rate),
                latest_path,
            )

            is_better = (
                success_rate > best_success_rate + 1e-9
                or (
                    success_rate >= best_success_rate - 1e-9
                    and avg_reward > best_eval_reward
                )
            )
            if is_better:
                best_success_rate = max(best_success_rate, success_rate)
                best_eval_reward = avg_reward
                best_path = os.path.join(args.save_dir, "dqn_attacker_guard_best_by_eval.pt")
                torch.save(
                    _make_checkpoint(episode, success_rate, avg_reward, defused_rate, all_dead_rate),
                    best_path,
                )
                print(
                    f"[SAVE] 新しいベストモデルを保存: {best_path} "
                    f"(success_rate={success_rate:.2%}, avg_reward={avg_reward:.2f})"
                )

    print("[DONE] 学習が完了しました。")


if __name__ == "__main__":
    main()