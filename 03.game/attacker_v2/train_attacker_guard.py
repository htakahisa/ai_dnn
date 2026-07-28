# train_attacker_guard.py
"""
ガードAI(guard、プラント後フェーズ専用)の学習スクリプト。
train_attacker_carry.py / train_attacker_escort.py / train_attacker_retrieve.py と同様、
外部の学習系モジュールには依存せず、必要なクラスはすべてこのファイル内に複製する。

guardの役割:
- プラント位置から2〜6マス程度の距離を保ちつつ、
  「プラント周囲8マス(解除可能マス)」+「サイトへの通路入口」への視線をチームで
  できるだけ多くカバーする(ただし1箇所に集中しすぎても極端な無駄にはならないよう、
  sqrt(n)による頭打ち+差分報酬でゆるく分散を誘導する)。
- 敵(解除に来るDefender)を視認したら即座にアビリティで反応する。
- 敵が解除中(busy)であれば、視認さえできれば確定で有利に交戦できる
  (run_game.py の process_battle と同じ優先順位: blind > busy > 五分五分)。

💡簡略化している点:
- 同時に相手をするDefenderは1体のみ(複数人が同時に解除に来るケースは扱わない)。
- Defenderはアビリティを持たない(現行の assign_abilities は team "A" のみに割り当てる仕様と一致)。
- Defender bot の移動はBFS勾配に沿った直進のみ(索敵・迂回等はしない)。
- 「通路入口」候補はサイトコンポーネント単位で事前計算し、実際のプラント位置に関わらず
  エピソードを通して固定する(プラント位置ごとに毎回計算すると学習が不安定になりやすいため)。
- 報酬の係数・重みはすべて仮置き。学習ログを見ながら調整すること。
"""

import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from map_data import NEW_MAZE_STR
from game_core import (
    WINNING_ROUNDS,
    TICK_TIME,
    MAX_HP,
    BODY_DAMAGE,
    HEADSHOT_DAMAGE,
    SHOOT_INTERVAL_TICKS,
    SIDE_PANEL_WIDTH,
    PLANT_REQUIRED_TICKS,
    DEFUSE_REQUIRED_TICKS,
    SMOKE_DURATION_TICKS,
    MOVING_ACCURACY,
    MOVING_TARGET_HIT_MULTIPLIER,
    BLIND_DURATION_TICKS,
    FLASH_BURST_DURATION_TICKS,
    BLIND_ACCURACY_MULTIPLIER,
    FLASH_SPEED_CELLS_PER_TICK,
    FLASH_MAX_FLIGHT_TICKS,
    RECON_SPEED_CELLS_PER_TICK,
    REVEAL_DURATION_TICKS,
    REVEALED_DODGE_MULTIPLIER,
    RECON_REVEAL_SIZE,
    COMBO_DISPLAY_TICKS,
    COMBO_BANNER_HEIGHT,
    ROUND_DURATION_TICKS,
    SPIKE_DETONATION_TICKS,
    RECON_BURST_DISPLAY_TICKS,
    SMOKE_WARNING_TICKS,
    ROUND_TRANSITION_TICKS,
    ABILITY_TYPES,
    FLASH_BLIND_TICKS,
    SMOKE_RADIUS,
    RECON_RADIUS,
)

OBS_DIM = 93   # 内訳: 自己位置2 + 壁4 + 直前行動5 + サイト距離1 + 距離帯フラグ1
               #       + 生存味方数1 + 味方スロット(最大2体)*4=8
               #       + ターゲットセル(12箇所)*5=60 + アビリティ4 + 敵情報6 + detonate残り1
N_ACTIONS = 5  # 0:上 1:下 2:左 3:右 4:アビリティ使用
NUM_EPISODES = 18000
SAVE_INTERVAL = 100
ABILITY_TYPES = ["flash", "smoke", "recon"]

MAX_GUARDS = 3              # 💡仮置き: 同時に学習させるguard最大人数
DETONATE_TICKS = 45         # run_game.py の self.detonate_timer と一致させる
GUARD_DEFUSE_REQUIRED = 6   # run_game.py の DEFUSE_REQUIRED(=6) と一致させる
MAX_STEPS = DETONATE_TICKS + 5  # 安全のためのバッファ

# ---- 距離維持(項目1) ----
GUARD_MIN_DIST = 0
GUARD_MAX_DIST = 3
IN_RANGE_REWARD = 0.3                 # 💡仮置き
OUT_OF_RANGE_PENALTY_SCALE = 0.5      # 💡仮置き

# ---- カバレッジ(項目2, 5) ----
NUM_PERIMETER_SLOTS = 8               # プラント位置周囲8マス(固定オフセット)
NUM_ENTRANCE_SLOTS = 4                # 通路入口候補(サイトごとに事前計算、固定)
NUM_TARGET_SLOTS = NUM_PERIMETER_SLOTS + NUM_ENTRANCE_SLOTS
PERIMETER_CELL_WEIGHT = 2.0           # 💡仮置き: 解除阻止に直結するため重め
ENTRANCE_CELL_WEIGHT = 0.1            # 💡仮置き
ENTRANCE_MIN_DIST = 2
ENTRANCE_MAX_DIST = 6
ENTRANCE_MIN_SEPARATION = 3
COVERAGE_REWARD_SCALE = 1.0           # 💡仮置き: 差分報酬全体のスケール

# ---- アビリティ(項目4: 視認したら即反応のみを評価) ----
ABILITY_USE_COST = -0.2
FLASH_SUCCESS_BONUS = 8.0
FLASH_REACTIVE_THROW_BONUS = 5.0      # 反応的に投げたが外れた場合の慰め程度のボーナス
RECON_SUCCESS_BONUS = 2.0
SMOKE_BLOCK_BONUS = 1.5               # 💡追加: 反応的に敵の視線を遮断できた場合のボーナス

# ---- 交戦・勝敗(項目3) ----
DEATH_PENALTY = -20.0                 # 💡仮置き
KILL_BOT_REWARD = 21.0                # 💡仮置き
GUARD_SUCCESS_REWARD = 50.0           # 💡仮置き: 解除阻止(タイマー切れ or 敵全滅)成功時
DEFUSE_FAIL_PENALTY = -50.0           # 💡仮置き: 解除成功されてしまった場合


# ===========================================================================
# 共通ヘルパー(train_attacker_escort.py / train_attacker_retrieve.py と同じロジックを複製)
# ===========================================================================
def bfs_distances(target, grid):
    height, width = grid.shape
    dist = np.full((height, width), np.inf)
    tr, tc = target
    dist[tr, tc] = 0
    q = deque([(tr, tc)])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
                if dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
    return dist


def split_site_components(cells):
    cells_set = set(cells)
    visited = set()
    components = []
    for cell in cells:
        if cell in visited:
            continue
        comp = []
        queue = deque([cell])
        visited.add(cell)
        while queue:
            r, c = queue.popleft()
            comp.append((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells_set and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        components.append(comp)
    return components


def multi_source_bfs(site_cells, grid):
    height, width = grid.shape
    dist = np.full((height, width), np.inf)
    q = deque()
    for (r, c) in site_cells:
        dist[r, c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
                if dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
    return dist


def find_corner_cells(grid):
    height, width = grid.shape
    corners = []
    for r in range(height):
        for c in range(width):
            if grid[r, c] == 1:
                continue
            north = grid[r - 1, c] == 1 if r - 1 >= 0 else True
            south = grid[r + 1, c] == 1 if r + 1 < height else True
            west = grid[r, c - 1] == 1 if c - 1 >= 0 else True
            east = grid[r, c + 1] == 1 if c + 1 < width else True
            pairs = [(north, west), (north, east), (south, west), (south, east)]
            if any(a and b for a, b in pairs):
                corners.append((r, c))
    return corners


def line_cells(p1, p2):
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


def has_line_of_sight(p1, p2, grid, smoke_cells=None):
    cells = line_cells(p1, p2)
    for (r, c) in cells:
        if grid[r, c] == 1:
            return False
    if smoke_cells:
        for cell in cells:
            if cell in smoke_cells:
                return False
    return True


def compute_projectile_path(start, aimed_cell, grid):
    height, width = grid.shape
    sr, sc = start
    ar, ac = aimed_cell
    dr, dc = ar - sr, ac - sc
    if dr == 0 and dc == 0:
        return [start]
    scale = max(height, width) * 3
    far = (sr + dr * scale, sc + dc * scale)
    raw = line_cells(start, far)
    path = [start]
    for rr, cc in raw[1:]:
        if not (0 <= rr < height and 0 <= cc < width):
            break
        if grid[rr, cc] == 1:
            break
        path.append((rr, cc))
    return path


def select_entrance_cells(grid, dist_map, corner_cells, min_dist, max_dist, num_points, min_separation):
    """サイトからの距離帯にある曲がり角(通路の入口らしいマス)を、
    間隔を空けながら最大num_points個選ぶ。💡簡略化: スコアは「曲がり角かどうか」の
    二値のみ(将来的に自動でより精緻なボトルネック判定に置き換えてもよい)。"""
    candidates = []
    for (r, c) in corner_cells:
        d = dist_map[r, c]
        if np.isfinite(d) and min_dist <= d <= max_dist:
            candidates.append((r, c))
    random.shuffle(candidates)  # 同点多数の中から偏りなく選ぶ

    selected = []
    for pos in candidates:
        if all(max(abs(pos[0] - s[0]), abs(pos[1] - s[1])) >= min_separation for s in selected):
            selected.append(pos)
        if len(selected) >= num_points:
            break
    return selected


PERIMETER_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]  # 💡固定順序。プラント位置からの相対オフセットとして毎エピソード同じ並びにする。


# ===========================================================================
# リプレイバッファ(mask / next_mask対応版)
# ===========================================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, obs, action, reward, next_obs, done, mask, next_mask):
        self.buffer.append((obs, action, reward, next_obs, done, mask, next_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done, mask, next_mask = zip(*batch)
        return (np.array(obs), np.array(action), np.array(reward),
                np.array(next_obs), np.array(done), np.array(mask), np.array(next_mask))


# ===========================================================================
# Dueling DQN構造
# ===========================================================================
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(128, 1)
        self.advantage_head = nn.Linear(128, action_dim)

    def forward(self, x):
        h = self.shared(x)
        value = self.value_head(h)
        advantage = self.advantage_head(h)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


# ===========================================================================
# guard専用マルチエージェント環境
# (同一の共有方策を、生存guard全員に対して独立Q学習の形で適用する)
# ===========================================================================
class GuardMultiEnv:
    def __init__(self, fixed_grid, plant_candidates, max_guards=MAX_GUARDS):
        self.grid = fixed_grid
        self.height, self.width = fixed_grid.shape
        self.max_guards = max_guards
        self.max_steps = MAX_STEPS

        self.corner_cells = find_corner_cells(self.grid)
        self.walkable_cells = [
            (r, c) for r in range(self.height) for c in range(self.width) if self.grid[r, c] != 1
        ]

        self.site_components = split_site_components([tuple(p) for p in plant_candidates])
        self.site_cell_sets = [set(comp) for comp in self.site_components]
        self.site_dist_maps = [multi_source_bfs(comp, self.grid) for comp in self.site_components]

        # 💡通路入口候補はサイトコンポーネントごとに一度だけ計算し、エピソード間で使い回す
        self.entrance_candidates = [
            select_entrance_cells(
                self.grid, dist_map, self.corner_cells,
                ENTRANCE_MIN_DIST, ENTRANCE_MAX_DIST, NUM_ENTRANCE_SLOTS, ENTRANCE_MIN_SEPARATION,
            )
            for dist_map in self.site_dist_maps
        ]

    def _is_walkable(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width and self.grid[r, c] != 1

    def _random_walkable(self):
        return random.choice(self.walkable_cells)

    # -----------------------------------------------------------------
    # 初期化
    # -----------------------------------------------------------------
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.detonate_timer = DETONATE_TICKS

        self.current_site_index = random.randrange(len(self.site_components))
        site_cells = self.site_cell_sets[self.current_site_index]
        self.site_dist_map = self.site_dist_maps[self.current_site_index]
        self.planted_pos = random.choice(list(site_cells))

        # ---- ターゲットセル(周囲8マス + 通路入口)の確定 ----
        self.target_positions = [None] * NUM_TARGET_SLOTS
        self.target_weights = [0.0] * NUM_TARGET_SLOTS
        for i, (dr, dc) in enumerate(PERIMETER_OFFSETS):
            r, c = self.planted_pos[0] + dr, self.planted_pos[1] + dc
            if self._is_walkable(r, c):
                self.target_positions[i] = (r, c)
                self.target_weights[i] = PERIMETER_CELL_WEIGHT
        entrances = self.entrance_candidates[self.current_site_index]
        for j in range(NUM_ENTRANCE_SLOTS):
            slot = NUM_PERIMETER_SLOTS + j
            if j < len(entrances):
                self.target_positions[slot] = entrances[j]
                self.target_weights[slot] = ENTRANCE_CELL_WEIGHT

        # ---- guard初期化 ----
        self.num_guards = random.randint(1, self.max_guards)
        self.guard_alive = [True] * self.num_guards
        self.guard_pos = [self._random_walkable() for _ in range(self.num_guards)]
        self.guard_last_action = [None] * self.num_guards
        self.guard_ability_type = [random.choice(ABILITY_TYPES) for _ in range(self.num_guards)]
        self.guard_ability_charge = [1.0] * self.num_guards
        self.guard_last_ability_result = [None] * self.num_guards

        # ---- 敵(解除に来るDefender bot)初期化 ----
        self.bot_alive = True
        self.bot_pos = random.choice(self.corner_cells) if self.corner_cells else self._random_walkable()
        self.bot_dist_map = bfs_distances(self.planted_pos, self.grid)
        self.bot_blind_remaining = 0
        self.bot_defuse_timer = 0

        # ---- 投射物・煙(全guard共有のフィールド状態) ----
        self.pending_flashes = []   # 各要素: {"owner":idx, "path":..., "progress":..., "ticks_alive":..., "is_reactive":...}
        self.pending_recons = []    # 各要素: {"owner":idx, "path":..., "progress":...}
        self.smokes = []            # 各要素: {"cells": set, "remaining_ticks": int}

        return self._get_all_obs(), {}

    # -----------------------------------------------------------------
    # 観測構築
    # -----------------------------------------------------------------
    def _smoke_cells(self):
        cells = set()
        for s in self.smokes:
            cells.update(s["cells"])
        return cells

    def _compute_guard_los_matrix(self):
        """guard_los[i][t] = guard i がターゲットセルtへの視線を持つか。敵がいる場合のLOSとは別物。"""
        smoke_cells = self._smoke_cells()
        los = [[False] * NUM_TARGET_SLOTS for _ in range(self.num_guards)]
        for i in range(self.num_guards):
            if not self.guard_alive[i]:
                continue
            for t in range(NUM_TARGET_SLOTS):
                pos = self.target_positions[t]
                if pos is None:
                    continue
                los[i][t] = has_line_of_sight(self.guard_pos[i], pos, self.grid, smoke_cells)
        return los

    def _get_obs_for(self, i, guard_los):
        if not self.guard_alive[i]:
            return np.zeros(OBS_DIM, dtype=np.float32)

        pr, pc = self.guard_pos[i]
        height, width = self.height, self.width
        smoke_cells = self._smoke_cells()

        base = [pr / (height - 1), pc / (width - 1)]
        walls = [0.0 if self._is_walkable(pr + dr, pc + dc) else 1.0
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        last_onehot = [0.0] * N_ACTIONS
        if self.guard_last_action[i] is not None:
            last_onehot[self.guard_last_action[i]] = 1.0

        site_max_dist = max(height, width) * 2
        d_site = self.site_dist_map[pr, pc]
        dist_to_site = [d_site / site_max_dist if np.isfinite(d_site) else 1.0]
        chebyshev_to_plant = max(abs(pr - self.planted_pos[0]), abs(pc - self.planted_pos[1]))
        in_range_flag = [1.0 if GUARD_MIN_DIST <= chebyshev_to_plant <= GUARD_MAX_DIST else 0.0]

        alive_others = [j for j in range(self.num_guards) if j != i and self.guard_alive[j]]
        num_alive = 1 + len(alive_others)  # 自分を含む
        teammate_count_norm = [num_alive / self.max_guards]

        # 味方スロット: 自分に近い順、最大 (max_guards-1) 体分
        others_sorted = sorted(
            alive_others,
            key=lambda j: max(abs(self.guard_pos[j][0] - pr), abs(self.guard_pos[j][1] - pc))
        )
        teammate_feats = []
        max_dist = max(height, width)
        for slot in range(self.max_guards - 1):
            if slot < len(others_sorted):
                j = others_sorted[slot]
                jr, jc = self.guard_pos[j]
                d = max(abs(jr - pr), abs(jc - pc))
                teammate_feats += [1.0, (jr - pr) / height, (jc - pc) / width, d / max_dist]
            else:
                teammate_feats += [0.0, 0.0, 0.0, 0.0]

        # ターゲットセル情報
        target_feats = []
        for t in range(NUM_TARGET_SLOTS):
            pos = self.target_positions[t]
            if pos is None:
                target_feats += [0.0, 0.0, 0.0, 0.0, 0.0]
                continue
            tr, tc = pos
            self_los = 1.0 if guard_los[i][t] else 0.0
            others_los_count = sum(
                1 for j in alive_others if guard_los[j][t]
            )
            others_norm = others_los_count / max(1, self.max_guards - 1)
            target_feats += [1.0, (tr - pr) / height, (tc - pc) / width, self_los, others_norm]

        ability_onehot = [1.0 if self.guard_ability_type[i] == t else 0.0 for t in ABILITY_TYPES]
        ability_charge = [self.guard_ability_charge[i]]

        enemy = [0.0, 0.0, 0.0]
        enemy_visible = False
        if self.bot_alive:
            enemy_visible = has_line_of_sight(self.guard_pos[i], self.bot_pos, self.grid, smoke_cells)
            if enemy_visible:
                br, bc = self.bot_pos
                enemy = [1.0, (br - pr) / height, (bc - pc) / width]
        bot_busy = [1.0 if (self.bot_alive and self._bot_is_defusing()) else 0.0]
        defuse_progress = [self.bot_defuse_timer / GUARD_DEFUSE_REQUIRED if self.bot_alive else 0.0]
        bot_blind_flag = [1.0 if (self.bot_alive and self.bot_blind_remaining > 0) else 0.0]
        enemy_info = enemy + bot_busy + defuse_progress + bot_blind_flag

        detonate_ratio = [self.detonate_timer / DETONATE_TICKS]

        return np.array(
            base + walls + last_onehot + dist_to_site + in_range_flag + teammate_count_norm +
            teammate_feats + target_feats + ability_onehot + ability_charge + enemy_info + detonate_ratio,
            dtype=np.float32
        )

    def _get_all_obs(self):
        guard_los = self._compute_guard_los_matrix()
        return [self._get_obs_for(i, guard_los) for i in range(self.num_guards)]

    def get_action_masks(self):
        masks = []
        for i in range(self.num_guards):
            if not self.guard_alive[i]:
                masks.append(np.zeros(N_ACTIONS, dtype=np.float32))
                continue
            pr, pc = self.guard_pos[i]
            moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
            mask = np.zeros(N_ACTIONS, dtype=np.float32)
            for a, (dr, dc) in moves.items():
                mask[a] = 1.0 if self._is_walkable(pr + dr, pc + dc) else 0.0
            mask[4] = 1.0 if self.guard_ability_charge[i] > 0 else 0.0
            masks.append(mask)
        return masks

    # -----------------------------------------------------------------
    # アビリティ
    # -----------------------------------------------------------------
    def _get_aim_direction(self, i):
        pr, pc = self.guard_pos[i]
        smoke_cells = self._smoke_cells()
        if self.bot_alive and has_line_of_sight(self.guard_pos[i], self.bot_pos, self.grid, smoke_cells):
            return self.bot_pos, True  # (狙うマス, 反応的かどうか)

        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        if self.guard_last_action[i] in moves:
            dr, dc = moves[self.guard_last_action[i]]
            return (pr + dr, pc + dc), False
        return (pr, pc + 1), False

    def _use_ability(self, i):
        if self.guard_ability_charge[i] <= 0:
            return -1.0

        self.guard_ability_charge[i] = 0.0
        ability_type = self.guard_ability_type[i]
        aimed_cell, is_reactive = self._get_aim_direction(i)
        path = compute_projectile_path(self.guard_pos[i], aimed_cell, self.grid)

        self.guard_last_ability_result[i] = {"type": ability_type, "success": False, "reactive": is_reactive}
        reward = ABILITY_USE_COST

        if ability_type == "flash":
            self.pending_flashes.append({
                "owner": i, "path": path, "progress": 0, "ticks_alive": 0, "is_reactive": is_reactive,
            })
        elif ability_type == "recon":
            self.pending_recons.append({"owner": i, "path": path, "progress": 0})
        elif ability_type == "smoke":
            tr, tc = path[-1]
            cells = {
                (rr, cc)
                for rr in range(tr - SMOKE_RADIUS, tr + SMOKE_RADIUS + 1)
                for cc in range(tc - SMOKE_RADIUS, tc + SMOKE_RADIUS + 1)
                if self._is_walkable(rr, cc)
            }
            self.smokes.append({"cells": cells, "remaining_ticks": SMOKE_DURATION_TICKS})
            if is_reactive and self.bot_alive:
                blocked = not has_line_of_sight(self.guard_pos[i], self.bot_pos, self.grid, self._smoke_cells())
                if blocked:
                    self.guard_last_ability_result[i]["success"] = True
                    reward += SMOKE_BLOCK_BONUS

        return reward

    def _advance_projectiles(self):
        rewards = [0.0] * self.num_guards
        smoke_cells = self._smoke_cells()

        remaining = []
        for p in self.pending_flashes:
            p["ticks_alive"] += 1
            next_progress = p["progress"] + FLASH_SPEED_CELLS_PER_TICK
            hit_wall_or_edge = next_progress >= len(p["path"]) - 1
            p["progress"] = min(next_progress, len(p["path"]) - 1)
            if hit_wall_or_edge or p["ticks_alive"] >= FLASH_MAX_FLIGHT_TICKS:
                impact = p["path"][p["progress"]]
                success = False
                if self.bot_alive and has_line_of_sight(impact, self.bot_pos, self.grid, smoke_cells):
                    self.bot_blind_remaining = max(self.bot_blind_remaining, BLIND_DURATION_TICKS)
                    success = True
                    rewards[p["owner"]] += FLASH_SUCCESS_BONUS
                elif p["is_reactive"]:
                    rewards[p["owner"]] += FLASH_REACTIVE_THROW_BONUS
                self.guard_last_ability_result[p["owner"]] = {"type": "flash", "success": success, "reactive": p["is_reactive"]}
            else:
                remaining.append(p)
        self.pending_flashes = remaining

        remaining = []
        for p in self.pending_recons:
            next_progress = p["progress"] + RECON_SPEED_CELLS_PER_TICK
            hit_wall_or_edge = next_progress >= len(p["path"]) - 1
            p["progress"] = min(next_progress, len(p["path"]) - 1)
            if hit_wall_or_edge:
                impact = p["path"][p["progress"]]
                ir, ic = impact
                radius = RECON_REVEAL_SIZE // 2
                success = False
                if self.bot_alive:
                    br, bc = self.bot_pos
                    if abs(br - ir) <= radius and abs(bc - ic) <= radius:
                        success = True
                        rewards[p["owner"]] += RECON_SUCCESS_BONUS
                self.guard_last_ability_result[p["owner"]] = {"type": "recon", "success": success, "reactive": True}
            else:
                remaining.append(p)
        self.pending_recons = remaining

        for s in self.smokes:
            s["remaining_ticks"] -= 1
        self.smokes = [s for s in self.smokes if s["remaining_ticks"] > 0]

        return rewards

    # -----------------------------------------------------------------
    # 敵(Defender bot)の移動・解除処理
    # -----------------------------------------------------------------
    def _bot_is_defusing(self):
        d = max(abs(self.bot_pos[0] - self.planted_pos[0]), abs(self.bot_pos[1] - self.planted_pos[1]))
        return d <= 1 and self.bot_blind_remaining <= 0

    def _move_bot(self):
        if self.bot_blind_remaining > 0:
            return  # 💡簡略化: フラッシュ中は行動不能(移動も解除も不可)
        if self._bot_is_defusing():
            return  # 解除範囲内なら移動しない

        br, bc = self.bot_pos
        candidates = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = br + dr, bc + dc
            if self._is_walkable(nr, nc):
                d = self.bot_dist_map[nr, nc]
                if np.isfinite(d):
                    candidates.append((d, (nr, nc)))
        if not candidates:
            return
        best_d = min(d for d, _ in candidates)
        best_moves = [pos for d, pos in candidates if d == best_d]
        self.bot_pos = random.choice(best_moves)

    # -----------------------------------------------------------------
    # 交戦解決(run_game.py の process_battle と同じ優先順位: blind > busy > 五分五分)
    # -----------------------------------------------------------------
    def _resolve_engagements(self):
        rewards = [0.0] * self.num_guards
        smoke_cells = self._smoke_cells()
        defuse_failed = False
        bot_killed_by = None

        if not self.bot_alive:
            return rewards, defuse_failed, bot_killed_by

        engaged = [
            i for i in range(self.num_guards)
            if self.guard_alive[i] and has_line_of_sight(self.guard_pos[i], self.bot_pos, self.grid, smoke_cells)
        ]
        random.shuffle(engaged)

        bot_busy = self._bot_is_defusing()
        bot_blind = self.bot_blind_remaining > 0

        for i in engaged:
            if not self.bot_alive or not self.guard_alive[i]:
                continue
            # guard側は blind/busy にならない(現行仕様上Defenderはアビリティを持たない)
            if bot_blind:
                bot_dies = True
            elif bot_busy:
                bot_dies = True
            else:
                bot_dies = random.random() < 0.5

            if bot_dies:
                self.bot_alive = False
                bot_killed_by = i
                rewards[i] += KILL_BOT_REWARD
                break
            else:
                self.guard_alive[i] = False
                rewards[i] += DEATH_PENALTY

        return rewards, defuse_failed, bot_killed_by

    # -----------------------------------------------------------------
    # カバレッジ差分報酬(項目2, 5)
    # -----------------------------------------------------------------
    def _coverage_rewards(self, guard_los):
        rewards = [0.0] * self.num_guards
        alive_idx = [i for i in range(self.num_guards) if self.guard_alive[i]]

        for t in range(NUM_TARGET_SLOTS):
            if self.target_positions[t] is None:
                continue
            weight = self.target_weights[t]
            n_t = sum(1 for i in alive_idx if guard_los[i][t])
            if n_t == 0:
                continue
            marginal = weight * (np.sqrt(n_t) - np.sqrt(n_t - 1))
            for i in alive_idx:
                if guard_los[i][t]:
                    rewards[i] += marginal * COVERAGE_REWARD_SCALE

        return rewards

    def _in_range_rewards(self):
        rewards = [0.0] * self.num_guards
        for i in range(self.num_guards):
            if not self.guard_alive[i]:
                continue
            pr, pc = self.guard_pos[i]
            d = max(abs(pr - self.planted_pos[0]), abs(pc - self.planted_pos[1]))
            if GUARD_MIN_DIST <= d <= GUARD_MAX_DIST:
                rewards[i] += IN_RANGE_REWARD
            elif d < GUARD_MIN_DIST:
                rewards[i] -= OUT_OF_RANGE_PENALTY_SCALE * (GUARD_MIN_DIST - d)
            else:
                rewards[i] -= OUT_OF_RANGE_PENALTY_SCALE * (d - GUARD_MAX_DIST)
        return rewards

    # -----------------------------------------------------------------
    # メインステップ
    # -----------------------------------------------------------------
    def step(self, actions):
        """actions: 長さ num_guards のリスト。死亡済みスロットの値は無視される。"""
        self.current_step += 1
        rewards = [0.0] * self.num_guards
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        # 1. guardの行動
        for i in range(self.num_guards):
            if not self.guard_alive[i]:
                continue
            action = actions[i]
            self.guard_last_action[i] = action
            if action == 4:
                rewards[i] += self._use_ability(i)
            else:
                dr, dc = moves[action]
                r, c = self.guard_pos[i]
                nr, nc = r + dr, c + dc
                if self._is_walkable(nr, nc):
                    self.guard_pos[i] = (nr, nc)
                else:
                    rewards[i] -= 1.5

        # 2. 敵bot移動
        self._move_bot()

        # 3. 投射物進行
        proj_rewards = self._advance_projectiles()
        for i in range(self.num_guards):
            rewards[i] += proj_rewards[i]

        # 4. 解除タイマー更新
        terminated = False
        if self.bot_alive:
            if self._bot_is_defusing():
                self.bot_defuse_timer += 1
                if self.bot_defuse_timer >= GUARD_DEFUSE_REQUIRED:
                    terminated = True
                    for i in range(self.num_guards):
                        if self.guard_alive[i]:
                            rewards[i] += DEFUSE_FAIL_PENALTY
            else:
                self.bot_defuse_timer = 0

        # 5. 交戦解決
        if not terminated and self.bot_alive:
            eng_rewards, _, bot_killed_by = self._resolve_engagements()
            for i in range(self.num_guards):
                rewards[i] += eng_rewards[i]
            if not self.bot_alive:
                terminated = True  # 敵全滅=このラウンドの防衛成功
                for i in range(self.num_guards):
                    if self.guard_alive[i]:
                        rewards[i] += GUARD_SUCCESS_REWARD

        # 6. 全滅チェック
        if not terminated and not any(self.guard_alive):
            terminated = True  # 誰も残っていない=敵の解除を防げない(追加報酬なし)

        # 7. カバレッジ・距離維持報酬(生存中のみ)
        if not terminated:
            guard_los = self._compute_guard_los_matrix()
            cov_rewards = self._coverage_rewards(guard_los)
            range_rewards = self._in_range_rewards()
            for i in range(self.num_guards):
                if self.guard_alive[i]:
                    rewards[i] += cov_rewards[i] + range_rewards[i]
        else:
            guard_los = self._compute_guard_los_matrix()

        # 8. detonateタイマー
        if not terminated:
            self.detonate_timer -= 1
            if self.detonate_timer <= 0:
                terminated = True
                for i in range(self.num_guards):
                    if self.guard_alive[i]:
                        rewards[i] += GUARD_SUCCESS_REWARD

        # 9. ブラインド減衰
        if self.bot_blind_remaining > 0:
            self.bot_blind_remaining -= 1

        truncated = self.current_step >= self.max_steps
        next_obs = [self._get_obs_for(i, guard_los) for i in range(self.num_guards)]

        info = {"alive_mask": list(self.guard_alive)}
        return next_obs, rewards, terminated, truncated, info


# ===========================================================================
# マスクを考慮した行動選択ヘルパー
# ===========================================================================
def masked_argmax(q_values, mask):
    if mask.sum() == 0:
        return int(np.argmax(q_values))
    masked = np.where(mask > 0, q_values, -np.inf)
    return int(np.argmax(masked))


def masked_random_action(mask):
    valid = np.where(mask > 0)[0]
    if len(valid) == 0:
        return random.randint(0, N_ACTIONS - 1)
    return int(random.choice(valid))


def masked_softmax_action(q_values, mask, temperature=0.5):
    if mask.sum() == 0:
        masked = q_values
    else:
        masked = np.where(mask > 0, q_values, -np.inf)
    probs = np.exp((masked - np.max(masked)) / temperature)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


# ===========================================================================
# 学習ループ(パラメータ共有・独立Q学習)
# ===========================================================================
def train():
    writer = SummaryWriter(log_dir="logs")

    EVAL_BEST_SAVE_DIR = "data/attacker_guard_data/"
    SAVE_DIR = "data_temp/attacker_guard_data/"
    os.makedirs(SAVE_DIR, exist_ok=True)

    lines = [line.strip() for line in NEW_MAZE_STR.strip("\n").split("\n") if line.strip()]
    fixed_grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
    plant_rows, plant_cols = np.where(fixed_grid == 2)
    plant_candidates = list(zip(plant_rows, plant_cols))
    if not plant_candidates:
        raise ValueError("プラントサイト(2)が定義されていません。")

    env = GuardMultiEnv(fixed_grid, plant_candidates)

    batch_size = 64
    gamma = 0.99
    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 0.9993
    lr = 0.0005
    IMPROVEMENT_MARGIN = 5.0

    device = torch.device("cpu")
    q_net = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(device)
    target_net = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=50000)

    epsilon = epsilon_start
    best_eval_reward = -float('inf')

    success_window = deque(maxlen=100)      # detonate成功 or 敵全滅で終了した割合
    defuse_fail_window = deque(maxlen=100)  # 解除されてしまった割合
    in_range_window = deque(maxlen=100)
    ability_used_window = {t: deque(maxlen=100) for t in ABILITY_TYPES}
    ability_success_window = {t: deque(maxlen=100) for t in ABILITY_TYPES}

    print(f"学習を開始します。デバイス: {device} | 入力次元: {OBS_DIM}")
    print("python -m tensorboard.main --logdir=logs")

    for episode in range(NUM_EPISODES):
        obs_list, _ = env.reset()
        mask_list = env.get_action_masks()
        episode_rewards = [0.0] * env.num_guards
        losses = []
        in_range_ticks = 0
        total_ticks = 0
        done_flags = [False] * env.num_guards  # 個々のguardスロットが既に終端遷移をpush済みか

        while True:
            actions = []
            for i in range(env.num_guards):
                if done_flags[i]:
                    actions.append(0)  # ダミー(使用されない)
                    continue
                if random.random() < epsilon:
                    actions.append(masked_random_action(mask_list[i]))
                else:
                    with torch.no_grad():
                        obs_t = torch.tensor(obs_list[i], dtype=torch.float32, device=device).unsqueeze(0)
                        q_values = q_net(obs_t).squeeze(0).cpu().numpy()
                    actions.append(masked_argmax(q_values, mask_list[i]))

            next_obs_list, rewards, terminated, truncated, info = env.step(actions)
            next_mask_list = env.get_action_masks()
            alive_mask = info["alive_mask"]

            total_ticks += 1
            in_range_this_tick = 0
            alive_count_this_tick = 0
            for i in range(env.num_guards):
                if env.guard_alive[i] or (i < len(alive_mask) and alive_mask[i]):
                    alive_count_this_tick += 1

            episode_done = terminated or truncated

            for i in range(env.num_guards):
                if done_flags[i]:
                    continue
                slot_done = episode_done or (not alive_mask[i])
                replay_buffer.push(
                    obs_list[i], actions[i], rewards[i], next_obs_list[i],
                    slot_done, mask_list[i], next_mask_list[i]
                )
                episode_rewards[i] += rewards[i]
                if slot_done:
                    done_flags[i] = True

            obs_list, mask_list = next_obs_list, next_mask_list

            if len(replay_buffer) >= batch_size:
                b_obs, b_act, b_rew, b_nobs, b_term, b_mask, b_next_mask = replay_buffer.sample(batch_size)
                b_obs_t = torch.tensor(b_obs, dtype=torch.float32, device=device)
                b_act_t = torch.tensor(b_act, dtype=torch.long, device=device).unsqueeze(1)
                b_rew_t = torch.tensor(b_rew, dtype=torch.float32, device=device).unsqueeze(1)
                b_nobs_t = torch.tensor(b_nobs, dtype=torch.float32, device=device)
                b_term_t = torch.tensor(b_term, dtype=torch.float32, device=device).unsqueeze(1)
                b_next_mask_t = torch.tensor(b_next_mask, dtype=torch.float32, device=device)

                current_q = q_net(b_obs_t).gather(1, b_act_t)
                with torch.no_grad():
                    next_q_online = q_net(b_nobs_t)
                    masked_next_q = next_q_online.masked_fill(b_next_mask_t == 0, -1e9)
                    next_actions = masked_next_q.argmax(dim=1, keepdim=True)
                    max_next_q = target_net(b_nobs_t).gather(1, next_actions)
                    target_q = b_rew_t + (1.0 - b_term_t) * gamma * max_next_q

                loss = nn.SmoothL1Loss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
                optimizer.step()
                losses.append(loss.item())

            if episode_done:
                if episode % 10 == 0:
                    target_net.load_state_dict(q_net.state_dict())
                break

        success = env.bot_alive is False or env.detonate_timer <= 0
        defuse_failed = env.bot_alive and env.bot_defuse_timer >= GUARD_DEFUSE_REQUIRED
        success_window.append(1.0 if success and not defuse_failed else 0.0)
        defuse_fail_window.append(1.0 if defuse_failed else 0.0)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        writer.add_scalar("Train/Episode_Reward_Mean", np.mean(episode_rewards), episode)
        writer.add_scalar("Train/Success_Rate", np.mean(success_window), episode)
        writer.add_scalar("Train/Defuse_Fail_Rate", np.mean(defuse_fail_window), episode)

        if (episode + 1) % 50 == 0:
            avg_loss = np.mean(losses) if losses else 0.0
            print(f"Episode {episode+1}/{NUM_EPISODES} | MeanReward: {np.mean(episode_rewards):.2f} "
                  f"| Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f} "
                  f"| SuccessRate(直近100): {np.mean(success_window):.0%} "
                  f"| DefuseFailRate: {np.mean(defuse_fail_window):.0%}")

            EVAL_EPISODES = 50
            eval_rewards = []
            eval_success_count = 0
            eval_defuse_fail_count = 0

            for _ in range(EVAL_EPISODES):
                eval_obs_list, _ = env.reset()
                eval_mask_list = env.get_action_masks()
                eval_done_flags = [False] * env.num_guards
                ep_reward = 0.0
                while True:
                    eval_actions = []
                    for i in range(env.num_guards):
                        if eval_done_flags[i]:
                            eval_actions.append(0)
                            continue
                        with torch.no_grad():
                            q_values = q_net(
                                torch.tensor(eval_obs_list[i], dtype=torch.float32, device=device).unsqueeze(0)
                            ).squeeze(0).cpu().numpy()
                        eval_actions.append(masked_softmax_action(q_values, eval_mask_list[i], temperature=0.5))

                    eval_obs_list, eval_rew, term, trunc, eval_info = env.step(eval_actions)
                    eval_mask_list = env.get_action_masks()
                    alive_mask = eval_info["alive_mask"]
                    ep_done = term or trunc
                    for i in range(env.num_guards):
                        if not eval_done_flags[i]:
                            ep_reward += eval_rew[i]
                            if ep_done or not alive_mask[i]:
                                eval_done_flags[i] = True
                    if ep_done:
                        break

                eval_rewards.append(ep_reward / max(1, env.num_guards))
                if env.bot_alive and env.bot_defuse_timer >= GUARD_DEFUSE_REQUIRED:
                    eval_defuse_fail_count += 1
                elif (not env.bot_alive) or env.detonate_timer <= 0:
                    eval_success_count += 1

            mean_eval = np.mean(eval_rewards)
            eval_success_rate = eval_success_count / EVAL_EPISODES
            eval_defuse_fail_rate = eval_defuse_fail_count / EVAL_EPISODES
            writer.add_scalar("Eval/Guard_Reward_Mean", mean_eval, episode)
            writer.add_scalar("Eval/Success_Rate", eval_success_rate, episode)
            writer.add_scalar("Eval/Defuse_Fail_Rate", eval_defuse_fail_rate, episode)
            print(f"   [Eval] mean_reward={mean_eval:.2f} success_rate={eval_success_rate:.0%} "
                  f"defuse_fail_rate={eval_defuse_fail_rate:.0%} n={EVAL_EPISODES}")

            if mean_eval > best_eval_reward + IMPROVEMENT_MARGIN:
                best_eval_reward = mean_eval
                best_path = os.path.join(EVAL_BEST_SAVE_DIR, "dqn_attacker_guard_best_by_eval.pt")
                torch.save(q_net.state_dict(), best_path)
                best_path = os.path.join(SAVE_DIR, "dqn_attacker_guard_best_by_eval.pt")
                torch.save(q_net.state_dict(), best_path)
                print(f"   [Eval Best] 保存しました (Eval Reward: {mean_eval:.2f}): {best_path}")

        if (episode + 1) % SAVE_INTERVAL == 0:
            save_path = os.path.join(SAVE_DIR, f"dqn_attacker_guard_ep{episode+1}.pt")
            torch.save(q_net.state_dict(), save_path)
            print(f"   [Save] 定期保存しました: {save_path}")

    final_path = os.path.join(SAVE_DIR, "dqn_attacker_guard_final.pt")
    torch.save(q_net.state_dict(), final_path)
    print(f"学習が完了しました。最終モデル: {final_path}")
    writer.close()


if __name__ == "__main__":
    train()