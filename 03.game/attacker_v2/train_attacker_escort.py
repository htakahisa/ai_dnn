# train_attacker_escort.py
"""
護衛AI(escort、carryフェーズ専用)の学習スクリプト。
train_attacker_carry.py と同様、外部の学習系モジュールには依存せず、
必要なクラスはすべてこのファイル内に複製する。

護衛の役割:
- carryフェーズのみ(プラント成功でguard突入、キャリアー死亡でretrieve突入する直前まで)
- キャリアーとの距離を2〜7マス(チェビシェフ距離)に保つ(前方でも可)
- 敵を視認したらその方向へアビリティ(反応的、常に許可)
- サイトに接近したらサイト方面へ予防的にアビリティ(ただし、味方が既にサイト内で
  使用済みならなるべく使用しない)
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

OBS_DIM = 28
N_ACTIONS = 5   # 0:上 1:下 2:左 3:右 4:アビリティ使用(設置行動は無い)
NUM_EPISODES = 9000
SAVE_INTERVAL = 100
ABILITY_TYPES = ["flash", "smoke", "recon"]
MAX_STEPS = 150

# ---- 敵・視認まわり(train_attacker_carry.pyと同じ考え方) ----
ENEMY_SPAWN_PROB = 0.5
SPOTTED_PENALTY = -5.0

# ---- アビリティ関連 ----
ABILITY_USE_COST = -0.2
RECON_SUCCESS_BONUS = 2.0
FLASH_SUCCESS_BONUS = 8.0
RECON_EMPTY_INFO_BONUS = 0.0  # 0で無効
CORNER_CHECK_RADIUS = 2

# flashが外れた場合の救済ボーナス(排他的、いずれか1つのみ加算)
FLASH_REACTIVE_THROW_BONUS = 5.0
FLASH_EMPTY_INFO_BONUS = 0.0  # 0で無効
FLASH_OPEN_THROW_MIN_PATH = 3
FLASH_OPEN_THROW_BONUS = 0.0  # 0で無効

SITE_APPROACH_BONUS = 1.0
SITE_APPROACH_DIST_THRESHOLD = 5
REDUNDANT_SITE_USE_PENALTY = -1.0     # 味方が既にサイト内で使用済みなのに予防的に使った場合の減点
TEAMMATE_ABILITY_USED_PROB = 0.4      # 学習用。エピソードごとに「味方が既に使用済み」を確率的に発生させる

# ---- 護衛の距離維持 ----
ESCORT_MIN_DIST = 2
ESCORT_MAX_DIST = 7
IN_RANGE_REWARD = 0.3
OUT_OF_RANGE_PENALTY_SCALE = 1.0
CARRIER_SITE_SUCCESS_BONUS = 100.0    # キャリアーがサイトに到達(プラント相当)したら護衛にも完了報酬


# ===========================================================================
# 共通ヘルパー(train_attacker_carry.pyと同じロジックをこのファイル内に複製)
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
    label = [[None] * width for _ in range(height)]
    q = deque()
    for (r, c) in site_cells:
        dist[r, c] = 0
        label[r][c] = (r, c)
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
                if dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    label[nr][nc] = label[r][c]
                    q.append((nr, nc))
    return dist, label


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
# 護衛専用環境
# ===========================================================================
class EscortOnlyEnv:
    """carryフェーズ限定で、キャリアー(BFSで自動的にサイトへ移動するシンプルなbot)に
    2〜7マス(チェビシェフ距離)ついていく護衛エージェントを学習する環境。

    💡簡略化している点:
    - キャリアーが敵に倒される(retrieveフェーズ突入)は再現していない。
      キャリアーがサイトに到達したら成功終了、タイムアウトのみが終了条件。
    - キャリアー自身の視認・被弾は判定するが、キャリアー自体は反撃も回避もしない
      (直進のBFS移動のみ)。護衛が「守る」対象として存在するだけ。
    """

    def __init__(self, fixed_grid, plant_candidates):
        self.grid = fixed_grid
        self.height, self.width = fixed_grid.shape
        self.plant_candidates = [tuple(p) for p in plant_candidates]
        self.max_steps = MAX_STEPS

        self.site_components = split_site_components(self.plant_candidates)
        self.site_maps = [multi_source_bfs(comp, self.grid) for comp in self.site_components]
        self.site_cell_sets = [set(comp) for comp in self.site_components]
        self.corner_cells = find_corner_cells(self.grid)

    def _is_walkable(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width and self.grid[r, c] != 1

    def _random_walkable(self):
        while True:
            p = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            if self._is_walkable(*p):
                return p

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.last_action = None

        self.escort_pos = self._random_walkable()
        self.carrier_pos = self._random_walkable()

        self.current_site_index = random.randrange(len(self.site_components))
        self.dist_map, self.label_map = self.site_maps[self.current_site_index]
        self.site_cells = self.site_cell_sets[self.current_site_index]

        self.own_ability_type = random.choice(ABILITY_TYPES)
        self.own_ability_charge = 1.0
        self.last_ability_result = None
        self.last_site_approach_triggered = False

        # 💡追加: 「味方(キャリアー)が既にサイト内でアビリティを使用済みか」を
        # エピソードごとに確率的にシミュレートする(実際のキャリアーAIの判断は再現していない)
        self.site_ability_used_by_teammate = random.random() < TEAMMATE_ABILITY_USED_PROB

        self.bot_present = random.random() < ENEMY_SPAWN_PROB
        if self.bot_present:
            self.bot_pos = random.choice(self.corner_cells) if self.corner_cells else self._random_walkable()
        else:
            self.bot_pos = None
        self.bot_blind_remaining = 0
        self.recon_reveal_remaining = 0
        self.smoke_cells = set()
        self.smoke_remaining = 0
        self.pending_flash = None
        self.pending_recon = None

        return self._get_obs(), {}

    def _get_obs(self):
        pr, pc = self.escort_pos
        height, width = self.height, self.width

        base = [pr / (height - 1), pc / (width - 1)]

        cr, cc = self.carrier_pos
        max_dist = max(height, width)
        carrier_dist = max(abs(cr - pr), abs(cc - pc))
        carrier_rel = [carrier_dist / max_dist, (cr - pr) / height, (cc - pc) / width]

        walls = [0.0 if self._is_walkable(pr + dr, pc + dc) else 1.0
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

        last_onehot = [0.0] * N_ACTIONS
        if self.last_action is not None:
            last_onehot[self.last_action] = 1.0

        site_max_dist = max(height, width) * 2
        site_dists = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pr + dr, pc + dc
            if self._is_walkable(nr, nc):
                d = self.dist_map[nr, nc]
                site_dists.append(d / site_max_dist if np.isfinite(d) else 1.0)
            else:
                site_dists.append(1.0)

        ability_onehot = [1.0 if self.own_ability_type == t else 0.0 for t in ABILITY_TYPES]
        ability_charge = [self.own_ability_charge]
        enemy_blinded_flag = [1.0 if self.bot_blind_remaining > 0 else 0.0]

        enemy = [0.0, 0.0, 0.0]
        if self.bot_present:
            los = has_line_of_sight(self.escort_pos, self.bot_pos, self.grid, self.smoke_cells)
            if los or self.recon_reveal_remaining > 0:
                br, bc = self.bot_pos
                enemy = [1.0, (br - pr) / height, (bc - pc) / width]

        own_site_dist = self.dist_map[pr, pc]
        own_site_dist_norm = [own_site_dist / site_max_dist if np.isfinite(own_site_dist) else 1.0]

        teammate_used_flag = [1.0 if self.site_ability_used_by_teammate else 0.0]

        return np.array(
            base + carrier_rel + walls + last_onehot + site_dists +
            ability_onehot + ability_charge + enemy_blinded_flag + enemy +
            own_site_dist_norm + teammate_used_flag,
            dtype=np.float32
        )

    def get_action_mask(self):
        r, c = self.escort_pos
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        for a, (dr, dc) in moves.items():
            mask[a] = 1.0 if self._is_walkable(r + dr, c + dc) else 0.0
        mask[4] = 1.0 if self.own_ability_charge > 0 else 0.0
        return mask

    def _move_carrier(self):
        """キャリアーはBFS勾配に沿ってサイトへ直進する(簡略化した固定ロジック)。
        タイ(同距離)は複数あればランダムに選び、多少の経路の揺れを出す。"""
        cr, cc = self.carrier_pos
        if (cr, cc) in self.site_cells:
            return
        candidates = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if self._is_walkable(nr, nc):
                d = self.dist_map[nr, nc]
                if np.isfinite(d):
                    candidates.append((d, (nr, nc)))
        if not candidates:
            return
        best_d = min(d for d, _ in candidates)
        best_moves = [pos for d, pos in candidates if d == best_d]
        self.carrier_pos = random.choice(best_moves)

    def _get_aim_direction(self):
        """狙う方向を決める。carryと同じ優先順位:
        1) 敵が見えている/リコン察知中ならその方向(反応的、is_reactive=True)
        2) 直前の移動方向
        3) BFS勾配上の最善方向(サイトに近づく向き)
        戻り値: (狙うマス, 反応的な使用かどうか)"""
        pr, pc = self.escort_pos

        if self.bot_present:
            visible = has_line_of_sight(self.escort_pos, self.bot_pos, self.grid, self.smoke_cells)
            if visible or self.recon_reveal_remaining > 0:
                return self.bot_pos, True

        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        if self.last_action in moves:
            dr, dc = moves[self.last_action]
            return (pr + dr, pc + dc), False

        best_dir, best_dist = None, self.dist_map[pr, pc]
        for dr, dc in moves.values():
            nr, nc = pr + dr, pc + dc
            if self._is_walkable(nr, nc):
                d = self.dist_map[nr, nc]
                if np.isfinite(d) and d < best_dist:
                    best_dist, best_dir = d, (dr, dc)
        if best_dir is None:
            best_dir = (0, 1)
        return (pr + best_dir[0], pc + best_dir[1]), False

    def _use_ability(self):
        if self.own_ability_charge <= 0:
            return -1.0

        self.own_ability_charge = 0.0
        ability_type = self.own_ability_type
        aimed_cell, is_reactive = self._get_aim_direction()
        path = compute_projectile_path(self.escort_pos, aimed_cell, self.grid)

        self.last_ability_result = {"type": ability_type, "success": False}
        self.last_site_approach_triggered = False
        reward = ABILITY_USE_COST

        # 💡追加: 反応的な使用(敵視認)でなければ、サイト方面への予防的使用として評価する
        if not is_reactive:
            player_dist_to_site = self.dist_map[self.escort_pos[0], self.escort_pos[1]]
            if np.isfinite(player_dist_to_site) and player_dist_to_site <= SITE_APPROACH_DIST_THRESHOLD:
                if any(cell in self.site_cells for cell in path):
                    if not self.site_ability_used_by_teammate:
                        reward += SITE_APPROACH_BONUS
                        self.site_ability_used_by_teammate = True
                        self.last_site_approach_triggered = True
                    else:
                        # 💡追加: 既に味方が使用済みなのに、さらに予防的に使ってしまった場合は減点
                        reward += REDUNDANT_SITE_USE_PENALTY

        if ability_type == "flash":
            self.pending_flash = {"path": path, "progress": 0, "ticks_alive": 0, "is_reactive": is_reactive}
        elif ability_type == "recon":
            self.pending_recon = {"path": path, "progress": 0}
        elif ability_type == "smoke":
            tr, tc = path[-1]
            self.smoke_cells = {
                (rr, cc)
                for rr in range(tr - SMOKE_RADIUS, tr + SMOKE_RADIUS + 1)
                for cc in range(tc - SMOKE_RADIUS, tc + SMOKE_RADIUS + 1)
                if self._is_walkable(rr, cc)
            }
            self.smoke_remaining = SMOKE_DURATION_TICKS
            if self.bot_present and not has_line_of_sight(self.escort_pos, self.bot_pos, self.grid, self.smoke_cells):
                self.last_ability_result["success"] = True

        return reward

    def _advance_projectiles(self):
        reward = 0.0

        if self.pending_flash is not None:
            p = self.pending_flash
            p["ticks_alive"] += 1
            next_progress = p["progress"] + FLASH_SPEED_CELLS_PER_TICK
            hit_wall_or_edge = next_progress >= len(p["path"]) - 1
            p["progress"] = min(next_progress, len(p["path"]) - 1)
            if hit_wall_or_edge or p["ticks_alive"] >= FLASH_MAX_FLIGHT_TICKS:
                impact = p["path"][p["progress"]]
                success = False
                if self.bot_present and has_line_of_sight(impact, self.bot_pos, self.grid, self.smoke_cells):
                    self.bot_blind_remaining = max(self.bot_blind_remaining, BLIND_DURATION_TICKS)
                    success = True
                    reward += FLASH_SUCCESS_BONUS

                if not success:
                    ir, ic = impact
                    if p["is_reactive"]:
                        reward += FLASH_REACTIVE_THROW_BONUS
                    elif self.corner_cells and min(
                        max(abs(ir - cr), abs(ic - cc)) for (cr, cc) in self.corner_cells
                    ) <= CORNER_CHECK_RADIUS:
                        reward += FLASH_EMPTY_INFO_BONUS
                    elif len(p["path"]) - 1 >= FLASH_OPEN_THROW_MIN_PATH:
                        reward += FLASH_OPEN_THROW_BONUS

                self.last_ability_result = {"type": "flash", "success": success}
                self.pending_flash = None

        if self.pending_recon is not None:
            p = self.pending_recon
            next_progress = p["progress"] + RECON_SPEED_CELLS_PER_TICK
            hit_wall_or_edge = next_progress >= len(p["path"]) - 1
            p["progress"] = min(next_progress, len(p["path"]) - 1)
            if hit_wall_or_edge:
                impact = p["path"][p["progress"]]
                ir, ic = impact
                radius = RECON_REVEAL_SIZE // 2
                success = False
                if self.bot_present:
                    br, bc = self.bot_pos
                    if abs(br - ir) <= radius and abs(bc - ic) <= radius:
                        self.recon_reveal_remaining = max(self.recon_reveal_remaining, REVEAL_DURATION_TICKS)
                        success = True
                        reward += RECON_SUCCESS_BONUS

                if not success and self.corner_cells:
                    nearest_corner_dist = min(
                        max(abs(ir - cr), abs(ic - cc)) for (cr, cc) in self.corner_cells
                    )
                    if nearest_corner_dist <= CORNER_CHECK_RADIUS:
                        reward += RECON_EMPTY_INFO_BONUS

                self.last_ability_result = {"type": "recon", "success": success}
                self.pending_recon = None

        return reward

    def _step_move(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        r, c = self.escort_pos
        dr, dc = moves[action]
        nr, nc = r + dr, c + dc
        if not self._is_walkable(nr, nc):
            return -1.5
        self.escort_pos = (nr, nc)
        return 0.0

    def step(self, action):
        self.current_step += 1
        self.last_action = action
        reward = 0.0
        terminated = False

        if action == 4:
            reward += self._use_ability()
        else:
            reward += self._step_move(action)

        # 💡キャリアーはescortのactionと無関係に毎tick自動で動く(実ゲームでは別キャラのため)
        self._move_carrier()

        reward += self._advance_projectiles()

        # 💡視認ペナルティ: escort自身 or キャリアーのどちらかが見られていれば発生
        if self.bot_present and self.bot_blind_remaining <= 0:
            escort_seen = has_line_of_sight(self.escort_pos, self.bot_pos, self.grid, self.smoke_cells)
            carrier_seen = has_line_of_sight(self.carrier_pos, self.bot_pos, self.grid, self.smoke_cells)
            if escort_seen or carrier_seen:
                reward += SPOTTED_PENALTY

        # 💡距離維持報酬(チェビシェフ距離、2〜7マスが理想帯)
        dist = max(abs(self.escort_pos[0] - self.carrier_pos[0]), abs(self.escort_pos[1] - self.carrier_pos[1]))
        if ESCORT_MIN_DIST <= dist <= ESCORT_MAX_DIST:
            reward += IN_RANGE_REWARD
        elif dist < ESCORT_MIN_DIST:
            reward -= OUT_OF_RANGE_PENALTY_SCALE * (ESCORT_MIN_DIST - dist)
        else:
            reward -= OUT_OF_RANGE_PENALTY_SCALE * (dist - ESCORT_MAX_DIST)

        if self.bot_blind_remaining > 0:
            self.bot_blind_remaining -= 1
        if self.recon_reveal_remaining > 0:
            self.recon_reveal_remaining -= 1
        if self.smoke_remaining > 0:
            self.smoke_remaining -= 1
            if self.smoke_remaining == 0:
                self.smoke_cells = set()

        # 💡終了条件: キャリアーがサイトに到達(プラント成功相当)
        if tuple(self.carrier_pos) in self.site_cells:
            terminated = True
            reward += CARRIER_SITE_SUCCESS_BONUS

        truncated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}


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
# 学習ループ
# ===========================================================================
def train():
    writer = SummaryWriter(log_dir="logs")

    EVAL_BEST_SAVE_DIR = "data/attacker_escort_data/"
    SAVE_DIR = "data_temp/attacker_escort_data/"
    os.makedirs(SAVE_DIR, exist_ok=True)

    lines = [line.strip() for line in NEW_MAZE_STR.strip("\n").split("\n") if line.strip()]
    fixed_grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
    plant_rows, plant_cols = np.where(fixed_grid == 2)
    plant_candidates = list(zip(plant_rows, plant_cols))
    if not plant_candidates:
        raise ValueError("プラントサイト(2)が定義されていません。")

    env = EscortOnlyEnv(fixed_grid, plant_candidates)

    batch_size = 64
    gamma = 0.99
    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 0.9985
    lr = 0.0005
    IMPROVEMENT_MARGIN = 5.0

    device = torch.device("cpu")
    q_net = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(device)
    target_net = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=30000)

    epsilon = epsilon_start
    best_eval_reward = -float('inf')

    success_window = deque(maxlen=100)
    ticks_window = deque(maxlen=100)
    in_range_window = deque(maxlen=100)  # 💡追加: 毎tickの「距離帯に収まっていた割合」

    ability_used_window = {t: deque(maxlen=100) for t in ABILITY_TYPES}
    ability_success_window = {t: deque(maxlen=100) for t in ABILITY_TYPES}
    site_approach_window = deque(maxlen=100)

    print(f"学習を開始します。デバイス: {device} | 入力次元: {OBS_DIM}")
    print("python -m tensorboard.main --logdir=logs")

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        mask = env.get_action_mask()
        episode_reward = 0.0
        losses = []
        in_range_ticks = 0
        total_ticks = 0

        while True:
            if random.random() < epsilon:
                action = masked_random_action(mask)
            else:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = q_net(obs_t).squeeze(0).cpu().numpy()
                action = masked_argmax(q_values, mask)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_mask = env.get_action_mask()

            total_ticks += 1
            dist = max(abs(env.escort_pos[0] - env.carrier_pos[0]), abs(env.escort_pos[1] - env.carrier_pos[1]))
            if ESCORT_MIN_DIST <= dist <= ESCORT_MAX_DIST:
                in_range_ticks += 1

            replay_buffer.push(obs, action, reward, next_obs, terminated, mask, next_mask)
            obs, mask = next_obs, next_mask
            episode_reward += reward

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

            if terminated or truncated:
                if episode % 10 == 0:
                    target_net.load_state_dict(q_net.state_dict())
                break

        success = terminated
        success_window.append(1.0 if success else 0.0)
        if success:
            ticks_window.append(env.current_step)
        if total_ticks > 0:
            in_range_window.append(in_range_ticks / total_ticks)

        used_result = env.last_ability_result
        for t in ABILITY_TYPES:
            used_flag = 1.0 if (used_result is not None and used_result["type"] == t) else 0.0
            ability_used_window[t].append(used_flag)
        if used_result is not None:
            ability_success_window[used_result["type"]].append(1.0 if used_result["success"] else 0.0)
            site_approach_window.append(1.0 if env.last_site_approach_triggered else 0.0)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        writer.add_scalar("Train/Episode_Reward", episode_reward, episode)
        writer.add_scalar("Train/Success_Rate", np.mean(success_window), episode)
        if ticks_window:
            writer.add_scalar("Train/Ticks_To_Complete", np.mean(ticks_window), episode)
        if in_range_window:
            writer.add_scalar("Train/InRangeRate", np.mean(in_range_window), episode)
        for t in ABILITY_TYPES:
            writer.add_scalar(f"Train/Ability_UsageRate_{t}", np.mean(ability_used_window[t]), episode)
            if ability_success_window[t]:
                writer.add_scalar(f"Train/Ability_SuccessRate_{t}", np.mean(ability_success_window[t]), episode)
        if site_approach_window:
            writer.add_scalar("Train/Ability_SiteApproachRate", np.mean(site_approach_window), episode)

        if (episode + 1) % 50 == 0:
            avg_loss = np.mean(losses) if losses else 0.0
            print(f"Episode {episode+1}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f}")

            ability_summary = " | ".join(
                f"{t}: use={np.mean(ability_used_window[t]):.0%}"
                f" succ={(np.mean(ability_success_window[t]) if ability_success_window[t] else float('nan')):.0%}"
                for t in ABILITY_TYPES
            )
            print(f"   [Ability] {ability_summary}")
            site_approach_rate = np.mean(site_approach_window) if site_approach_window else float('nan')
            print(f"   [Ability SiteApproach] rate={site_approach_rate:.0%} (n_used={len(site_approach_window)})")
            print(f"   [InRange] rate={np.mean(in_range_window) if in_range_window else float('nan'):.0%}")

            EVAL_EPISODES = 100
            eval_rewards = []
            eval_success_count = 0
            eval_success_ticks = []
            eval_in_range_rates = []
            eval_ability_used = {t: 0 for t in ABILITY_TYPES}
            eval_ability_success = {t: 0 for t in ABILITY_TYPES}
            eval_site_approach_count = 0

            for _ in range(EVAL_EPISODES):
                eval_obs, _ = env.reset()
                eval_mask = env.get_action_mask()
                eval_reward = 0.0
                e_in_range, e_total = 0, 0
                while True:
                    with torch.no_grad():
                        q_values = q_net(torch.tensor(eval_obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
                    a = masked_softmax_action(q_values, eval_mask, temperature=0.5)
                    eval_obs, r, term, trunc, _ = env.step(a)
                    eval_mask = env.get_action_mask()
                    eval_reward += r

                    e_total += 1
                    d = max(abs(env.escort_pos[0] - env.carrier_pos[0]), abs(env.escort_pos[1] - env.carrier_pos[1]))
                    if ESCORT_MIN_DIST <= d <= ESCORT_MAX_DIST:
                        e_in_range += 1

                    if term or trunc:
                        if term:
                            eval_success_count += 1
                            eval_success_ticks.append(env.current_step)
                        break
                eval_rewards.append(eval_reward)
                if e_total > 0:
                    eval_in_range_rates.append(e_in_range / e_total)

                if env.last_ability_result is not None:
                    t = env.last_ability_result["type"]
                    eval_ability_used[t] += 1
                    if env.last_ability_result["success"]:
                        eval_ability_success[t] += 1
                    if env.last_site_approach_triggered:
                        eval_site_approach_count += 1

            mean_eval = np.mean(eval_rewards)
            eval_success_rate = eval_success_count / EVAL_EPISODES
            writer.add_scalar("Eval/Escort_Reward", mean_eval, episode)
            writer.add_scalar("Eval/Success_Rate", eval_success_rate, episode)
            if eval_success_ticks:
                writer.add_scalar("Eval/Ticks_To_Complete", np.mean(eval_success_ticks), episode)
            if eval_in_range_rates:
                writer.add_scalar("Eval/InRangeRate", np.mean(eval_in_range_rates), episode)
            for t in ABILITY_TYPES:
                writer.add_scalar(f"Eval/Ability_UsedCount_{t}", eval_ability_used[t], episode)
                if eval_ability_used[t] > 0:
                    writer.add_scalar(f"Eval/Ability_SuccessRate_{t}", eval_ability_success[t] / eval_ability_used[t], episode)
            total_used = sum(eval_ability_used.values())
            if total_used > 0:
                writer.add_scalar("Eval/Ability_SiteApproachRate", eval_site_approach_count / total_used, episode)

            print(f"   [Eval] mean={mean_eval:.2f} success_rate={eval_success_rate:.2%} "
                  f"avg_ticks={np.mean(eval_success_ticks) if eval_success_ticks else float('nan'):.1f} "
                  f"in_range={np.mean(eval_in_range_rates) if eval_in_range_rates else float('nan'):.0%} n={EVAL_EPISODES}")
            eval_site_rate = (eval_site_approach_count / total_used) if total_used > 0 else float('nan')
            print(f"   [Eval SiteApproach] rate={eval_site_rate:.0%} (used={total_used})")

            if mean_eval > best_eval_reward + IMPROVEMENT_MARGIN:
                best_eval_reward = mean_eval
                best_path = os.path.join(EVAL_BEST_SAVE_DIR, "dqn_attacker_escort_best_by_eval.pt")
                torch.save(q_net.state_dict(), best_path)
                best_path = os.path.join(SAVE_DIR, "dqn_attacker_escort_best_by_eval.pt")
                torch.save(q_net.state_dict(), best_path)
                print(f"   [Eval Best] 保存しました (Eval Reward: {mean_eval:.2f}): {best_path}")

        if (episode + 1) % SAVE_INTERVAL == 0:
            save_path = os.path.join(SAVE_DIR, f"dqn_attacker_escort_ep{episode+1}.pt")
            torch.save(q_net.state_dict(), save_path)
            print(f"   [Save] 定期保存しました: {save_path}")

    final_path = os.path.join(SAVE_DIR, "dqn_attacker_escort_final.pt")
    torch.save(q_net.state_dict(), final_path)
    print(f"学習が完了しました。最終モデル: {final_path}")
    writer.close()


if __name__ == "__main__":
    train()