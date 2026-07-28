# train_attacker_retrieve.py
"""
スパイク回収AI(retrieveフェーズ専用)の学習スクリプト。
train_attacker_carry.py と同様、外部の学習系モジュールには依存せず、
必要なクラスはすべてこのファイル内に複製する。

retrieveの役割:
- マップ全体(壁以外)のランダムな位置に落ちているスパイクまで移動し、回収する
  (実ゲームでは該当マスに乗った時点で自動回収されるため、専用の「拾う」アクションは無い)
- 敵を視認したらその方向へアビリティ(反応的)
- スパイクに接近したら、スパイク方面へ予防的にアビリティ(拾いやすくするための下準備)
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
from abilities import (
    FLASH_SPEED_CELLS_PER_TICK,
    FLASH_MAX_FLIGHT_TICKS,
    BLIND_DURATION_TICKS,
    RECON_SPEED_CELLS_PER_TICK,
    RECON_REVEAL_SIZE,
    REVEAL_DURATION_TICKS,
    SMOKE_DURATION_TICKS,
    SMOKE_RADIUS,
)

OBS_DIM = 25
N_ACTIONS = 5   # 0:上 1:下 2:左 3:右 4:アビリティ使用(専用の「拾う」アクションは無い)
NUM_EPISODES = 13000
SAVE_INTERVAL = 100
ABILITY_TYPES = ["flash", "smoke", "recon"]
MAX_STEPS = 90

ENEMY_SPAWN_PROB = 0.5
SPOTTED_PENALTY = -5.0

ABILITY_USE_COST = -0.2
RECON_SUCCESS_BONUS = 2.0
FLASH_SUCCESS_BONUS = 8.0
RECON_EMPTY_INFO_BONUS = 0.8
CORNER_CHECK_RADIUS = 2

# 💡追加: flashが外れた場合の救済ボーナス(排他的、いずれか1つのみ加算)
FLASH_REACTIVE_THROW_BONUS = 1.0   # 敵視認中に反応して投げた場合(命中しなくても反応自体は正しい)
FLASH_EMPTY_INFO_BONUS = 0.5       # 曲がり角付近を狙っていた場合(reconより情報価値が低いため小さめ)
FLASH_OPEN_THROW_MIN_PATH = 3      # この距離以上飛べば「開けた場所へ投げた」とみなす
FLASH_OPEN_THROW_BONUS = 0.3       # 開けた通路へ適切に投げた場合(壁際1マスでの無駄撃ちと区別)

# 💡追加: スパイクに接近した状態でスパイク方面へアビリティを使った場合のボーナス。
# 「拾う前に周囲を確認/制圧しておく」行動を評価するための項目で、命中判定とは独立に加算される。
SPIKE_APPROACH_BONUS = 1.0
SPIKE_APPROACH_DIST_THRESHOLD = 5
SPIKE_APPROACH_HIT_RADIUS = 1  # 投射経路がスパイク位置からこの距離以内を通れば「方面へ撃った」とみなす

RETRIEVE_SUCCESS_REWARD = 200.0


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
# retrieve専用環境
# ===========================================================================
class RetrieveOnlyEnv:
    """マップ全体(壁以外)のランダムな位置に落ちているスパイクまで移動し、回収するまでを
    学習する環境。実ゲームでは該当マスに乗った時点で自動回収されるため、専用の
    「拾う」アクションは無く、到達=成功終了とする。"""

    def __init__(self, fixed_grid):
        self.grid = fixed_grid
        self.height, self.width = fixed_grid.shape
        self.max_steps = MAX_STEPS
        self.corner_cells = find_corner_cells(self.grid)
        self.walkable_cells = [
            (r, c) for r in range(self.height) for c in range(self.width) if self.grid[r, c] != 1
        ]

    def _is_walkable(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width and self.grid[r, c] != 1

    def _random_walkable(self):
        return random.choice(self.walkable_cells)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.last_action = None

        self.player_pos = self._random_walkable()
        self.spike_pos = self._random_walkable()
        while self.spike_pos == self.player_pos:
            self.spike_pos = self._random_walkable()

        # 💡毎エピソード、スパイク位置が変わるためBFSも都度計算し直す
        self.dist_map = bfs_distances(self.spike_pos, self.grid)
        self.prev_dist = self.dist_map[self.player_pos[0], self.player_pos[1]]

        self.own_ability_type = random.choice(ABILITY_TYPES)
        self.own_ability_charge = 1.0
        self.last_ability_result = None
        self.last_spike_approach_triggered = False

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

        self.pos_history = deque(maxlen=6)
        return self._get_obs(), {}

    def _get_obs(self):
        pr, pc = self.player_pos
        sr, sc = self.spike_pos
        height, width = self.height, self.width

        base = [pr / (height - 1), pc / (width - 1), sr / (height - 1), sc / (width - 1)]

        walls = [0.0 if self._is_walkable(pr + dr, pc + dc) else 1.0
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

        last_onehot = [0.0] * N_ACTIONS
        if self.last_action is not None:
            last_onehot[self.last_action] = 1.0

        max_dist = max(height, width) * 2
        dists = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pr + dr, pc + dc
            if self._is_walkable(nr, nc):
                d = self.dist_map[nr, nc]
                dists.append(d / max_dist if np.isfinite(d) else 1.0)
            else:
                dists.append(1.0)

        ability_onehot = [1.0 if self.own_ability_type == t else 0.0 for t in ABILITY_TYPES]
        ability_charge = [self.own_ability_charge]
        enemy_blinded_flag = [1.0 if self.bot_blind_remaining > 0 else 0.0]

        enemy = [0.0, 0.0, 0.0]
        if self.bot_present:
            los = has_line_of_sight(self.player_pos, self.bot_pos, self.grid, self.smoke_cells)
            if los or self.recon_reveal_remaining > 0:
                br, bc = self.bot_pos
                enemy = [1.0, (br - pr) / height, (bc - pc) / width]

        return np.array(
            base + walls + last_onehot + dists + ability_onehot + ability_charge + enemy_blinded_flag + enemy,
            dtype=np.float32
        )

    def get_action_mask(self):
        pr, pc = self.player_pos
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        for a, (dr, dc) in moves.items():
            mask[a] = 1.0 if self._is_walkable(pr + dr, pc + dc) else 0.0
        mask[4] = 1.0 if self.own_ability_charge > 0 else 0.0
        return mask

    def _get_aim_direction(self):
        pr, pc = self.player_pos

        if self.bot_present:
            visible = has_line_of_sight(self.player_pos, self.bot_pos, self.grid, self.smoke_cells)
            if visible or self.recon_reveal_remaining > 0:
                return self.bot_pos, True  # (狙うマス, 反応的な使用かどうか)

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
        path = compute_projectile_path(self.player_pos, aimed_cell, self.grid)

        self.last_ability_result = {"type": ability_type, "success": False}
        self.last_spike_approach_triggered = False
        reward = ABILITY_USE_COST

        # 💡追加: 反応的な使用(敵視認)でなければ、スパイク方面への予防的使用として評価する
        if not is_reactive:
            player_dist_to_spike = self.dist_map[self.player_pos[0], self.player_pos[1]]
            if np.isfinite(player_dist_to_spike) and player_dist_to_spike <= SPIKE_APPROACH_DIST_THRESHOLD:
                near_spike = any(
                    max(abs(cell[0] - self.spike_pos[0]), abs(cell[1] - self.spike_pos[1])) <= SPIKE_APPROACH_HIT_RADIUS
                    for cell in path
                )
                if near_spike:
                    reward += SPIKE_APPROACH_BONUS
                    self.last_spike_approach_triggered = True

        if ability_type == "flash":
            # 💡追加: is_reactive(敵視認中の反応かどうか)を保持し、着弾解決時の救済判定に使う
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
            if self.bot_present and not has_line_of_sight(self.player_pos, self.bot_pos, self.grid, self.smoke_cells):
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

                # 💡追加: 外れた場合の救済ボーナス(いずれか1つのみ、優先順位あり)
                if not success:
                    ir, ic = impact
                    if p["is_reactive"]:
                        # 敵視認中に反応して投げていた(命中しなくても反応自体は正しい)
                        reward += FLASH_REACTIVE_THROW_BONUS
                    elif self.corner_cells and min(
                        max(abs(ir - cr), abs(ic - cc)) for (cr, cc) in self.corner_cells
                    ) <= CORNER_CHECK_RADIUS:
                        # 曲がり角付近を狙っていた
                        reward += FLASH_EMPTY_INFO_BONUS
                    elif len(p["path"]) - 1 >= FLASH_OPEN_THROW_MIN_PATH:
                        # 開けた通路へある程度の距離を飛ばして投げていた(壁際での無駄撃ちではない)
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
        r, c = self.player_pos
        dr, dc = moves[action]
        nr, nc = r + dr, c + dc

        if not self._is_walkable(nr, nc):
            return -1.5, False

        self.player_pos = (nr, nc)

        if (nr, nc) == self.spike_pos:
            return RETRIEVE_SUCCESS_REWARD, True

        new_dist = self.dist_map[nr, nc]
        shaping = (self.prev_dist - new_dist) * 0.5
        reward = -1.0 + shaping

        if np.isfinite(new_dist) and new_dist <= 3:
            reward += 1.5

        pos_tuple = (nr, nc)
        near_goal = np.isfinite(new_dist) and new_dist <= 3
        if not near_goal and pos_tuple in self.pos_history and new_dist >= self.prev_dist:
            reward -= 2.0
        self.pos_history.append(pos_tuple)

        self.prev_dist = new_dist
        return reward, False

    def step(self, action):
        self.current_step += 1
        self.last_action = action

        if action == 4:
            reward = self._use_ability()
            terminated = False
        else:
            reward, terminated = self._step_move(action)

        reward += self._advance_projectiles()

        if self.bot_present and self.bot_blind_remaining <= 0:
            if has_line_of_sight(self.player_pos, self.bot_pos, self.grid, self.smoke_cells):
                reward += SPOTTED_PENALTY

        if self.bot_blind_remaining > 0:
            self.bot_blind_remaining -= 1
        if self.recon_reveal_remaining > 0:
            self.recon_reveal_remaining -= 1
        if self.smoke_remaining > 0:
            self.smoke_remaining -= 1
            if self.smoke_remaining == 0:
                self.smoke_cells = set()

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

    EVAL_BEST_SAVE_DIR = "data_temp/attacker_retrieve_data/"
    SAVE_DIR = "data_temp/attacker_retrieve_data/"
    os.makedirs(SAVE_DIR, exist_ok=True)

    lines = [line.strip() for line in NEW_MAZE_STR.strip("\n").split("\n") if line.strip()]
    fixed_grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)

    env = RetrieveOnlyEnv(fixed_grid)

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

    ability_used_window = {t: deque(maxlen=100) for t in ABILITY_TYPES}
    ability_success_window = {t: deque(maxlen=100) for t in ABILITY_TYPES}
    spike_approach_window = deque(maxlen=100)

    print(f"学習を開始します。デバイス: {device} | 入力次元: {OBS_DIM}")
    print("python -m tensorboard.main --logdir=logs")

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        mask = env.get_action_mask()
        episode_reward = 0.0
        losses = []

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

        used_result = env.last_ability_result
        for t in ABILITY_TYPES:
            used_flag = 1.0 if (used_result is not None and used_result["type"] == t) else 0.0
            ability_used_window[t].append(used_flag)
        if used_result is not None:
            ability_success_window[used_result["type"]].append(1.0 if used_result["success"] else 0.0)
            spike_approach_window.append(1.0 if env.last_spike_approach_triggered else 0.0)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        writer.add_scalar("Train/Episode_Reward", episode_reward, episode)
        writer.add_scalar("Train/Success_Rate", np.mean(success_window), episode)
        if ticks_window:
            writer.add_scalar("Train/Ticks_To_Retrieve", np.mean(ticks_window), episode)
        for t in ABILITY_TYPES:
            writer.add_scalar(f"Train/Ability_UsageRate_{t}", np.mean(ability_used_window[t]), episode)
            if ability_success_window[t]:
                writer.add_scalar(f"Train/Ability_SuccessRate_{t}", np.mean(ability_success_window[t]), episode)
        if spike_approach_window:
            writer.add_scalar("Train/Ability_SpikeApproachRate", np.mean(spike_approach_window), episode)

        if (episode + 1) % 50 == 0:
            avg_loss = np.mean(losses) if losses else 0.0
            print(f"Episode {episode+1}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f}")

            ability_summary = " | ".join(
                f"{t}: use={np.mean(ability_used_window[t]):.0%}"
                f" succ={(np.mean(ability_success_window[t]) if ability_success_window[t] else float('nan')):.0%}"
                for t in ABILITY_TYPES
            )
            print(f"   [Ability] {ability_summary}")
            spike_approach_rate = np.mean(spike_approach_window) if spike_approach_window else float('nan')
            print(f"   [Ability SpikeApproach] rate={spike_approach_rate:.0%} (n_used={len(spike_approach_window)})")

            EVAL_EPISODES = 100
            eval_rewards = []
            eval_success_count = 0
            eval_success_ticks = []
            eval_ability_used = {t: 0 for t in ABILITY_TYPES}
            eval_ability_success = {t: 0 for t in ABILITY_TYPES}
            eval_spike_approach_count = 0

            for _ in range(EVAL_EPISODES):
                eval_obs, _ = env.reset()
                eval_mask = env.get_action_mask()
                eval_reward = 0.0
                while True:
                    with torch.no_grad():
                        q_values = q_net(torch.tensor(eval_obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
                    a = masked_softmax_action(q_values, eval_mask, temperature=0.5)
                    eval_obs, r, term, trunc, _ = env.step(a)
                    eval_mask = env.get_action_mask()
                    eval_reward += r
                    if term or trunc:
                        if term:
                            eval_success_count += 1
                            eval_success_ticks.append(env.current_step)
                        break
                eval_rewards.append(eval_reward)

                if env.last_ability_result is not None:
                    t = env.last_ability_result["type"]
                    eval_ability_used[t] += 1
                    if env.last_ability_result["success"]:
                        eval_ability_success[t] += 1
                    if env.last_spike_approach_triggered:
                        eval_spike_approach_count += 1

            mean_eval = np.mean(eval_rewards)
            eval_success_rate = eval_success_count / EVAL_EPISODES
            writer.add_scalar("Eval/Retrieve_Reward", mean_eval, episode)
            writer.add_scalar("Eval/Success_Rate", eval_success_rate, episode)
            if eval_success_ticks:
                writer.add_scalar("Eval/Ticks_To_Retrieve", np.mean(eval_success_ticks), episode)
            for t in ABILITY_TYPES:
                writer.add_scalar(f"Eval/Ability_UsedCount_{t}", eval_ability_used[t], episode)
                if eval_ability_used[t] > 0:
                    writer.add_scalar(f"Eval/Ability_SuccessRate_{t}", eval_ability_success[t] / eval_ability_used[t], episode)
            total_used = sum(eval_ability_used.values())
            if total_used > 0:
                writer.add_scalar("Eval/Ability_SpikeApproachRate", eval_spike_approach_count / total_used, episode)

            print(f"   [Eval] mean={mean_eval:.2f} success_rate={eval_success_rate:.2%} "
                  f"avg_ticks={np.mean(eval_success_ticks) if eval_success_ticks else float('nan'):.1f} n={EVAL_EPISODES}")
            eval_spike_rate = (eval_spike_approach_count / total_used) if total_used > 0 else float('nan')
            print(f"   [Eval SpikeApproach] rate={eval_spike_rate:.0%} (used={total_used})")

            if mean_eval > best_eval_reward + IMPROVEMENT_MARGIN:
                best_eval_reward = mean_eval
                best_path = os.path.join(EVAL_BEST_SAVE_DIR, "dqn_attacker_retrieve_best_by_eval.pt")
                torch.save(q_net.state_dict(), best_path)
                best_path = os.path.join(SAVE_DIR, "dqn_attacker_retrieve_best_by_eval.pt")
                torch.save(q_net.state_dict(), best_path)
                print(f"   [Eval Best] 保存しました (Eval Reward: {mean_eval:.2f}): {best_path}")

        if (episode + 1) % SAVE_INTERVAL == 0:
            save_path = os.path.join(SAVE_DIR, f"dqn_attacker_retrieve_ep{episode+1}.pt")
            torch.save(q_net.state_dict(), save_path)
            print(f"   [Save] 定期保存しました: {save_path}")

    final_path = os.path.join(SAVE_DIR, "dqn_attacker_retrieve_final.pt")
    torch.save(q_net.state_dict(), final_path)
    print(f"学習が完了しました。最終モデル: {final_path}")
    writer.close()


if __name__ == "__main__":
    train()