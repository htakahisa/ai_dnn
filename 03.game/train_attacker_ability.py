# train_attacker_ability.py
# 💡このファイルは他のtrain_*.pyに依存しない。必要なクラス・関数は全てここに複製してある。
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from map_data import NEW_MAZE_STR
from controllers import BaseController

# =====================================================================
# 🔧 調整用定数
# =====================================================================
K_ENEMIES = 3
ENEMY_MEMORY_TICKS = 15
ESCORT_OFFSET_MAX = 12
ATTACKER_SIDE_MARGIN = 10
GUARD_ATTACKER_MAX_DIST = 20
GUARD_DEFENDER_MAX_DIST = 20
N_DEFENDERS_GUARD = 5
DEFUSE_REQUIRED = 6
COVERAGE_REWARD_SCALE = 0.4
DEFUSE_BLIND_PENALTY_SCALE = 1.5
SPOT_DURING_DEFUSE_BONUS = 60.0

ABILITY_TYPES = ["flash", "smoke", "recon", "none"]
FLASH_BLIND_TICKS = 3
SMOKE_DURATION_TICKS = 15
SMOKE_RADIUS = 2
RECON_RADIUS = 4
BLIND_WIN_PROB = 0.85
NORMAL_WIN_PROB = 0.5
PLANT_REQUIRED_TICKS = 1  # 1回選択で即完了(run_game.pyの仕様と一致)

OBS_DIM = 46
N_ACTIONS = 6

NUM_EPISODES = 6000
SAVE_INTERVAL = 1000
EPISODE_SPLIT = 1000
EVAL_EPISODES_PER_PHASE = 1000

# =====================================================================
# 🧠 共有ユーティリティ(train_attacker_combined.pyから複製)
# =====================================================================
def has_line_of_sight(p1, p2, grid):
    x0, y0, x1, y1 = p1[1], p1[0], p2[1], p2[0]
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    curr_x, curr_y = x0, y0
    while True:
        if grid[curr_y, curr_x] == 1:
            return False
        if curr_x == x1 and curr_y == y1:
            return True
        e2 = 2 * err
        if e2 >= dy: err += dy; curr_x += sx
        if e2 <= dx: err += dx; curr_y += sy


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


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_actions))

    def forward(self, x):
        feat = self.feature(x)
        value = self.value_stream(feat)
        advantage = self.advantage_stream(feat)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class BaseControllerMinimal:
    """move_towards_target等、FixedEscortControllerが必要とする最小限のBFS移動ロジックのみ複製。"""

    def move_towards_target(self, pos, target, grid):
        start = tuple(pos)
        goal = tuple(target)
        if start == goal:
            return list(pos)

        height, width = grid.shape
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queue = deque([start])
        parent = {start: None}
        found = False
        while queue:
            curr = queue.popleft()
            if curr == goal:
                found = True
                break
            r, c = curr
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    if grid[nr, nc] != 1 and (nr, nc) not in parent:
                        parent[(nr, nc)] = curr
                        queue.append((nr, nc))

        if found:
            curr = goal
            while parent[curr] != start:
                curr = parent[curr]
            return list(curr)

        # 経路が無ければランダム移動にフォールバック
        r, c = pos
        valid = [
            (r + dr, c + dc) for dr, dc in moves
            if 0 <= r + dr < height and 0 <= c + dc < width and grid[r + dr, c + dc] != 1
        ]
        return list(random.choice(valid)) if valid else pos


class FixedEscortController:
    def __init__(self, offset_max=ESCORT_OFFSET_MAX):
        self.offset_max = offset_max
        self._base = BaseController()

    def _gradient_walk(self, start, dist_map, grid, steps, seek_smaller, min_dist_from_goal=1):
        height, width = grid.shape
        pos = tuple(start)
        for _ in range(steps):
            r, c = pos
            if seek_smaller and dist_map[r, c] <= min_dist_from_goal:
                break
            best = None
            best_val = dist_map[r, c]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < height and 0 <= nc < width) or grid[nr, nc] == 1:
                    continue
                val = dist_map[nr, nc]
                if not np.isfinite(val):
                    continue
                if seek_smaller and val < min_dist_from_goal:
                    continue
                if seek_smaller and val < best_val:
                    best_val, best = val, (nr, nc)
                elif not seek_smaller and val > best_val:
                    best_val, best = val, (nr, nc)
            if best is None:
                break
            pos = best
        return pos

    def compute_target(self, escort_pos, dist_map, grid, role, steps_override=None, min_dist_from_goal=1):
        seek_smaller = (role == "front")
        steps = steps_override if steps_override is not None else self.offset_max
        return self._gradient_walk(escort_pos, dist_map, grid, steps, seek_smaller, min_dist_from_goal=min_dist_from_goal)

    def next_move(self, escort_pos, target, grid, occupied_positions):
        blocked_grid = grid.copy()
        for pos in occupied_positions:
            pr, pc = pos
            if tuple(pos) != tuple(escort_pos):
                blocked_grid[pr, pc] = 1
        next_pos = self._base.move_towards_target(escort_pos, target, blocked_grid)
        if tuple(next_pos) == tuple(escort_pos):
            next_pos = self._base.move_towards_target(escort_pos, target, grid)
        return next_pos


# =====================================================================
# 🧠 敵メモリ管理(スモーク対応版)
# =====================================================================
class EnemyMemoryTracker:
    def __init__(self, memory_ticks=ENEMY_MEMORY_TICKS, k_enemies=K_ENEMIES, los_fn=None):
        self.memory_ticks = memory_ticks
        self.k_enemies = k_enemies
        self.los_fn = los_fn or has_line_of_sight
        self.memory = {}

    def reset(self):
        self.memory.clear()

    def update(self, observer_positions, enemies, grid):
        visible_ids = set()
        for enemy_id, pos, is_alive in enemies:
            if not is_alive:
                self.memory.pop(enemy_id, None)
                continue
            seen = any(self.los_fn(obs_pos, pos, grid) for obs_pos in observer_positions)
            if seen:
                self.memory[enemy_id] = {"pos": tuple(pos), "age": 0}
                visible_ids.add(enemy_id)
        stale_ids = [eid for eid in self.memory if eid not in visible_ids]
        for eid in stale_ids:
            self.memory[eid]["age"] += 1
            if self.memory[eid]["age"] > self.memory_ticks:
                del self.memory[eid]
        return visible_ids

    def build_features(self, self_pos, height, width, visible_ids):
        pr, pc = self_pos
        entries = []
        for eid, info in self.memory.items():
            er, ec = info["pos"]
            dist = max(abs(er - pr), abs(ec - pc))
            entries.append((dist, eid, info))
        entries.sort(key=lambda x: x[0])
        feats = []
        for i in range(self.k_enemies):
            if i < len(entries):
                _, eid, info = entries[i]
                er, ec = info["pos"]
                visible = 1.0 if eid in visible_ids else 0.0
                stale = 0.0 if visible else 1.0
                dr = (er - pr) / height
                dc = (ec - pc) / width
                age_norm = min(info["age"] / self.memory_ticks, 1.0)
                feats.extend([visible, stale, dr, dc, age_norm])
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        return feats


# =====================================================================
# 🎮 ability対応環境（AttackerMultiEnvを継承せず、単独クラスとして定義）
# =====================================================================
class AttackerAbilityEnv:
    """retrieve / carry(護衛付き・人数可変) / guard(複数敵・ability) を学習する統合環境。
    他ファイルのEnvクラスを継承しない、完全独立実装。"""

    def __init__(self, fixed_grid, plant_candidates):
        self.grid = fixed_grid
        self.height, self.width = fixed_grid.shape
        self.plant_candidates = [tuple(p) for p in plant_candidates]
        spawn_rows, spawn_cols = np.where(fixed_grid == 3)
        self.attacker_spawn_candidates = list(zip(spawn_rows.tolist(), spawn_cols.tolist()))
        self.max_steps = 150
        self.detonate_limit = 45
        self.policy_net = None
        self.escort_ctrl = FixedEscortController()
        self.enemy_memory = EnemyMemoryTracker(los_fn=self._smoke_aware_los)

        self.ability_type = "none"
        self.ability_charges = 0
        self.player_blind_remaining = 0
        self.active_smoke_cells = set()
        self.smoke_remaining_ticks = 0
        self.escort_positions = {}

    def _is_walkable(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width and self.grid[r, c] != 1

    def _random_walkable(self):
        while True:
            p = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            if self._is_walkable(*p):
                return p

    def _sample_guard_positions(self, dist_map):
        site_r, site_c = self.goal_pos
        attacker_candidates = [
            (r, c) for r in range(site_r, self.height)
            for c in range(self.width)
            if self._is_walkable(r, c)
            and (c <= ATTACKER_SIDE_MARGIN or c >= self.width - 1 - ATTACKER_SIDE_MARGIN)
            and np.isfinite(dist_map[r, c]) and dist_map[r, c] <= GUARD_ATTACKER_MAX_DIST
        ]
        defender_candidates = [
            (r, c) for r in range(0, site_r + 1)
            for c in range(self.width)
            if self._is_walkable(r, c)
            and np.isfinite(dist_map[r, c]) and dist_map[r, c] <= GUARD_DEFENDER_MAX_DIST
        ]
        if not attacker_candidates:
            attacker_candidates = [(site_r, site_c)]
        if not defender_candidates:
            defender_candidates = [(site_r, site_c)]
        return attacker_candidates, defender_candidates

    # -------------------------------------------------------------
    def reset(self, seed=None, options=None, phase=None):
        self.current_step = 0
        self.last_action = None
        # 💡 guardフェーズを発生させず、retrieveとcarryのみを学習させる
        self.phase = phase if phase is not None else random.choices(
            ["retrieve", "carry"], weights=[0.4, 0.6]
        )[0]

        self.carrying = False
        self.is_planted = False
        self.pos_history = deque(maxlen=6)
        self.escort_positions = {}

        self.player_blind_remaining = 0
        self.active_smoke_cells = set()
        self.smoke_remaining_ticks = 0
        self.ability_type = "none"
        self.ability_charges = 0
        self.enemy_memory.reset()

        if self.phase == "retrieve":
            self.player_pos = self._random_walkable()
            self.goal_pos = self._random_walkable()
            while self.goal_pos == self.player_pos:
                self.goal_pos = self._random_walkable()

        elif self.phase == "carry":
            self.player_pos = random.choice(self.attacker_spawn_candidates)
            self.carrying = True
            self.goal_pos = random.choice(self.plant_candidates)
            # 💡multi版と同じ、護衛2体固定に戻す
            self.escort_positions = {
                "front": random.choice(self.attacker_spawn_candidates),
                "back": random.choice(self.attacker_spawn_candidates),
            }
            self.carry_end_reason = None
            self.carry_arrival_step = None

        else:  # guard
            self.is_planted = True
            self.goal_pos = random.choice(self.plant_candidates)
            self.dist_map = bfs_distances(self.goal_pos, self.grid)

            attacker_pool, defender_pool = self._sample_guard_positions(self.dist_map)
            self.player_pos = random.choice(attacker_pool)
            self.teammate_pos = random.choice(attacker_pool)
            self.teammate_alive = True

            self.defenders = []
            for i in range(N_DEFENDERS_GUARD):
                self.defenders.append({
                    "id": i, "pos": random.choice(defender_pool), "alive": True,
                    "last_action": None, "defuse_timer": 0, "blind_remaining": 0,
                })
            self.detonate_timer = self.detonate_limit
            self.guard_end_reason = None

            self.ability_type = random.choice(ABILITY_TYPES)
            self.ability_charges = 1 if self.ability_type != "none" else 0

        self.dist_map = bfs_distances(self.goal_pos, self.grid)
        self.prev_dist = self.dist_map[self.player_pos[0]][self.player_pos[1]]

        self._choke_points = []
        if self.phase == "guard":
            self._compute_choke_points()

        return self._get_obs(), {}

    # -------------------------------------------------------------
    def _get_obs(self):
        pr, pc = self.player_pos
        gr, gc = self.goal_pos
        base = [pr / (self.height - 1), pc / (self.width - 1),
                gr / (self.height - 1), gc / (self.width - 1)]
        walls = [0.0 if self._is_walkable(pr + dr, pc + dc) else 1.0
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

        last_onehot = [0.0] * N_ACTIONS
        if self.last_action is not None:
            last_onehot[self.last_action] = 1.0

        max_dist = max(self.height, self.width) * 2
        dists = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pr + dr, pc + dc
            if self._is_walkable(nr, nc):
                d = self.dist_map[nr][nc]
                dists.append(d / max_dist if np.isfinite(d) else 1.0)
            else:
                dists.append(1.0)

        phase_flags = [1.0 if self.carrying else 0.0, 1.0 if self.is_planted else 0.0]

        visible_ids = set()
        if self.phase == "guard":
            observer_positions = [tuple(self.player_pos)]
            if self.teammate_alive:
                observer_positions.append(tuple(self.teammate_pos))
            enemies = [(d["id"], d["pos"], d["alive"]) for d in self.defenders]
            visible_ids = self.enemy_memory.update(observer_positions, enemies, self.grid)
        enemy_feats = self.enemy_memory.build_features((pr, pc), self.height, self.width, visible_ids)

        # 💡変更: carry中はallyを使わない(スパイク保持者の最優先事項はサイト到達であり、
        #    護衛の位置に判断を左右されるべきではない)
        ally = [0.0, 0.0, 0.0]
        if self.phase == "guard" and self.teammate_alive:
            tr, tc = self.teammate_pos
            ally = [1.0, (tr - pr) / self.height, (tc - pc) / self.width]

        defuse_info = [0.0, 0.0]
        if self.phase == "guard":
            max_timer = max((d["defuse_timer"] for d in self.defenders), default=0)
            if max_timer > 0:
                defuse_info = [1.0, min(max_timer / DEFUSE_REQUIRED, 1.0)]

        ability_onehot = [0.0] * 4
        ability_onehot[ABILITY_TYPES.index(self.ability_type)] = 1.0
        charges = [1.0 if self.ability_charges > 0 else 0.0]
        self_blind = [1.0 if self.player_blind_remaining > 0 else 0.0]

        return np.array(base + walls + last_onehot + dists + phase_flags
                         + enemy_feats + ally + defuse_info
                         + ability_onehot + charges + self_blind, dtype=np.float32)

    def _get_teammate_obs(self):
        pr, pc = self.teammate_pos
        gr, gc = self.goal_pos
        base = [pr / (self.height - 1), pc / (self.width - 1),
                gr / (self.height - 1), gc / (self.width - 1)]
        walls = [0.0 if self._is_walkable(pr + dr, pc + dc) else 1.0
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        last_onehot = [0.0] * N_ACTIONS
        max_dist = max(self.height, self.width) * 2
        dists = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pr + dr, pc + dc
            if self._is_walkable(nr, nc):
                d = self.dist_map[nr][nc]
                dists.append(d / max_dist if np.isfinite(d) else 1.0)
            else:
                dists.append(1.0)
        phase_flags = [0.0, 1.0]
        enemy_feats = [0.0] * (K_ENEMIES * 5)
        sr, sc = self.player_pos
        ally = [1.0, (sr - pr) / self.height, (sc - pc) / self.width]
        defuse_info = [0.0, 0.0]
        
        ability_onehot = [0.0, 0.0, 0.0, 1.0]
        charges = [0.0]
        self_blind = [0.0]
        return np.array(base + walls + last_onehot + dists + phase_flags
                         + enemy_feats + ally + defuse_info
                         + ability_onehot + charges + self_blind, dtype=np.float32)

    def _move_teammate(self, action):
        if action >= 4:
            return
        moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        r, c = self.teammate_pos
        nr, nc = r + moves[action][0], c + moves[action][1]
        if self._is_walkable(nr, nc):
            self.teammate_pos = (nr, nc)

    def _move_escorts(self):
        """護衛2体を固定方策で1歩動かす(multi版と同じ、front/back固定)。"""
        occupied = [tuple(self.player_pos)] + list(self.escort_positions.values())
        for i, role in enumerate(("front", "back")):
            pos = self.escort_positions[role]
            offset_steps = max(2, ESCORT_OFFSET_MAX - i * 3)
            target = self.escort_ctrl.compute_target(pos, self.dist_map, self.grid, "front", steps_override=offset_steps)
            next_pos = self.escort_ctrl.next_move(pos, target, self.grid,
                                                    [p for p in occupied if p != tuple(pos)])
            self.escort_positions[role] = tuple(next_pos)

    # -------------------------------------------------------------
    def step(self, action):
        self.current_step += 1
        reward = 0.0
        terminated = False
        self.last_action = action

        if self.phase == "retrieve":
            if action in (4, 5):
                reward = -1.0
            else:
                reward, terminated = self._step_move(action, arrival_reward=100.0)

        elif self.phase == "carry":
            pr, pc = self.player_pos
            if action == 4:
                if self.grid[pr, pc] == 2:
                    dist_from_ideal = self.dist_map[pr][pc]
                    reward = 200.0 - min(dist_from_ideal * 5.0, 100.0)
                    terminated = True
                    self.carry_end_reason = "planted"
                    self.carry_arrival_step = self.current_step
                else:
                    reward = -1.0
            elif action == 5:
                reward = -1.0
            else:
                reward, _ = self._step_move(action, arrival_reward=None)
            self._move_escorts()

        else:  # guard
            reward, terminated = self._step_guard(action)

        truncated = self.current_step >= self.max_steps
        if self.phase == "carry" and truncated and self.carry_end_reason is None:
            self.carry_end_reason = "timeout"

        self._advance_ability_timers()
        return self._get_obs(), reward, terminated, truncated, {}

    def _step_guard(self, action):
        reward = 0.0
        terminated = False
        self.guard_end_reason = None

        if action == 4:
            reward = -0.1
        elif action == 5:
            reward += self._use_ability_guard()
        else:
            reward, _ = self._step_move(action, arrival_reward=None, guard_mode=True)

        if self.teammate_alive and self.policy_net is not None:
            tobs = self._get_teammate_obs()
            with torch.no_grad():
                t_q = self.policy_net(torch.tensor(tobs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy().copy()
            t_q[4] = -np.inf
            t_q[5] = -np.inf
            t_action = int(np.argmax(t_q))
            self._move_teammate(t_action)

        any_defusing = False
        for d in self.defenders:
            if not d["alive"]:
                continue
            if d.get("blind_remaining", 0) > 0:
                d["blind_remaining"] -= 1
                d["defuse_timer"] = 0
                walkable_dirs = [
                    (dr, dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if self._is_walkable(d["pos"][0] + dr, d["pos"][1] + dc)
                ]
                if walkable_dirs and random.random() < 0.5:
                    ddr, ddc = random.choice(walkable_dirs)
                    d["pos"] = (d["pos"][0] + ddr, d["pos"][1] + ddc)
                continue

            dr_, dc_ = d["pos"]
            dist_to_spike = max(abs(dr_ - self.goal_pos[0]), abs(dc_ - self.goal_pos[1]))
            if dist_to_spike <= 1:
                d["defuse_timer"] += 1
                any_defusing = True
                if d["defuse_timer"] >= DEFUSE_REQUIRED:
                    reward -= 100.0
                    terminated = True
                    self.guard_end_reason = "defused"
            else:
                best, best_d = None, self.dist_map[dr_][dc_]
                for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = dr_ + ddr, dc_ + ddc
                    if self._is_walkable(nr, nc) and self.dist_map[nr][nc] < best_d:
                        best_d = self.dist_map[nr][nc]
                        best = (nr, nc)
                if best is not None:
                    d["pos"] = best
                d["defuse_timer"] = 0

        observer_positions = [tuple(self.player_pos)]
        if self.teammate_alive:
            observer_positions.append(tuple(self.teammate_pos))

        visible_defenders = [
            d for d in self.defenders
            if d["alive"] and any(self._smoke_aware_los(op, d["pos"], self.grid) for op in observer_positions)
        ]

        if visible_defenders:
            n_visible = len(visible_defenders)
            for d in visible_defenders:
                progress = d["defuse_timer"] / DEFUSE_REQUIRED
                win_prob = BLIND_WIN_PROB if d.get("blind_remaining", 0) > 0 else NORMAL_WIN_PROB
                if random.random() < win_prob:
                    d["alive"] = False
                    reward += (100.0 + SPOT_DURING_DEFUSE_BONUS * (1.0 - progress) * (1.0 if progress > 0 else 0.0)) / n_visible
                else:
                    reward -= 30.0 / n_visible

        coverage = self._spike_neighbor_coverage_smoke(self.player_pos)
        reward += coverage * COVERAGE_REWARD_SCALE

        if any_defusing:
            visible_defuser = any(
                self._smoke_aware_los(tuple(self.player_pos), d["pos"], self.grid)
                for d in self.defenders if d["alive"] and d["defuse_timer"] > 0
            )
            if not visible_defuser:
                max_progress = max(
                    (d["defuse_timer"] / DEFUSE_REQUIRED for d in self.defenders if d["alive"]),
                    default=0.0
                )
                reward -= DEFUSE_BLIND_PENALTY_SCALE * max_progress

        if not any(d["alive"] for d in self.defenders) and not terminated:
            reward += 50.0
            terminated = True
            self.guard_end_reason = "annihilated"
        elif not terminated:
            self.detonate_timer -= 1
            if self.detonate_timer <= 0:
                reward += 50.0
                terminated = True
                self.guard_end_reason = "survived_timeout"

        return reward, terminated

    def _use_ability_guard(self):
        if self.ability_charges <= 0 or self.ability_type == "none":
            return -1.0
        self.ability_charges -= 1
        target_cell = self._pick_ability_target()

        if self.ability_type == "flash":
            hit = 0
            for d in self.defenders:
                if d["alive"] and has_line_of_sight(target_cell, d["pos"], self.grid):
                    d["blind_remaining"] = FLASH_BLIND_TICKS
                    hit += 1
            return 15.0 * hit if hit > 0 else -1.0

        if self.ability_type == "smoke":
            tr, tc = target_cell
            self.active_smoke_cells = {
                (r, c) for r in range(tr - SMOKE_RADIUS, tr + SMOKE_RADIUS + 1)
                for c in range(tc - SMOKE_RADIUS, tc + SMOKE_RADIUS + 1)
                if self._is_walkable(r, c)
            }
            self.smoke_remaining_ticks = SMOKE_DURATION_TICKS
            return 3.0

        if self.ability_type == "recon":
            revealed = 0
            rr, rc = target_cell
            for d in self.defenders:
                if d["alive"] and max(abs(d["pos"][0] - rr), abs(d["pos"][1] - rc)) <= RECON_RADIUS:
                    self.enemy_memory.memory[d["id"]] = {"pos": tuple(d["pos"]), "age": 0}
                    revealed += 1
            return 8.0 * revealed if revealed > 0 else -1.0

        return 0.0

    def _compute_choke_points(self):
        candidates = []
        for r in range(self.height):
            for c in range(self.width):
                if not self._is_walkable(r, c):
                    continue
                d = self.dist_map[r][c]
                if not np.isfinite(d) or d < 2 or d > 10:
                    continue
                n = self._is_walkable(r - 1, c)
                s = self._is_walkable(r + 1, c)
                w = self._is_walkable(r, c - 1)
                e = self._is_walkable(r, c + 1)
                vertical_corridor = n and s and not w and not e
                horizontal_corridor = w and e and not n and not s
                if vertical_corridor or horizontal_corridor:
                    candidates.append((d, (r, c)))
        candidates.sort(key=lambda x: x[0])
        self._choke_points = [pos for _, pos in candidates]

    def _pick_ability_target(self):
        if self.enemy_memory.memory:
            nearest = min(
                self.enemy_memory.memory.items(),
                key=lambda kv: max(
                    abs(kv[1]["pos"][0] - self.player_pos[0]),
                    abs(kv[1]["pos"][1] - self.player_pos[1]),
                ),
            )
            return nearest[1]["pos"]
        if self._choke_points:
            return self._choke_points[0]
        return self.goal_pos

    def _smoke_aware_los(self, p1, p2, grid):
        if not has_line_of_sight(p1, p2, grid):
            return False
        if not self.active_smoke_cells:
            return True
        return not self._smoke_blocks_line(p1, p2)

    def _smoke_blocks_line(self, p1, p2):
        cells = self._line_cells(p1, p2)
        interior = cells[1:-1] if len(cells) > 2 else []
        return any(c in self.active_smoke_cells for c in interior)

    def _line_cells(self, start, end):
        y0, x0 = start
        y1, x1 = end
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        cells = []
        while True:
            cells.append((y0, x0))
            if x0 == x1 and y0 == y1:
                return cells
            e2 = 2 * err
            if e2 >= dy: err += dy; x0 += sx
            if e2 <= dx: err += dx; y0 += sy

    def _spike_neighbor_coverage_smoke(self, from_pos):
        gr, gc = self.goal_pos
        neighbors = [(gr + dr, gc + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]
        walkable = [n for n in neighbors if self._is_walkable(*n)]
        if not walkable:
            return 0.0
        visible = sum(1 for n in walkable if self._smoke_aware_los(tuple(from_pos), n, self.grid))
        return visible / len(walkable)

    def _advance_ability_timers(self):
        if self.player_blind_remaining > 0:
            self.player_blind_remaining -= 1
        if self.smoke_remaining_ticks > 0:
            self.smoke_remaining_ticks -= 1
            if self.smoke_remaining_ticks <= 0:
                self.active_smoke_cells = set()

    def _step_move(self, action, arrival_reward, guard_mode=False):
        moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        r, c = self.player_pos
        nr, nc = r + moves[action][0], c + moves[action][1]

        if not self._is_walkable(nr, nc):
            return -1.5, False

        self.player_pos = (nr, nc)
        new_dist = self.dist_map[nr][nc]
        shaping = (self.prev_dist - new_dist) * 0.5

        if (nr, nc) == self.goal_pos and arrival_reward is not None:
            self.prev_dist = new_dist
            return arrival_reward, True

        reward = -1.0 + shaping
        if arrival_reward is not None and np.isfinite(new_dist) and new_dist <= 3:
            reward += 1.5

        pos_tuple = (nr, nc)
        near_goal = np.isfinite(new_dist) and new_dist <= 3
        if not near_goal and pos_tuple in self.pos_history and new_dist >= self.prev_dist:
            reward -= 2.0
        self.pos_history.append(pos_tuple)

        if guard_mode:
            d_spike = max(abs(nr - self.goal_pos[0]), abs(nc - self.goal_pos[1]))
            if d_spike > 6:
                reward -= 0.3

        self.prev_dist = new_dist
        return reward, False



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    # mask と next_mask を追加
    def push(self, obs, action, reward, next_obs, done, mask, next_mask):
        self.buffer.append((obs, action, reward, next_obs, done, mask, next_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done, mask, next_mask = zip(*batch)
        return (np.array(obs), np.array(action), np.array(reward), 
                np.array(next_obs), np.array(done), np.array(mask), np.array(next_mask))

def get_action_mask(env):
    # 0.0 は有効、-np.inf は無効
    mask = np.zeros(N_ACTIONS, dtype=np.float32)
    if env.phase == "retrieve":
        mask[4] = -np.inf
        mask[5] = -np.inf
    elif env.phase == "carry":
        if env.grid[env.player_pos[0], env.player_pos[1]] != 2:
            mask[4] = -np.inf
        mask[5] = -np.inf
    else:
        if env.ability_charges <= 0:
            mask[5] = -np.inf
    return mask

def mask_invalid_actions(q_values, mask):
    return q_values + mask

# =====================================================================
# 🏋️ 学習ループ(前バージョンと同一。省略せず記載)
# =====================================================================
def train():
    writer = SummaryWriter(log_dir="logs")
    SAVE_DIR = "data_temp/attacker_ability_data"
    os.makedirs(SAVE_DIR, exist_ok=True)

    lines = [line.strip() for line in NEW_MAZE_STR.strip("\n").split("\n") if line.strip()]
    fixed_grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
    plant_rows, plant_cols = np.where(fixed_grid == 2)
    plant_candidates = list(zip(plant_rows, plant_cols))
    if not plant_candidates:
        raise ValueError("プラントサイト(2)が定義されていません。")

    env = AttackerAbilityEnv(fixed_grid, plant_candidates)

    batch_size = 64
    gamma = 0.99
    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 0.9985
    lr = 0.0005
    IMPROVEMENT_MARGIN = 5.0
    

    device = torch.device("cpu")
    q_net = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(device)
    target_net = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(device)
    target_net.load_state_dict(q_net.state_dict())
    env.policy_net = target_net

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=30000)
    epsilon = epsilon_start
    best_eval_reward = -float('inf')

    print(f"学習を開始します。デバイス: {device} | 入力次元: {OBS_DIM} | 行動数: {N_ACTIONS}")
    print("python -m tensorboard.main --logdir=logs")

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        current_mask = get_action_mask(env)

        episode_reward = 0.0
        losses = []

        while True:
            # 行動選択時
            if random.random() < epsilon:
                max_action = N_ACTIONS if env.phase == "guard" else N_ACTIONS - 1
                action = random.randint(0, max_action - 1)
            else:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = q_net(obs_t).squeeze(0).cpu().numpy()
                    eval_mask = get_action_mask(env) # マスクを取得する
                    q_values = mask_invalid_actions(q_values, eval_mask)
                    action = int(np.argmax(q_values))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_mask = get_action_mask(env) # 修正

            # Bufferに mask と next_mask も記録
            replay_buffer.push(obs, action, reward, next_obs, terminated, current_mask, next_mask)
            
            obs = next_obs
            current_mask = next_mask
            episode_reward += reward

            if len(replay_buffer) >= batch_size:
                # バッチサンプリング
                b_obs, b_act, b_rew, b_nobs, b_term, b_mask, b_nmask = replay_buffer.sample(batch_size)
                
                b_obs_t = torch.tensor(b_obs, dtype=torch.float32, device=device)
                b_act_t = torch.tensor(b_act, dtype=torch.long, device=device).unsqueeze(1)
                b_rew_t = torch.tensor(b_rew, dtype=torch.float32, device=device).unsqueeze(1)
                b_nobs_t = torch.tensor(b_nobs, dtype=torch.float32, device=device)
                b_term_t = torch.tensor(b_term, dtype=torch.float32, device=device).unsqueeze(1)
                b_nmask_t = torch.tensor(b_nmask, dtype=torch.float32, device=device) # 次の状態のマスク

                current_q = q_net(b_obs_t).gather(1, b_act_t)
                with torch.no_grad():
                    # 💡 ここで next_mask を足し合わせることで、無効な行動のQ値を -np.inf にする
                    next_q_values = q_net(b_nobs_t) + b_nmask_t
                    next_actions = next_q_values.argmax(dim=1, keepdim=True)
                    
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

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        writer.add_scalar("Train/Episode_Reward", episode_reward, episode)

        if (episode + 1) % EPISODE_SPLIT == 0:
            avg_loss = np.mean(losses) if losses else 0.0
            print(f"Episode {episode+1}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f}")

            # 💡 guardの辞書キーを削除
            phase_rewards = {"retrieve": [], "carry": []} 
            guard_end_reasons = {"defused": 0, "annihilated": 0, "survived_timeout": 0, "truncated": 0}
            carry_end_reasons = {"planted": 0, "timeout": 0}
            carry_arrival_steps = []
            ability_use_count = 0
            ability_positive_count = 0

            # 💡 評価対象を retrieve と carry のみに変更
            for phase_name in ["retrieve", "carry"]:
                for _ in range(EVAL_EPISODES_PER_PHASE):
                    eval_obs, _ = env.reset(phase=phase_name)
                    phase = env.phase
                    eval_reward = 0.0
                    while True:
                        with torch.no_grad():
                            q_values = q_net(
                                torch.tensor(eval_obs, dtype=torch.float32, device=device).unsqueeze(0)
                            ).squeeze(0).cpu().numpy()
                        eval_mask = get_action_mask(env) # マスクを取得する
                        q_values = mask_invalid_actions(q_values, eval_mask)
                        probs = np.exp((q_values - np.max(q_values)) / 0.5)
                        probs = probs / probs.sum()
                        a = np.random.choice(len(probs), p=probs)

                        if phase == "guard" and a == 5:
                            ability_use_count += 1

                        eval_obs, r, term, trunc, _ = env.step(a)
                        if phase == "guard" and a == 5 and r > 0:
                            ability_positive_count += 1
                        eval_reward += r
                        if term or trunc:
                            if phase == "guard":
                                reason = env.guard_end_reason if term else "truncated"
                                guard_end_reasons[reason] += 1
                            if phase == "carry":
                                reason = env.carry_end_reason if env.carry_end_reason else "timeout"
                                carry_end_reasons[reason] += 1
                                if reason == "planted" and env.carry_arrival_step is not None:
                                    carry_arrival_steps.append(env.carry_arrival_step)
                            break
                    phase_rewards[phase].append(eval_reward)

            greedy_carry_success = 0
            greedy_carry_total = 0
            for _ in range(EVAL_EPISODES_PER_PHASE):
                eval_obs, _ = env.reset(phase="carry")
                while True:
                    with torch.no_grad():
                        q_values = q_net(
                            torch.tensor(eval_obs, dtype=torch.float32, device=device).unsqueeze(0)
                        ).squeeze(0).cpu().numpy()
                    eval_mask = get_action_mask(env) # マスクを取得する
                    q_values = mask_invalid_actions(q_values, eval_mask)
                    a = int(np.argmax(q_values))
                    eval_obs, r, term, trunc, _ = env.step(a)
                    if term or trunc:
                        greedy_carry_total += 1
                        if env.carry_end_reason == "planted":
                            greedy_carry_success += 1
                        break

            if greedy_carry_total > 0:
                greedy_rate = greedy_carry_success / greedy_carry_total
                print(f"   [carry greedy] success_rate={greedy_rate*100:.1f}%")
                writer.add_scalar("Eval/carry_greedy_success_rate", greedy_rate, episode)

            for phase_name, rewards_list in phase_rewards.items():
                if rewards_list:
                    writer.add_scalar(f"Eval/{phase_name}_Reward", np.mean(rewards_list), episode)
                    print(f"   [{phase_name}] mean={np.mean(rewards_list):.2f} n={len(rewards_list)}")

            carry_total = sum(carry_end_reasons.values())
            if carry_total > 0:
                success_rate = carry_end_reasons["planted"] / carry_total
                avg_arrival = np.mean(carry_arrival_steps) if carry_arrival_steps else 0.0
                print(f"   [carry breakdown] planted={carry_end_reasons['planted']} / timeout={carry_end_reasons['timeout']} "
                      f"(success_rate={success_rate*100:.1f}%, avg_arrival_step={avg_arrival:.1f})")
                writer.add_scalar("Eval/carry_success_rate", success_rate, episode)

            guard_total = sum(guard_end_reasons.values())
            if guard_total > 0:
                breakdown_str = " / ".join(f"{k}={v}" for k, v in guard_end_reasons.items())
                print(f"   [guard breakdown] {breakdown_str} (total={guard_total})")
                for k, v in guard_end_reasons.items():
                    writer.add_scalar(f"Eval/guard_{k}_rate", v / guard_total, episode)

            if ability_use_count > 0:
                hit_rate = ability_positive_count / ability_use_count
                print(f"   [ability] used={ability_use_count} positive_result={ability_positive_count} (hit_rate={hit_rate*100:.1f}%)")
                writer.add_scalar("Eval/ability_hit_rate", hit_rate, episode)
                writer.add_scalar("Eval/ability_use_count", ability_use_count, episode)

            # 💡 guardの計算を除外し、retrieveとcarryの平均を最終評価スコアとする
            mean_eval = np.mean(phase_rewards["retrieve"] + phase_rewards["carry"]) if (phase_rewards["retrieve"] + phase_rewards["carry"]) else 0.0
            writer.add_scalar("Eval/Weighted_Reward", mean_eval, episode)

            if mean_eval > best_eval_reward + IMPROVEMENT_MARGIN:
                best_eval_reward = mean_eval
                best_path = os.path.join(SAVE_DIR, "dqn_attacker_ability_best_by_eval.pt")
                torch.save(q_net.state_dict(), best_path)
                print(f"   [Eval Best] 保存しました (Eval Reward: {mean_eval:.2f}): {best_path}")

        if (episode + 1) % SAVE_INTERVAL == 0:
            save_path = os.path.join(SAVE_DIR, f"dqn_attacker_ability_ep{episode+1}.pt")
            torch.save(q_net.state_dict(), save_path)
            print(f"   [Save] 定期保存しました: {save_path}")

    final_path = os.path.join(SAVE_DIR, "dqn_attacker_ability_final.pt")
    torch.save(q_net.state_dict(), final_path)
    print(f"学習が完了しました。最終モデル: {final_path}")
    writer.close()


if __name__ == "__main__":
    train()