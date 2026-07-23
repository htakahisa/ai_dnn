# train_attacker_multi.py
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from map_data import NEW_MAZE_STR
from train_defender_combined import QNetwork, ReplayBuffer
from train_attacker_combined import DuelingQNetwork, has_line_of_sight, bfs_distances, DefenderBotAI
from controllers import BaseController

# =====================================================================
# 🔧 調整用定数(このあたりを触って挙動をチューニングする想定)
# =====================================================================

# --- 敵メモリ(複数敵の視認・記憶) ---
K_ENEMIES = 3               # 観測に含める敵の最大人数
ENEMY_MEMORY_TICKS = 15     # 見失ってから記憶を保持する最大tick数

# --- carryフェーズ: 護衛フォーメーション ---
ESCORT_OFFSET_MAX = 12      # 前衛/後衛のオフセット距離(最大)

# --- guardフェーズ: 初期配置 ---
ATTACKER_SIDE_MARGIN = 10   # アタッカー配置の横方向許容範囲(画面端からのマス数)
GUARD_ATTACKER_MAX_DIST = 20  # サイトからの近さ制限(BFS距離。あまり遠くから開始させない)
GUARD_DEFENDER_MAX_DIST = 20
N_DEFENDERS_GUARD = 5       # guard学習でシミュレートするdefender人数(簡略化版・要調整)

# --- guardフェーズ: 解除阻止まわり ---
DEFUSE_REQUIRED = 6
COVERAGE_REWARD_SCALE = 0.4      # スパイク隣接8マスのLOSカバー率に応じた報酬係数
DEFUSE_BLIND_PENALTY_SCALE = 1.5  # 見えない状態で解除が進んでいる時のペナルティ係数
SPOT_DURING_DEFUSE_BONUS = 60.0   # 解除中に発見・撃破できた場合の追加ボーナス(早期発見ほど大)

OBS_DIM = 39  # 内訳は下記 _get_obs のコメント参照
N_ACTIONS = 5

# 学習 episode 数
NUM_EPISODES = 2000
SAVE_INTERVAL = 100

# =====================================================================
# 🧠 敵メモリ管理(train/inference共通)
# =====================================================================
class EnemyMemoryTracker:
    """視認した敵の位置を数tick保持するための共有ロジック。
    学習環境と本番コントローラの両方から同じクラスを使うことで、obsの意味がズレないようにする。"""

    def __init__(self, memory_ticks=ENEMY_MEMORY_TICKS, k_enemies=K_ENEMIES):
        self.memory_ticks = memory_ticks
        self.k_enemies = k_enemies
        self.memory = {}  # enemy_id -> {"pos": (r, c), "age": int}

    def reset(self):
        self.memory.clear()

    def update(self, observer_positions, enemies, grid):
        """
        observer_positions: 視認判定を行う自陣キャラのpos一覧 [(r,c), ...]
        enemies: [(enemy_id, pos, is_alive), ...] 現在の敵の実際の状態
        戻り値: このtickで実際に視認できたenemy_idのset
        """
        visible_ids = set()
        for enemy_id, pos, is_alive in enemies:
            if not is_alive:
                self.memory.pop(enemy_id, None)
                continue
            seen = any(has_line_of_sight(obs_pos, pos, grid) for obs_pos in observer_positions)
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
        """K枠分の敵特徴量を [visible, stale, dr, dc, age_norm] × K のフラットなリストで返す"""
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
# 🚶 carryフェーズ: 護衛(前衛/後衛)の固定方策(train/inference共通)
# =====================================================================
class FixedEscortController:
    """carryフェーズの護衛を固定方策(BFS勾配追従)で動かす。学習・本番で同じロジックを使う。"""

    def __init__(self, offset_max=ESCORT_OFFSET_MAX):
        self.offset_max = offset_max
        self._base = BaseController()

    def _gradient_walk(self, start, dist_map, grid, steps, seek_smaller, min_dist_from_goal=1):
        """dist_map(ゴールからの距離マップ)上を勾配に沿って歩き、offset分だけ離れた地点を返す。
        seek_smaller=True: ゴールに近づく方向(前衛用)
        seek_smaller=False: ゴールから離れる方向(後衛用)
        min_dist_from_goal: ゴール自体からこの距離未満には近づかない(将来のマス排他制御を見据えた予約)"""
        height, width = grid.shape
        pos = tuple(start)
        for _ in range(steps):
            r, c = pos
            # 💡ゴールに既に十分近ければ、これ以上は進まない(ゴールマス自体を専有しない)
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
                # 💡ゴールに近づきすぎる移動は候補から除外
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
        """role: 'front' or 'back'。steps_overrideでオフセット距離を上書き可能(団子防止用)"""
        seek_smaller = (role == "front")
        steps = steps_override if steps_override is not None else self.offset_max
        return self._gradient_walk(escort_pos, dist_map, grid, steps, seek_smaller, min_dist_from_goal=min_dist_from_goal)

    def next_move(self, escort_pos, target, grid, occupied_positions):
        """occupied_positions(他キャラの現在地)を一時的に壁扱いして渋滞・重なりを回避しつつ1歩進める"""
        blocked_grid = grid.copy()
        for pos in occupied_positions:
            pr, pc = pos
            if tuple(pos) != tuple(escort_pos):
                blocked_grid[pr, pc] = 1

        next_pos = self._base.move_towards_target(escort_pos, target, blocked_grid)
        if tuple(next_pos) == tuple(escort_pos):
            # 完全にブロックされていた場合は通常gridでフォールバック(足止め回避)
            next_pos = self._base.move_towards_target(escort_pos, target, grid)
        return next_pos


# =====================================================================
# 🧠 Dueling Q-Network は train_attacker_combined.py のものをそのまま使う
# =====================================================================


class AttackerMultiEnv:
    """retrieve / carry(護衛付き) / guard(複数敵・解除阻止) を学習する統合環境"""

    def __init__(self, fixed_grid, plant_candidates):
        self.grid = fixed_grid
        self.height, self.width = fixed_grid.shape
        self.plant_candidates = [tuple(p) for p in plant_candidates]
        spawn_rows, spawn_cols = np.where(fixed_grid == 3)
        self.attacker_spawn_candidates = list(zip(spawn_rows.tolist(), spawn_cols.tolist()))
        self.max_steps = 150
        self.detonate_limit = 45
        self.policy_net = None          # guardのteammate self-play用
        self.defender_bot = None        # guardの単一defender評価に使う既存モデル(互換用・未使用でも可)
        self.escort_ctrl = FixedEscortController()
        self.enemy_memory = EnemyMemoryTracker()

    def _is_walkable(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width and self.grid[r, c] != 1

    def _random_walkable(self):
        while True:
            p = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            if self._is_walkable(*p):
                return p

    # -----------------------------------------------------------------
    # guardフェーズ用の配置候補計算
    # アタッカー: サイト含みそこから下方向、横は画面端からATTACKER_SIDE_MARGIN以内
    # ディフェンダー: サイト含みそこから上方向、横の制限なし
    # -----------------------------------------------------------------
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

    def reset(self, seed=None, options=None, phase=None):
        self.current_step = 0
        self.last_action = None
        self.phase = phase if phase is not None else random.choices(
            ["retrieve", "carry", "guard"], weights=[0.3, 0.4, 0.3]
        )[0]

        self.carrying = False
        self.is_planted = False
        self.pos_history = deque(maxlen=6)
        self.escort_positions = {}   # role -> pos (carryフェーズのみ使用)
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
            self.dist_map = bfs_distances(self.goal_pos, self.grid)
            self.escort_positions["front"] = random.choice(self.attacker_spawn_candidates)
            self.escort_positions["back"] = random.choice(self.attacker_spawn_candidates)
            self.carry_end_reason = None          # 💡追加
            self.carry_arrival_step = None        # 💡追加: 到達できた場合、何tick目だったか

        else:  # guard
            self.is_planted = True
            self.goal_pos = random.choice(self.plant_candidates)  # スパイク位置
            self.dist_map = bfs_distances(self.goal_pos, self.grid)

            attacker_pool, defender_pool = self._sample_guard_positions(self.dist_map)
            self.player_pos = random.choice(attacker_pool)
            self.teammate_pos = random.choice(attacker_pool)
            self.teammate_alive = True

            self.defenders = []
            for i in range(N_DEFENDERS_GUARD):
                self.defenders.append({
                    "id": i,
                    "pos": random.choice(defender_pool),
                    "alive": True,
                    "last_action": None,
                    "defuse_timer": 0,
                })
            self.detonate_timer = self.detonate_limit
            self.guard_end_reason = None

        if self.phase != "carry":
            self.dist_map = bfs_distances(self.goal_pos, self.grid)
        self.prev_dist = self.dist_map[self.player_pos[0]][self.player_pos[1]]

        return self._get_obs(), {}

    # -----------------------------------------------------------------
    # 観測ベクトル構築
    # 内訳: base(4) + walls(4) + last_onehot(5) + dists(4) + phase_flags(2)
    #       + enemies(K_ENEMIES*5=15) + ally(3) + defuse_info(2) = 39
    # -----------------------------------------------------------------
    def _get_obs(self):
        pr, pc = self.player_pos
        gr, gc = self.goal_pos
        base = [pr / (self.height - 1), pc / (self.width - 1),
                gr / (self.height - 1), gc / (self.width - 1)]

        walls = [0.0 if self._is_walkable(pr + dr, pc + dc) else 1.0
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

        last_onehot = [0.0] * 5
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

        # --- 敵情報(guardフェーズのみ意味を持つ。それ以外は全て0埋め) ---
        visible_ids = set()
        if self.phase == "guard":
            observer_positions = [tuple(self.player_pos)]
            if self.teammate_alive:
                observer_positions.append(tuple(self.teammate_pos))
            enemies = [(d["id"], d["pos"], d["alive"]) for d in self.defenders]
            visible_ids = self.enemy_memory.update(observer_positions, enemies, self.grid)
        enemy_feats = self.enemy_memory.build_features((pr, pc), self.height, self.width, visible_ids)

        # --- 味方(guardの同僚 / carryの護衛のうち近い方)---
        ally = [0.0, 0.0, 0.0]
        if self.phase == "guard" and self.teammate_alive:
            tr, tc = self.teammate_pos
            ally = [1.0, (tr - pr) / self.height, (tc - pc) / self.width]
        elif self.phase == "carry" and self.escort_positions:
            nearest = min(self.escort_positions.values(),
                          key=lambda p: max(abs(p[0] - pr), abs(p[1] - pc)))
            ally = [1.0, (nearest[0] - pr) / self.height, (nearest[1] - pc) / self.width]

        # --- 解除進行フラグ(guardフェーズのみ) ---
        defuse_info = [0.0, 0.0]
        if self.phase == "guard":
            max_timer = max((d["defuse_timer"] for d in self.defenders), default=0)
            if max_timer > 0:
                defuse_info = [1.0, min(max_timer / DEFUSE_REQUIRED, 1.0)]

        return np.array(base + walls + last_onehot + dists + phase_flags
                         + enemy_feats + ally + defuse_info, dtype=np.float32)

    # -----------------------------------------------------------------
    def step(self, action):
        self.current_step += 1
        reward = 0.0
        terminated = False
        self.last_action = action

        if self.phase == "retrieve":
            if action == 4:
                reward = -1.0
            else:
                reward, terminated = self._step_move(action, arrival_reward=100.0)

        elif self.phase == "carry":
            pr, pc = self.player_pos
            if action == 4:
                if self.grid[pr, pc] == 2:   # 💡サイト内であれば設置成立
                    dist_from_ideal = self.dist_map[pr][pc]
                    reward = 200.0 - min(dist_from_ideal * 5.0, 100.0)
                    terminated = True
                    self.carry_end_reason = "planted"
                    self.carry_arrival_step = self.current_step
                else:
                    reward = -1.0   # サイト外でaction=4は無駄行動なので引き続きペナルティ
            else:
                reward, _ = self._step_move(action, arrival_reward=None)
            self._move_escorts()

        else:  # guard
            reward, terminated = self._step_guard(action)

        truncated = self.current_step >= self.max_steps
        if self.phase == "carry" and truncated and self.carry_end_reason is None:
            self.carry_end_reason = "timeout"   # 💡追加: 時間内にプラントできなかった
        return self._get_obs(), reward, terminated, truncated, {}

    def _move_escorts(self):
        """護衛2体を固定方策で1歩動かす(渋滞回避付き、front/back区別なし)"""
        occupied = [tuple(self.player_pos)] + list(self.escort_positions.values())
        for i, role in enumerate(("front", "back")):  # キー名は維持、動きだけ統一
            pos = self.escort_positions[role]
            offset_steps = max(2, ESCORT_OFFSET_MAX - i * 3)
            target = self.escort_ctrl.compute_target(pos, self.dist_map, self.grid, "front", steps_override=offset_steps)
            next_pos = self.escort_ctrl.next_move(pos, target, self.grid,
                                                    [p for p in occupied if p != tuple(pos)])
            self.escort_positions[role] = tuple(next_pos)

    def _step_guard(self, action):
        pr, pc = self.player_pos
        reward = 0.0
        terminated = False
        self.guard_end_reason = None   # 💡追加: このtickで決着した理由

        if action == 4:
            reward = -0.1
        else:
            reward, _ = self._step_move(action, arrival_reward=None, guard_mode=True)

        # teammate self-play(従来通り)
        if self.teammate_alive and self.policy_net is not None:
            tobs = self._get_teammate_obs()
            with torch.no_grad():
                t_action = self.policy_net(torch.tensor(tobs, dtype=torch.float32).unsqueeze(0)).argmax(dim=1).item()
            self._move_teammate(t_action)

        # --- defender(複数)の行動 ---
        any_defusing = False
        for d in self.defenders:
            if not d["alive"]:
                continue
            dr_, dc_ = d["pos"]
            dist_to_spike = max(abs(dr_ - self.goal_pos[0]), abs(dc_ - self.goal_pos[1]))
            if dist_to_spike <= 1:
                d["defuse_timer"] += 1
                any_defusing = True
                if d["defuse_timer"] >= DEFUSE_REQUIRED:
                    reward -= 100.0
                    terminated = True
                    self.guard_end_reason = "defused"   # 💡追加
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

        # --- 視認判定・撃破処理 ---
        observer_positions = [tuple(self.player_pos)]
        if self.teammate_alive:
            observer_positions.append(tuple(self.teammate_pos))

        visible_defenders = [
            d for d in self.defenders
            if d["alive"] and any(has_line_of_sight(op, d["pos"], self.grid) for op in observer_positions)
        ]

        if visible_defenders:
            n_visible = len(visible_defenders)
            for d in visible_defenders:
                progress = d["defuse_timer"] / DEFUSE_REQUIRED
                if random.random() < 0.5:
                    d["alive"] = False
                    reward += (100.0 + SPOT_DURING_DEFUSE_BONUS * (1.0 - progress) * (1.0 if progress > 0 else 0.0)) / n_visible
                else:
                    reward -= 30.0 / n_visible

        # --- スパイク隣接8マスのLOSカバー率報酬 ---
        coverage = self._spike_neighbor_coverage(self.player_pos)
        reward += coverage * COVERAGE_REWARD_SCALE

        # --- 見えない状態で解除が進んでいる場合のペナルティ ---
        if any_defusing:
            visible_defuser = any(
                has_line_of_sight(tuple(self.player_pos), d["pos"], self.grid)
                for d in self.defenders if d["alive"] and d["defuse_timer"] > 0
            )
            if not visible_defuser:
                max_progress = max((d["defuse_timer"] / DEFUSE_REQUIRED for d in self.defenders if d["alive"]), default=0.0)
                reward -= DEFUSE_BLIND_PENALTY_SCALE * max_progress

        if not any(d["alive"] for d in self.defenders) and not terminated:
            reward += 50.0
            terminated = True
            self.guard_end_reason = "annihilated"   # 💡追加: defender全滅(attacker 勝利(
        elif not terminated:
            self.detonate_timer -= 1
            if self.detonate_timer <= 0:
                reward += 50.0
                terminated = True
                self.guard_end_reason = "survived_timeout"   # 💡追加: 時間切れ守り切り

        return reward, terminated

    def _spike_neighbor_coverage(self, from_pos):
        gr, gc = self.goal_pos
        neighbors = [(gr + dr, gc + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]
        walkable = [n for n in neighbors if self._is_walkable(*n)]
        if not walkable:
            return 0.0
        visible = sum(1 for n in walkable if has_line_of_sight(tuple(from_pos), n, self.grid))
        return visible / len(walkable)

    def _get_teammate_obs(self):
        pr, pc = self.teammate_pos
        gr, gc = self.goal_pos
        base = [pr / (self.height - 1), pc / (self.width - 1),
                gr / (self.height - 1), gc / (self.width - 1)]
        walls = [0.0 if self._is_walkable(pr + dr, pc + dc) else 1.0
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        last_onehot = [0.0] * 5
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
        enemy_feats = [0.0] * (K_ENEMIES * 5)  # teammate視点の簡略化(必要なら独自トラッカーを持たせる)
        sr, sc = self.player_pos
        ally = [1.0, (sr - pr) / self.height, (sc - pc) / self.width]
        defuse_info = [0.0, 0.0]
        return np.array(base + walls + last_onehot + dists + phase_flags
                         + enemy_feats + ally + defuse_info, dtype=np.float32)

    def _move_teammate(self, action):
        if action == 4:
            return
        moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        r, c = self.teammate_pos
        nr, nc = r + moves[action][0], c + moves[action][1]
        if self._is_walkable(nr, nc):
            self.teammate_pos = (nr, nc)

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
        # near_goal の判定から arrival_reward is not None の条件を外す。
        #    carry/guardでも目的地手前では後戻りペナルティを免除する。
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


def train():
    writer = SummaryWriter(log_dir="logs")
    SAVE_DIR = "attacker_multi_data"
    os.makedirs(SAVE_DIR, exist_ok=True)

    lines = [line.strip() for line in NEW_MAZE_STR.strip("\n").split("\n") if line.strip()]
    fixed_grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
    plant_rows, plant_cols = np.where(fixed_grid == 2)
    plant_candidates = list(zip(plant_rows, plant_cols))
    if not plant_candidates:
        raise ValueError("プラントサイト(2)が定義されていません。")

    env = AttackerMultiEnv(fixed_grid, plant_candidates)

    batch_size = 64
    gamma = 0.99
    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 0.9985
    lr = 0.0005
    IMPROVEMENT_MARGIN = 5.0
    EVAL_EPISODES_PER_PHASE = 20 # フェーズごとの固定eval試行回数

    device = torch.device("cpu")
    q_net = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(device)
    target_net = DuelingQNetwork(OBS_DIM, N_ACTIONS).to(device)
    target_net.load_state_dict(q_net.state_dict())
    env.policy_net = target_net

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=30000)
    epsilon = epsilon_start
    best_eval_reward = -float('inf')

    print(f"学習を開始します。デバイス: {device} | 入力次元: {OBS_DIM}")
    print("python -m tensorboard.main --logdir=logs")

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        episode_reward = 0.0
        losses = []

        while True:
            if random.random() < epsilon:
                action = random.randint(0, N_ACTIONS - 1)
            else:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = q_net(obs_t).squeeze(0).cpu().numpy().copy()
                    if env.phase == "retrieve":
                        q_values[4] = -np.inf
                    elif env.phase == "carry" and env.grid[env.player_pos[0], env.player_pos[1]] != 2:
                        q_values[4] = -np.inf
                    action = int(np.argmax(q_values))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            replay_buffer.push(obs, action, reward, next_obs, terminated)
            obs = next_obs
            episode_reward += reward

            if len(replay_buffer) >= batch_size:
                b_obs, b_act, b_rew, b_nobs, b_term = replay_buffer.sample(batch_size)
                b_obs_t = torch.tensor(b_obs, dtype=torch.float32, device=device)
                b_act_t = torch.tensor(b_act, dtype=torch.long, device=device).unsqueeze(1)
                b_rew_t = torch.tensor(b_rew, dtype=torch.float32, device=device).unsqueeze(1)
                b_nobs_t = torch.tensor(b_nobs, dtype=torch.float32, device=device)
                b_term_t = torch.tensor(b_term, dtype=torch.float32, device=device).unsqueeze(1)

                current_q = q_net(b_obs_t).gather(1, b_act_t)
                with torch.no_grad():
                    next_actions = q_net(b_nobs_t).argmax(dim=1, keepdim=True)
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

        # ========= エピソード終了時のログと定期保存 =========
        if (episode + 1) % 50 == 0:
            avg_loss = np.mean(losses) if losses else 0.0
            print(f"Episode {episode+1}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f}")

            # 💡 フェーズ別evalログ(softmax評価+carry強制PLANT対応、固定回数で評価しブレを抑える)
            phase_rewards = {"retrieve": [], "carry": [], "guard": []}
            guard_end_reasons = {"defused": 0, "annihilated": 0, "survived_timeout": 0, "truncated": 0}
            carry_end_reasons = {"planted": 0, "timeout": 0}      # 💡追加
            carry_arrival_steps = []                               # 💡追加: 到達できた場合のtick数を集める

            for phase_name in ["retrieve", "carry", "guard"]:
                for _ in range(EVAL_EPISODES_PER_PHASE):
                    eval_obs, _ = env.reset(phase=phase_name)
                    phase = env.phase
                    eval_reward = 0.0
                    while True:
                        with torch.no_grad():
                            q_values = q_net(torch.tensor(eval_obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
                        q_values = q_values.copy()
                        # 💡action=4(PLANT/DEFUSE)を無効化する条件:
                        #    retrieve: 常に無効
                        #    carry: サイト内にいなければ無効(サイト内ならモデルの判断に委ねる)
                        #    guard: 常に有効
                        if phase == "retrieve":
                            q_values[4] = -np.inf
                        elif phase == "carry" and env.grid[env.player_pos[0], env.player_pos[1]] != 2:
                            q_values[4] = -np.inf
                        probs = np.exp((q_values - np.max(q_values)) / 0.5)
                        probs = probs / probs.sum()
                        a = np.random.choice(len(probs), p=probs)

                        eval_obs, r, term, trunc, _ = env.step(a)
                        eval_reward += r
                        if term or trunc:
                            if phase == "guard":
                                reason = env.guard_end_reason if term else "truncated"
                                guard_end_reasons[reason] += 1
                            if phase == "carry":                                          # 💡追加
                                reason = env.carry_end_reason if env.carry_end_reason else "timeout"
                                carry_end_reasons[reason] += 1
                                if reason == "planted" and env.carry_arrival_step is not None:
                                    carry_arrival_steps.append(env.carry_arrival_step)
                            break
                    phase_rewards[phase].append(eval_reward)

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
                if carry_arrival_steps:
                    writer.add_scalar("Eval/carry_avg_arrival_step", avg_arrival, episode)

            # 💡追加: guardの決着理由の内訳を表示
            guard_total = sum(guard_end_reasons.values())
            if guard_total > 0:
                breakdown_str = " / ".join(f"{k}={v}" for k, v in guard_end_reasons.items())
                print(f"   [guard breakdown] {breakdown_str} (total={guard_total})")
                for k, v in guard_end_reasons.items():
                    writer.add_scalar(f"Eval/guard_{k}_rate", v / guard_total, episode)

            # 加重平均(guardを重視)でbest model判定
            guard_mean = np.mean(phase_rewards["guard"]) if phase_rewards["guard"] else 0.0
            other_mean = np.mean(phase_rewards["retrieve"] + phase_rewards["carry"]) if (phase_rewards["retrieve"] + phase_rewards["carry"]) else 0.0
            mean_eval = 0.5 * guard_mean + 0.5 * other_mean

            writer.add_scalar("Eval/Weighted_Reward", mean_eval, episode)

            if mean_eval > best_eval_reward + IMPROVEMENT_MARGIN:
                best_eval_reward = mean_eval
                best_path = os.path.join(SAVE_DIR, "dqn_attacker_multi_best_by_eval.pt")
                torch.save(q_net.state_dict(), best_path)
                print(f"   [Eval Best] 保存しました (Eval Reward: {mean_eval:.2f}): {best_path}")

        # 指定エピソードごとにモデルを保存
        if (episode + 1) % SAVE_INTERVAL == 0:
            save_path = os.path.join(SAVE_DIR, f"dqn_attacker_multi_ep{episode+1}.pt")
            torch.save(q_net.state_dict(), save_path)
            print(f"   [Save] 定期保存しました: {save_path}")
        # ==========================================

    final_path = os.path.join(SAVE_DIR, "dqn_attacker_multi_final.pt")
    torch.save(q_net.state_dict(), final_path)
    print(f"学習が完了しました。最終モデル: {final_path}")
    writer.close()


if __name__ == "__main__":
    train()