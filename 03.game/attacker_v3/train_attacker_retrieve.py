"""train_attacker_retrieve.py

Retrieve フェーズ（落下スパイク回収）用アタッカーAI学習スクリプト。
完全に自己完結。他のfeatureモジュール(controllers.py, battle_logic.py等)は
importせず、必要なロジックはこのファイル内に持つ。run_game.py / controllers.py
は変更しない。
"""

import random
import sys
import math
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from map_data import NEW_MAZE_STR

from game_core import (
    MAX_HP,
    BODY_DAMAGE,
    HEADSHOT_DAMAGE,
    MOVING_ACCURACY,
    MOVING_TARGET_HIT_MULTIPLIER,
    BLIND_ACCURACY_MULTIPLIER,
    REVEALED_DODGE_MULTIPLIER,
    BLIND_DURATION_TICKS,
    REVEAL_DURATION_TICKS,
)

EPISODE_COUNT = 9000

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")

MAX_TICKS = 70
CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTIONS = ["MOVE", "ABILITY"]
N_ACTIONS = len(ACTIONS)
ROLES = ["FLASH", "SMOKE", "RECON"]

# 敵スタブ(学習用の簡易ディフェンダー)
ENEMY_SPAWN_PROB = 0.6          # そのエピソードで敵が出現するか
ENEMY_PRE_AFFECTED_PROB = 0.3   # 出現時、すでに味方が炙り出し済みという想定
ENEMY_ACCURACY = 0.55
ENEMY_DODGE = 0.15
ENEMY_HS_RATE = 0.25
ENEMY_REACTION_HIT_CHANCE = 0.5  # そのTickに敵が反撃してくる確率(簡易反応速度)

# 味方渋滞スタブ(狭い通路で先頭が塞ぐ状況を再現する)
ALLY_BLOCK_PROB = 0.35          # このエピソードで味方が経路上に立っている確率
ALLY_BLOCK_MIN_CLEAR_TICKS = 4  # 塞がれてから解除されるまでの最短tick数
ALLY_BLOCK_MAX_CLEAR_TICKS = 12 # 塞がれてから解除されるまでの最長tick数
STALL_TICKS_PENALTY = -0.05     # 同じマスに留まり続けている場合の追加ペナルティ
STALL_TICKS_THRESHOLD = 3       # 何tick同じ位置に留まったらペナルティを課すか

# 報酬
STEP_PENALTY = -0.02
GOAL_REWARD = 12.0
TIMEOUT_PENALTY = -4.0
DEATH_PENALTY = -8.0
ABILITY_GOOD_FIRE = 0.6          # 未使用の敵に初めて当てた
ABILITY_EMPTY_FIRE = -1.2        # 敵が見えないのに撃った(空撃ち)
ABILITY_WASTED_ON_AFFECTED = -0.8  # すでに炙り出されている敵に撃った(重複)
KILL_REWARD = 3.0
KILL_WHILE_DEBUFFED_BONUS = 1.5  # フラッシュ/リコン状態の敵を倒した追加ボーナス

GREEDY_EPISODE = 200   # 評価に使用する episode数

Transition = namedtuple("Transition", ["s", "a", "r", "s2", "done", "mask2"])


# ---------------------------------------------------------------------------
# マップ読み込み・BFS距離
# ---------------------------------------------------------------------------
def load_grid():
    lines = [l.strip() for l in NEW_MAZE_STR.strip("\n").split("\n") if l.strip()]
    return np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)


GRID = load_grid()
HEIGHT, WIDTH = GRID.shape
WALKABLE = [
    (r, c)
    for r in range(HEIGHT)
    for c in range(WIDTH)
    if GRID[r, c] != 1
]


def bfs_distance_map(goal):
    """goalから各セルへの最短距離マップ(壁越え不可)。"""
    dist = np.full((HEIGHT, WIDTH), -1, dtype=np.int32)
    dist[goal[0], goal[1]] = 0
    queue = deque([goal])
    while queue:
        r, c = queue.popleft()
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            if 0 <= nr < HEIGHT and 0 <= nc < WIDTH and GRID[nr, nc] != 1 and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def line_cells(p1, p2):
    y0, x0 = p1
    y1, x1 = p2
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


def has_los(p1, p2):
    for r, c in line_cells(p1, p2):
        if GRID[r, c] == 1:
            return False
    return True


# ---------------------------------------------------------------------------
# 環境
# ---------------------------------------------------------------------------
class RetrieveEnv:
    """1体のリトリーブ役アタッカー + 0〜1体の簡易敵スタブ + 味方渋滞スタブ。"""

    OBS_DIM = 20

    def reset(self):
        self.tick = 0
        self.role = random.choice(ROLES)
        self.charge = 1  # このラウンドのアビリティ残数

        self.spike_pos = random.choice(WALKABLE)
        self.dist_map = bfs_distance_map(self.spike_pos)

        # スポーン地点はspikeから到達可能な場所からランダムに選ぶ
        reachable = [p for p in WALKABLE if self.dist_map[p] > 0]
        self.pos = list(random.choice(reachable)) if reachable else list(self.spike_pos)
        start_dist = self.dist_map[tuple(self.pos)]

        self.hp = MAX_HP
        self.alive = True
        self.ability_used_this_episode = False

        # 敵スタブ
        self.enemy_alive = random.random() < ENEMY_SPAWN_PROB
        self.enemy_pos = None
        self.enemy_hp = MAX_HP
        self.enemy_blind = 0
        self.enemy_reveal = 0
        if self.enemy_alive:
            candidates = [p for p in WALKABLE if p != tuple(self.pos) and p != self.spike_pos]
            self.enemy_pos = list(random.choice(candidates))
            if random.random() < ENEMY_PRE_AFFECTED_PROB:
                if random.random() < 0.5:
                    self.enemy_blind = BLIND_DURATION_TICKS
                else:
                    self.enemy_reveal = REVEAL_DURATION_TICKS

        # 味方渋滞スタブ: 経路上(自分より少しゴールに近いセル)に味方が立っている
        # 想定を作る。一定tick後に自動で解除する(先頭が動いて道が空く想定)。
        self.ally_blocked_cell = None
        self.ally_block_clear_tick = None
        if random.random() < ALLY_BLOCK_PROB and start_dist > 1:
            path_candidates = [
                p for p in WALKABLE
                if p != tuple(self.pos)
                and p != self.spike_pos
                and 0 <= self.dist_map[p] < start_dist
            ]
            if path_candidates:
                self.ally_blocked_cell = random.choice(path_candidates)
                self.ally_block_clear_tick = self.tick + random.randint(
                    ALLY_BLOCK_MIN_CLEAR_TICKS, ALLY_BLOCK_MAX_CLEAR_TICKS
                )

        self.stall_ticks = 0

        return self._obs(), self._action_mask()

    # -- 観測 --------------------------------------------------------------
    def _visible_enemy(self):
        if not self.enemy_alive:
            return False
        return has_los(tuple(self.pos), tuple(self.enemy_pos))

    def _is_ally_blocked(self, cell):
        return self.ally_blocked_cell is not None and cell == self.ally_blocked_cell

    def _obs(self):
        r, c = self.pos
        wall_up = 1.0 if GRID[r - 1, c] == 1 else 0.0
        wall_down = 1.0 if GRID[r + 1, c] == 1 else 0.0
        wall_left = 1.0 if GRID[r, c - 1] == 1 else 0.0
        wall_right = 1.0 if GRID[r, c + 1] == 1 else 0.0

        # 隣接4マスの「実際のBFS距離」を渡す。
        # 壁 または 味方に塞がれているセルは最悪値(1.0)扱いにして避けさせる。
        max_dist_scale = float(HEIGHT + WIDTH)
        neighbor_dists = []
        for (dr_, dc_), is_wall in zip(
            CARDINAL, [wall_up, wall_down, wall_left, wall_right]
        ):
            nr, nc = r + dr_, c + dc_
            blocked = is_wall or not (0 <= nr < HEIGHT and 0 <= nc < WIDTH)
            if not blocked and self._is_ally_blocked((nr, nc)):
                blocked = True
            if blocked:
                neighbor_dists.append(1.0)
            else:
                neighbor_dists.append(min(1.0, self.dist_map[nr, nc] / max_dist_scale))

        dist_norm = min(1.0, self.dist_map[r, c] / max_dist_scale)
        role_onehot = [1.0 if self.role == role else 0.0 for role in ROLES]

        visible = self._visible_enemy()
        if visible:
            er, ec = self.enemy_pos
            edr = np.clip((er - r) / HEIGHT, -1, 1)
            edc = np.clip((ec - c) / WIDTH, -1, 1)
            e_present = 1.0
            e_blind = 1.0 if self.enemy_blind > 0 else 0.0
            e_reveal = 1.0 if self.enemy_reveal > 0 else 0.0
        else:
            edr = edc = 0.0
            e_present = e_blind = e_reveal = 0.0

        obs = [
            r / HEIGHT, c / WIDTH,
            wall_up, wall_down, wall_left, wall_right,
            *neighbor_dists,
            dist_norm,
            *role_onehot,
            float(self.charge),
            e_present, edr, edc, e_blind, e_reveal,
        ]
        return np.array(obs, dtype=np.float32)

    def _action_mask(self):
        """チャージ0でのABILITYのみ禁止する。移動は常にBFS最短距離で自動移動するためマスク不要。"""
        mask = [True] * N_ACTIONS
        if self.charge <= 0:
            mask[1] = False
        return np.array(mask, dtype=bool)

    def _bfs_move_to_spike(self):
        """スパイクの位置へBFS距離で1マス進む(常に距離が最小になる隣接セルへ移動)。"""
        r, c = self.pos
        best_cell = None
        best_dist = self.dist_map[r, c]
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < HEIGHT and 0 <= nc < WIDTH) or GRID[nr, nc] == 1:
                continue
            d = self.dist_map[nr, nc]
            if d != -1 and d < best_dist:
                best_dist = d
                best_cell = (nr, nc)
        if best_cell is None:
            return False
        self.pos = [best_cell[0], best_cell[1]]
        return True

    # -- ステップ ------------------------------------------------------------
    def step(self, action):
        self.tick += 1
        reward = STEP_PENALTY
        done = False
        info = {}

        # 味方渋滞の自動解除(先頭が動いて道が空く想定)
        if (
            self.ally_blocked_cell is not None
            and self.ally_block_clear_tick is not None
            and self.tick >= self.ally_block_clear_tick
        ):
            self.ally_blocked_cell = None

        moved = False
        if action == 0:
            old_dist = self.dist_map[self.pos[0], self.pos[1]]
            moved = self._bfs_move_to_spike()
            if moved:
                new_dist = self.dist_map[self.pos[0], self.pos[1]]
                reward += 0.3 * (old_dist - new_dist)
        elif action == 1:
            reward += self._resolve_ability()

        # 足踏み(同じマスに留まり続ける)ペナルティ。
        # 迂回路を探させる/待つべき時と無駄な足踏みを区別するための緩い誘導。
        if moved:
            self.stall_ticks = 0
        else:
            self.stall_ticks += 1
            if self.stall_ticks > STALL_TICKS_THRESHOLD:
                reward += STALL_TICKS_PENALTY

        # スパイク到達判定
        if tuple(self.pos) == self.spike_pos:
            reward += GOAL_REWARD
            done = True
            info["result"] = "reached"

        # 敵の反撃(簡易) — LOSが通っていれば毎tick必ず撃ち合いが発生する
        if not done and self._visible_enemy():
            reward += self._resolve_enemy_shot()
            if self.hp <= 0:
                reward += DEATH_PENALTY
                done = True
                info["result"] = "died"

        if not done and self.tick >= MAX_TICKS:
            reward += TIMEOUT_PENALTY
            done = True
            info["result"] = "timeout"

        # 効果減衰
        self.enemy_blind = max(0, self.enemy_blind - 1)
        self.enemy_reveal = max(0, self.enemy_reveal - 1)

        return self._obs(), reward, done, self._action_mask(), info

    def _resolve_ability(self):
        if self.charge <= 0:
            return 0.0
        self.charge -= 1
        self.ability_used_this_episode = True

        if not self._visible_enemy():
            return ABILITY_EMPTY_FIRE

        already_affected = self.enemy_blind > 0 or self.enemy_reveal > 0
        if already_affected:
            return ABILITY_WASTED_ON_AFFECTED

        # 新規にデバフを付与
        if self.role == "FLASH":
            self.enemy_blind = BLIND_DURATION_TICKS
        elif self.role == "RECON":
            self.enemy_reveal = REVEAL_DURATION_TICKS
        else:  # SMOKE: 攻撃的効果はないが射線を切る想定 -> 敵の反撃率を下げる扱い
            self.enemy_blind = max(self.enemy_blind, 1)
        return ABILITY_GOOD_FIRE

    def _resolve_enemy_shot(self):
        """自分が敵を撃つ側(反撃合戦)を簡易シミュレート。デバフ中の敵は倒しやすい。"""
        accuracy = MOVING_ACCURACY
        hit_chance = accuracy * (1.0 - ENEMY_DODGE)

        # 敵がデバフ中なら、こちらの命中率を有利に補正(簡易表現)
        debuffed = self.enemy_blind > 0 or self.enemy_reveal > 0
        if debuffed:
            hit_chance = min(1.0, hit_chance * 1.4)

        killed_enemy = False
        if random.random() < hit_chance:
            dmg = HEADSHOT_DAMAGE if random.random() < 0.3 else BODY_DAMAGE
            self.enemy_hp -= dmg
            if self.enemy_hp <= 0:
                self.enemy_alive = False
                killed_enemy = True

        reward = 0.0
        if killed_enemy:
            reward += KILL_REWARD
            if debuffed:
                reward += KILL_WHILE_DEBUFFED_BONUS

        # 敵からの反撃
        enemy_hit_chance = ENEMY_ACCURACY * (1.0 - REVEALED_DODGE_MULTIPLIER if debuffed else ENEMY_ACCURACY)
        if debuffed:
            enemy_hit_chance *= BLIND_ACCURACY_MULTIPLIER if self.enemy_blind > 0 else 1.0
        if self.enemy_alive and random.random() < enemy_hit_chance:
            dmg = HEADSHOT_DAMAGE if random.random() < ENEMY_HS_RATE else BODY_DAMAGE
            self.hp -= dmg

        return reward


# ---------------------------------------------------------------------------
# Dueling DQN
# ---------------------------------------------------------------------------
class DuelingQNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, n_actions))

    def forward(self, x):
        feat = self.feature(x)
        v = self.value(feat)
        a = self.advantage(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


def masked_argmax(q_values, mask):
    q = q_values.clone()
    q[~mask] = -1e9
    return int(torch.argmax(q).item())


# ---------------------------------------------------------------------------
# 学習ループ
# ---------------------------------------------------------------------------
def train(
    episodes=EPISODE_COUNT,
    batch_size=128,
    gamma=0.97,
    lr=3e-4,
    buffer_size=50000,
    target_update_every=1000,
    eps_start=1.0,
    eps_end=0.10,
    eps_decay_episodes=int(EPISODE_COUNT * 0.8),
    warmup_steps=2000,
    save_path="data/attacker_retrieve_data/dqn_attacker_retrieve_best_by_eval.pt",
):
    env = RetrieveEnv()
    policy_net = DuelingQNet(RetrieveEnv.OBS_DIM, N_ACTIONS).to(DEVICE)
    target_net = DuelingQNet(RetrieveEnv.OBS_DIM, N_ACTIONS).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    replay = deque(maxlen=buffer_size)
    recent_rewards = deque(maxlen=200)
    step_count = 0
    best_eval_reward = float("-inf")

    for ep in range(episodes):
        obs, mask = env.reset()
        done = False
        ep_reward = 0.0
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / eps_decay_episodes)

        while not done:
            state_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
            mask_t = torch.from_numpy(mask).to(DEVICE)

            if random.random() < eps:
                valid_actions = np.where(mask)[0]
                action = int(random.choice(valid_actions))
            else:
                with torch.no_grad():
                    q_values = policy_net(state_t).squeeze(0)
                    action = masked_argmax(q_values, mask_t)

            next_obs, reward, done, next_mask, info = env.step(action)
            replay.append(Transition(obs, action, reward, next_obs, done, next_mask))
            obs, mask = next_obs, next_mask
            ep_reward += reward
            step_count += 1

            if len(replay) >= batch_size:
                batch = random.sample(replay, batch_size)
                s = torch.from_numpy(np.stack([t.s for t in batch])).float().to(DEVICE)
                a = torch.tensor([t.a for t in batch], device=DEVICE).unsqueeze(1)
                r = torch.tensor([t.r for t in batch], device=DEVICE, dtype=torch.float32).unsqueeze(1)
                s2 = torch.from_numpy(np.stack([t.s2 for t in batch])).float().to(DEVICE)
                d = torch.tensor([t.done for t in batch], device=DEVICE, dtype=torch.float32).unsqueeze(1)
                mask2 = torch.from_numpy(np.stack([t.mask2 for t in batch])).to(DEVICE)

                q_sa = policy_net(s).gather(1, a)

                with torch.no_grad():
                    next_q_policy = policy_net(s2)
                    next_q_policy_masked = next_q_policy.masked_fill(~mask2, -1e9)
                    next_actions = next_q_policy_masked.argmax(dim=1, keepdim=True)
                    next_q_target = target_net(s2).gather(1, next_actions)
                    y = r + gamma * (1 - d) * next_q_target

                loss = nn.functional.smooth_l1_loss(q_sa, y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

            if step_count % target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

        recent_rewards.append(ep_reward)
        if ep % GREEDY_EPISODE == 0:
            eval_reward, success_rate, death_rate = evaluate_greedy(policy_net, episodes=GREEDY_EPISODE)
            print(f"[EP {ep}/{EPISODE_COUNT}] eps={eps:.3f} "
                f"eval_reward={eval_reward:.3f} success={success_rate:.2%} death={death_rate:.2%}")
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(policy_net.state_dict(), save_path)
                print(f"  -> best model saved (eval_reward={eval_reward:.3f})")

    torch.save(policy_net.state_dict(), save_path.replace("best", "final"))
    print("Training complete.")


def evaluate_greedy(policy_net, episodes=GREEDY_EPISODE):
    """探索なし(eps=0)でN episode実行し、成功率・死亡率・平均報酬を計測する。"""
    env = RetrieveEnv()
    total_reward = 0.0
    reached = 0
    died = 0
    for _ in range(episodes):
        obs, mask = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            state_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
            mask_t = torch.from_numpy(mask).to(DEVICE)
            with torch.no_grad():
                q_values = policy_net(state_t).squeeze(0)
                action = masked_argmax(q_values, mask_t)
            obs, reward, done, mask, info = env.step(action)
            ep_reward += reward
        total_reward += ep_reward
        if info.get("result") == "reached":
            reached += 1
        elif info.get("result") == "died":
            died += 1
    n = episodes
    return total_reward / n, reached / n, died / n


if __name__ == "__main__":
    train()