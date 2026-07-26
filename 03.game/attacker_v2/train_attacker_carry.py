# train_attacker_carry.py
"""
スパイクキャリアーAI(carryフェーズ専用)の学習スクリプト。
外部の学習系モジュール(train_defender_combined.py等)には依存せず、
必要なクラスはすべてこのファイル内に複製する。
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

# このファイルの1階層上をパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from map_data import NEW_MAZE_STR

OBS_DIM = 25
N_ACTIONS = 5   # 0:上 1:下 2:左 3:右 4:設置
NUM_EPISODES = 2000
SAVE_INTERVAL = 100
ABILITY_TYPES = ["flash", "smoke", "recon"]


# ===========================================================================
# BFS距離マップ
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


# 💡追加: grid==2のセル群を隣接関係(4方向)の連結成分ごとに分割する。
# 左サイト/右サイトのように離れた場所に複数のサイトが存在する場合、
# それぞれ別サイトとして区別するために必要。
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


# 💡追加: 元々はCarryOnlyEnvのメソッドだったが、learning_attacker_carry.py側でも
# 同じロジックが必要になったため、モジュールレベル関数として共通化する。
def multi_source_bfs(site_cells, grid):
    """サイト内の全マスを起点とした同時BFS。
    dist[r,c] = 最寄りサイトマスまでの距離
    label[r,c] = そのマスから見た最寄りサイトマスの座標(観測の目標点として利用)"""
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
# carry専用環境
# ===========================================================================
class CarryOnlyEnv:
    def __init__(self, fixed_grid, plant_candidates):
        self.grid = fixed_grid
        self.height, self.width = fixed_grid.shape
        self.plant_candidates = [tuple(p) for p in plant_candidates]
        self.max_steps = 150

        # 💡変更: サイトが複数(左/右)ある場合、全部まとめて1つのBFSにすると
        # 「一番近いサイト」に常に誘導されてしまい、ラウンドごとに指定されるべき
        # target_plant_pos(オレンジの目標)を無視する挙動になる。サイトごとに分離する。
        self.site_components = split_site_components(self.plant_candidates)
        self.site_maps = [multi_source_bfs(comp, self.grid) for comp in self.site_components]
        self.site_cell_sets = [set(comp) for comp in self.site_components]

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
        self.player_pos = self._random_walkable()

        # 💡変更: ラウンドごとにどちらのサイトを狙うかをランダムに選び、そのサイト専用の
        # dist_map/label_mapを使う(実ゲームのtarget_plant_posランダム割当を再現するため)
        self.current_site_index = random.randrange(len(self.site_components))
        self.dist_map, self.label_map = self.site_maps[self.current_site_index]
        self.site_cells = self.site_cell_sets[self.current_site_index]

        self.prev_dist = self.dist_map[self.player_pos[0]][self.player_pos[1]]

        self.own_ability_type = random.choice(ABILITY_TYPES)
        self.own_ability_charge = 1.0
        self.own_blind_remaining = 0

        self.pos_history = deque(maxlen=6)
        return self._get_obs(), {}

    def _get_obs(self):
        pr, pc = self.player_pos
        # 💡変更: 「目標位置」は現在地から見た最寄りのサイトマス(移動するたびに更新されうる)
        gr, gc = self.label_map[pr][pc]

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

        ability_onehot = [1.0 if self.own_ability_type == t else 0.0 for t in ABILITY_TYPES]
        ability_charge = [self.own_ability_charge]
        blind_flag = [1.0 if self.own_blind_remaining > 0 else 0.0]
        enemy = [0.0, 0.0, 0.0]

        return np.array(
            base + walls + last_onehot + dists + ability_onehot + ability_charge + blind_flag + enemy,
            dtype=np.float32
        )

    def get_action_mask(self):
        """移動: 壁でなければ1。設置(4): 現在地がこのラウンドで割り当てられたサイト内ならどこでも1。
        (別サイトのgrid==2マスにいる場合は設置不可扱いにする)"""
        pr, pc = self.player_pos
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        for a, (dr, dc) in moves.items():
            mask[a] = 1.0 if self._is_walkable(pr + dr, pc + dc) else 0.0
        # 💡変更: grid==2判定 → 割り当てサイトの成分に属しているかの判定に変更
        mask[4] = 1.0 if (pr, pc) in self.site_cells else 0.0
        return mask

    def step(self, action):
        self.current_step += 1
        self.last_action = action
        pr, pc = self.player_pos

        if action == 4:
            # 💡変更: 割り当てられたサイトの成分内であればどこでも設置成功
            if (pr, pc) in self.site_cells:
                reward, terminated = 200.0, True
            else:
                reward, terminated = -1.0, False
        else:
            reward, terminated = self._step_move(action)

        truncated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def _step_move(self, action):
        moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        r, c = self.player_pos
        nr, nc = r + moves[action][0], c + moves[action][1]

        if not self._is_walkable(nr, nc):
            return -1.5, False

        self.player_pos = (nr, nc)
        new_dist = self.dist_map[nr][nc]  # 💡最寄りサイトマスまでの距離(マルチソース)
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

# ===========================================================================
# マスクを考慮した行動選択ヘルパー
# ===========================================================================
def masked_argmax(q_values, mask):
    """maskが0の行動を-infにしてargmaxを取る(全滅マスク対策として全0なら素のargmax)"""
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

    SAVE_DIR = "data_temp/attacker_carry_data/"
    os.makedirs(SAVE_DIR, exist_ok=True)

    lines = [line.strip() for line in NEW_MAZE_STR.strip("\n").split("\n") if line.strip()]
    fixed_grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
    plant_rows, plant_cols = np.where(fixed_grid == 2)
    plant_candidates = list(zip(plant_rows, plant_cols))
    if not plant_candidates:
        raise ValueError("プラントサイト(2)が定義されていません。")

    env = CarryOnlyEnv(fixed_grid, plant_candidates)

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
    # 💡追加: 成功率・到達tick数トラッキング用
    success_window = deque(maxlen=100)
    ticks_window = deque(maxlen=100)  # 成功時のtickのみ格納

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
                    # 💡マスク: 次状態で不可能な行動をDouble DQNのargmax候補から除外
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

        # 💡追加: このenvではterminated=Trueは設置成功時のみ発生する
        # (交戦などの失敗要因が未実装のため、truncated=タイムアウト失敗と等価)
        success = terminated
        success_window.append(1.0 if success else 0.0)
        if success:
            ticks_window.append(env.current_step)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        writer.add_scalar("Train/Episode_Reward", episode_reward, episode)
        writer.add_scalar("Train/Success_Rate", np.mean(success_window), episode)
        if ticks_window:
            writer.add_scalar("Train/Ticks_To_Plant", np.mean(ticks_window), episode)

        if (episode + 1) % 50 == 0:
            avg_loss = np.mean(losses) if losses else 0.0
            print(f"Episode {episode+1}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f}")

            eval_rewards = []
            eval_success_count = 0
            eval_success_ticks = []  # 💡追加

            for _ in range(30):
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
                        if term:  # 💡追加: 成功時のみカウント
                            eval_success_count += 1
                            eval_success_ticks.append(env.current_step)
                        break
                eval_rewards.append(eval_reward)

            mean_eval = np.mean(eval_rewards)
            eval_success_rate = eval_success_count / 30
            writer.add_scalar("Eval/Carry_Reward", mean_eval, episode)
            writer.add_scalar("Eval/Success_Rate", eval_success_rate, episode)
            if eval_success_ticks:
                writer.add_scalar("Eval/Mean_Ticks_To_Plant", np.mean(eval_success_ticks), episode)

            print(f"   [Eval] mean={mean_eval:.2f} success_rate={eval_success_rate:.2%} "
                  f"avg_ticks={np.mean(eval_success_ticks) if eval_success_ticks else float('nan'):.1f} n=30")

            if mean_eval > best_eval_reward + IMPROVEMENT_MARGIN:
                best_eval_reward = mean_eval
                best_path = os.path.join(SAVE_DIR, "dqn_attacker_carry_best_by_eval.pt")
                torch.save(q_net.state_dict(), best_path)
                print(f"   [Eval Best] 保存しました (Eval Reward: {mean_eval:.2f}): {best_path}")

        if (episode + 1) % SAVE_INTERVAL == 0:
            save_path = os.path.join(SAVE_DIR, f"dqn_attacker_carry_ep{episode+1}.pt")
            torch.save(q_net.state_dict(), save_path)
            print(f"   [Save] 定期保存しました: {save_path}")

    final_path = os.path.join(SAVE_DIR, "dqn_attacker_carry_final.pt")
    torch.save(q_net.state_dict(), final_path)
    print(f"学習が完了しました。最終モデル: {final_path}")
    writer.close()


if __name__ == "__main__":
    train()