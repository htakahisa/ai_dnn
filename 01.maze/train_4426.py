import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import tkinter as tk
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter


class GridWorldEnv(gym.Env):
    def __init__(self, wall_ratio=0.10):
        super(GridWorldEnv, self).__init__()
        self.width, self.height = 44, 26
        self.action_space = spaces.Discrete(4)
        # 観測: [player_r, player_c, goal_r, goal_c, up, down, left, right, last_action(4次元one-hot), neighbor_dist(4次元)] (16次元, 正規化済み)
        self.obs_dim = 16
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.max_steps = 300
        self.wall_ratio = wall_ratio
        self.player_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array([self.height - 1, self.width - 1], dtype=np.int32)
        self.last_action = None
        self._generate_fixed_maze()

    def _generate_fixed_maze(self):
        while True:
            self.grid = np.zeros((self.height, self.width), dtype=np.int32)
            for r in range(self.height):
                for c in range(self.width):
                    if (r == 0 and c == 0) or (r == self.height - 1 and c == self.width - 1):
                        continue
                    if np.random.rand() < self.wall_ratio:
                        self.grid[r, c] = 1
            if self._is_reachable(self.player_pos, self.goal_pos):
                break

    def _is_reachable(self, start, goal):
        queue = [tuple(start)]; visited = {tuple(start)}
        while queue:
            curr = queue.pop(0)
            if curr == tuple(goal): return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if self.grid[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc)); queue.append((nr, nc))
        return False

    def _random_free_cell(self):
        while True:
            r = np.random.randint(0, self.height)
            c = np.random.randint(0, self.width)
            if self.grid[r, c] == 0:
                return np.array([r, c], dtype=np.int32)

    def _local_walls(self):
        # 上下左右が壁(または場外)なら1, 通行可なら0
        r, c = self.player_pos
        out = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width and self.grid[nr, nc] == 0:
                out.append(0.0)
            else:
                out.append(1.0)
        return out

    def _get_obs(self):
        pr, pc = self.player_pos
        gr, gc = self.goal_pos
        base = [pr / (self.height - 1), pc / (self.width - 1),
                gr / (self.height - 1), gc / (self.width - 1)]
        last_action_onehot = [0.0, 0.0, 0.0, 0.0]
        if self.last_action is not None:
            last_action_onehot[self.last_action] = 1.0
        return np.array(
            base + self._local_walls() + last_action_onehot + self._neighbor_distances(),
            dtype=np.float32
        )

    def _neighbor_distances(self):
        r, c = self.player_pos
        max_dist = self.height * self.width
        out = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width and self.grid[nr, nc] == 0:
                d = self.dist_map[nr, nc]
                out.append(d / max_dist if np.isfinite(d) else 1.0)
            else:
                out.append(1.0)  # 壁 or 場外は最大距離扱い
        return out

    def _bfs_distances_from_goal(self):
        dist = np.full((self.height, self.width), np.inf)
        gr, gc = self.goal_pos
        dist[gr, gc] = 0
        q = deque([(gr, gc)])
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.height and 0 <= nc < self.width and self.grid[nr, nc] == 0:
                    if dist[nr, nc] > dist[r, c] + 1:
                        dist[nr, nc] = dist[r, c] + 1
                        q.append((nr, nc))
        return dist

    def reset(self, seed=None, options=None):
        while True:
            self.player_pos = self._random_free_cell()
            self.goal_pos = self._random_free_cell()
            if not np.array_equal(self.player_pos, self.goal_pos) and \
               self._is_reachable(self.player_pos, self.goal_pos):
                break
        self.current_step = 0
        self.last_action = None
        self.dist_map = self._bfs_distances_from_goal()  # ゴール確定後に1回計算
        self.prev_dist = self.dist_map[self.player_pos[0], self.player_pos[1]]
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        next_pos = self.player_pos + moves[action]

        if 0 <= next_pos[0] < self.height and 0 <= next_pos[1] < self.width and self.grid[next_pos[0], next_pos[1]] == 0:
            self.player_pos = next_pos
            terminated = np.array_equal(self.player_pos, self.goal_pos)
            new_dist = self.dist_map[self.player_pos[0], self.player_pos[1]]
            # 距離が縮まったら+0.5, 遠ざかったら-0.5 のシェイピング報酬を追加
            shaping = (self.prev_dist - new_dist) * 0.5
            self.prev_dist = new_dist
            reward = 1000.0 if terminated else (-2.0 + shaping)
        else:
            reward = -20.0
            terminated = False

        self.last_action = action
        truncated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)


def run_visual_test(env, policy_net, device):
    root = tk.Tk()
    root.title("DQN Visualization")
    cell_size = 18
    canvas = tk.Canvas(root, width=env.width * cell_size, height=env.height * cell_size)
    canvas.pack()
    obs, _ = env.reset()
    done = False

    def update():
        nonlocal obs, done
        if not root.winfo_exists(): return
        if not done:
            with torch.no_grad():
                state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy_net(state_t).argmax(dim=1).item()
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
            canvas.delete("all")
            for r in range(env.height):
                for c in range(env.width):
                    x1, y1 = c * cell_size, r * cell_size
                    x2, y2 = x1 + cell_size, y1 + cell_size
                    color = "white"
                    if env.grid[r, c] == 1: color = "#34495e"
                    elif r == env.goal_pos[0] and c == env.goal_pos[1]: color = "#2ecc71"
                    elif (r, c) == (env.player_pos[0], env.player_pos[1]): color = "#e74c3c"
                    canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#eee")
            root.after(30, update)
        else:
            root.destroy()

    root.after(500, update)
    root.mainloop()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridWorldEnv()

    policy_net = QNetwork(env.obs_dim, 4).to(device)
    target_net = QNetwork(env.obs_dim, 4).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer()
    writer = SummaryWriter(log_dir="logs")

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.9995
    batch_size = 128
    target_update_freq = 500  # 学習ステップ単位
    min_buffer_size = 2000

    n_episodes = 20001
    global_step = 0

    print("学習を開始します。TensorBoardを確認してください。")
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done, total_reward = False, 0

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action = policy_net(state_t).argmax(dim=1).item()

            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            global_step += 1

            if len(replay_buffer) >= min_buffer_size and global_step % 4 == 0:
                s, a, r, s2, d = replay_buffer.sample(batch_size)
                s = torch.tensor(s, dtype=torch.float32, device=device)
                a = torch.tensor(a, dtype=torch.long, device=device).unsqueeze(1)
                r = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
                s2 = torch.tensor(s2, dtype=torch.float32, device=device)
                d = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)

                q_values = policy_net(s).gather(1, a)
                with torch.no_grad():
                    next_q = target_net(s2).max(dim=1, keepdim=True)[0]
                    target_q = r + gamma * next_q * (1 - d)

                loss = nn.functional.smooth_l1_loss(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        writer.add_scalar("Reward/episode", total_reward, episode)
        writer.add_scalar("Epsilon", epsilon, episode)

        if episode % 1000 == 0:
            print(f"Ep {episode} | Reward: {total_reward:.2f} | Eps: {epsilon:.2f}")
            run_visual_test(env, policy_net, device)

    writer.close()
    torch.save(policy_net.state_dict(), "dqn_gridworld.pt")