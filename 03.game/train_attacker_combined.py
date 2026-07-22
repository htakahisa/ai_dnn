# train_attacker_combined.py
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

OBS_DIM = 25
N_ACTIONS = 5  # 0:上 1:下 2:左 3:右 4:設置/見張り(局面依存)
NUM_EPISODES = 1000
SAVE_INTERVAL = 100

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


class DefenderBotAI:
    """学習環境内で使う、本物のdefender AIをラップしたクラス。
    LearningDefenderAllAIControllerのプラント後ロジックと同じ挙動を再現する。
    (観測の次元・スケールは train_defender_combined.py の CombinedGridWorldEnv と厳密に一致させる)"""

    def __init__(self, model_path="dqn_defender_combined_best.pt", obs_dim=20, n_actions=5):
        self.device = torch.device("cpu")
        model_path_obj = Path(model_path)
        full_path = model_path_obj if model_path_obj.is_absolute() else Path(__file__).resolve().parent / model_path_obj
        self.model = QNetwork(obs_dim, n_actions).to(self.device)
        self.model.load_state_dict(torch.load(str(full_path), map_location=self.device))
        self.model.eval()

    def _is_walkable(self, r, c, grid):
        h, w = grid.shape
        return 0 <= r < h and 0 <= c < w and grid[r, c] != 1

    def decide_action(self, bot_pos, goal_pos, dist_map, last_action, grid):
        br, bc = bot_pos
        gr, gc = goal_pos

        # 💡本物と同じ: 解除可能範囲(距離1以内)なら強制DEFUSE
        dist_to_spike = max(abs(br - gr), abs(bc - gc))
        if dist_to_spike <= 1:
            return 4

        height, width = grid.shape
        base = [br / (height - 1), bc / (width - 1), gr / (height - 1), gc / (width - 1)]
        walls = [0.0 if self._is_walkable(br + dr, bc + dc, grid) else 1.0
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        last_onehot = [1.0 if last_action == i else 0.0 for i in range(5)]

        # 💡注意: defenderモデルの学習時スケールに合わせる(attacker用のmax(h,w)*2とは異なる)
        max_dist = height * width
        dists = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = br + dr, bc + dc
            if self._is_walkable(nr, nc, grid):
                d = dist_map[nr][nc]
                dists.append(d / max_dist if np.isfinite(d) else 1.0)
            else:
                dists.append(1.0)

        # プラント後は常にスパイク位置が分かっている(本物の spotted_info と同じ扱い)
        enemy_info = [1.0, gr / (height - 1), gc / (width - 1)]

        obs = np.array(base + walls + last_onehot + dists + enemy_info, dtype=np.float32)
        with torch.no_grad():
            state_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.model(state_t).squeeze(0)
            probs = torch.softmax(q_values / 0.5, dim=-1).cpu().numpy()
        return int(np.random.choice(len(probs), p=probs))


class DuelingQNetwork(nn.Module):
    """Attacker用のDueling DQN構造。状態価値V(s)と行動優位性A(s,a)を分離することで、
    guardフェーズのように『多くの行動でほぼ同じ価値』の局面での学習を安定させる狙い。"""
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




class AttackerCombinedEnv:
    """1体のattackerが retrieve(拾得) / carry(運搬+設置) / guard(見張り) の
    3局面をランダムに経験して学習する、単一エージェント・単一モデルの統合環境"""

    def __init__(self, fixed_grid, plant_candidates):
        self.grid = fixed_grid
        self.height, self.width = fixed_grid.shape
        self.plant_candidates = [tuple(p) for p in plant_candidates]
        self.max_steps = 150
        self.detonate_limit = 45
        self.defuse_required = 6
        self.policy_net = None
        self.defender_bot = None

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
        self.phase = random.choices(["retrieve", "carry", "guard"], weights=[0.3, 0.4, 0.3])[0]

        self.carrying = False
        self.is_planted = False
        self.bot_alive = False
        self.bot_defuse_timer = 0
        self.detonate_timer = self.detonate_limit
        self.pos_history = deque(maxlen=6)

        if self.phase == "retrieve":
            self.player_pos = self._random_walkable()
            self.goal_pos = self._random_walkable()
            while self.goal_pos == self.player_pos:
                self.goal_pos = self._random_walkable()

        elif self.phase == "carry":
            self.player_pos = self._random_walkable()
            self.carrying = True
            self.goal_pos = random.choice(self.plant_candidates)

        else:  # guard
            self.is_planted = True
            self.goal_pos = random.choice(self.plant_candidates)  # スパイク位置
            candidates = [
                (r, c) for r in range(self.height) for c in range(self.width)
                if self._is_walkable(r, c) and 1 <= max(abs(r - self.goal_pos[0]), abs(c - self.goal_pos[1])) <= 5
            ]
            pool = candidates if candidates else [self._random_walkable()]
            self.player_pos = random.choice(pool)
            self.teammate_pos = random.choice(pool)  # 💡追加: 味方guardも同時スポーン
            self.teammate_alive = True                # 💡追加

            self.bot_pos = self._random_walkable()
            self.bot_alive = True
            self.bot_last_action = None

        self.dist_map = bfs_distances(self.goal_pos, self.grid)
        self.prev_dist = self.dist_map[self.player_pos[0]][self.player_pos[1]]

        return self._get_obs(), {}

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

        enemy = [0.0, 0.0, 0.0]
        if self.phase == "guard" and self.bot_alive and has_line_of_sight((pr, pc), self.bot_pos, self.grid):
            edr = (self.bot_pos[0] - pr) / self.height
            edc = (self.bot_pos[1] - pc) / self.width
            enemy = [1.0, edr, edc]

        # 💡追加: 味方guardの位置(guardフェーズのみ有効、それ以外は0埋め)
        teammate = [0.0, 0.0, 0.0]
        if self.phase == "guard":
            tr, tc = self.teammate_pos
            teammate = [1.0, (tr - pr) / self.height, (tc - pc) / self.width]

        return np.array(base + walls + last_onehot + dists + phase_flags + enemy + teammate, dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        terminated = False
        self.last_action = action

        pr, pc = self.player_pos

        if self.phase == "retrieve":
            if action == 4:
                reward = -1.0  # retrieve中はaction4(設置/見張り)は無意味なのでペナルティのみ
            else:
                reward, terminated = self._step_move(action, arrival_reward=100.0)

        elif self.phase == "carry":
            if action == 4:
                if (pr, pc) == self.goal_pos:
                    reward, terminated = 200.0, True
                else:
                    reward = -1.0
            else:
                reward, _ = self._step_move(action, arrival_reward=None)

        else:  # guard
            if action == 4:
                reward = -0.1  # 見張り(停止)は基本コスト小
            else:
                reward, _ = self._step_move(action, arrival_reward=None, guard_mode=True)

            #  teammateも同じネットワークで行動選択(self-play)
            if self.teammate_alive and self.policy_net is not None:
                tobs = self._get_teammate_obs()
                with torch.no_grad():
                    t_t = torch.tensor(tobs, dtype=torch.float32).unsqueeze(0)
                    t_action = self.policy_net(t_t).argmax(dim=1).item()
                self._move_teammate(t_action)

            #  味方と近すぎ(重なり含む)たら軽いペナルティで分散を促す
            if self.teammate_alive:
                tr, tc = self.teammate_pos
                pr, pc = self.player_pos
                d_team = max(abs(pr - tr), abs(pc - tc))
                if d_team == 0:
                    reward -= 1.0
                elif d_team == 1:
                    reward -= 0.3

            # --- defender botの行動 ---
            if self.bot_alive:
                if self.defender_bot is not None:
                    # 💡変更: 本物のdefender AIで行動決定
                    bot_action = self.defender_bot.decide_action(
                        self.bot_pos, self.goal_pos, self.dist_map, self.bot_last_action, self.grid
                    )
                    self.bot_last_action = bot_action
                    if bot_action == 4:
                        dist_to_spike = max(abs(self.bot_pos[0] - self.goal_pos[0]), abs(self.bot_pos[1] - self.goal_pos[1]))
                        if dist_to_spike <= 1:
                            self.bot_defuse_timer += 1
                        else:
                            self.bot_defuse_timer = 0
                    else:
                        moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
                        br, bc = self.bot_pos
                        nr, nc = br + moves[bot_action][0], bc + moves[bot_action][1]
                        if self._is_walkable(nr, nc):
                            self.bot_pos = (nr, nc)
                        self.bot_defuse_timer = 0
                else:
                    # 従来のフォールバック(単純直進bot)
                    br, bc = self.bot_pos
                    best, best_d = None, self.dist_map[br][bc]
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = br + dr, bc + dc
                        if self._is_walkable(nr, nc) and self.dist_map[nr][nc] < best_d:
                            best_d = self.dist_map[nr][nc]
                            best = (nr, nc)
                    dist_to_spike = max(abs(br - self.goal_pos[0]), abs(bc - self.goal_pos[1]))
                    if dist_to_spike <= 1:
                        self.bot_defuse_timer += 1
                    elif best is not None:
                        self.bot_pos = best
                        self.bot_defuse_timer = 0

            # --- 視認判定: 自分 or 味方どちらが見つけても迎撃扱い ---
            spotted_by = None
            if self.bot_alive and has_line_of_sight(tuple(self.player_pos), self.bot_pos, self.grid):
                spotted_by = "self"
            elif self.bot_alive and self.teammate_alive and has_line_of_sight(tuple(self.teammate_pos), self.bot_pos, self.grid):
                spotted_by = "teammate"

            if spotted_by is not None:
                bot_busy = self.bot_defuse_timer > 0
                if bot_busy or random.random() < 0.5:
                    self.bot_alive = False
                    reward += 100.0 if spotted_by == "self" else 30.0  # 味方の手柄なら分け前程度
                else:
                    if spotted_by == "self":
                        reward -= 50.0
                        terminated = True  # 自分が返り討ちにあった
                    else:
                        self.teammate_alive = False  # 味方が返り討ちにあった(自分は継続)

            if not self.bot_alive and not terminated:
                terminated = True
            elif self.bot_defuse_timer >= self.defuse_required:
                reward -= 100.0
                terminated = True
            elif not terminated:
                self.detonate_timer -= 1
                if self.detonate_timer <= 0:
                    reward += 50.0
                    terminated = True

        truncated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}


    def _get_teammate_obs(self):
        """teammate視点での観測(自分と役割を入れ替えて同じ関数を使い回す)"""
        pr, pc = self.teammate_pos
        gr, gc = self.goal_pos
        base = [pr / (self.height - 1), pc / (self.width - 1),
                gr / (self.height - 1), gc / (self.width - 1)]
        walls = [0.0 if self._is_walkable(pr + dr, pc + dc) else 1.0
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        last_onehot = [0.0] * 5  # teammateの前回行動は簡略化のため追跡しない
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
        enemy = [0.0, 0.0, 0.0]
        if self.bot_alive and has_line_of_sight((pr, pc), self.bot_pos, self.grid):
            enemy = [1.0, (self.bot_pos[0] - pr) / self.height, (self.bot_pos[1] - pc) / self.width]
        sr, sc = self.player_pos
        teammate = [1.0, (sr - pr) / self.height, (sc - pc) / self.width]
        return np.array(base + walls + last_onehot + dists + phase_flags + enemy + teammate, dtype=np.float32)

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

        # 💡追加: ゴール手前(BFS距離3以内)は詰める価値を強めに評価し、
        # 最終アプローチの精度を上げる
        if arrival_reward is not None and np.isfinite(new_dist) and new_dist <= 3:
            reward += 1.5

        pos_tuple = (nr, nc)
        # 💡変更: ゴール手前(距離3以内)では後戻りペナルティを外す。
        # 狭い入り口や折れ曲がった経路では最短ルートでも近くを通り直すことがあり、
        # このペナルティが最終アプローチの精度を妨げていた可能性があるため。
        near_goal = arrival_reward is not None and np.isfinite(new_dist) and new_dist <= 3
        if not near_goal and pos_tuple in self.pos_history and new_dist >= self.prev_dist:
            reward -= 2.0
        self.pos_history.append(pos_tuple)

        if guard_mode:
            d_spike = max(abs(nr - self.goal_pos[0]), abs(nc - self.goal_pos[1]))
            if d_spike > 6:
                reward -= 0.3
            
            # 💡追加: スパイク位置への視線が通っているかを評価
            # (解除に来た敵は必ずスパイクの隣接マスに立つため、スパイクへの視線が
            #  通っていれば、その敵をほぼ確実に視認できることになる)
            if has_line_of_sight((nr, nc), self.goal_pos, self.grid):
                reward += 0.3
            else:
                reward -= 0.5

        self.prev_dist = new_dist
        return reward, False


def train():
    writer = SummaryWriter(log_dir="logs")
    
    # ========= 保存用フォルダの作成 =========
    SAVE_DIR = "attacker_data"
    os.makedirs(SAVE_DIR, exist_ok=True)


    lines = [line.strip() for line in NEW_MAZE_STR.strip("\n").split("\n") if line.strip()]
    fixed_grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
    plant_rows, plant_cols = np.where(fixed_grid == 2)
    plant_candidates = list(zip(plant_rows, plant_cols))
    if not plant_candidates:
        raise ValueError("プラントサイト(2)が定義されていません。")

    env = AttackerCombinedEnv(fixed_grid, plant_candidates)
    env.defender_bot = DefenderBotAI(model_path="dqn_defender_combined_best.pt")

    
    batch_size = 64
    gamma = 0.99
    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 0.9985
    lr = 0.0005
    IMPROVEMENT_MARGIN = 5.0

    #  attacker側のみDueling構造に変更(DefenderBotAIのQNetworkはそのまま)
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
    # 以下 print文削除禁止
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
                    action = q_net(obs_t).argmax(dim=1).item()

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

        # ========= 変更: エピソード終了時のログと定期保存 =========
        if (episode + 1) % 50 == 0:
            avg_loss = np.mean(losses) if losses else 0.0
            print(f"Episode {episode+1}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f}")

            # 💡追加: 局面別evalログ(softmax評価+carry強制PLANT対応)
            phase_rewards = {"retrieve": [], "carry": [], "guard": []}
            for _ in range(30):
                eval_obs, _ = env.reset()
                phase = env.phase
                eval_reward = 0.0
                while True:
                    # carry中、設置サイトに到達していたら強制的にPLANT(action=4)を選ぶ
                    # (learning_attacker.pyの推論ロジックと一致させる)
                    if phase == "carry" and tuple(env.player_pos) == env.goal_pos:
                        a = 4
                    else:
                        with torch.no_grad():
                            q_values = q_net(torch.tensor(eval_obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
                        if phase != "guard":
                            q_values = q_values.copy()
                            q_values[4] = -np.inf
                        probs = np.exp((q_values - np.max(q_values)) / 0.5)
                        probs = probs / probs.sum()
                        a = np.random.choice(len(probs), p=probs)

                    eval_obs, r, term, trunc, _ = env.step(a)
                    eval_reward += r
                    if term or trunc:
                        break
                phase_rewards[phase].append(eval_reward)

            for phase_name, rewards_list in phase_rewards.items():
                if rewards_list:
                    writer.add_scalar(f"Eval/{phase_name}_Reward", np.mean(rewards_list), episode)
                    print(f"   [{phase_name}] mean={np.mean(rewards_list):.2f} n={len(rewards_list)}")

            # 加重平均(guardを重視)でbest model判定
            guard_mean = np.mean(phase_rewards["guard"]) if phase_rewards["guard"] else 0.0
            other_mean = np.mean(phase_rewards["retrieve"] + phase_rewards["carry"]) if (phase_rewards["retrieve"] + phase_rewards["carry"]) else 0.0
            mean_eval = 0.5 * guard_mean + 0.5 * other_mean

            writer.add_scalar("Eval/Weighted_Reward", mean_eval, episode)

            if mean_eval > best_eval_reward + IMPROVEMENT_MARGIN:
                best_eval_reward = mean_eval
                best_path = os.path.join(SAVE_DIR, "dqn_attacker_combined_best_by_eval.pt")
                torch.save(q_net.state_dict(), best_path)
                print(f"   [Eval Best] 保存しました (Eval Reward: {mean_eval:.2f}): {best_path}")

        # 指定エピソードごとにモデルを保存
        if (episode + 1) % SAVE_INTERVAL == 0:
            save_path = os.path.join(SAVE_DIR, f"dqn_attacker_ep{episode+1}.pt")
            torch.save(q_net.state_dict(), save_path)
            print(f"   [Save] 定期保存しました: {save_path}")
        # ==========================================

    # 最終モデルの保存
    final_path = os.path.join(SAVE_DIR, "dqn_attacker_combined_final.pt")
    torch.save(q_net.state_dict(), final_path)
    print(f"学習が完了しました。最終モデル: {final_path}")
    writer.close()

if __name__ == "__main__":
    train()