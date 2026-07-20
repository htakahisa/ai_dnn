# train_defender_combined.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# 🎯 run_game.py からマスターのマップデータを直接インポート
from map_data import NEW_MAZE_STR

# ==========================================
# 🧠 ネットワーク & バッファ定義
# ==========================================
class QNetwork(nn.Module):
    """ディフェンダー用Qネットワーク構造（20次元入力対応）"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    """学習用経験再生バッファ"""
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state, dtype=np.float32),
                np.array(action, dtype=np.int64),
                np.array(reward, dtype=np.float32),
                np.array(next_state, dtype=np.float32),
                np.array(done, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)

# ==========================================
# 🗺️ 環境定義
# ==========================================
class CombinedGridWorldEnv(gym.Env):
    """プラント前・後をシミュレートする統合学習環境（敵の目撃情報・ローテーション・ガチ解除学習版）"""
    def __init__(self, fixed_grid, goal_candidates):
        super().__init__()
        self.height, self.width = fixed_grid.shape
        self.grid = fixed_grid
        self.goal_candidates = [np.array(g, dtype=np.int32) for g in goal_candidates]

        self.action_space = spaces.Discrete(5) # 0:上, 1:下, 2:左, 3:右, 4:解除
        # 💡 [基本4][壁4][前回行動5][距離4][敵情報3] = 20次元
        self.obs_dim = 20
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.max_steps = 300
        self.detonate_limit = 45

    def _is_walkable(self, r, c):
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.grid[r, c] != 1
        return False

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.last_action = None
        self.plant_step_count = 0

        self.true_enemy_target = random.choice(self.goal_candidates)
        self.plant_site = random.choice(self.goal_candidates)

        self.enemy_spotted = 0.0
        self.spotted_site_pos = np.array([self.height // 2, self.width // 2], dtype=np.int32)

        while True:
            self.player_pos = np.array([random.randint(0, self.height-1), random.randint(0, self.width-1)], dtype=np.int32)
            if self._is_walkable(self.player_pos[0], self.player_pos[1]):
                break

        self.pos_history = deque(maxlen=6)   

        # 💡【変更】50%の確率でプラント済み状態からスタートし、解除フェーズのサンプルを増やす
        self.is_planted = (random.random() < 0.5)
        self.defuse_timer = 0

        if self.is_planted:
            self.goal_pos = self.true_enemy_target
            self.enemy_spotted = 1.0
            self.spotted_site_pos = self.true_enemy_target

            # 💡 プラント後スタートは、解除可能な距離内にスポーンさせる
            dist_map_for_spawn = self._bfs_distances_from_goal(self.goal_pos)
            max_spawn_dist = self.detonate_limit - 6 - 5  # 移動+解除+多少の余裕
            candidates = [
                (r, c) for r in range(self.height) for c in range(self.width)
                if self._is_walkable(r, c) and np.isfinite(dist_map_for_spawn[r, c])
                and dist_map_for_spawn[r, c] <= max_spawn_dist
            ]
            self.player_pos = np.array(random.choice(candidates), dtype=np.int32)
        else:
            while True:
                self.player_pos = np.array([random.randint(0, self.height-1), random.randint(0, self.width-1)], dtype=np.int32)
                if self._is_walkable(self.player_pos[0], self.player_pos[1]):
                    break
            self.goal_pos = self.plant_site

        self.dist_map = self._bfs_distances_from_goal(self.goal_pos)
        self.prev_dist = self.dist_map[self.player_pos[0], self.player_pos[1]]

        self.enemy_target_dist_map = self._bfs_distances_from_goal(self.true_enemy_target)
        self.prev_enemy_target_dist = self.enemy_target_dist_map[self.player_pos[0], self.player_pos[1]]

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        shaping_pre_plant = 0.0
        reward = 0.0
        terminated = False
        
        # ---------------------------------------------------------------------
        # 👁️ プラント前のタイムライン・情報アップデート
        # ---------------------------------------------------------------------
        if not self.is_planted:
            if self.current_step > 15 or random.random() < 0.05:
                self.enemy_spotted = 1.0
                self.spotted_site_pos = self.true_enemy_target 
            
            if self.current_step > 40 or random.random() < 0.02:
                self.is_planted = True
                self.goal_pos = self.true_enemy_target 
                self.dist_map = self._bfs_distances_from_goal(self.goal_pos)
                self.prev_dist = self.dist_map[self.player_pos[0], self.player_pos[1]]

        if self.is_planted:
            self.plant_step_count += 1
        
        # ---------------------------------------------------------------------
        # ⚖️ アクション処理ロジック（移動 or 解除）
        # ---------------------------------------------------------------------
        if action == 4:
            # 🖐 解除アクションが選択された場合
            if self.is_planted:
                # スパイクとのチェビシェフ距離を計算（本番環境の max(abs, abs) 基準に合わせる）
                dist_to_spike = max(abs(self.goal_pos[0] - self.player_pos[0]), abs(self.goal_pos[1] - self.player_pos[1]))
                if dist_to_spike <= 1:
                    # 解除エリア内ならタイマー進行
                    self.defuse_timer += 1
                    reward = 1.0 # 解除を進めていることへのステップ報酬
                    if self.defuse_timer >= 6:
                        reward = 200.0 # 🏆 解除完了特大ボーナス！
                        terminated = True
                else:
                    # スパイクから離れた場所で無意味に解除を押した（お仕置きペナルティ）
                    self.defuse_timer = 0
                    reward = -1.0
            else:
                # まだプラントされてないのに解除を押した（無駄行動ペナルティ）
                self.defuse_timer = 0
                reward = -1.0
        else:
            # 🏃 移動アクション（0~3）が選択された場合
            self.defuse_timer = 0 # 移動したら解除タイマーはリセット
            moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
            next_pos = self.player_pos + moves[action]

            if self._is_walkable(next_pos[0], next_pos[1]):
                self.player_pos = next_pos
                
                # 移動によるゴールへの距離報酬計算
                new_dist = self.dist_map[self.player_pos[0], self.player_pos[1]]
                shaping = (self.prev_dist - new_dist) * 0.5
                
                if not self.is_planted and self.enemy_spotted > 0.5:
                    new_enemy_target_dist = self.enemy_target_dist_map[self.player_pos[0], self.player_pos[1]]
                    shaping_pre_plant = (self.prev_enemy_target_dist - new_enemy_target_dist) * 0.4 
                    self.prev_enemy_target_dist = new_enemy_target_dist

                # 移動だけでゴール（スパイク位置）に重なっても即終了にはせず、
                # あくまで「解除アクションでタイマーを貯めること」をゴールとするため terminated=False のまま。
                # ただしプラント前フェーズの「サイト割り当て防衛」の到着は従来通り許可しても良いですが、
                # 今回は統一して「その場に留まるだけ」か「解除のみ終了」にするため、ここではベース報酬のみ。
                if np.array_equal(self.player_pos, self.goal_pos) and not self.is_planted:
                    reward = 50.0 # プラント前に割り当てサイトを守り切った報酬
                    terminated = True
                else:
                    reward = -1.0 + shaping + shaping_pre_plant

                    # ゴール距離が縮まっていない「かつ」訪問済みマスに戻った場合のみ罰する
                    pos_tuple = tuple(self.player_pos)
                    if pos_tuple in self.pos_history and new_dist >= self.prev_dist:
                        reward -= 2.0
                    self.pos_history.append(pos_tuple)

                self.prev_dist = new_dist
                
            else:
                reward = -1.5 # 壁衝突ペナルティ

        self.last_action = action
        
        # detonate_limitは本当のラウンド終了なのでterminatedにする
        if self.is_planted and self.plant_step_count >= self.detonate_limit and not terminated:
            reward = -5.0
            terminated = True   # truncatedではなくterminated

        truncated = self.current_step >= self.max_steps
        
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        pr, pc = self.player_pos
        gr, gc = self.goal_pos
        base = [pr / (self.height - 1), pc / (self.width - 1),
                gr / (self.height - 1), gc / (self.width - 1)]
        
        #  前回行動のOne-hot表現を5次元へ拡張
        last_action_onehot = [0.0, 0.0, 0.0, 0.0, 0.0]
        if self.last_action is not None:
            last_action_onehot[self.last_action] = 1.0
            
        sr, sc = self.spotted_site_pos
        #  enemy_info を 3次元 に拡張（目撃フラグ, 正規化R, 正規化C）
        enemy_info = [self.enemy_spotted, sr / (self.height - 1), sc / (self.width - 1)]

        return np.array(base + self._local_walls() + last_action_onehot + self._neighbor_distances() + enemy_info, dtype=np.float32)

    def _local_walls(self):
        r, c = self.player_pos
        out = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if self._is_walkable(r + dr, c + dc):
                out.append(0.0)
            else:
                out.append(1.0)
        return out

    def _neighbor_distances(self):
        r, c = self.player_pos
        max_dist = self.height * self.width
        out = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self._is_walkable(nr, nc):
                d = self.dist_map[nr, nc]
                out.append(d / max_dist if np.isfinite(d) else 1.0)
            else:
                out.append(1.0)
        return out

    def _bfs_distances_from_goal(self, target_pos):
        dist = np.full((self.height, self.width), np.inf)
        gr, gc = target_pos
        dist[gr, gc] = 0
        q = deque([(gr, gc)])
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if self._is_walkable(nr, nc):
                    if dist[nr, nc] > dist[r, c] + 1:
                        dist[nr, nc] = dist[r, c] + 1
                        q.append((nr, nc))
        return dist


# ==========================================
# 📈 学習実行メインロジック
# ==========================================
def train():
    writer = SummaryWriter(log_dir="logs")

    lines = [line.strip() for line in NEW_MAZE_STR.strip("\n").split("\n") if line.strip()]
    fixed_grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)

    plant_rows, plant_cols = np.where(fixed_grid == 2)
    goal_candidates = list(zip(plant_rows, plant_cols))

    if not goal_candidates:
        raise ValueError("Error: NEW_MAZE_STR 内にプラントサイト（値: 2）が定義されていません。")

    env = CombinedGridWorldEnv(fixed_grid, goal_candidates)

    # ハイパーパラメータ
    num_episodes = 4000 # 状態空間が少し広がったため、エピソード数をやや拡張
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.996
    lr = 0.0005
    IMPROVEMENT_MARGIN = 10.0

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    q_net = QNetwork(env.obs_dim, env.action_space.n).to(device)
    target_net = QNetwork(env.obs_dim, env.action_space.n).to(device)
    target_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=20000)
    
    epsilon = epsilon_start
    total_steps = 0
    best_eval_reward = -float('inf')

    print(f"学習を開始します。デバイス: {device} | 入力次元: {env.obs_dim}")
    # 以下 print文削除禁止
    print("python -m tensorboard.main --logdir=logs")

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        losses = []

        while True:
            total_steps += 1
            
            if random.random() < epsilon:
                action = random.randint(0, env.action_space.n - 1)
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
                    # 💡 行動選択はオンラインネットワーク、価値評価はターゲットネットワークで行う
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
        writer.add_scalar("Train/Epsilon", epsilon, episode)
        if losses:
            writer.add_scalar("Train/Loss", np.mean(losses), episode)

        if (episode + 1) % 50 == 0:
            avg_loss = np.mean(losses) if losses else 0.0
            print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f}")

        
        if (episode + 1) % 50 == 0:
            eval_rewards = []
            for _ in range(10):  # best評価のepisode件数
                eval_obs, _ = env.reset()
                eval_reward = 0.0
                while True:
                    with torch.no_grad():
                        eval_action = q_net(torch.tensor(eval_obs, dtype=torch.float32, device=device).unsqueeze(0)).argmax(dim=1).item()
                    eval_obs, r, term, trunc, _ = env.step(eval_action)
                    eval_reward += r
                    if term or trunc:
                        break
                eval_rewards.append(eval_reward)
                
            mean_eval = np.mean(eval_rewards) 
            writer.add_scalar("Eval/Greedy_Reward", np.mean(eval_rewards), episode)
            writer.add_scalar("Eval/Greedy_Reward_Std", np.std(eval_rewards), episode)
    
            if mean_eval > best_eval_reward + IMPROVEMENT_MARGIN:
                best_eval_reward = mean_eval
                torch.save(q_net.state_dict(), "dqn_defender_combined_best.pt")
                print(f"   New best model saved (Eval Reward: {mean_eval:.2f})")
    torch.save(q_net.state_dict(), "dqn_defender_combined.pt")
    print("モデルの保存が完了しました: dqn_defender_combined.pt")
    writer.close()

if __name__ == "__main__":
    train()