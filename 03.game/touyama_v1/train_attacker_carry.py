"""train_attacker_carry.py

Attacker Carry Phase 専用の学習スクリプト（Dueling DQN）。

【目的】
スパイクを持ったキャリアーが、マップ上のどちらかのプラントサイト(grid==2)へ
たどり着き、そこでスパイクを設置(PLANT)できるようになるまで学習する。
さらに、学習専用マップ(map_data_carry.py)に定義された「優先プラント地点」
(grid==5)への設置を緩やかに優先しつつ、残り時間が少ない場合はサイト内の
どこでも（grid==2）設置してよい、というトレードオフを学習させる。

【重要：ラウンドごとのランダムな目標地点(target_plant_pos)】
実際のゲーム(run_game.py の init_round())は、ラウンド開始時に
grid==2のセルからランダムに1点を選び target_plant_pos としている。
以前のバージョンでは「サイト全体（左右問わず）への最短距離」を
ナビゲーション目標にしていたため、スポーン地点から近い方のサイトだけに
収束してしまう問題があった（近い方が常に効率的なため、RLとしては
合理的な結果だが、左右均等に運ぶ挙動を学習させたい場合は不適切）。

このバージョンでは、実ゲームと同じように「エピソードごとにランダムな
目標地点を1つ選び、そこまでの距離」をナビゲーション目標にする。
これにより目標が左右どちらのサイトからも均等に選ばれるようになり、
両方向への到達方法を学習する。
（設置自体は従来通りサイト内のどこでも成功扱い。targetは
「誘導目標」であって「強制ゴール」ではない点も実ゲームのルールと一致）

【設計方針】
- このファイルは完全に自己完結している（他のfeatureモジュールをimportしない）。
  map_data_carry.py / map_data.py（マップ座標データのみ）以外の
  ゲーム本体コードには依存しない。
- run_game.py / controllers.py など既存の共有インフラは変更・複製しない。
- キャリアーは常時発動パッシブ「ハンター」を持つ想定のため、
  アクティブなアビリティ選択は行わない（移動 + PLANT のみ）。
- 敵(Defender)は未実装。このフェーズでは「サイトまで運んでプラントする」
  という移動・意思決定能力の獲得のみを対象とする。
"""

import argparse
import os
import random
import sys
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

EPISODE_COUNT = 8000

# ---------------------------------------------------------------------------
# map_data_carry.py（優先プラント地点=5を含む学習専用マップ）を優先ロードし、
# 無ければ従来の map_data.py にフォールバックする。
# このファイルが attacker_v3/ 配下・プロジェクト直下のどちらに置かれても
# 解決できるようにする。
# ---------------------------------------------------------------------------
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = (
    os.path.dirname(_CURRENT_DIR)
    if os.path.basename(_CURRENT_DIR) == "attacker_v3"
    else _CURRENT_DIR
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from map_data_carry import NEW_MAZE_STR  # noqa: E402
    print("[MAP] map_data_carry.py を使用します（優先プラント地点=5対応・学習専用）")
except ImportError:
    from map_data import NEW_MAZE_STR  # noqa: E402
    print("[MAP] map_data_carry.py が見つからないため map_data.py を使用します（優先地点なし）")


SITE_CELL_VALUE = 2         # 本番にも存在する通常のプラント可能セル
PRIORITY_CELL_VALUE = 5     # 学習専用マップにのみ存在する優先プラント地点


# ---------------------------------------------------------------------------
# BFSによる距離マップ（指定した座標群を始点としたマルチソースBFS）
# ---------------------------------------------------------------------------
def _build_distance_map_from_coords(grid, source_cells):
    """指定した座標群を始点としたマルチソースBFS。壁(1)は通行不可。"""
    height, width = grid.shape
    dist = np.full((height, width), np.inf, dtype=np.float32)
    q = deque()

    for r, c in source_cells:
        if 0 <= r < height and 0 <= c < width and grid[r, c] != 1:
            dist[r, c] = 0.0
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
                if dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))

    return dist


def _cells_with_value(grid, value):
    return list(zip(*np.where(grid == value)))


# ---------------------------------------------------------------------------
# 環境
# ---------------------------------------------------------------------------
class CarryEnv:
    """スパイクキャリアーの「目標地点まで運んでプラントする」を扱う軽量環境。

    毎エピソード、実ゲームの target_plant_pos と同じ選び方
    （grid==2 セルからランダムに1点）で目標地点を決め、そこまでの距離を
    ナビゲーション目標にする。これにより左右どちらのサイトへ向かう
    経験も均等に得られる。

    優先プラント地点(grid==5、学習専用マップにのみ存在)が定義されている場合、
    そこへの設置をやんわり優先しつつ、通常セル(grid==2)への設置も
    「残り時間が少ないほど遜色ない報酬」になるよう設計する
    （こちらは target とは独立した仕組み）。
    """

    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY, ACTION_PLANT = range(6)
    N_ACTIONS = 6
    _MOVE_DELTA = {
        ACTION_UP: (-1, 0),
        ACTION_DOWN: (1, 0),
        ACTION_LEFT: (0, -1),
        ACTION_RIGHT: (0, 1),
        ACTION_STAY: (0, 0),
    }

    def __init__(
        self,
        maze_str=NEW_MAZE_STR,
        plant_required_ticks=4,
        max_ticks=90,
        random_spawn_ratio=0.25,
        shaping_coef=0.15,
        priority_shaping_coef=0.05,
        plant_tick_reward_priority=0.2,
        plant_tick_reward_generic=0.1,
        plant_success_reward_priority=20.0,
        plant_success_reward_generic_base=12.0,
        seed=None,
    ):
        lines = [l.strip() for l in maze_str.strip("\n").split("\n") if l.strip()]
        self.grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
        self.height, self.width = self.grid.shape

        # 実ゲームの target_plant_pos と同じ選び方（grid==2から選ぶ）に
        # 合わせるため、目標候補は grid==2 のセルのみとする。
        self.target_candidate_cells = _cells_with_value(self.grid, SITE_CELL_VALUE)
        if not self.target_candidate_cells:
            raise ValueError("マップに grid==2（プラントサイト）が存在しません")

        # 正規化用の定数。「全サイトセルへの最短距離」の最大値を、
        # target距離の正規化スケールとして流用する
        # （どのtargetが選ばれてもスケールが揺れないように固定基準を使う）。
        all_site_cells = self.target_candidate_cells + _cells_with_value(self.grid, PRIORITY_CELL_VALUE)
        _dist_for_norm = _build_distance_map_from_coords(self.grid, all_site_cells)
        finite = _dist_for_norm[np.isfinite(_dist_for_norm)]
        self.norm_max_dist = float(finite.max()) if finite.size else 1.0

        # 優先地点(5)の座標。学習専用マップにのみ存在する。target選択とは独立。
        self.priority_cells = _cells_with_value(self.grid, PRIORITY_CELL_VALUE)
        self.has_priority_cells = len(self.priority_cells) > 0

        if self.has_priority_cells:
            self.priority_dist_map = _build_distance_map_from_coords(self.grid, self.priority_cells)
        else:
            self.priority_dist_map = _dist_for_norm
        finite_p = self.priority_dist_map[np.isfinite(self.priority_dist_map)]
        self.max_finite_priority_dist = float(finite_p.max()) if finite_p.size else 1.0

        self.spawn_cells = list(zip(*np.where(self.grid == 3)))
        self.walkable_cells = list(zip(*np.where(self.grid != 1)))

        self.plant_required_ticks = plant_required_ticks
        self.max_ticks = max_ticks
        self.random_spawn_ratio = random_spawn_ratio
        self.shaping_coef = shaping_coef
        self.priority_shaping_coef = priority_shaping_coef
        self.plant_tick_reward_priority = plant_tick_reward_priority
        self.plant_tick_reward_generic = plant_tick_reward_generic
        self.plant_success_reward_priority = plant_success_reward_priority
        self.plant_success_reward_generic_base = plant_success_reward_generic_base

        self.rng = random.Random(seed)

        # target座標(r, c) -> 距離マップ のキャッシュ。
        # 目標候補数は有限（サイトのセル数）なので、繰り返し再利用できる。
        self._target_dist_cache = {}

        self.pos = (0, 0)
        self.target_pos = self.target_candidate_cells[0]
        self.target_dist_map = self._get_or_build_target_map(self.target_pos)
        self.plant_timer = 0
        self.tick = 0
        self.last_delta = (0, 0)
        self.stuck_counter = 0
        self._prev_dist = 0.0
        self._prev_priority_dist = 0.0

    def _get_or_build_target_map(self, target_pos):
        cached = self._target_dist_cache.get(target_pos)
        if cached is None:
            cached = _build_distance_map_from_coords(self.grid, [target_pos])
            self._target_dist_cache[target_pos] = cached
        return cached

    def _is_wall(self, r, c):
        if not (0 <= r < self.height and 0 <= c < self.width):
            return True
        return self.grid[r, c] == 1

    def _is_plantable(self, r, c):
        """通常セル(2) または 優先セル(5、学習専用) のどちらでも設置可能。"""
        if not (0 <= r < self.height and 0 <= c < self.width):
            return False
        return self.grid[r, c] in (SITE_CELL_VALUE, PRIORITY_CELL_VALUE)

    def _is_priority_cell(self, r, c):
        if not (0 <= r < self.height and 0 <= c < self.width):
            return False
        return self.grid[r, c] == PRIORITY_CELL_VALUE

    def _target_distance(self, pos):
        r, c = pos
        if not (0 <= r < self.height and 0 <= c < self.width):
            return self.norm_max_dist
        d = self.target_dist_map[r, c]
        return self.norm_max_dist if not np.isfinite(d) else float(d)

    def _priority_distance(self, pos):
        r, c = pos
        d = self.priority_dist_map[r, c]
        return self.max_finite_priority_dist if not np.isfinite(d) else float(d)

    def reset(self):
        if self.spawn_cells and self.rng.random() > self.random_spawn_ratio:
            self.pos = self.rng.choice(self.spawn_cells)
        else:
            self.pos = self.rng.choice(self.walkable_cells)

        # 実ゲームの init_round() と同じ選び方（grid==2からランダムに1点）で
        # 今エピソードの目標地点を決める。これが左右均等な経験を生む。
        self.target_pos = self.rng.choice(self.target_candidate_cells)
        self.target_dist_map = self._get_or_build_target_map(self.target_pos)

        self.plant_timer = 0
        self.tick = 0
        self.last_delta = (0, 0)
        self.stuck_counter = 0
        self._prev_dist = self._target_distance(self.pos)
        self._prev_priority_dist = self._priority_distance(self.pos)
        return self._get_obs()

    def _get_obs(self):
        r, c = self.pos

        obs = []
        obs.append(r / max(1, self.height - 1))
        obs.append(c / max(1, self.width - 1))

        cur_dist = self._target_distance(self.pos)
        obs.append(min(1.0, cur_dist / self.norm_max_dist))

        # 上下左右：壁フラグ + その方向へ進んだ場合の「今回の目標地点」までの距離
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            wall = self._is_wall(nr, nc)
            obs.append(1.0 if wall else 0.0)
            obs.append(1.0 if wall else min(1.0, self._target_distance((nr, nc)) / self.norm_max_dist))

        # 斜め4方向の壁フラグ（局所形状の把握用）
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            obs.append(1.0 if self._is_wall(r + dr, c + dc) else 0.0)

        obs.append(1.0 if self._is_plantable(r, c) else 0.0)  # on_site（通常+優先、どのサイトでもよい）
        obs.append(1.0 if self.plant_timer > 0 else 0.0)  # is_planting
        obs.append(self.plant_timer / max(1, self.plant_required_ticks))  # plant progress
        obs.append(1.0 - min(1.0, self.tick / max(1, self.max_ticks)))  # 残り時間比
        obs.append(self.last_delta[0])
        obs.append(self.last_delta[1])
        obs.append(min(1.0, self.stuck_counter / 10.0))

        # --- 優先プラント地点関連の追加特徴量（targetとは独立） ---
        cur_priority_dist = self._priority_distance(self.pos)
        obs.append(min(1.0, cur_priority_dist / self.max_finite_priority_dist))
        obs.append(1.0 if self._is_priority_cell(r, c) else 0.0)

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            wall = self._is_wall(nr, nc)
            obs.append(1.0 if wall else min(1.0, self._priority_distance((nr, nc)) / self.max_finite_priority_dist))

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.tick += 1
        reward = -0.01  # 時間経過ペナルティ（速さを促す）
        done = False
        info = {"success": False, "priority": False}

        r, c = self.pos
        remaining_ratio = 1.0 - min(1.0, self.tick / max(1, self.max_ticks))
        time_pressure = 1.0 - remaining_ratio  # 0(開始直後) -> 1(残り時間僅か)

        if action == self.ACTION_PLANT:
            on_priority = self._is_priority_cell(r, c)
            on_plantable = self._is_plantable(r, c)

            if on_plantable:
                self.plant_timer += 1
                reward += (
                    self.plant_tick_reward_priority if on_priority
                    else self.plant_tick_reward_generic
                )

                if self.plant_timer >= self.plant_required_ticks:
                    if on_priority or not self.has_priority_cells:
                        # 優先セル、または優先セルが存在しないマップ（＝優先地点
                        # という概念自体が学習に含まれていない）では満額。
                        success_reward = self.plant_success_reward_priority
                    else:
                        # 通常セル：基礎報酬 + 時間切迫度に応じたボーナスで
                        # 満額に近づける（時間がなければ実質同等の価値にする）。
                        bonus = time_pressure * (
                            self.plant_success_reward_priority
                            - self.plant_success_reward_generic_base
                        )
                        success_reward = self.plant_success_reward_generic_base + bonus

                    reward += success_reward
                    done = True
                    info["success"] = True
                    info["priority"] = bool(on_priority or not self.has_priority_cells)
            else:
                self.plant_timer = 0
                reward -= 0.05
            self.last_delta = (0, 0)
            self.stuck_counter += 1
        else:
            self.plant_timer = 0
            dr, dc = self._MOVE_DELTA[action]
            nr, nc = r + dr, c + dc

            if self._is_wall(nr, nc):
                reward -= 0.05
                self.last_delta = (0, 0)
                self.stuck_counter += 1
            else:
                self.pos = (nr, nc)
                if (dr, dc) == (0, 0):
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = 0
                self.last_delta = (dr, dc)

            new_dist = self._target_distance(self.pos)
            reward += (self._prev_dist - new_dist) * self.shaping_coef
            self._prev_dist = new_dist

            # 優先地点への緩やかな誘導（targetとは独立、弱い係数で主導線は乱さない）
            new_priority_dist = self._priority_distance(self.pos)
            reward += (self._prev_priority_dist - new_priority_dist) * self.priority_shaping_coef
            self._prev_priority_dist = new_priority_dist

        if not done and self.tick >= self.max_ticks:
            done = True
            reward -= 5.0
            info["success"] = False

        return self._get_obs(), reward, done, info


def get_action_mask(env):
    """壁に向かう移動だけを無効化する。STAY / PLANT は常に有効。"""
    r, c = env.pos
    mask = np.ones(env.N_ACTIONS, dtype=bool)
    for a, (dr, dc) in env._MOVE_DELTA.items():
        if a == env.ACTION_STAY:
            continue
        if env._is_wall(r + dr, c + dc):
            mask[a] = False
    return mask


# ---------------------------------------------------------------------------
# Dueling DQN
# ---------------------------------------------------------------------------
class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x):
        feat = self.feature(x)
        value = self.value_head(feat)
        advantage = self.advantage_head(feat)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "next_mask", "done"))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


def select_action(net, state, mask, epsilon, device):
    if random.random() < epsilon:
        valid_actions = np.flatnonzero(mask)
        return int(random.choice(valid_actions))
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = net(state_t).squeeze(0).cpu().numpy()
    q = np.where(mask, q, -1e9)
    return int(np.argmax(q))


def optimize_model(policy_net, target_net, optimizer, buffer, batch_size, gamma, device):
    if len(buffer) < batch_size:
        return None

    batch = buffer.sample(batch_size)

    states = torch.as_tensor(np.array(batch.state), dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.as_tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
    dones = torch.as_tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)
    next_masks = torch.as_tensor(np.array(batch.next_mask), dtype=torch.bool, device=device)

    q_values = policy_net(states).gather(1, actions)

    with torch.no_grad():
        next_q_policy = policy_net(next_states)
        # next_state側で無効なアクションのQ値をマスクしてからargmaxを取る。
        # ここを忘れるとTD誤差計算がマスク外のQ値に汚染される。
        next_q_policy = next_q_policy.masked_fill(~next_masks, -1e9)
        next_actions = next_q_policy.argmax(dim=1, keepdim=True)

        next_q_target = target_net(next_states).gather(1, next_actions)
        target = rewards + gamma * next_q_target * (1.0 - dones)

    loss = F.smooth_l1_loss(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()

    return float(loss.item())


def evaluate(env, policy_net, device, episodes=30):
    successes = 0
    total_ticks = 0
    total_reward = 0.0
    priority_plants = 0
    info = {"success": False, "priority": False}

    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            mask = get_action_mask(env)
            with torch.no_grad():
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q = policy_net(state_t).squeeze(0).cpu().numpy()
            q = np.where(mask, q, -1e9)
            action = int(np.argmax(q))
            state, reward, done, info = env.step(action)
            episode_reward += reward
        total_reward += episode_reward
        if info["success"]:
            successes += 1
            total_ticks += env.tick
            if info.get("priority"):
                priority_plants += 1

    success_rate = successes / episodes
    avg_ticks = total_ticks / successes if successes else float("nan")
    avg_reward = total_reward / episodes
    priority_rate = priority_plants / successes if successes else float("nan")
    return success_rate, avg_ticks, avg_reward, priority_rate


def main():
    parser = argparse.ArgumentParser(description="Attacker Carry Phase 学習スクリプト")
    parser.add_argument("--episodes", type=int, default=EPISODE_COUNT)
    parser.add_argument("--max-ticks", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-episodes", type=int, default=int(EPISODE_COUNT * 0.8))
    parser.add_argument("--target-update-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--random-spawn-ratio", type=float, default=0.25)
    parser.add_argument("--shaping-coef", type=float, default=0.15)
    parser.add_argument("--priority-shaping-coef", type=float, default=0.05)
    parser.add_argument("--plant-tick-reward-priority", type=float, default=0.2)
    parser.add_argument("--plant-tick-reward-generic", type=float, default=0.1)
    parser.add_argument("--plant-success-reward-priority", type=float, default=20.0)
    parser.add_argument("--plant-success-reward-generic-base", type=float, default=12.0)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "attacker_v3", "data", "attacker_carry_data"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    env_kwargs = dict(
        max_ticks=args.max_ticks,
        random_spawn_ratio=args.random_spawn_ratio,
        shaping_coef=args.shaping_coef,
        priority_shaping_coef=args.priority_shaping_coef,
        plant_tick_reward_priority=args.plant_tick_reward_priority,
        plant_tick_reward_generic=args.plant_tick_reward_generic,
        plant_success_reward_priority=args.plant_success_reward_priority,
        plant_success_reward_generic_base=args.plant_success_reward_generic_base,
    )

    env = CarryEnv(seed=args.seed, **env_kwargs)
    eval_env = CarryEnv(seed=args.seed + 1, **env_kwargs)

    print(
        f"[INFO] has_priority_cells={env.has_priority_cells} "
        f"target_candidates={len(env.target_candidate_cells)}"
    )

    obs_dim = env.reset().shape[0]
    n_actions = env.N_ACTIONS
    print(f"[INFO] obs_dim={obs_dim} n_actions={n_actions} device={device}")

    policy_net = DuelingQNetwork(obs_dim, n_actions).to(device)
    target_net = DuelingQNetwork(obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)

    best_success_rate = -1.0
    best_eval_reward = float("-inf")
    global_step = 0

    def _make_checkpoint(episode, success_rate, avg_reward, priority_rate):
        return {
            "model_state_dict": policy_net.state_dict(),
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "episode": episode,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "priority_rate": priority_rate,
            # 本番マップ(map_data.py)には grid==5 は存在しない。
            # 推論側は、この座標リストを使って本番マップ上でBFS距離を
            # 計算するため、gridの値そのものには依存しない。
            # numpy.int64 のまま保存すると torch.load(weights_only=True) で
            # 読み込めなくなるため、必ず素の Python int に変換しておく。
            "priority_cells": [(int(r), int(c)) for r, c in env.priority_cells],
            "has_priority_cells": bool(env.has_priority_cells),
        }

    for episode in range(1, args.episodes + 1):
        progress = min(1.0, episode / args.eps_decay_episodes)
        epsilon = args.eps_start + (args.eps_end - args.eps_start) * progress

        state = env.reset()
        done = False
        episode_reward = 0.0
        info = {"success": False, "priority": False}

        while not done:
            mask = get_action_mask(env)
            action = select_action(policy_net, state, mask, epsilon, device)
            next_state, reward, done, info = env.step(action)
            next_mask = get_action_mask(env)

            buffer.push(state, action, reward, next_state, next_mask, done)
            state = next_state
            episode_reward += reward
            global_step += 1

            if len(buffer) >= max(args.batch_size, args.warmup_steps):
                optimize_model(
                    policy_net, target_net, optimizer, buffer,
                    args.batch_size, args.gamma, device,
                )

            if global_step % args.target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if episode % 50 == 0:
            print(
                f"[EP {episode}/{EPISODE_COUNT}] reward={episode_reward:.2f} "
                f"eps={epsilon:.3f} success={info['success']} "
                f"priority={info.get('priority')} ticks={env.tick} "
                f"target={env.target_pos}"
            )

        if episode % args.eval_every == 0:
            success_rate, avg_ticks, avg_reward, priority_rate = evaluate(
                eval_env, policy_net, device, args.eval_episodes
            )
            print(
                f"[EVAL @ EP {episode}/{EPISODE_COUNT}] success_rate={success_rate:.2%} "
                f"avg_ticks_on_success={avg_ticks:.1f} avg_reward={avg_reward:.2f} "
                f"priority_rate={priority_rate:.2%}"
            )

            latest_path = os.path.join(args.save_dir, "dqn_attacker_carry_latest.pt")
            torch.save(
                _make_checkpoint(episode, success_rate, avg_reward, priority_rate),
                latest_path,
            )

            # 成功率が高い方を優先し、成功率が同等（実質100%到達後）なら
            # 平均報酬（速さ・効率を反映）が高い方をベストとする複合基準。
            is_better = (
                success_rate > best_success_rate + 1e-9
                or (
                    success_rate >= best_success_rate - 1e-9
                    and avg_reward > best_eval_reward
                )
            )

            if is_better:
                best_success_rate = max(best_success_rate, success_rate)
                best_eval_reward = avg_reward
                best_path = os.path.join(args.save_dir, "dqn_attacker_carry_best_by_eval.pt")
                torch.save(
                    _make_checkpoint(episode, success_rate, avg_reward, priority_rate),
                    best_path,
                )
                print(
                    f"[SAVE] 新しいベストモデルを保存: {best_path} "
                    f"(success_rate={success_rate:.2%}, avg_reward={avg_reward:.2f}, "
                    f"priority_rate={priority_rate:.2%})"
                )

    print("[DONE] 学習が完了しました。")


if __name__ == "__main__":
    main()