# train_attacker_ai_v3.py
"""AI Ver3: 設置学習用・完全デバッグ版。

目的:
- アタッカー1人、ディフェンダー0人
- 常にスパイク保持
- サイトへ移動し、エリアを保持してPLANTを4tick連続で選んで設置
- WAITとPLANTを別アクションとして学習
- OBS_DIM=125 / N_ACTIONS=14
- Dueling DQN構造は維持するが、出力数が14なのでVer2へ直接ロードは不可

実行例:
    python train_attacker_ai_v3_site_hold_full.py --episodes 3000
    python train_attacker_ai_v3_site_hold_full.py --episodes 3000 --eval-q-trace-steps 20
    python train_attacker_ai_v3_site_hold_full.py --resume attacker_ai_v3_data/training_state_latest.pt

注意:
- Phase1では移動4方向、WAIT、PLANTだけを選択します。
- アビリティ用の出力ヘッドは未学習のまま残ります。
- 旧Ver3のcheckpointはN_ACTIONSが13なので再開には使えません。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from map_data import NEW_MAZE_STR


Position = Tuple[int, int]

# ---------------------------------------------------------------------
# Ver2互換定数
# ---------------------------------------------------------------------

ABILITY_NONE = "NONE"
ABILITY_FLASH = "FLASH"
ABILITY_SMOKE = "SMOKE"
ABILITY_RECON = "RECON"
ABILITY_HUNT = "HUNT"

ABILITY_TO_INDEX = {
    ABILITY_NONE: 0,
    ABILITY_FLASH: 1,
    ABILITY_SMOKE: 2,
    ABILITY_RECON: 3,
    ABILITY_HUNT: 4,
}

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_WAIT = 4
ACTION_PLANT = 5
ACTION_ABILITY_UP_NEAR = 6
ACTION_ABILITY_DOWN_NEAR = 7
ACTION_ABILITY_LEFT_NEAR = 8
ACTION_ABILITY_RIGHT_NEAR = 9
ACTION_ABILITY_UP_FAR = 10
ACTION_ABILITY_DOWN_FAR = 11
ACTION_ABILITY_LEFT_FAR = 12
ACTION_ABILITY_RIGHT_FAR = 13

N_ACTIONS = 14
VALID_PHASE1_ACTIONS = (
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_WAIT,
    ACTION_PLANT,
)

ACTION_NAMES = {
    0: "MOVE_UP",
    1: "MOVE_DOWN",
    2: "MOVE_LEFT",
    3: "MOVE_RIGHT",
    4: "WAIT",
    5: "PLANT",
    6: "ABILITY_UP_NEAR",
    7: "ABILITY_DOWN_NEAR",
    8: "ABILITY_LEFT_NEAR",
    9: "ABILITY_RIGHT_NEAR",
    10: "ABILITY_UP_FAR",
    11: "ABILITY_DOWN_FAR",
    12: "ABILITY_LEFT_FAR",
    13: "ABILITY_RIGHT_FAR",
}

MOVE_DELTAS: Dict[int, Position] = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}

MAX_HP = 100
ROUND_DURATION_TICKS = 120
PLANT_REQUIRED_TICKS = 4
OBS_DIM = 125

# ---------------------------------------------------------------------
# 学習設定
# ---------------------------------------------------------------------

DEFAULT_EPISODES = 3000
DEFAULT_SEED = 42

GAMMA = 0.99
LEARNING_RATE = 2.5e-4
BATCH_SIZE = 128
REPLAY_CAPACITY = 150_000
LEARNING_STARTS = 2_000
TRAIN_EVERY_STEPS = 4
TARGET_UPDATE_INTERVAL = 1_000
GRADIENT_CLIP_NORM = 10.0

EPSILON_START = 1.0
EPSILON_END = 0.03
EPSILON_DECAY_STEPS = 120_000

EVAL_INTERVAL_EPISODES = 100
EVAL_EPISODES = 50
SAVE_INTERVAL_EPISODES = 100

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "attacker_ai_v3_data"
LATEST_MODEL_PATH = MODEL_DIR / "dqn_attacker_ai_v3_latest.pt"
BEST_MODEL_PATH = MODEL_DIR / "dqn_attacker_ai_v3_best.pt"
FINAL_MODEL_PATH = MODEL_DIR / "dqn_attacker_ai_v3_final.pt"
TRAINING_STATE_PATH = MODEL_DIR / "training_state_latest.pt"
CONFIG_PATH = MODEL_DIR / "training_config.json"

# ---------------------------------------------------------------------
# 報酬
# ---------------------------------------------------------------------

R_STEP = -0.01
R_WAIT = -0.04
R_TOWARD = 1.0
R_AWAY = -1.0
R_SAME_DISTANCE = -0.03

R_OUT = -1.0
R_WALL = -0.8
R_MOVE_FAILURE_STREAK_CAP = 5
R_REVISIT = -0.5

# サイト確保・保持
R_ENTER_SITE = 12.0
R_LEAVE_SITE = -12.0
R_LEAVE_SITE_AFTER_PLANT = -2.0
R_SITE_HOLD = 0.05

# 設置
R_PLANT_PROGRESS = 0.5
R_PLANT_COMPLETE = 120.0
R_INVALID_PLANT = -0.6
R_PLANT_INTERRUPTED = -4.0

R_TIMEOUT = -60.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_grid(text: str) -> np.ndarray:
    rows = [row.strip() for row in text.strip().splitlines() if row.strip()]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("NEW_MAZE_STRの形が不正です")

    try:
        return np.asarray(
            [[int(ch) for ch in row] for row in rows],
            dtype=np.int8,
        )
    except ValueError as exc:
        raise ValueError("NEW_MAZE_STRには数字以外が含まれています") from exc


def bfs(goal: Position, grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    distance = np.full((h, w), np.inf, dtype=np.float32)

    if (
        not (0 <= goal[0] < h and 0 <= goal[1] < w)
        or grid[goal] == 1
    ):
        return distance

    queue: Deque[Position] = deque([goal])
    distance[goal] = 0.0

    while queue:
        row, col = queue.popleft()
        for dr, dc in MOVE_DELTAS.values():
            nr, nc = row + dr, col + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and grid[nr, nc] != 1
                and distance[nr, nc] > distance[row, col] + 1
            ):
                distance[nr, nc] = distance[row, col] + 1
                queue.append((nr, nc))

    return distance


def epsilon_by_steps(steps: int) -> float:
    fraction = min(max(steps / EPSILON_DECAY_STEPS, 0.0), 1.0)
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


class PlantOnlyEnv:
    """1人でサイト移動と設置だけを学習する環境。"""

    def __init__(self, seed: int = DEFAULT_SEED):
        self.grid = parse_grid(NEW_MAZE_STR)
        self.h, self.w = self.grid.shape
        self.rng = random.Random(seed)

        self.attacker_spawns = self.cells(3)
        self.plant_cells = self.cells(2)
        self.walk_cells = [
            (r, c)
            for r in range(self.h)
            for c in range(self.w)
            if self.grid[r, c] != 1
        ]

        if not self.attacker_spawns:
            raise ValueError("map_data.pyにアタッカースポーン値3がありません")
        if not self.plant_cells:
            raise ValueError("map_data.pyに設置地点値2がありません")

        self.pos: Position = self.attacker_spawns[0]
        self.target_plant: Position = self.plant_cells[0]

        self.tick = 0
        self.planting = False
        self.plant_timer = 0
        self.bomb_planted = False

        # 現在サイト内を保持しているか。
        # Phase1ではプレイヤー1人なので、現在サイト内にいることと同義。
        self.site_control = False

        # 壁・範囲外への連続移動失敗数。
        # 成功移動するまで維持し、ペナルティ倍率に使用する。
        self.move_failure_streak = 0

        self.last_action = ACTION_WAIT

        self.moved = False
        self.move_failed = False
        self.fail_reason = ""

        self.done = False
        self.success = False
        self.end_reason = ""

        self.visits: Dict[Position, int] = {}

        # 1エピソード内の報酬内訳。step()ごとに加算し、reset()で初期化する。
        self.reward_stats: Dict[str, float] = defaultdict(float)
        self.action_counts: Dict[str, int] = defaultdict(int)

        # 設置が何tick目で中断されたかを記録する。
        # 例: PLANTを3回押した後に移動した場合はキー3を1増やす。
        self.plant_interrupt_counts: Dict[int, int] = {
            depth: 0
            for depth in range(1, PLANT_REQUIRED_TICKS)
        }

    def cells(self, value: int) -> List[Position]:
        rows, cols = np.where(self.grid == value)
        return list(zip(rows.tolist(), cols.tolist()))

    def inside(self, pos: Position) -> bool:
        return 0 <= pos[0] < self.h and 0 <= pos[1] < self.w

    def walkable(self, pos: Position) -> bool:
        return self.inside(pos) and self.grid[pos] != 1

    def distance(self, start: Position, goal: Position) -> float:
        return float(bfs(goal, self.grid)[start])

    def curriculum_level(self, episode: int) -> int:
        if episode <= 300:
            return 0
        if episode <= 1000:
            return 1
        return 2

    def choose_start(self, level: int) -> Position:
        distance_map = bfs(self.target_plant, self.grid)

        if level == 0:
            candidates = [
                p
                for p in self.walk_cells
                if 2 <= distance_map[p] <= 6
            ]
        elif level == 1:
            candidates = [
                p
                for p in self.walk_cells
                if 5 <= distance_map[p] <= 14
            ]
        else:
            candidates = [
                p
                for p in self.attacker_spawns
                if np.isfinite(distance_map[p])
            ]

        if not candidates:
            candidates = [
                p
                for p in self.walk_cells
                if np.isfinite(distance_map[p])
                and p not in self.plant_cells
            ]

        if not candidates:
            raise RuntimeError("設置地点へ到達可能な開始位置がありません")

        return self.rng.choice(candidates)

    def reset(
        self,
        episode: int = 1,
        seed: Optional[int] = None,
        force_full_map: bool = False,
    ) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)

        self.target_plant = self.rng.choice(self.plant_cells)
        level = 2 if force_full_map else self.curriculum_level(episode)
        self.pos = self.choose_start(level)

        self.tick = 0
        self.planting = False
        self.plant_timer = 0
        self.bomb_planted = False
        self.site_control = self.pos in self.plant_cells
        self.move_failure_streak = 0
        self.last_action = ACTION_WAIT

        self.moved = False
        self.move_failed = False
        self.fail_reason = ""

        self.done = False
        self.success = False
        self.end_reason = ""

        self.visits = {self.pos: 1}
        self.reward_stats = defaultdict(float)
        self.action_counts = defaultdict(int)
        self.plant_interrupt_counts = {
            depth: 0
            for depth in range(1, PLANT_REQUIRED_TICKS)
        }
        return self.observation()

    def observation(self) -> np.ndarray:
        row, col = self.pos
        obj_row, obj_col = self.target_plant
        distance = self.distance(self.pos, self.target_plant)

        out: List[float] = [
            row / max(self.h - 1, 1),
            col / max(self.w - 1, 1),
            1.0,  # hp
            1.0,  # alive
            float(self.moved),
            0.0,  # blind
            0.0,  # reveal
            1.0,  # has_spike
            float(self.planting),
            min(self.plant_timer / PLANT_REQUIRED_TICKS, 1.0),
        ]

        # ability one-hot: NONE, FLASH, SMOKE, RECON, HUNT
        out += [1.0, 0.0, 0.0, 0.0, 0.0]
        out += [0.0]  # charge available

        out += [
            float(self.move_failed),
            float(self.fail_reason == "OUT"),
            float(self.fail_reason == "WALL"),
            0.0,  # ally/conflict
        ]

        out += [
            float(i == self.last_action)
            for i in range(N_ACTIONS)
        ]

        out += [
            float(
                not self.walkable((row + dr, col + dc))
            )
            for dr, dc in MOVE_DELTAS.values()
        ]

        out += [
            (obj_row - row) / max(self.h - 1, 1),
            (obj_col - col) / max(self.w - 1, 1),
            (
                1.0
                if not np.isfinite(distance)
                else min(distance / (self.h + self.w), 1.0)
            ),
        ]

        # not_planted, planted, spike_dropped, round_timer, spike_timer
        out += [
            float(not self.bomb_planted),
            float(self.bomb_planted),
            0.0,
            min(self.tick / ROUND_DURATION_TICKS, 1.0),
            0.0,
        ]

        # 味方4人分。Phase1では全て0。
        out += [0.0] * 24

        # 既存の未使用領域へPhase1用の追加状態を格納する。
        out += [
            float(self.site_control),
            min(
                self.move_failure_streak / R_MOVE_FAILURE_STREAK_CAP,
                1.0,
            ),
        ]

        # Ver2と同じ位置で76要素まで埋める。
        while len(out) < 76:
            out.append(0.0)

        if len(out) != 76:
            raise RuntimeError(
                f"敵情報前の観測次元が不正です: {len(out)}"
            )

        # 敵5人分: 5 * 8 = 40
        out += [0.0] * 40

        # smoke: own tile + 4 directions
        out += [0.0] * 5

        # defuse info
        out += [0.0] * 4

        arr = np.asarray(out, dtype=np.float32)
        if arr.shape != (OBS_DIM,):
            raise RuntimeError(
                f"OBS_DIM mismatch: actual={arr.shape}, expected={(OBS_DIM,)}"
            )
        return arr

    def step(self, action: int):
        if self.done:
            raise RuntimeError("終了済み環境へstep()が呼ばれました")

        action = int(action)
        reward = 0.0

        def add_reward(name: str, value: float) -> None:
            """報酬を合計値と内訳へ同時に加算する。"""
            nonlocal reward
            value = float(value)
            reward += value
            self.reward_stats[name] += value

        def interrupt_planting() -> None:
            """設置中なら、中断深度を記録して設置状態を解除する。"""
            if not self.planting:
                return

            depth = int(self.plant_timer)
            if depth in self.plant_interrupt_counts:
                self.plant_interrupt_counts[depth] += 1

            self.planting = False
            self.plant_timer = 0
            add_reward("PLANT_INTERRUPTED", R_PLANT_INTERRUPTED)

        add_reward("STEP", R_STEP)
        self.action_counts[ACTION_NAMES.get(action, f"UNKNOWN_{action}")] += 1

        self.tick += 1
        self.moved = False
        self.move_failed = False
        self.fail_reason = ""
        self.last_action = action

        old_pos = self.pos
        old_distance = self.distance(self.pos, self.target_plant)
        was_in_site = self.pos in self.plant_cells

        if action in MOVE_DELTAS:
            if self.planting:
                interrupt_planting()

            dr, dc = MOVE_DELTAS[action]
            new_pos = (self.pos[0] + dr, self.pos[1] + dc)

            if not self.inside(new_pos):
                self.move_failed = True
                self.fail_reason = "OUT"
                self.move_failure_streak += 1
                multiplier = min(
                    self.move_failure_streak,
                    R_MOVE_FAILURE_STREAK_CAP,
                )
                add_reward("OUT", R_OUT * multiplier)

            elif not self.walkable(new_pos):
                self.move_failed = True
                self.fail_reason = "WALL"
                self.move_failure_streak += 1
                multiplier = min(
                    self.move_failure_streak,
                    R_MOVE_FAILURE_STREAK_CAP,
                )
                add_reward("WALL", R_WALL * multiplier)

            else:
                self.pos = new_pos
                self.moved = True
                self.move_failure_streak = 0

                now_in_site = self.pos in self.plant_cells
                new_distance = self.distance(
                    self.pos,
                    self.target_plant,
                )

                if np.isfinite(old_distance) and np.isfinite(new_distance):
                    difference = old_distance - new_distance
                    if difference > 0:
                        add_reward("TOWARD", R_TOWARD * difference)
                    elif difference < 0:
                        add_reward("AWAY", R_AWAY * abs(difference))
                    else:
                        add_reward("SAME_DISTANCE", R_SAME_DISTANCE)

                # 未設置時は、サイト進入で確保報酬、退出で同額の損失。
                # そのため出入りを繰り返しても差し引き0になる。
                if not was_in_site and now_in_site:
                    self.site_control = True
                    if not self.bomb_planted:
                        add_reward("ENTER_SITE", R_ENTER_SITE)

                elif was_in_site and not now_in_site:
                    self.site_control = False
                    if self.bomb_planted:
                        add_reward(
                            "LEAVE_SITE_AFTER_PLANT",
                            R_LEAVE_SITE_AFTER_PLANT,
                        )
                    else:
                        add_reward("LEAVE_SITE", R_LEAVE_SITE)

                visit_count = self.visits.get(self.pos, 0)
                if visit_count > 0:
                    add_reward("REVISIT", R_REVISIT * visit_count)
                self.visits[self.pos] = visit_count + 1

        elif action == ACTION_WAIT:
            # WAITは何もしない。設置中なら連続PLANTが途切れる。
            if self.planting:
                interrupt_planting()
            add_reward("WAIT", R_WAIT)

        elif action == ACTION_PLANT:
            if self.pos in self.plant_cells:
                self.planting = True
                self.plant_timer += 1
                add_reward("PLANT_PROGRESS", R_PLANT_PROGRESS)

                if self.plant_timer >= PLANT_REQUIRED_TICKS:
                    self.bomb_planted = True
                    self.site_control = True
                    self.done = True
                    self.success = True
                    self.end_reason = "PLANTED"
                    add_reward("PLANT_COMPLETE", R_PLANT_COMPLETE)
            else:
                self.planting = False
                self.plant_timer = 0
                add_reward("INVALID_PLANT", R_INVALID_PLANT)

        else:
            # Phase1ではアビリティを選択対象にしないが、
            # 誤って渡された場合も安全に処理する。
            if self.planting:
                interrupt_planting()
            add_reward("INVALID_ACTION", -1.0)

        # サイトを保持している間は小さな滞在ボーナス。
        # ENTER_SITEより十分小さくし、長時間待機だけで設置を上回らない値にする。
        if self.pos in self.plant_cells:
            self.site_control = True
            add_reward("SITE_HOLD", R_SITE_HOLD)

        if not self.done and self.tick >= ROUND_DURATION_TICKS:
            self.done = True
            self.success = False
            self.end_reason = "ROUND_TIMEOUT"
            add_reward("TIMEOUT", R_TIMEOUT)

        # 内訳の合計と実際のstep報酬が一致することを検証する。
        # 浮動小数点誤差を除き、食い違った場合は実装漏れを即座に発見できる。
        cumulative_reward = float(sum(self.reward_stats.values()))

        info = {
            "success": self.success,
            "end_reason": self.end_reason,
            "tick": self.tick,
            "position": self.pos,
            "old_position": old_pos,
            "target_plant": self.target_plant,
            "plant_timer": self.plant_timer,
            "bomb_planted": self.bomb_planted,
            "site_control": self.site_control,
            "move_failure_streak": self.move_failure_streak,
            "step_reward": float(reward),
            "reward_stats": dict(self.reward_stats),
            "reward_stats_total": cumulative_reward,
            "action_counts": dict(self.action_counts),
            "plant_interrupt_counts": dict(self.plant_interrupt_counts),
        }

        return self.observation(), float(reward), self.done, info


class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_CAPACITY):
        self.capacity = capacity
        self.states = np.empty(
            (capacity, OBS_DIM),
            dtype=np.float32,
        )
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty(
            (capacity, OBS_DIM),
            dtype=np.float32,
        )
        self.dones = np.empty(capacity, dtype=np.float32)

        self.index = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        i = self.index
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        indices = np.random.randint(
            0,
            self.size,
            size=batch_size,
        )

        return (
            torch.as_tensor(
                self.states[indices],
                dtype=torch.float32,
                device=device,
            ),
            torch.as_tensor(
                self.actions[indices],
                dtype=torch.int64,
                device=device,
            ),
            torch.as_tensor(
                self.rewards[indices],
                dtype=torch.float32,
                device=device,
            ),
            torch.as_tensor(
                self.next_states[indices],
                dtype=torch.float32,
                device=device,
            ),
            torch.as_tensor(
                self.dones[indices],
                dtype=torch.float32,
                device=device,
            ),
        )


class DuelingQNetwork(nn.Module):
    """Ver2と同じ構造。state_dictを直接流用できる。"""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
    ):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.adv = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value(features)
        advantage = self.adv(features)
        return value + advantage - advantage.mean(
            dim=1,
            keepdim=True,
        )


def choose_action(
    model: DuelingQNetwork,
    observation: np.ndarray,
    epsilon: float,
    device: torch.device,
    rng: random.Random,
) -> int:
    if rng.random() < epsilon:
        return rng.choice(VALID_PHASE1_ACTIONS)

    with torch.no_grad():
        tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        q_values = model(tensor)[0]

        valid_indices = torch.as_tensor(
            VALID_PHASE1_ACTIONS,
            dtype=torch.long,
            device=device,
        )
        valid_q = q_values.index_select(0, valid_indices)
        best_local_index = int(valid_q.argmax().item())
        return int(VALID_PHASE1_ACTIONS[best_local_index])


def optimize(
    policy: DuelingQNetwork,
    target: DuelingQNetwork,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer,
    device: torch.device,
) -> Optional[float]:
    if len(replay) < max(BATCH_SIZE, LEARNING_STARTS):
        return None

    states, actions, rewards, next_states, dones = replay.sample(
        BATCH_SIZE,
        device,
    )

    current_q = policy(states).gather(
        1,
        actions.unsqueeze(1),
    ).squeeze(1)

    valid_indices = torch.as_tensor(
        VALID_PHASE1_ACTIONS,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        policy_next_all = policy(next_states)
        policy_next_valid = policy_next_all.index_select(
            1,
            valid_indices,
        )
        best_valid_local = policy_next_valid.argmax(
            dim=1,
            keepdim=True,
        )
        best_actions = valid_indices[
            best_valid_local.squeeze(1)
        ].unsqueeze(1)

        next_q = target(next_states).gather(
            1,
            best_actions,
        ).squeeze(1)

        targets = rewards + GAMMA * (1.0 - dones) * next_q

    loss = nn.functional.smooth_l1_loss(
        current_q,
        targets,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(
        policy.parameters(),
        GRADIENT_CLIP_NORM,
    )
    optimizer.step()

    return float(loss.item())


def save_model(path: Path, model: DuelingQNetwork) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def q_values_for_valid_actions(
    model: DuelingQNetwork,
    observation: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    """有効行動だけのQ値を取得する。"""
    with torch.no_grad():
        tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        q_values = model(tensor)[0]

    return {
        ACTION_NAMES[action]: float(q_values[action].item())
        for action in VALID_PHASE1_ACTIONS
    }


def print_q_values(
    title: str,
    q_values: Dict[str, float],
    chosen_action: Optional[int] = None,
) -> None:
    """Q値を高い順ではなく、行動定義順に表示する。"""
    print(f"[{title}] Q values")
    for action in VALID_PHASE1_ACTIONS:
        name = ACTION_NAMES[action]
        marker = " <- chosen" if chosen_action == action else ""
        print(f"  {name:20s} {q_values[name]:10.3f}{marker}")


def evaluate(
    model: DuelingQNetwork,
    device: torch.device,
    episodes: int = EVAL_EPISODES,
    seed: int = 10_000,
    q_trace_steps: int = 12,
) -> Dict[str, object]:
    env = PlantOnlyEnv(seed)
    rng = random.Random(seed)

    successes = 0
    total_ticks = 0
    reasons: Dict[str, int] = {}
    reward_totals: Dict[str, float] = defaultdict(float)
    action_totals: Dict[str, int] = defaultdict(int)
    interrupt_totals: Dict[int, int] = defaultdict(int)

    # 最初の評価エピソードだけ、各stepのQ値を保存する。
    q_trace: List[Dict[str, object]] = []

    was_training = model.training
    model.eval()

    for episode in range(1, episodes + 1):
        observation = env.reset(
            episode=10_000,
            seed=seed + episode,
            force_full_map=True,
        )
        done = False

        while not done:
            q_values = q_values_for_valid_actions(
                model,
                observation,
                device,
            )
            action = choose_action(
                model,
                observation,
                epsilon=0.0,
                device=device,
                rng=rng,
            )

            if episode == 1 and len(q_trace) < q_trace_steps:
                q_trace.append(
                    {
                        "tick": env.tick,
                        "position": env.pos,
                        "target_plant": env.target_plant,
                        "in_site": env.pos in env.plant_cells,
                        "planting": env.planting,
                        "plant_timer": env.plant_timer,
                        "site_control": env.site_control,
                        "move_failure_streak": env.move_failure_streak,
                        "chosen_action": action,
                        "chosen_action_name": ACTION_NAMES[action],
                        "q_values": q_values,
                    }
                )

            observation, _, done, info = env.step(action)

        for key, value in info["reward_stats"].items():
            reward_totals[key] += float(value)
        for key, value in info["action_counts"].items():
            action_totals[key] += int(value)
        for depth, count in info["plant_interrupt_counts"].items():
            interrupt_totals[int(depth)] += int(count)

        successes += int(env.success)
        total_ticks += env.tick
        reasons[env.end_reason] = reasons.get(env.end_reason, 0) + 1

    if was_training:
        model.train()

    return {
        "plant_rate": successes / episodes,
        "avg_ticks": total_ticks / episodes,
        "reasons": reasons,
        "reward_breakdown_avg": {
            key: value / episodes
            for key, value in sorted(reward_totals.items())
        },
        "action_counts_avg": {
            key: value / episodes
            for key, value in sorted(action_totals.items())
        },
        "plant_interrupt_counts_avg": {
            depth: interrupt_totals.get(depth, 0) / episodes
            for depth in range(1, PLANT_REQUIRED_TICKS)
        },
        "q_trace": q_trace,
    }



def mean_dict(items) -> Dict[str, float]:
    """辞書列について、欠けているキーを0として平均を返す。"""
    if not items:
        return {}

    keys = set()
    for item in items:
        keys.update(item.keys())

    count = len(items)
    return {
        key: sum(float(item.get(key, 0.0)) for item in items) / count
        for key in sorted(keys)
    }


def print_debug_breakdown(
    title: str,
    reward_breakdown: Dict[str, float],
    action_counts: Optional[Dict[str, float]] = None,
) -> None:
    """報酬と行動回数を読みやすい表形式で表示する。"""
    print(f"[{title}] reward breakdown (average per episode)")
    if reward_breakdown:
        for key, value in reward_breakdown.items():
            print(f"  {key:20s} {value:10.2f}")
        print(f"  {'TOTAL':20s} {sum(reward_breakdown.values()):10.2f}")
    else:
        print("  (no reward data)")

    if action_counts is not None:
        print(f"[{title}] action counts (average per episode)")
        if action_counts:
            for key, value in action_counts.items():
                print(f"  {key:20s} {value:10.2f}")
        else:
            print("  (no action data)")


def print_plant_interruptions(
    title: str,
    counts: Dict[int, float],
) -> None:
    """設置が何tick目で途切れたかを表示する。"""
    print(f"[{title}] plant interruptions (average per episode)")
    total = 0.0
    for depth in range(1, PLANT_REQUIRED_TICKS):
        value = float(counts.get(depth, 0.0))
        total += value
        print(f"  after {depth} PLANT{'s' if depth != 1 else ' ':12s} {value:10.2f}")
    print(f"  {'TOTAL':20s} {total:10.2f}")


def print_eval_q_trace(q_trace: List[Dict[str, object]]) -> None:
    """最初の評価エピソードについて、stepごとのQ値と選択行動を表示する。"""
    if not q_trace:
        print("[EVAL-Q] no trace data")
        return

    print("[EVAL-Q] first evaluation episode")
    for item in q_trace:
        print(
            "  "
            f"tick={item['tick']:3d} "
            f"pos={item['position']} "
            f"site={item['in_site']} "
            f"planting={item['planting']} "
            f"timer={item['plant_timer']} "
            f"control={item['site_control']} "
            f"fail_streak={item['move_failure_streak']} "
            f"chosen={item['chosen_action_name']}"
        )
        print_q_values(
            "EVAL-Q-STEP",
            item["q_values"],
            int(item["chosen_action"]),
        )

def train(
    episodes: int = DEFAULT_EPISODES,
    seed: int = DEFAULT_SEED,
    device_name: Optional[str] = None,
    resume: Optional[Path] = None,
    tensorboard: bool = True,
    eval_q_trace_steps: int = 12,
) -> None:
    seed_everything(seed)

    device = torch.device(
        device_name
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "phase": "plant_only_wait_plant_split",
                "obs_dim": OBS_DIM,
                "n_actions": N_ACTIONS,
                "valid_phase1_actions": list(
                    VALID_PHASE1_ACTIONS
                ),
                "action_names": ACTION_NAMES,
                "plant_required_ticks": PLANT_REQUIRED_TICKS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    env = PlantOnlyEnv(seed)

    policy = DuelingQNetwork().to(device)
    target = DuelingQNetwork().to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.AdamW(
        policy.parameters(),
        lr=LEARNING_RATE,
        eps=1e-5,
    )
    replay = ReplayBuffer()

    start_episode = 1
    global_steps = 0
    best_plant_rate = -math.inf

    if resume is not None:
        checkpoint = torch.load(
            resume,
            map_location=device,
        )

        if checkpoint.get("obs_dim") != OBS_DIM:
            raise ValueError(
                "checkpointのOBS_DIMが一致しません"
            )
        if checkpoint.get("n_actions") != N_ACTIONS:
            raise ValueError(
                "checkpointのN_ACTIONSが一致しません"
            )

        policy.load_state_dict(checkpoint["policy"])
        target.load_state_dict(checkpoint["target"])
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        start_episode = checkpoint["episode"] + 1
        global_steps = checkpoint["steps"]
        best_plant_rate = checkpoint.get(
            "best_plant_rate",
            best_plant_rate,
        )

    writer = (
        SummaryWriter(str(MODEL_DIR / "tensorboard"))
        if tensorboard and SummaryWriter is not None
        else None
    )

    if tensorboard and SummaryWriter is None:
        print(
            "[WARN] tensorboard未導入のため"
            "ログを無効化します"
        )

    rng = random.Random(seed + 99)
    recent_rewards: Deque[float] = deque(maxlen=100)
    recent_successes: Deque[float] = deque(maxlen=100)
    recent_reward_breakdowns: Deque[Dict[str, float]] = deque(maxlen=100)
    recent_action_counts: Deque[Dict[str, int]] = deque(maxlen=100)
    recent_interrupt_counts: Deque[Dict[int, int]] = deque(maxlen=100)

    print(
        f"device={device} episodes={episodes} "
        f"OBS_DIM={OBS_DIM} N_ACTIONS={N_ACTIONS}"
    )
    print(
        "Phase1 valid actions:",
        [ACTION_NAMES[a] for a in VALID_PHASE1_ACTIONS],
    )

    try:
        for episode in range(
            start_episode,
            episodes + 1,
        ):
            observation = env.reset(
                episode=episode,
                seed=seed + episode,
            )
            done = False
            total_reward = 0.0
            losses: List[float] = []

            while not done:
                epsilon = epsilon_by_steps(global_steps)
                action = choose_action(
                    policy,
                    observation,
                    epsilon,
                    device,
                    rng,
                )

                (
                    next_observation,
                    reward,
                    done,
                    info,
                ) = env.step(action)

                replay.add(
                    observation,
                    action,
                    reward,
                    next_observation,
                    done,
                )

                total_reward += reward
                global_steps += 1

                if (
                    global_steps >= LEARNING_STARTS
                    and global_steps % TRAIN_EVERY_STEPS == 0
                ):
                    loss = optimize(
                        policy,
                        target,
                        optimizer,
                        replay,
                        device,
                    )
                    if loss is not None:
                        losses.append(loss)

                if (
                    global_steps
                    % TARGET_UPDATE_INTERVAL
                    == 0
                ):
                    target.load_state_dict(
                        policy.state_dict()
                    )

                observation = next_observation

            breakdown_total = float(sum(info["reward_stats"].values()))
            if not math.isclose(
                total_reward,
                breakdown_total,
                rel_tol=1e-6,
                abs_tol=1e-5,
            ):
                raise RuntimeError(
                    "報酬合計と内訳が一致しません: "
                    f"total_reward={total_reward}, "
                    f"breakdown_total={breakdown_total}"
                )

            recent_rewards.append(total_reward)
            recent_successes.append(float(env.success))
            recent_reward_breakdowns.append(dict(info["reward_stats"]))
            recent_action_counts.append(dict(info["action_counts"]))
            recent_interrupt_counts.append(
                dict(info["plant_interrupt_counts"])
            )
            mean_loss = (
                float(np.mean(losses))
                if losses
                else 0.0
            )

            if writer is not None:
                writer.add_scalar(
                    "train/reward",
                    total_reward,
                    episode,
                )
                writer.add_scalar(
                    "train/planted",
                    float(env.success),
                    episode,
                )
                writer.add_scalar(
                    "train/loss",
                    mean_loss,
                    episode,
                )
                writer.add_scalar(
                    "train/epsilon",
                    epsilon_by_steps(global_steps),
                    episode,
                )
                writer.add_scalar(
                    "train/curriculum_level",
                    env.curriculum_level(episode),
                    episode,
                )
                for key, value in info["reward_stats"].items():
                    writer.add_scalar(
                        f"reward_component/{key}",
                        float(value),
                        episode,
                    )
                for key, value in info["action_counts"].items():
                    writer.add_scalar(
                        f"action_count/{key}",
                        int(value),
                        episode,
                    )
                for depth, count in info["plant_interrupt_counts"].items():
                    writer.add_scalar(
                        f"plant_interrupt/after_{depth}",
                        int(count),
                        episode,
                    )

            if episode % 10 == 0 or episode == start_episode:
                print(
                    f"ep {episode}/{episodes} "
                    f"plant100={np.mean(recent_successes):.3f} "
                    f"reward100={np.mean(recent_rewards):.1f} "
                    f"eps={epsilon_by_steps(global_steps):.3f} "
                    f"loss={mean_loss:.4f} "
                    f"ticks={env.tick} "
                    f"end={env.end_reason}"
                )

            if episode % 100 == 0:
                print_debug_breakdown(
                    "TRAIN100",
                    mean_dict(recent_reward_breakdowns),
                    mean_dict(recent_action_counts),
                )
                print_plant_interruptions(
                    "TRAIN100",
                    mean_dict(recent_interrupt_counts),
                )

            if episode % SAVE_INTERVAL_EPISODES == 0:
                save_model(LATEST_MODEL_PATH, policy)
                torch.save(
                    {
                        "episode": episode,
                        "steps": global_steps,
                        "best_plant_rate": best_plant_rate,
                        "obs_dim": OBS_DIM,
                        "n_actions": N_ACTIONS,
                        "policy": policy.state_dict(),
                        "target": target.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    TRAINING_STATE_PATH,
                )

            if episode % EVAL_INTERVAL_EPISODES == 0:
                result = evaluate(
                    policy,
                    device,
                    q_trace_steps=eval_q_trace_steps,
                )
                print(
                    "[EVAL]",
                    {
                        "plant_rate": result["plant_rate"],
                        "avg_ticks": result["avg_ticks"],
                        "reasons": result["reasons"],
                    },
                )
                print_debug_breakdown(
                    "EVAL",
                    result["reward_breakdown_avg"],
                    result["action_counts_avg"],
                )
                print_plant_interruptions(
                    "EVAL",
                    result["plant_interrupt_counts_avg"],
                )
                print_eval_q_trace(result["q_trace"])

                if writer is not None:
                    writer.add_scalar(
                        "eval/plant_rate",
                        result["plant_rate"],
                        episode,
                    )
                    writer.add_scalar(
                        "eval/avg_ticks",
                        result["avg_ticks"],
                        episode,
                    )
                    for depth, count in result[
                        "plant_interrupt_counts_avg"
                    ].items():
                        writer.add_scalar(
                            f"eval_plant_interrupt/after_{depth}",
                            float(count),
                            episode,
                        )
                    if result["q_trace"]:
                        first_q = result["q_trace"][0]["q_values"]
                        for action_name, q_value in first_q.items():
                            writer.add_scalar(
                                f"eval_initial_q/{action_name}",
                                float(q_value),
                                episode,
                            )

                plant_rate = float(result["plant_rate"])
                if plant_rate > best_plant_rate:
                    best_plant_rate = plant_rate
                    save_model(BEST_MODEL_PATH, policy)
                    print(
                        "[BEST] plant_rate=",
                        best_plant_rate,
                    )

        save_model(FINAL_MODEL_PATH, policy)
        save_model(LATEST_MODEL_PATH, policy)

    except KeyboardInterrupt:
        print("中断されたため最新モデルを保存します")
        save_model(LATEST_MODEL_PATH, policy)

    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=None,
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
    )
    parser.add_argument(
        "--eval-q-trace-steps",
        type=int,
        default=12,
        help=(
            "評価1エピソード目で表示するstepごとのQ値数。"
            "0でQトレース表示を無効化します"
        ),
    )

    args = parser.parse_args()

    if args.resume is not None and not args.resume.is_file():
        raise SystemExit(
            f"resume file not found: {args.resume}"
        )

    train(
        episodes=args.episodes,
        seed=args.seed,
        device_name=args.device,
        resume=args.resume,
        tensorboard=not args.no_tensorboard,
        eval_q_trace_steps=max(args.eval_q_trace_steps, 0),
    )


if __name__ == "__main__":
    main()