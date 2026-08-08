"""AI Ver6 Recon: Flash学習済み方策を引き継ぎ、Recon行動を学習する1vs2環境。

構成:
- 学習対象: アタッカー1人（Reconロール）
- 対戦相手: 解除役／カバー役に分かれるルールベースディフェンダー2人
- アクション: 4方向移動 / WAIT / PLANT / Recon 8候補
- OBS_DIM=131 / N_ACTIONS=14を維持
- Reconは近距離3マス／遠距離6マスへ投射し、壁手前または最大距離で起爆
- 起爆中心から9×9（Chebyshev半径4）を5tickリビール、2チャージ
- リビールは攻撃側の情報取得だけに作用し、射撃そのものにはLOSが必要
- --flash-modelでFlashベストモデルを読み込み、Recon専用Headだけ0初期化して学習

整理方針:
- Ability共通処理（投射地点・有効行動）とRecon固有処理をまとまったメソッド群に配置
- 既存Flash Head（ability_adv_adapter）は固定して保持
- 新規Recon Head（recon_adv_adapter）の2048パラメータだけ更新
- Recon QはFlash v2で成功した mean(normal Q)-cost+delta 方式

実行例:
    python train_attacker_ai_v6_recon.py --episodes 1000 --flash-model attacker_ai_v5_flash_v2_data/dqn_attacker_ai_v5_flash_v2_best.pt
    python train_attacker_ai_v6_recon.py --resume attacker_ai_v6_recon_data/training_state_latest.pt
    python train_attacker_ai_v6_recon.py --flash-model attacker_ai_v5_flash_v2_data/dqn_attacker_ai_v5_flash_v2_best.pt --eval
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

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
ABILITY_RECON = "RECON"
ABILITY_SMOKE = "SMOKE"
ABILITY_RECON = "RECON"
ABILITY_HUNT = "HUNT"

ABILITY_TO_INDEX = {
    ABILITY_NONE: 0,
    ABILITY_RECON: 1,
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
VALID_PHASE1_ACTIONS = tuple(range(N_ACTIONS))

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
PHASE2_OBS_DIM = 125
PHASE3_EXTRA_OBS = 6
OBS_DIM = PHASE2_OBS_DIM + PHASE3_EXTRA_OBS

# ---------------------------------------------------------------------
# 学習設定
# ---------------------------------------------------------------------

DEFAULT_EPISODES = 6000
DEFAULT_SEED = 42

GAMMA = 0.99
LEARNING_RATE = 5.0e-5
BATCH_SIZE = 128
REPLAY_CAPACITY = 250_000
LEARNING_STARTS = 2_000
TRAIN_EVERY_STEPS = 4
TARGET_UPDATE_INTERVAL = 1_000
GRADIENT_CLIP_NORM = 10.0

EPSILON_START = 1.0
EPSILON_END = 0.03
EPSILON_DECAY_STEPS = 220_000

# Phase2.7からの転移学習では、完成済み方策を壊さないよう探索率を抑える。
TRANSFER_EPSILON_START = 0.10
TRANSFER_EPSILON_END = 0.02
TRANSFER_EPSILON_DECAY_STEPS = 50_000

EVAL_INTERVAL_EPISODES = 100
EVAL_EPISODES = 500
SAVE_INTERVAL_EPISODES = 100

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "attacker_ai_v6_recon_data"
LATEST_MODEL_PATH = MODEL_DIR / "dqn_attacker_ai_v6_recon_latest.pt"
BEST_MODEL_PATH = MODEL_DIR / "dqn_attacker_ai_v6_recon_best.pt"
FINAL_MODEL_PATH = MODEL_DIR / "dqn_attacker_ai_v6_recon_final.pt"
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


# ---------------------------------------------------------------------
# Phase2.6: 1vs2戦闘設定
# ---------------------------------------------------------------------

TEAM_ATTACKER = "ATTACKER"
TEAM_DEFENDER = "DEFENDER"

SPIKE_DETONATION_TICKS = 35
DEFUSE_REQUIRED_TICKS = 7

# Phase6 Recon
RECON_MAX_CHARGES = 2
RECON_REVEAL_TICKS = 5
RECON_RADIUS = 4
RECON_NEAR_DISTANCE = 3
RECON_FAR_DISTANCE = 6
RECON_OUTCOME_WINDOW_TICKS = 7
RECON_Q_BASELINE_COST = 2.0


ATTACKER_ACCURACY = 0.62
DEFENDER_ACCURACY = 0.55
ATTACKER_HEADSHOT_RATE = 0.28
DEFENDER_HEADSHOT_RATE = 0.22
BODY_DAMAGE = 40
HEAD_DAMAGE = 100

# 移動したtickは命中率を下げる。
MOVING_ACCURACY_MULTIPLIER = 0.50

# 報酬: 設置単体よりも最終勝利を重視する。
R_DAMAGE_DEALT = 0.08
R_DAMAGE_TAKEN = -0.06
R_KILL = 40.0
R_DEATH = -45.0
R_WIN = 80.0
R_LOSS = -80.0
R_DETONATE = 25.0
R_DEFUSE_START = -2.0
R_DEFUSE_PROGRESS = -1.0
R_DEFUSER_KILL = 12.0

R_RECON_USE = -0.05
R_RECON_INVALID = -0.8
R_RECON_REVEAL = 2.0
R_RECON_MULTI_REVEAL = 1.5
R_RECON_POINTLESS = 0.0
R_RECON_REPEAT = -0.2
R_RECON_DAMAGE_BONUS = 0.08
R_RECON_KILL_BONUS = 10.0
R_RECON_ENABLE_PLANT = 3.0
R_RECON_NEW_CELL = 0.01
# Recon使用後、情報を活用して結果につなげた場合の小さな追加報酬。
# 敵を発見できなかったReconも価値を持つため、対象がrevealedかどうかは条件にしない。
R_RECON_FOLLOWUP_KILL = 6.0
R_RECON_SAFE_REPOSITION = 0.5
R_RECON_DEFUSER_FOUND = 5.0
R_WAIT_STREAK_STEP = -0.03
WAIT_STREAK_CAP = 10


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

    if not (0 <= goal[0] < h and 0 <= goal[1] < w) or grid[goal] == 1:
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


def chebyshev(a: Position, b: Position) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def line_cells(a: Position, b: Position) -> List[Position]:
    """Bresenham法でaからbまでのセルを返す。"""
    r0, c0 = a
    r1, c1 = b
    cells: List[Position] = []

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1

    if dc > dr:
        error = dc // 2
        while c0 != c1:
            cells.append((r0, c0))
            error -= dr
            if error < 0:
                r0 += sr
                error += dc
            c0 += sc
    else:
        error = dr // 2
        while r0 != r1:
            cells.append((r0, c0))
            error -= dc
            if error < 0:
                c0 += sc
                error += dr
            r0 += sr

    cells.append((r1, c1))
    return cells


def has_los(
    a: Position,
    b: Position,
    grid: np.ndarray,
    smoke_cells: Optional[set[Position]] = None,
) -> bool:
    """壁またはSmokeに遮られていない場合だけTrue。隣接セルは常に視認可能。"""
    if chebyshev(a, b) <= 1:
        return True

    cells = line_cells(a, b)
    smoke_cells = smoke_cells or set()

    # 壁は始点・終点を除外。Smokeは始点・終点を含め、通過した時点で遮断。
    for row, col in cells[1:-1]:
        if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
            return False
        if grid[row, col] == 1:
            return False
    return not any(cell in smoke_cells for cell in cells)


def linear_epsilon_schedule(
    steps: int,
    start: float,
    end: float,
    decay_steps: int,
) -> float:
    if decay_steps <= 0:
        return float(end)
    fraction = min(max(steps / decay_steps, 0.0), 1.0)
    return float(start + fraction * (end - start))


def epsilon_by_steps(steps: int) -> float:
    return linear_epsilon_schedule(
        steps,
        EPSILON_START,
        EPSILON_END,
        EPSILON_DECAY_STEPS,
    )


def transfer_epsilon_by_steps(
    steps: int,
    start: float = TRANSFER_EPSILON_START,
    end: float = TRANSFER_EPSILON_END,
    decay_steps: int = TRANSFER_EPSILON_DECAY_STEPS,
) -> float:
    return linear_epsilon_schedule(steps, start, end, decay_steps)


class Team(str, Enum):
    ATTACKER = TEAM_ATTACKER
    DEFENDER = TEAM_DEFENDER


@dataclass(frozen=True)
class CharacterStats:
    name: str
    accuracy: float
    headshot_rate: float
    reaction: float = 1.0


@dataclass
class Player:
    player_id: str
    team: Team
    stats: CharacterStats
    pos: Position
    hp: int = MAX_HP
    alive: bool = True
    moved: bool = False
    is_defusing: bool = False
    last_seen_enemy: Optional[Position] = None
    recon_reveal_ticks: int = 0

    def reset(self, pos: Position) -> None:
        self.pos = pos
        self.hp = MAX_HP
        self.alive = True
        self.moved = False
        self.is_defusing = False
        self.last_seen_enemy = None
        self.recon_reveal_ticks = 0

    @property
    def revealed(self) -> bool:
        return self.recon_reveal_ticks > 0

    def take_damage(self, damage: int) -> int:
        if not self.alive:
            return 0
        actual = min(max(int(damage), 0), self.hp)
        self.hp -= actual
        if self.hp <= 0:
            self.hp = 0
            self.alive = False
            self.is_defusing = False
        return actual


class PlayerRoster:
    def __init__(self, players: Iterable[Player]):
        self.players = list(players)
        if not self.players:
            raise ValueError("PlayerRosterには1人以上必要です")
        ids = [p.player_id for p in self.players]
        if len(ids) != len(set(ids)):
            raise ValueError("player_idが重複しています")

    @property
    def attackers(self) -> List[Player]:
        return [p for p in self.players if p.team == Team.ATTACKER]

    @property
    def defenders(self) -> List[Player]:
        return [p for p in self.players if p.team == Team.DEFENDER]

    def alive(self, team: Optional[Team] = None) -> List[Player]:
        return [p for p in self.players if p.alive and (team is None or p.team == team)]

    def enemies_of(self, player: Player) -> List[Player]:
        return [p for p in self.players if p.alive and p.team != player.team]

    def occupied_positions(
        self,
        exclude: Optional[Player] = None,
    ) -> set[Position]:
        return {p.pos for p in self.players if p.alive and p is not exclude}

    def shooting_order(
        self,
        rng: random.Random,
        use_reaction: bool = False,
    ) -> List[Player]:
        shooters = self.alive()
        if not use_reaction:
            rng.shuffle(shooters)
            return shooters

        buckets: Dict[float, List[Player]] = defaultdict(list)
        for player in shooters:
            buckets[player.stats.reaction].append(player)

        ordered: List[Player] = []
        for reaction in sorted(buckets, reverse=True):
            group = buckets[reaction]
            rng.shuffle(group)
            ordered.extend(group)
        return ordered


class Phase3OneVsTwoEnv:
    """学習アタッカー1人対、ルールベースディフェンダー2人。"""

    def __init__(self, seed: int = DEFAULT_SEED):
        self.grid = parse_grid(NEW_MAZE_STR)
        self.h, self.w = self.grid.shape
        self.rng = random.Random(seed)

        self.attacker_spawns = self.cells(3)
        self.defender_spawns = self.cells(4)
        self.plant_cells = self.cells(2)
        self.walk_cells = [
            (r, c) for r in range(self.h) for c in range(self.w) if self.grid[r, c] != 1
        ]

        if not self.attacker_spawns:
            raise ValueError("map_data.pyにアタッカースポーン値3がありません")
        if not self.plant_cells:
            raise ValueError("map_data.pyに設置地点値2がありません")
        if not self.defender_spawns:
            self.defender_spawns = [
                pos for pos in self.walk_cells if pos[0] < self.h // 2
            ]

        self.distance_maps = {cell: bfs(cell, self.grid) for cell in self.plant_cells}

        attacker_stats = CharacterStats(
            "Attacker",
            ATTACKER_ACCURACY,
            ATTACKER_HEADSHOT_RATE,
        )
        defender_stats = CharacterStats(
            "Defender",
            DEFENDER_ACCURACY,
            DEFENDER_HEADSHOT_RATE,
        )

        self.attacker = Player(
            "A0",
            Team.ATTACKER,
            attacker_stats,
            self.attacker_spawns[0],
        )
        self.defenders = [
            Player(
                f"D{i}",
                Team.DEFENDER,
                defender_stats,
                self.defender_spawns[min(i, len(self.defender_spawns) - 1)],
            )
            for i in range(2)
        ]
        self.roster = PlayerRoster([self.attacker, *self.defenders])

        self.target_plant = self.plant_cells[0]
        self.tick = 0
        self.done = False
        self.success = False
        self.winner: Optional[str] = None
        self.end_reason = ""

        self.planting = False
        self.plant_timer = 0
        self.bomb_planted = False
        self.spike_pos: Optional[Position] = None
        self.spike_timer = 0
        self.defuse_timer = 0
        self.defuser_id: Optional[str] = None
        self.assigned_defuser_id: Optional[str] = None

        self.site_control = False
        self.move_failure_streak = 0
        self.last_action = ACTION_WAIT
        self.move_failed = False
        self.fail_reason = ""

        self.visits: Dict[Position, int] = {}
        self.reward_stats: Dict[str, float] = defaultdict(float)
        self.action_counts: Dict[str, int] = defaultdict(int)
        self.plant_interrupt_counts = {
            depth: 0 for depth in range(1, PLANT_REQUIRED_TICKS)
        }
        self.defender_last_seen = {d.player_id: None for d in self.defenders}
        self.recon_charges = RECON_MAX_CHARGES
        self.wait_streak = 0
        self.pending_recon_outcomes: List[Dict[str, object]] = []

    @property
    def pos(self) -> Position:
        return self.attacker.pos

    @property
    def hp(self) -> int:
        return self.attacker.hp

    @property
    def alive(self) -> bool:
        return self.attacker.alive

    @property
    def moved(self) -> bool:
        return self.attacker.moved

    @property
    def defender_pos(self) -> Position:
        return self.defenders[0].pos

    @property
    def defender_hp(self) -> int:
        return self.defenders[0].hp

    @property
    def defender_alive(self) -> bool:
        return self.defenders[0].alive

    @property
    def defender_moved(self) -> bool:
        return self.defenders[0].moved

    def cells(self, value: int) -> List[Position]:
        rows, cols = np.where(self.grid == value)
        return list(zip(rows.tolist(), cols.tolist()))

    @staticmethod
    def curriculum_level(episode: int) -> int:
        return 0 if episode < 800 else 1

    def inside(self, pos: Position) -> bool:
        return 0 <= pos[0] < self.h and 0 <= pos[1] < self.w

    def walkable(self, pos: Position) -> bool:
        return self.inside(pos) and self.grid[pos] != 1

    def distance(self, start: Position, goal: Position) -> float:
        if goal in self.distance_maps:
            return float(self.distance_maps[goal][start])
        return float(bfs(goal, self.grid)[start])

    def random_reachable_spawn(
        self,
        pool: List[Position],
        avoid: Optional[set[Position]] = None,
    ) -> Position:
        avoid = avoid or set()
        candidates = [
            pos
            for pos in pool
            if pos not in avoid and np.isfinite(self.distance(pos, self.target_plant))
        ]
        if not candidates:
            candidates = [pos for pos in self.walk_cells if pos not in avoid]
        if not candidates:
            raise RuntimeError("重複しないスポーン地点を確保できません")
        return self.rng.choice(candidates)

    def reset(
        self,
        episode: int = 0,
        seed: Optional[int] = None,
        force_full_map: bool = False,
    ) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)

        self.target_plant = self.rng.choice(self.plant_cells)

        if force_full_map or episode >= 800:
            attacker_pool = self.attacker_spawns
        else:
            distance_map = self.distance_maps[self.target_plant]
            finite = [
                pos
                for pos in self.walk_cells
                if np.isfinite(distance_map[pos]) and 5 <= distance_map[pos] <= 16
            ]
            attacker_pool = finite or self.attacker_spawns

        occupied: set[Position] = set()
        attacker_pos = self.random_reachable_spawn(attacker_pool, occupied)
        occupied.add(attacker_pos)
        self.attacker.reset(attacker_pos)

        for defender in self.defenders:
            defender_pos = self.random_reachable_spawn(
                self.defender_spawns,
                occupied,
            )
            occupied.add(defender_pos)
            defender.reset(defender_pos)

        self.tick = 0
        self.done = False
        self.success = False
        self.winner = None
        self.end_reason = ""

        self.planting = False
        self.plant_timer = 0
        self.bomb_planted = False
        self.spike_pos = None
        self.spike_timer = 0
        self.defuse_timer = 0
        self.defuser_id = None
        self.assigned_defuser_id = None

        self.site_control = self.attacker.pos in self.plant_cells
        self.move_failure_streak = 0
        self.last_action = ACTION_WAIT
        self.move_failed = False
        self.fail_reason = ""

        self.visits = {self.attacker.pos: 1}
        self.reward_stats = defaultdict(float)
        self.action_counts = defaultdict(int)
        self.plant_interrupt_counts = {
            depth: 0 for depth in range(1, PLANT_REQUIRED_TICKS)
        }
        self.defender_last_seen = {d.player_id: None for d in self.defenders}
        self.recon_charges = RECON_MAX_CHARGES
        self.wait_streak = 0
        self.pending_recon_outcomes = []
        self.recon_explored_cells = {self.attacker.pos}
        return self.observation()

    def has_los(self, a: Position, b: Position, grid: Optional[np.ndarray] = None) -> bool:
        return has_los(a, b, self.grid, set())

    def ability_target(self, action: int) -> Position:
        mapping = {
            ACTION_ABILITY_UP_NEAR: (-1, 0, RECON_NEAR_DISTANCE),
            ACTION_ABILITY_DOWN_NEAR: (1, 0, RECON_NEAR_DISTANCE),
            ACTION_ABILITY_LEFT_NEAR: (0, -1, RECON_NEAR_DISTANCE),
            ACTION_ABILITY_RIGHT_NEAR: (0, 1, RECON_NEAR_DISTANCE),
            ACTION_ABILITY_UP_FAR: (-1, 0, RECON_FAR_DISTANCE),
            ACTION_ABILITY_DOWN_FAR: (1, 0, RECON_FAR_DISTANCE),
            ACTION_ABILITY_LEFT_FAR: (0, -1, RECON_FAR_DISTANCE),
            ACTION_ABILITY_RIGHT_FAR: (0, 1, RECON_FAR_DISTANCE),
        }
        if action not in mapping:
            return self.attacker.pos
        dr, dc, distance = mapping[action]
        current = self.attacker.pos
        for step in range(1, distance + 1):
            probe = (self.attacker.pos[0] + dr * step, self.attacker.pos[1] + dc * step)
            if not self.walkable(probe):
                break
            current = probe
        return current

    def recon_can_reach(self, center: Position, defender: Player) -> bool:
        return bool(defender.alive and chebyshev(center, defender.pos) <= RECON_RADIUS)

    def recon_targets_for_action(self, action: int) -> List[Player]:
        """実際にRecon範囲へ入る敵。効果解決専用で、行動マスクには使わない。"""

        center = self.ability_target(action)
        return [
            defender
            for defender in self.roster.alive(Team.DEFENDER)
            if self.recon_can_reach(center, defender)
        ]

    def predicted_recon_target_count(self, action: int) -> int:
        """視認中または最後に確認した位置だけから命中候補数を推定する。"""
        center = self.ability_target(action)
        count = 0
        for defender in self.roster.alive(Team.DEFENDER):
            if self.player_visible(self.attacker, defender):
                predicted_pos = defender.pos
            else:
                predicted_pos = self.defender_last_seen.get(defender.player_id)
            if predicted_pos is None:
                continue
            if chebyshev(center, predicted_pos) > RECON_RADIUS:
                continue
            if not has_los(center, predicted_pos, self.grid, set()):
                continue
            if defender.recon_reveal_ticks <= 1:
                count += 1
        return count

    def recon_tactically_available(self) -> bool:
        return bool(self.attacker.alive and self.recon_charges > 0)

    def recon_action_available(self, action: int) -> bool:
        if not self.recon_tactically_available():
            return False
        return self.ability_target(action) != self.attacker.pos

    def valid_actions(self) -> Tuple[int, ...]:
        base = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_WAIT, ACTION_PLANT]
        for action in range(ACTION_ABILITY_UP_NEAR, ACTION_ABILITY_RIGHT_FAR + 1):
            if self.recon_action_available(action):
                base.append(action)
        return tuple(base)

    def tick_reveals(self) -> None:
        for defender in self.defenders:
            if defender.recon_reveal_ticks > 0:
                defender.recon_reveal_ticks -= 1

    def handle_recon(self, action: int, add_reward) -> None:
        self.interrupt_planting(add_reward)
        if self.recon_charges <= 0 or not self.recon_action_available(action):
            add_reward("RECON_INVALID", R_RECON_INVALID)
            return

        center = self.ability_target(action)

        # Reconで新しく確認できた歩行可能セルに情報取得報酬を与える。
        newly_seen = 0
        for rr in range(center[0] - RECON_RADIUS, center[0] + RECON_RADIUS + 1):
            for cc in range(center[1] - RECON_RADIUS, center[1] + RECON_RADIUS + 1):
                cell = (rr, cc)
                if not self.inside(cell) or not self.walkable(cell):
                    continue
                if cell not in self.recon_explored_cells:
                    self.recon_explored_cells.add(cell)
                    newly_seen += 1

        if newly_seen > 0:
            add_reward(
                "RECON_NEW_CELL",
                newly_seen * R_RECON_NEW_CELL,
            )

        targets = self.recon_targets_for_action(action)
        fresh_targets = [d for d in targets if d.recon_reveal_ticks <= 1]

        # 最後に見た位置へ投げた場合は外れることもある。投射自体は成立し、チャージを消費する。
        self.recon_charges -= 1
        add_reward("RECON_USE", R_RECON_USE)
        add_reward("RECON_REVEAL", R_RECON_REVEAL * len(fresh_targets))
        if len(fresh_targets) >= 2:
            add_reward("RECON_MULTI_REVEAL", R_RECON_MULTI_REVEAL)

        affected_ids: set[str] = set()
        found_defuser = False
        for defender in fresh_targets:
            affected_ids.add(defender.player_id)
            found_defuser = found_defuser or defender.is_defusing
            defender.recon_reveal_ticks = max(
                defender.recon_reveal_ticks,
                RECON_REVEAL_TICKS,
            )

        # Reconは解除を止めず、解除役を発見した情報にだけ報酬を与える。
        if found_defuser:
            add_reward("RECON_DEFUSER_FOUND", R_RECON_DEFUSER_FOUND)

        self.pending_recon_outcomes.append(
            {
                "expires_tick": self.tick + RECON_OUTCOME_WINDOW_TICKS,
                "affected_ids": affected_ids,
                "plant_rewarded": False,
                "kill_rewarded": False,
                "safe_reposition_rewarded": False,
                "bomb_planted_at_use": self.bomb_planted,
                "attacker_hp_at_use": self.attacker.hp,
                "attacker_pos_at_use": self.attacker.pos,
                "moved_after_use": False,
                # 敵を検出したかどうかだけでReconの価値を決めない。
                "positive": bool(affected_ids),
            }
        )

    def evaluate_recon_outcomes(self, add_reward) -> None:
        remaining: List[Dict[str, object]] = []
        for event in self.pending_recon_outcomes:
            if (
                self.bomb_planted
                and not bool(event["bomb_planted_at_use"])
                and not bool(event["plant_rewarded"])
            ):
                add_reward("RECON_ENABLE_PLANT", R_RECON_ENABLE_PLANT)
                event["plant_rewarded"] = True
                event["positive"] = True

            if self.tick >= int(event["expires_tick"]):
                if (
                    bool(event["moved_after_use"])
                    and self.attacker.alive
                    and self.attacker.hp >= int(event["attacker_hp_at_use"])
                    and not bool(event["safe_reposition_rewarded"])
                ):
                    add_reward(
                        "RECON_SAFE_REPOSITION",
                        R_RECON_SAFE_REPOSITION,
                    )
                    event["safe_reposition_rewarded"] = True
                    event["positive"] = True

                # R_RECON_POINTLESSは0.0。敵がいなかった確認も情報なので罰しない。
                if not bool(event["positive"]):
                    add_reward("RECON_POINTLESS", R_RECON_POINTLESS)
            else:
                remaining.append(event)
        self.pending_recon_outcomes = remaining

    def player_visible(self, observer: Player, target: Player) -> bool:
        if not observer.alive or not target.alive:
            return False
        if observer.team == Team.ATTACKER and target.team == Team.DEFENDER and target.revealed:
            return True
        return self.has_los(observer.pos, target.pos, self.grid)

    def visible_defenders(self) -> List[Player]:
        visible = [d for d in self.defenders if self.player_visible(self.attacker, d)]
        for defender in visible:
            self.defender_last_seen[defender.player_id] = defender.pos
        return visible

    def defender_visible(self) -> bool:
        return bool(self.visible_defenders())

    def enemy_observation_block(
        self,
        defender: Player,
        visible: bool,
    ) -> List[float]:
        r, c = self.attacker.pos
        remembered = self.defender_last_seen.get(defender.player_id)
        enemy_pos = defender.pos if visible else remembered
        if enemy_pos is None:
            return [0.0] * 12

        return [
            float(defender.alive),
            float(visible),
            (enemy_pos[0] - r) / max(self.h - 1, 1),
            (enemy_pos[1] - c) / max(self.w - 1, 1),
            defender.hp / MAX_HP,
            min(
                chebyshev(self.attacker.pos, enemy_pos) / max(self.h, self.w),
                1.0,
            ),
            float(visible),
            float(defender.is_defusing),
            (
                min(self.defuse_timer / DEFUSE_REQUIRED_TICKS, 1.0)
                if self.defuser_id == defender.player_id
                else 0.0
            ),
            float(self.bomb_planted),
            float(self.attacker.hp < 50),
            float(defender.hp < 50),
        ]

    def enemy_memory_block(
        self,
        defender: Player,
        visible: bool,
    ) -> List[float]:
        r, c = self.attacker.pos
        remembered = self.defender_last_seen.get(defender.player_id)
        enemy_pos = defender.pos if visible else remembered
        if enemy_pos is None:
            return [0.0] * 8

        return [
            float(defender.alive),
            float(visible),
            float(not visible),
            (enemy_pos[0] - r) / max(self.h - 1, 1),
            (enemy_pos[1] - c) / max(self.w - 1, 1),
            defender.hp / MAX_HP,
            0.0 if visible else 0.5,
            float(defender.is_defusing),
        ]

    def observation(self) -> np.ndarray:
        r, c = self.attacker.pos
        visible_ids = {d.player_id for d in self.visible_defenders()}

        target = (
            self.spike_pos
            if self.bomb_planted and self.spike_pos is not None
            else self.target_plant
        )
        distance_to_target = self.distance(self.attacker.pos, target)

        out: List[float] = [
            r / max(self.h - 1, 1),
            c / max(self.w - 1, 1),
            self.attacker.hp / MAX_HP,
            float(self.attacker.alive),
            float(self.attacker.moved),
            0.0,
            0.0,
            float(not self.bomb_planted),
            float(self.planting),
            min(self.plant_timer / PLANT_REQUIRED_TICKS, 1.0),
        ]
        # アタッカーはReconロール。既存125次元内の予約領域を使用する。
        out += [0.0, 0.0, 0.0, 1.0, 0.0]
        out += [self.recon_charges / max(RECON_MAX_CHARGES, 1)]
        out += [
            float(self.move_failed),
            float(self.fail_reason == "OUT"),
            float(self.fail_reason == "WALL"),
            float(self.fail_reason == "OCCUPIED"),
        ]
        out += [float(action == self.last_action) for action in range(N_ACTIONS)]

        occupied = self.roster.occupied_positions(exclude=self.attacker)
        for dr, dc in MOVE_DELTAS.values():
            next_pos = (r + dr, c + dc)
            out.append(float(not self.walkable(next_pos) or next_pos in occupied))

        out += [
            (target[0] - r) / max(self.h - 1, 1),
            (target[1] - c) / max(self.w - 1, 1),
            (
                1.0
                if not np.isfinite(distance_to_target)
                else min(
                    distance_to_target / (self.h + self.w),
                    1.0,
                )
            ),
        ]
        out += [
            float(not self.bomb_planted),
            float(self.bomb_planted),
            0.0,
            min(self.tick / ROUND_DURATION_TICKS, 1.0),
            (
                min(self.spike_timer / SPIKE_DETONATION_TICKS, 1.0)
                if self.bomb_planted
                else 0.0
            ),
        ]

        for defender in self.defenders:
            out += self.enemy_observation_block(
                defender,
                defender.player_id in visible_ids,
            )

        out += [
            float(self.site_control),
            min(
                self.move_failure_streak / R_MOVE_FAILURE_STREAK_CAP,
                1.0,
            ),
        ]

        while len(out) < 76:
            out.append(0.0)

        for defender in self.defenders:
            out += self.enemy_memory_block(
                defender,
                defender.player_id in visible_ids,
            )
        ability_actions = list(range(ACTION_ABILITY_UP_NEAR, ACTION_ABILITY_RIGHT_FAR + 1))
        recon_features: List[float] = [
            min(self.predicted_recon_target_count(action) / 2.0, 1.0)
            for action in ability_actions
        ]
        # 自分と隣接4方向のブラインド状態。味方無効なので常に0。
        recon_features += [0.0] * 5
        recon_features += [
            float(self.recon_action_available(action))
            for action in ability_actions
        ]
        alive_defenders_for_recon = self.roster.alive(Team.DEFENDER)
        recon_features += [
            (
                sum(float(defender.revealed) for defender in alive_defenders_for_recon)
                / max(len(alive_defenders_for_recon), 1)
            ),
            max((defender.recon_reveal_ticks for defender in alive_defenders_for_recon), default=0)
            / max(RECON_REVEAL_TICKS, 1),
            self.recon_charges / max(RECON_MAX_CHARGES, 1),
        ]
        if len(recon_features) != 24:
            raise RuntimeError(f"recon feature mismatch: {len(recon_features)}")
        out += recon_features

        alive_defenders = self.roster.alive(Team.DEFENDER)
        out.append(
            float(
                any(
                    self.has_los(self.attacker.pos, d.pos, self.grid)
                    for d in alive_defenders
                )
            )
        )
        for dr, dc in MOVE_DELTAS.values():
            probe = (r + dr, c + dc)
            out.append(
                float(
                    self.walkable(probe)
                    and any(self.has_los(probe, d.pos, self.grid) for d in alive_defenders)
                )
            )

        defuser = self.get_defuser()
        out += [
            float(defuser is not None and self.defuse_timer > 0),
            min(self.defuse_timer / DEFUSE_REQUIRED_TICKS, 1.0),
            ((defuser.pos[0] - r) / max(self.h - 1, 1) if defuser is not None else 0.0),
            ((defuser.pos[1] - c) / max(self.w - 1, 1) if defuser is not None else 0.0),
        ]

        # Phase3.0戦術観測。
        # 必ずPhase2.7の125要素の「末尾」に追加し、既存特徴の添字を維持する。
        if len(out) != PHASE2_OBS_DIM:
            raise RuntimeError(
                "Phase2 observation prefix mismatch: "
                f"{len(out)}, expected {PHASE2_OBS_DIM}"
            )

        round_remaining = (
            max(
                ROUND_DURATION_TICKS - self.tick,
                0,
            )
            / ROUND_DURATION_TICKS
        )

        spike_remaining = (
            max(self.spike_timer, 0) / SPIKE_DETONATION_TICKS
            if self.bomb_planted
            else 0.0
        )

        alive_defender_ratio = len(self.roster.alive(Team.DEFENDER)) / max(
            len(self.defenders), 1
        )

        active_defuser = self.get_defuser()
        defuse_active = float(active_defuser is not None and self.defuse_timer > 0)

        if self.bomb_planted and self.spike_pos is not None:
            spike_distance_raw = self.distance(
                self.attacker.pos,
                self.spike_pos,
            )
            spike_distance = (
                1.0
                if not np.isfinite(spike_distance_raw)
                else min(
                    spike_distance_raw / (self.h + self.w),
                    1.0,
                )
            )
        else:
            spike_distance = 0.0

        out += [
            float(round_remaining),
            float(spike_remaining),
            float(alive_defender_ratio),
            defuse_active,
            float(spike_distance),
            float(self.bomb_planted),
        ]

        array = np.asarray(out, dtype=np.float32)
        if array.shape != (OBS_DIM,):
            raise RuntimeError(
                f"OBS_DIM mismatch: {array.shape}, expected {(OBS_DIM,)}"
            )
        return array

    def best_move_toward(
        self,
        start: Position,
        target: Position,
        occupied: Optional[set[Position]] = None,
    ) -> Optional[int]:
        occupied = occupied or set()
        distance_map = bfs(target, self.grid)
        choices: List[Tuple[float, int]] = []

        for action, (dr, dc) in MOVE_DELTAS.items():
            pos = (start[0] + dr, start[1] + dc)
            if not self.walkable(pos) or pos in occupied:
                continue
            choices.append((float(distance_map[pos]), action))

        if not choices:
            return None

        best_distance = min(value for value, _ in choices)
        best_actions = [action for value, action in choices if value == best_distance]
        return self.rng.choice(best_actions)

    def ensure_postplant_roles(self) -> None:
        """設置後の解除役を決める。担当が死亡した場合だけ再選出する。"""
        if not self.bomb_planted or self.spike_pos is None:
            self.assigned_defuser_id = None
            return

        alive_defenders = self.roster.alive(Team.DEFENDER)
        if not alive_defenders:
            self.assigned_defuser_id = None
            return

        current = next(
            (
                defender
                for defender in alive_defenders
                if defender.player_id == self.assigned_defuser_id
            ),
            None,
        )
        if current is not None:
            return

        self.assigned_defuser_id = min(
            alive_defenders,
            key=lambda defender: (
                self.distance(defender.pos, self.spike_pos),
                defender.hp,
                defender.player_id,
            ),
        ).player_id

    def is_assigned_defuser(self, defender: Player) -> bool:
        self.ensure_postplant_roles()
        return defender.player_id == self.assigned_defuser_id

    def cover_defenders(self) -> List[Player]:
        self.ensure_postplant_roles()
        return [
            defender
            for defender in self.roster.alive(Team.DEFENDER)
            if defender.player_id != self.assigned_defuser_id
        ]

    def defender_action(self, defender: Player) -> Optional[int]:
        if not defender.alive:
            return None

        attacker_visible = self.player_visible(defender, self.attacker)
        if attacker_visible:
            defender.last_seen_enemy = self.attacker.pos

        if self.bomb_planted and self.spike_pos is not None:
            self.ensure_postplant_roles()

            if self.is_assigned_defuser(defender):
                # 解除役はスパイクへ直行。範囲内では停止して解除を試みる。
                if chebyshev(defender.pos, self.spike_pos) <= 1:
                    return None
                return self.best_move_toward(
                    defender.pos,
                    self.spike_pos,
                    self.roster.occupied_positions(exclude=defender),
                )

            # カバー役は解除地点に重ならず、敵を見た／見失った位置へ圧力をかける。
            if attacker_visible:
                target = self.attacker.pos
            elif defender.last_seen_enemy is not None:
                target = defender.last_seen_enemy
                if defender.pos == target:
                    defender.last_seen_enemy = None
                    return None
            else:
                # 情報がなければスパイク周辺へ寄るが、解除役のセルは避ける。
                target = self.spike_pos

            # カバー役は少し静止して射撃精度を確保する。
            if attacker_visible and self.rng.random() < 0.45:
                return None
            if not attacker_visible and self.rng.random() < 0.20:
                return None

            return self.best_move_toward(
                defender.pos,
                target,
                self.roster.occupied_positions(exclude=defender),
            )

        # 設置前：見えている敵、最後に見た位置、サイトの順に追う。
        if attacker_visible:
            target = self.attacker.pos
        elif defender.last_seen_enemy is not None:
            target = defender.last_seen_enemy
            if defender.pos == target:
                defender.last_seen_enemy = None
                return None
        else:
            target = self.target_plant

        if self.rng.random() < 0.5:
            return None

        return self.best_move_toward(
            defender.pos,
            target,
            self.roster.occupied_positions(exclude=defender),
        )

    def interrupt_planting(self, add_reward) -> None:
        if not self.planting:
            return
        depth = int(self.plant_timer)
        if depth in self.plant_interrupt_counts:
            self.plant_interrupt_counts[depth] += 1
        self.planting = False
        self.plant_timer = 0
        add_reward("PLANT_INTERRUPTED", R_PLANT_INTERRUPTED)

    def resolve_attacker_move(self, action: int, add_reward) -> None:
        self.interrupt_planting(add_reward)
        target_before = (
            self.spike_pos
            if self.bomb_planted and self.spike_pos is not None
            else self.target_plant
        )
        old_distance = self.distance(self.attacker.pos, target_before)
        was_in_site = self.attacker.pos in self.plant_cells

        dr, dc = MOVE_DELTAS[action]
        new_pos = (self.attacker.pos[0] + dr, self.attacker.pos[1] + dc)

        if not self.inside(new_pos):
            self.move_failed = True
            self.fail_reason = "OUT"
        elif not self.walkable(new_pos):
            self.move_failed = True
            self.fail_reason = "WALL"
        elif new_pos in self.roster.occupied_positions(exclude=self.attacker):
            self.move_failed = True
            self.fail_reason = "OCCUPIED"

        if self.move_failed:
            self.move_failure_streak += 1
            multiplier = min(
                self.move_failure_streak,
                R_MOVE_FAILURE_STREAK_CAP,
            )
            reward_name = self.fail_reason
            base = R_OUT if reward_name == "OUT" else R_WALL
            add_reward(reward_name, base * multiplier)
            return

        self.attacker.pos = new_pos
        self.attacker.moved = True
        self.move_failure_streak = 0

        # Recon後の情報を使って位置を変えたかを記録する。
        for event in self.pending_recon_outcomes:
            if self.tick <= int(event["expires_tick"]):
                event["moved_after_use"] = True

        now_in_site = self.attacker.pos in self.plant_cells
        target = (
            self.spike_pos
            if self.bomb_planted and self.spike_pos is not None
            else self.target_plant
        )
        new_distance = self.distance(self.attacker.pos, target)

        if np.isfinite(old_distance) and np.isfinite(new_distance):
            difference = old_distance - new_distance
            if difference > 0:
                add_reward("TOWARD", R_TOWARD * difference)
            elif difference < 0:
                add_reward("AWAY", R_AWAY * abs(difference))
            else:
                add_reward("SAME_DISTANCE", R_SAME_DISTANCE)

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

        visit_count = self.visits.get(self.attacker.pos, 0)
        if visit_count > 0:
            add_reward("REVISIT", R_REVISIT * visit_count)
        self.visits[self.attacker.pos] = visit_count + 1

    def resolve_defender_move(
        self,
        defender: Player,
        action: Optional[int],
    ) -> None:
        if action not in MOVE_DELTAS or not defender.alive:
            return
        dr, dc = MOVE_DELTAS[action]
        new_pos = (defender.pos[0] + dr, defender.pos[1] + dc)
        occupied = self.roster.occupied_positions(exclude=defender)
        if self.walkable(new_pos) and new_pos not in occupied:
            defender.pos = new_pos
            defender.moved = True

    def handle_plant(self, add_reward) -> None:
        if self.bomb_planted:
            add_reward("WAIT", R_WAIT)
            return
        if self.attacker.pos not in self.plant_cells:
            self.planting = False
            self.plant_timer = 0
            add_reward("INVALID_PLANT", R_INVALID_PLANT)
            return

        self.planting = True
        self.plant_timer += 1
        add_reward("PLANT_PROGRESS", R_PLANT_PROGRESS)

        if self.plant_timer >= PLANT_REQUIRED_TICKS:
            self.bomb_planted = True
            self.spike_pos = self.attacker.pos
            self.spike_timer = SPIKE_DETONATION_TICKS
            self.planting = False
            self.plant_timer = 0
            self.site_control = True
            add_reward("PLANT_COMPLETE", R_PLANT_COMPLETE)

    def get_defuser(self) -> Optional[Player]:
        if self.defuser_id is None:
            return None
        return next(
            (d for d in self.defenders if d.player_id == self.defuser_id and d.alive),
            None,
        )

    def clear_defuse(self) -> None:
        for defender in self.defenders:
            defender.is_defusing = False
        self.defuser_id = None
        self.assigned_defuser_id = None
        self.defuse_timer = 0

    def handle_defuse(self, add_reward) -> None:
        if not self.bomb_planted or self.spike_pos is None:
            self.clear_defuse()
            self.assigned_defuser_id = None
            return

        self.ensure_postplant_roles()
        assigned = next(
            (
                defender
                for defender in self.roster.alive(Team.DEFENDER)
                if defender.player_id == self.assigned_defuser_id
            ),
            None,
        )

        if (
            assigned is None
            or chebyshev(assigned.pos, self.spike_pos) > 1
        ):
            self.clear_defuse()
            return

        current = self.get_defuser()
        if current is not assigned:
            self.clear_defuse()
            assigned.is_defusing = True
            self.defuser_id = assigned.player_id

        self.defuse_timer += 1
        if self.defuse_timer == 1:
            add_reward("DEFUSE_START", R_DEFUSE_START)
        add_reward("DEFUSE_PROGRESS", R_DEFUSE_PROGRESS)

        if self.defuse_timer >= DEFUSE_REQUIRED_TICKS:
            self.finish_round(TEAM_DEFENDER, "DEFUSED", add_reward)

    def select_target(
        self,
        shooter: Player,
        enemies: List[Player],
    ) -> Player:
        if not enemies:
            raise ValueError("select_target() に敵が渡されていません")

        return min(
            enemies,
            key=lambda enemy: (
                0 if enemy.is_defusing else 1,
                chebyshev(shooter.pos, enemy.pos),
                enemy.hp,
                enemy.player_id,
            ),
        )

    def defender_should_shoot(self, defender: Player) -> bool:
        """解除役は味方がカバーできる間は解除を継続する。"""
        if not defender.is_defusing:
            return True

        cover_has_los = any(
            teammate.alive
            and teammate is not defender
            and self.has_los(teammate.pos, self.attacker.pos, self.grid)
            for teammate in self.cover_defenders()
        )
        return not cover_has_los

    def shoot_player(
        self,
        shooter: Player,
        target: Player,
        add_reward,
    ) -> None:
        if not shooter.alive or not target.alive:
            return
        if not self.has_los(shooter.pos, target.pos, self.grid):
            return

        if shooter.is_defusing:
            self.clear_defuse()

        accuracy = shooter.stats.accuracy
        if shooter.moved:
            accuracy *= MOVING_ACCURACY_MULTIPLIER
        if self.rng.random() > accuracy:
            return

        damage = (
            HEAD_DAMAGE
            if self.rng.random() < shooter.stats.headshot_rate
            else BODY_DAMAGE
        )

        target_was_defusing = target.is_defusing
        target_was_revealed = target.revealed
        actual = target.take_damage(damage)

        if shooter.team == Team.ATTACKER:
            add_reward("DAMAGE_DEALT", actual * R_DAMAGE_DEALT)
            if target_was_revealed:
                add_reward("RECON_DAMAGE_BONUS", actual * R_RECON_DAMAGE_BONUS)
        else:
            add_reward("DAMAGE_TAKEN", actual * R_DAMAGE_TAKEN)

        if target.alive:
            return

        if target_was_defusing:
            self.clear_defuse()

        if shooter.team == Team.ATTACKER:
            add_reward("KILL", R_KILL)
            if target_was_revealed:
                add_reward("RECON_KILL_BONUS", R_RECON_KILL_BONUS)

            # Reconで敵が見つかった場合だけでなく、「いない」と分かった情報を
            # 利用して有利な撃ち合いへ進んだ場合も評価する。
            for event in self.pending_recon_outcomes:
                if (
                    self.tick <= int(event["expires_tick"])
                    and not bool(event["kill_rewarded"])
                ):
                    add_reward("RECON_FOLLOWUP_KILL", R_RECON_FOLLOWUP_KILL)
                    event["kill_rewarded"] = True
                    event["positive"] = True

            if target_was_defusing:
                add_reward("DEFUSER_KILL", R_DEFUSER_KILL)
        else:
            add_reward("DEATH", R_DEATH)

    def combat(self, add_reward) -> None:
        order = self.roster.shooting_order(
            self.rng,
            use_reaction=False,
        )
        for shooter in order:
            if not shooter.alive:
                continue

            # カバー役が射線を通している間、解除役は射撃せず解除を続ける。
            if shooter.team == Team.DEFENDER and not self.defender_should_shoot(
                shooter
            ):
                continue

            enemies = [
                enemy
                for enemy in self.roster.enemies_of(shooter)
                if self.has_los(shooter.pos, enemy.pos, self.grid)
            ]
            if not enemies:
                continue
            self.shoot_player(
                shooter,
                self.select_target(shooter, enemies),
                add_reward,
            )

    def finish_round(
        self,
        winner: str,
        reason: str,
        add_reward,
    ) -> None:
        if self.done:
            return
        self.done = True
        self.winner = winner
        self.end_reason = reason
        self.success = winner == TEAM_ATTACKER
        add_reward(
            "WIN" if self.success else "LOSS",
            R_WIN if self.success else R_LOSS,
        )

    def check_end(self, add_reward) -> None:
        if self.done:
            return

        if not self.attacker.alive and not self.bomb_planted:
            self.finish_round(
                TEAM_DEFENDER,
                "ATTACKER_ELIMINATED",
                add_reward,
            )
            return

        alive_defenders = self.roster.alive(Team.DEFENDER)

        if self.bomb_planted:
            if not alive_defenders:
                self.finish_round(
                    TEAM_ATTACKER,
                    (
                        "DEFENDERS_ELIMINATED_POST_PLANT"
                        if self.attacker.alive
                        else "POST_PLANT_TRADE"
                    ),
                    add_reward,
                )
                return

            self.spike_timer -= 1
            if self.spike_timer <= 0:
                add_reward("DETONATE", R_DETONATE)
                self.finish_round(
                    TEAM_ATTACKER,
                    "DETONATED",
                    add_reward,
                )
            return

        if self.tick >= ROUND_DURATION_TICKS:
            self.finish_round(
                TEAM_DEFENDER,
                "ROUND_TIMEOUT",
                add_reward,
            )

    def step(self, action: int):
        reward = 0.0

        def add_reward(name: str, value: float) -> None:
            nonlocal reward
            value = float(value)
            reward += value
            self.reward_stats[name] += value

        self.tick += 1
        self.attacker.moved = False
        for defender in self.defenders:
            defender.moved = False

        self.move_failed = False
        self.fail_reason = ""

        if action not in self.valid_actions():
            action = ACTION_WAIT
            add_reward("INVALID_ACTION", -1.0)

        self.last_action = action
        self.action_counts[ACTION_NAMES[action]] += 1
        add_reward("STEP", R_STEP)

        if not self.attacker.alive:
            action = ACTION_WAIT

        if action == ACTION_WAIT:
            self.wait_streak += 1
            if not self.bomb_planted:
                add_reward(
                    "WAIT_STREAK",
                    R_WAIT_STREAK_STEP * min(self.wait_streak, WAIT_STREAK_CAP),
                )
        else:
            self.wait_streak = 0

        if action in MOVE_DELTAS and self.attacker.alive:
            self.resolve_attacker_move(action, add_reward)
        elif action == ACTION_PLANT and self.attacker.alive:
            self.handle_plant(add_reward)
        elif ACTION_ABILITY_UP_NEAR <= action <= ACTION_ABILITY_RIGHT_FAR and self.attacker.alive:
            self.handle_recon(action, add_reward)
        else:
            if self.planting:
                self.interrupt_planting(add_reward)
            add_reward("WAIT", R_WAIT)

        defender_order = list(self.defenders)
        self.rng.shuffle(defender_order)
        for defender in defender_order:
            self.resolve_defender_move(
                defender,
                self.defender_action(defender),
            )

        self.combat(add_reward)

        if not self.done:
            self.handle_defuse(add_reward)

        self.evaluate_recon_outcomes(add_reward)

        if self.attacker.pos in self.plant_cells and self.attacker.alive:
            self.site_control = True
            add_reward("SITE_HOLD", R_SITE_HOLD)

        self.check_end(add_reward)
        self.tick_reveals()

        info = {
            "success": self.success,
            "winner": self.winner,
            "end_reason": self.end_reason,
            "tick": self.tick,
            "position": self.attacker.pos,
            "hp": self.attacker.hp,
            "defenders": [
                {
                    "player_id": d.player_id,
                    "position": d.pos,
                    "hp": d.hp,
                    "alive": d.alive,
                    "is_defusing": d.is_defusing,
                }
                for d in self.defenders
            ],
            "defender_position": self.defenders[0].pos,
            "defender_hp": self.defenders[0].hp,
            "plant_timer": self.plant_timer,
            "bomb_planted": self.bomb_planted,
            "site_control": self.site_control,
            "move_failure_streak": self.move_failure_streak,
            "defuse_timer": self.defuse_timer,
            "defuser_id": self.defuser_id,
            "assigned_defuser_id": self.assigned_defuser_id,
            "recon_charges": self.recon_charges,
            "defender_reveal_ticks": {
                defender.player_id: defender.recon_reveal_ticks for defender in self.defenders
            },
            "wait_streak": self.wait_streak,
            "pending_recon_outcomes": len(self.pending_recon_outcomes),
            "step_reward": float(reward),
            "reward_stats": dict(self.reward_stats),
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
    """Phase2本体を固定し、Phase3の追加6観測だけをAdapterで注入する。"""

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

        # 追加6観測専用Adapter。
        # 0初期化により、転移直後はPhase2と完全に同じQ値になる。
        self.extra_adapter = nn.Linear(
            PHASE3_EXTRA_OBS,
            256,
            bias=False,
        )
        nn.init.zeros_(self.extra_adapter.weight)

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
        # Phase3の既存Q値を維持したまま、Recon 8行動だけを追加学習する補正Head。
        # Flash学習済みHead。転移時に読み込み、固定して保持する。
        self.ability_adv_adapter = nn.Linear(256, 8, bias=False)
        nn.init.zeros_(self.ability_adv_adapter.weight)

        # Recon専用Head。今回ここだけを学習する。
        self.recon_adv_adapter = nn.Linear(256, 8, bias=False)
        nn.init.zeros_(self.recon_adv_adapter.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != OBS_DIM:
            raise ValueError(f"入力次元が不正です: {x.shape[-1]} != {OBS_DIM}")

        # Phase2本体の入力層出力へ、追加6観測のAdapter出力を加える。
        base_hidden = self.feature[0](x)
        extra = x[..., PHASE2_OBS_DIM:]
        extra_hidden = 0.1 * self.extra_adapter(extra)
        hidden = base_hidden + extra_hidden
        hidden = self.feature[1](hidden)
        hidden = self.feature[2](hidden)
        hidden = self.feature[3](hidden)
        features = self.feature[4](hidden)

        value = self.value(features)
        advantage = self.adv(features)
        base_q = value + advantage - advantage.mean(dim=1, keepdim=True)
        normal_q = base_q[..., :6]
        ability_baseline = normal_q.mean(dim=-1, keepdim=True) - RECON_Q_BASELINE_COST

        flash_delta = self.ability_adv_adapter(features)
        recon_delta = self.recon_adv_adapter(features)
        # role one-hotは観測index 10..14。Flash=index11、Recon=index13。
        flash_role = x[..., 11:12]
        recon_role = x[..., 13:14]
        selected_delta = flash_role * flash_delta + recon_role * recon_delta
        ability_q = ability_baseline + selected_delta
        return torch.cat([normal_q, ability_q], dim=-1)


def adapter_debug_metrics(
    model: DuelingQNetwork,
    observation: np.ndarray,
    device: torch.device,
) -> Dict[str, object]:
    """Phase2本体と追加Adapterの寄与を同じ観測で測定する。

    base_norm: feature.0(x) のL2ノルム
    adapter_norm: extra_adapter(extra) のL2ノルム
    ratio: adapter_norm / base_norm
    q_shift_max: Adapterを無効化した場合からの最大Q値変化
    action_changed: Adapterの有無でgreedy行動が変わるか
    """
    with torch.no_grad():
        x = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(
            0
        )

        base_hidden = model.feature[0](x)
        extra = x[..., PHASE2_OBS_DIM:]
        adapter_hidden = 0.1 * model.extra_adapter(extra)
        combined_hidden = base_hidden + adapter_hidden

        def finish_from_hidden(hidden: torch.Tensor) -> torch.Tensor:
            hidden = model.feature[1](hidden)
            hidden = model.feature[2](hidden)
            hidden = model.feature[3](hidden)
            features = model.feature[4](hidden)
            value = model.value(features)
            advantage = model.adv(features)
            base_q = value + advantage - advantage.mean(dim=1, keepdim=True)
            normal_q = base_q[..., :6]
            baseline = normal_q.mean(dim=-1, keepdim=True) - RECON_Q_BASELINE_COST
            flash_delta = model.ability_adv_adapter(features)
            recon_delta = model.recon_adv_adapter(features)
            flash_role = x[..., 11:12]
            recon_role = x[..., 13:14]
            ability_q = baseline + flash_role * flash_delta + recon_role * recon_delta
            return torch.cat([normal_q, ability_q], dim=-1)

        q_base = finish_from_hidden(base_hidden)[0]
        q_combined = finish_from_hidden(combined_hidden)[0]

        valid_indices = torch.as_tensor(
            VALID_PHASE1_ACTIONS, dtype=torch.long, device=device
        )
        base_valid = q_base.index_select(0, valid_indices)
        combined_valid = q_combined.index_select(0, valid_indices)

        base_action = int(VALID_PHASE1_ACTIONS[int(base_valid.argmax().item())])
        combined_action = int(VALID_PHASE1_ACTIONS[int(combined_valid.argmax().item())])

        base_norm = float(torch.linalg.vector_norm(base_hidden).item())
        adapter_norm = float(torch.linalg.vector_norm(adapter_hidden).item())
        combined_norm = float(torch.linalg.vector_norm(combined_hidden).item())
        ratio = adapter_norm / max(base_norm, 1e-12)
        q_delta = combined_valid - base_valid

        return {
            "base_norm": base_norm,
            "adapter_norm": adapter_norm,
            "combined_norm": combined_norm,
            "adapter_base_ratio": ratio,
            "adapter_weight_norm": float(
                torch.linalg.vector_norm(model.extra_adapter.weight).item()
            ),
            "extra_features": [float(v) for v in extra[0].tolist()],
            "q_shift_mean_abs": float(q_delta.abs().mean().item()),
            "q_shift_max_abs": float(q_delta.abs().max().item()),
            "base_action": base_action,
            "base_action_name": ACTION_NAMES[base_action],
            "combined_action": combined_action,
            "combined_action_name": ACTION_NAMES[combined_action],
            "action_changed": bool(base_action != combined_action),
        }


def freeze_phase2_base_for_adapter(model: DuelingQNetwork) -> None:
    """Phase3方策を完全固定し、Recon 8行動専用Headだけを学習する。"""
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.recon_adv_adapter.parameters():
        parameter.requires_grad = True


def print_trainable_parameters(model: DuelingQNetwork) -> None:
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    frozen_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    trainable_count = sum(count for _, count in trainable)

    print("[ADAPTER] trainable parameters:")
    for name, count in trainable:
        print(f"  {name:30s} {count:8d}")
    print(f"[ADAPTER] trainable={trainable_count} frozen={frozen_count}")


def choose_action(
    model: DuelingQNetwork,
    observation: np.ndarray,
    epsilon: float,
    device: torch.device,
    rng: random.Random,
    valid_actions: Optional[Tuple[int, ...]] = None,
) -> int:
    valid_actions = valid_actions or VALID_PHASE1_ACTIONS
    if rng.random() < epsilon:
        return rng.choice(valid_actions)

    with torch.no_grad():
        tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        q_values = model(tensor)[0]

        valid_indices = torch.as_tensor(
            valid_actions,
            dtype=torch.long,
            device=device,
        )
        valid_q = q_values.index_select(0, valid_indices)
        best_local_index = int(valid_q.argmax().item())
        return int(valid_actions[best_local_index])


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

    current_q = (
        policy(states)
        .gather(
            1,
            actions.unsqueeze(1),
        )
        .squeeze(1)
    )

    valid_indices = torch.as_tensor(
        VALID_PHASE1_ACTIONS,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        policy_next_all = policy(next_states).clone()
        # index15はRecon残数。index105..112は8方向ごとのRecon有効フラグ。
        no_recon = next_states[:, 15] <= 0.0
        policy_next_all[no_recon, 6:14] = -torch.inf
        recon_available = next_states[:, 105:113] > 0.5
        policy_next_all[:, 6:14] = policy_next_all[:, 6:14].masked_fill(
            ~recon_available,
            -torch.inf,
        )
        policy_next_valid = policy_next_all.index_select(
            1,
            valid_indices,
        )
        best_valid_local = policy_next_valid.argmax(
            dim=1,
            keepdim=True,
        )
        best_actions = valid_indices[best_valid_local.squeeze(1)].unsqueeze(1)

        next_q = (
            target(next_states)
            .gather(
                1,
                best_actions,
            )
            .squeeze(1)
        )

        targets = rewards + GAMMA * (1.0 - dones) * next_q

    loss = nn.functional.smooth_l1_loss(
        current_q,
        targets,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        GRADIENT_CLIP_NORM,
    )
    optimizer.step()

    return float(loss.item())


def extract_model_state_dict(loaded: object) -> Dict[str, torch.Tensor]:
    """モデル単体またはtraining checkpointからpolicyのstate_dictを取り出す。"""
    if not isinstance(loaded, dict):
        raise ValueError("読み込んだモデルが辞書形式ではありません")

    if "policy" in loaded and isinstance(loaded["policy"], dict):
        return loaded["policy"]
    if "model_state_dict" in loaded and isinstance(loaded["model_state_dict"], dict):
        return loaded["model_state_dict"]

    # state_dict単体は値がTensorであることを確認する。
    if loaded and all(isinstance(value, torch.Tensor) for value in loaded.values()):
        return loaded

    raise ValueError("policy/model_state_dict/state_dict単体のいずれも見つかりません")


def load_phase2_partial_weights(
    model: DuelingQNetwork,
    phase2_state: Dict[str, torch.Tensor],
) -> Dict[str, object]:
    """Phase2.7の125次元モデルをPhase3の131次元モデルへ部分転移する。"""
    current = model.state_dict()
    copied_exact: List[str] = []
    skipped: List[str] = []

    first_weight_key = "feature.0.weight"
    old_first = phase2_state.get(first_weight_key)
    new_first = current[first_weight_key]

    if old_first is None:
        raise ValueError(f"Phase2モデルに {first_weight_key} がありません")
    if old_first.ndim != 2 or new_first.ndim != 2:
        raise ValueError("入力層weightの次元が不正です")
    if old_first.shape[0] != new_first.shape[0]:
        raise ValueError(
            "入力層の出力幅が一致しません: "
            f"phase2={tuple(old_first.shape)}, "
            f"phase3={tuple(new_first.shape)}"
        )
    if old_first.shape[1] != PHASE2_OBS_DIM:
        raise ValueError(
            "Phase2入力次元が想定と異なります: "
            f"{old_first.shape[1]} != {PHASE2_OBS_DIM}"
        )
    if new_first.shape[1] != OBS_DIM:
        raise ValueError(
            "Phase3入力次元が想定と異なります: " f"{new_first.shape[1]} != {OBS_DIM}"
        )

    # 先頭125列をPhase2からコピーし、追加6列は0にする。
    # 新特徴の影響はextra_adapterだけが担当する。
    expanded_first = torch.zeros_like(new_first)
    expanded_first[:, :PHASE2_OBS_DIM] = old_first.to(
        dtype=expanded_first.dtype,
        device=expanded_first.device,
    )
    current[first_weight_key] = expanded_first

    for key, value in phase2_state.items():
        if key == first_weight_key:
            continue
        if key in current and current[key].shape == value.shape:
            current[key] = value.to(
                dtype=current[key].dtype,
                device=current[key].device,
            )
            copied_exact.append(key)
        else:
            skipped.append(key)

    model.load_state_dict(current)

    return {
        "phase2_obs_dim": int(old_first.shape[1]),
        "phase3_obs_dim": int(new_first.shape[1]),
        "copied_input_columns": PHASE2_OBS_DIM,
        "new_input_columns": OBS_DIM - PHASE2_OBS_DIM,
        "copied_exact_count": len(copied_exact),
        "adapter_zero_initialized": True,
        "skipped": skipped,
    }


def load_base_weights_for_recon(
    model: DuelingQNetwork,
    phase3_state: Dict[str, torch.Tensor],
) -> Dict[str, object]:
    """Phase3/Smokeモデルをコピーし、Recon専用Headだけ0初期値で残す。"""
    current = model.state_dict()
    copied: List[str] = []
    skipped: List[str] = []
    for key, value in phase3_state.items():
        if key == "recon_adv_adapter.weight":
            skipped.append(key)
            continue
        if key in current and current[key].shape == value.shape:
            current[key] = value.to(dtype=current[key].dtype, device=current[key].device)
            copied.append(key)
        else:
            skipped.append(key)
    current["recon_adv_adapter.weight"] = torch.zeros_like(
        current["recon_adv_adapter.weight"]
    )
    model.load_state_dict(current)
    return {
        "copied_count": len(copied),
        "skipped": skipped,
        "recon_head_zero_initialized": True,
    }


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
    env = Phase3OneVsTwoEnv(seed)
    rng = random.Random(seed)

    successes = 0
    total_ticks = 0
    reasons: Dict[str, int] = {}
    reward_totals: Dict[str, float] = defaultdict(float)
    action_totals: Dict[str, int] = defaultdict(int)
    interrupt_totals: Dict[int, int] = defaultdict(int)
    adapter_metric_sums: Dict[str, float] = defaultdict(float)
    adapter_metric_max: Dict[str, float] = defaultdict(float)
    adapter_metric_steps = 0
    adapter_action_changes = 0

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
            adapter_debug = adapter_debug_metrics(model, observation, device)
            adapter_metric_steps += 1
            for metric_name in (
                "base_norm",
                "adapter_norm",
                "combined_norm",
                "adapter_base_ratio",
                "q_shift_mean_abs",
                "q_shift_max_abs",
            ):
                value = float(adapter_debug[metric_name])
                adapter_metric_sums[metric_name] += value
                adapter_metric_max[metric_name] = max(
                    adapter_metric_max[metric_name], value
                )
            adapter_action_changes += int(adapter_debug["action_changed"])

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
                valid_actions=env.valid_actions(),
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
                        "hp": env.hp,
                        "defender_hp": [d.hp for d in env.defenders],
                        "defender_position": [d.pos for d in env.defenders],
                        "defuse_timer": env.defuse_timer,
                        "bomb_planted": env.bomb_planted,
                        "round_remaining": float(observation[-6]),
                        "spike_remaining": float(observation[-5]),
                        "alive_defender_ratio": float(observation[-4]),
                        "defuse_active": float(observation[-3]),
                        "spike_distance": float(observation[-2]),
                        "post_plant": float(observation[-1]),
                        "chosen_action": action,
                        "chosen_action_name": ACTION_NAMES[action],
                        "q_values": q_values,
                        "adapter_debug": adapter_debug,
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
        "win_rate": successes / episodes,
        "avg_ticks": total_ticks / episodes,
        "reasons": reasons,
        "reward_breakdown_avg": {
            key: value / episodes for key, value in sorted(reward_totals.items())
        },
        "action_counts_avg": {
            key: value / episodes for key, value in sorted(action_totals.items())
        },
        "plant_interrupt_counts_avg": {
            depth: interrupt_totals.get(depth, 0) / episodes
            for depth in range(1, PLANT_REQUIRED_TICKS)
        },
        "q_trace": q_trace,
        "adapter_debug_summary": {
            "steps": adapter_metric_steps,
            "weight_norm": float(
                torch.linalg.vector_norm(model.extra_adapter.weight).item()
            ),
            "action_change_rate": (
                adapter_action_changes / max(adapter_metric_steps, 1)
            ),
            "averages": {
                key: value / max(adapter_metric_steps, 1)
                for key, value in sorted(adapter_metric_sums.items())
            },
            "maxima": dict(sorted(adapter_metric_max.items())),
        },
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


def print_adapter_debug_summary(summary: Dict[str, object]) -> None:
    print("[ADAPTER-DEBUG] evaluation-wide summary")
    print(f"  steps                    {int(summary.get('steps', 0)):10d}")
    print(f"  adapter_weight_norm      {float(summary.get('weight_norm', 0.0)):10.6f}")
    print(
        f"  action_change_rate       {float(summary.get('action_change_rate', 0.0)):10.4%}"
    )
    averages = summary.get("averages", {})
    maxima = summary.get("maxima", {})
    for key in (
        "base_norm",
        "adapter_norm",
        "combined_norm",
        "adapter_base_ratio",
        "q_shift_mean_abs",
        "q_shift_max_abs",
    ):
        print(
            f"  {key + '_avg':25s} {float(averages.get(key, 0.0)):10.6f} "
            f"max={float(maxima.get(key, 0.0)):10.6f}"
        )


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
            f"hp={item['hp']} "
            f"enemy_hp={item['defender_hp']} "
            f"enemy={item['defender_position']} "
            f"planted={item['bomb_planted']} "
            f"defuse={item['defuse_timer']} "
            f"round_left={item['round_remaining']:.2f} "
            f"spike_left={item['spike_remaining']:.2f} "
            f"alive_def={item['alive_defender_ratio']:.2f} "
            f"defusing={item['defuse_active']:.0f} "
            f"spike_dist={item['spike_distance']:.2f} "
            f"chosen={item['chosen_action_name']}"
        )
        debug = item.get("adapter_debug", {})
        print(
            "[ADAPTER-STEP] "
            f"base_norm={float(debug.get('base_norm', 0.0)):.6f} "
            f"adapter_norm={float(debug.get('adapter_norm', 0.0)):.6f} "
            f"ratio={float(debug.get('adapter_base_ratio', 0.0)):.6f} "
            f"q_shift_mean={float(debug.get('q_shift_mean_abs', 0.0)):.6f} "
            f"q_shift_max={float(debug.get('q_shift_max_abs', 0.0)):.6f} "
            f"base_action={debug.get('base_action_name', '?')} "
            f"combined_action={debug.get('combined_action_name', '?')} "
            f"changed={debug.get('action_changed', False)} "
            f"extra={debug.get('extra_features', [])}"
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
    phase2_model: Optional[Path] = None,
    phase3_model: Optional[Path] = None,
    flash_model: Optional[Path] = None,
    tensorboard: bool = True,
    eval_q_trace_steps: int = 12,
    eval_only: bool = False,
    transfer_epsilon_start: float = TRANSFER_EPSILON_START,
    transfer_epsilon_end: float = TRANSFER_EPSILON_END,
    transfer_epsilon_decay_steps: int = TRANSFER_EPSILON_DECAY_STEPS,
) -> None:
    seed_everything(seed)

    if not 0.0 <= transfer_epsilon_end <= transfer_epsilon_start <= 1.0:
        raise ValueError(
            "転移用epsilonは 0 <= end <= start <= 1 を満たす必要があります"
        )
    if transfer_epsilon_decay_steps < 0:
        raise ValueError("転移用epsilon decay stepsは0以上である必要があります")

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "phase": "phase6_recon",
                "obs_dim": OBS_DIM,
                "phase2_obs_dim": PHASE2_OBS_DIM,
                "phase3_extra_obs": PHASE3_EXTRA_OBS,
                "extra_observation_names": [
                    "round_time_remaining",
                    "spike_time_remaining",
                    "alive_defender_ratio",
                    "defuse_active",
                    "distance_to_spike",
                    "post_plant",
                ],
                "n_actions": N_ACTIONS,
                "valid_actions": list(VALID_PHASE1_ACTIONS),
                "action_names": ACTION_NAMES,
                "plant_required_ticks": PLANT_REQUIRED_TICKS,
                "recon_max_charges": RECON_MAX_CHARGES,
                "recon_reveal_ticks": RECON_REVEAL_TICKS,
                "recon_radius": RECON_RADIUS,
                "recon_q_baseline": "mean(normal_q)-cost",
                "recon_q_baseline_cost": RECON_Q_BASELINE_COST,
                "transfer_epsilon_start": transfer_epsilon_start,
                "transfer_epsilon_end": transfer_epsilon_end,
                "transfer_epsilon_decay_steps": transfer_epsilon_decay_steps,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    env = Phase3OneVsTwoEnv(seed)

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
    best_win_rate = -math.inf

    selected_inputs = sum(x is not None for x in (resume, phase2_model, phase3_model, flash_model))
    if selected_inputs > 1:
        raise ValueError("--resume / --phase2-model / --phase3-model / --flash-modelは同時指定できません")

    # Phase2モデルからの新規転移とPhase3 checkpoint再開では、
    # 完成済み方策を維持する低epsilonスケジュールを使う。
    use_transfer_epsilon = phase2_model is not None or phase3_model is not None or flash_model is not None or resume is not None

    def current_epsilon(steps: int) -> float:
        if use_transfer_epsilon:
            return transfer_epsilon_by_steps(
                steps,
                start=transfer_epsilon_start,
                end=transfer_epsilon_end,
                decay_steps=transfer_epsilon_decay_steps,
            )
        return epsilon_by_steps(steps)

    if phase2_model is not None:
        loaded = torch.load(
            phase2_model,
            map_location=device,
            weights_only=False,
        )
        phase2_state = extract_model_state_dict(loaded)
        transfer_info = load_phase2_partial_weights(
            policy,
            phase2_state,
        )
        target.load_state_dict(policy.state_dict())
        print(f"[PHASE2] loaded partial weights: {phase2_model}")
        print("[PHASE2] transfer info:", transfer_info)

    if phase3_model is not None:
        loaded = torch.load(phase3_model, map_location=device, weights_only=False)
        phase3_state = extract_model_state_dict(loaded)
        transfer_info = load_base_weights_for_recon(policy, phase3_state)
        target.load_state_dict(policy.state_dict())
        print(f"[PHASE3] loaded full weights: {phase3_model}")
        print("[PHASE3] recon transfer info:", transfer_info)

    if flash_model is not None:
        loaded = torch.load(flash_model, map_location=device, weights_only=False)
        flash_state = extract_model_state_dict(loaded)
        transfer_info = load_base_weights_for_recon(policy, flash_state)
        target.load_state_dict(policy.state_dict())
        print(f"[FLASH] loaded base weights: {flash_model}")
        print("[FLASH] recon transfer info:", transfer_info)

    if resume is not None:
        checkpoint = torch.load(
            resume,
            map_location=device,
        )

        if checkpoint.get("obs_dim") != OBS_DIM:
            raise ValueError("checkpointのOBS_DIMが一致しません")
        if checkpoint.get("n_actions") != N_ACTIONS:
            raise ValueError("checkpointのN_ACTIONSが一致しません")

        policy.load_state_dict(checkpoint["policy"])
        target.load_state_dict(checkpoint["target"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_episode = checkpoint["episode"] + 1
        global_steps = checkpoint["steps"]
        best_win_rate = checkpoint.get(
            "best_win_rate",
            best_win_rate,
        )
        transfer_epsilon_start = float(
            checkpoint.get("transfer_epsilon_start", transfer_epsilon_start)
        )
        transfer_epsilon_end = float(
            checkpoint.get("transfer_epsilon_end", transfer_epsilon_end)
        )
        transfer_epsilon_decay_steps = int(
            checkpoint.get(
                "transfer_epsilon_decay_steps",
                transfer_epsilon_decay_steps,
            )
        )

    # Phase2から転移した本体（入力層・bias・LayerNorm・後段を含む）を
    # すべて固定し、追加6観測用Adapterだけを学習する。
    freeze_phase2_base_for_adapter(policy)
    freeze_phase2_base_for_adapter(target)
    print_trainable_parameters(policy)

    if eval_only:
        result = evaluate(
            policy,
            device,
            q_trace_steps=eval_q_trace_steps,
        )

        print(
            "[EVAL]",
            {
                "win_rate": result["win_rate"],
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
        print_adapter_debug_summary(result["adapter_debug_summary"])

        print_eval_q_trace(result["q_trace"])
        return

    writer = (
        SummaryWriter(str(MODEL_DIR / "tensorboard"))
        if tensorboard and SummaryWriter is not None
        else None
    )

    if tensorboard and SummaryWriter is None:
        print("[WARN] tensorboard未導入のため" "ログを無効化します")

    rng = random.Random(seed + 99)
    recent_rewards: Deque[float] = deque(maxlen=100)
    recent_wins: Deque[float] = deque(maxlen=100)
    recent_reward_breakdowns: Deque[Dict[str, float]] = deque(maxlen=100)
    recent_action_counts: Deque[Dict[str, int]] = deque(maxlen=100)
    recent_interrupt_counts: Deque[Dict[int, int]] = deque(maxlen=100)

    print(
        f"device={device} episodes={episodes} "
        f"OBS_DIM={OBS_DIM} N_ACTIONS={N_ACTIONS}"
    )
    print(
        "Phase6 Recon valid actions:",
        [ACTION_NAMES[a] for a in VALID_PHASE1_ACTIONS],
    )
    print(
        f"[RECON-Q] baseline=mean(normal Q)-{RECON_Q_BASELINE_COST:.1f} "
        "+ learned Recon delta"
    )
    if use_transfer_epsilon:
        print(
            "[EPSILON] transfer schedule "
            f"start={transfer_epsilon_start:.3f} "
            f"end={transfer_epsilon_end:.3f} "
            f"decay_steps={transfer_epsilon_decay_steps}"
        )
    else:
        print(
            "[EPSILON] scratch schedule "
            f"start={EPSILON_START:.3f} end={EPSILON_END:.3f} "
            f"decay_steps={EPSILON_DECAY_STEPS}"
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
                epsilon = current_epsilon(global_steps)
                action = choose_action(
                    policy,
                    observation,
                    epsilon,
                    device,
                    rng,
                    valid_actions=env.valid_actions(),
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

                if global_steps % TARGET_UPDATE_INTERVAL == 0:
                    target.load_state_dict(policy.state_dict())

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
            recent_wins.append(float(env.success))
            recent_reward_breakdowns.append(dict(info["reward_stats"]))
            recent_action_counts.append(dict(info["action_counts"]))
            recent_interrupt_counts.append(dict(info["plant_interrupt_counts"]))
            mean_loss = float(np.mean(losses)) if losses else 0.0

            if writer is not None:
                writer.add_scalar(
                    "train/reward",
                    total_reward,
                    episode,
                )
                writer.add_scalar(
                    "train/win",
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
                    current_epsilon(global_steps),
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
                    f"win100={np.mean(recent_wins):.3f} "
                    f"reward100={np.mean(recent_rewards):.1f} "
                    f"eps={current_epsilon(global_steps):.3f} "
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
                        "best_win_rate": best_win_rate,
                        "obs_dim": OBS_DIM,
                        "n_actions": N_ACTIONS,
                        "policy": policy.state_dict(),
                        "target": target.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "transfer_epsilon_start": transfer_epsilon_start,
                        "transfer_epsilon_end": transfer_epsilon_end,
                        "transfer_epsilon_decay_steps": transfer_epsilon_decay_steps,
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
                        "win_rate": result["win_rate"],
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
                        "eval/win_rate",
                        result["win_rate"],
                        episode,
                    )
                    writer.add_scalar(
                        "eval/avg_ticks",
                        result["avg_ticks"],
                        episode,
                    )
                    for depth, count in result["plant_interrupt_counts_avg"].items():
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

                win_rate = float(result["win_rate"])
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    save_model(BEST_MODEL_PATH, policy)
                    print(
                        "[BEST] win_rate=",
                        best_win_rate,
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
        "--phase2-model",
        type=Path,
        default=None,
        help=(
            "Phase2.7モデルを部分転移の初期重みとして読み込みます。"
            "--resumeと同時指定はできません"
        ),
    )
    parser.add_argument(
        "--phase3-model",
        type=Path,
        default=None,
        help="Phase3モデルを読み込み、Recon補正Headを0初期化して学習します",
    )
    parser.add_argument(
        "--flash-model",
        type=Path,
        default=None,
        help="Flash学習済みモデルを基礎方策として読み込み、Recon専用Headを0初期化します",
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
    parser.add_argument(
        "--transfer-epsilon-start",
        type=float,
        default=TRANSFER_EPSILON_START,
        help="Phase2転移／Phase3再開時の初期epsilon（既定: 0.10）",
    )
    parser.add_argument(
        "--transfer-epsilon-end",
        type=float,
        default=TRANSFER_EPSILON_END,
        help="Phase2転移／Phase3再開時の最終epsilon（既定: 0.02）",
    )
    parser.add_argument(
        "--transfer-epsilon-decay-steps",
        type=int,
        default=TRANSFER_EPSILON_DECAY_STEPS,
        help="転移用epsilonを減衰させるstep数（既定: 50000）",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="学習せず評価のみ実行",
    )

    args = parser.parse_args()

    if args.resume is not None and not args.resume.is_file():
        raise SystemExit(f"resume file not found: {args.resume}")
    if args.phase2_model is not None and not args.phase2_model.is_file():
        raise SystemExit(f"phase2 model file not found: {args.phase2_model}")
    if args.phase3_model is not None and not args.phase3_model.is_file():
        raise SystemExit(f"phase3 model file not found: {args.phase3_model}")
    if args.flash_model is not None and not args.flash_model.is_file():
        raise SystemExit(f"flash model file not found: {args.flash_model}")

    train(
        episodes=args.episodes,
        seed=args.seed,
        device_name=args.device,
        resume=args.resume,
        phase2_model=args.phase2_model,
        phase3_model=args.phase3_model,
        flash_model=args.flash_model,
        tensorboard=not args.no_tensorboard,
        eval_q_trace_steps=max(args.eval_q_trace_steps, 0),
        eval_only=args.eval,
        transfer_epsilon_start=args.transfer_epsilon_start,
        transfer_epsilon_end=args.transfer_epsilon_end,
        transfer_epsilon_decay_steps=args.transfer_epsilon_decay_steps,
    )


if __name__ == "__main__":
    main()
