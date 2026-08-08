# train_attacker_ai_v3_phase2.py
"""AI Ver3 Phase2.6: 1vs2の設置・戦闘学習用フルコード。

構成:
- 学習対象: アタッカー1人
- 対戦相手: ルールベースのディフェンダー2人
- アクション: 4方向移動 / WAIT / PLANT
- 射撃: 射線が通る場合に毎tick自動射撃
- 目的: 生存しながらサイトを確保し、設置してラウンドに勝つ
- サイト保持、設置中断、壁ループ対策はPhase1から継承
- OBS_DIM=125 / N_ACTIONS=14
- アビリティ出力は残すがPhase2では無効

実行例:
    python train_attacker_ai_v3_phase2.py --episodes 6000
    python train_attacker_ai_v3_phase2.py --episodes 6000 --phase1-model attacker_ai_v3_data/dqn_attacker_ai_v3_phase2_6_final.pt
    python train_attacker_ai_v3_phase2.py --eval-q-trace-steps 20

Phase1モデルの利用:
- --phase1-model にはモデルstate_dictだけの.pt、またはtraining_state_latest.ptを指定できます。
- 出力構造と観測次元が同じなので、Phase1の移動・設置知識を初期値として利用できます。
- Phase2専用checkpointを再開する場合は --resume を使用してください。
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

DEFAULT_EPISODES = 6000
DEFAULT_SEED = 42

GAMMA = 0.99
LEARNING_RATE = 2.5e-4
BATCH_SIZE = 128
REPLAY_CAPACITY = 250_000
LEARNING_STARTS = 2_000
TRAIN_EVERY_STEPS = 4
TARGET_UPDATE_INTERVAL = 1_000
GRADIENT_CLIP_NORM = 10.0

EPSILON_START = 1.0
EPSILON_END = 0.03
EPSILON_DECAY_STEPS = 220_000

EVAL_INTERVAL_EPISODES = 100
EVAL_EPISODES = 50
SAVE_INTERVAL_EPISODES = 100

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "attacker_ai_v3_phase2_6_data"
LATEST_MODEL_PATH = MODEL_DIR / "dqn_attacker_ai_v3_phase2_6_latest.pt"
BEST_MODEL_PATH = MODEL_DIR / "dqn_attacker_ai_v3_phase2_6_best.pt"
FINAL_MODEL_PATH = MODEL_DIR / "dqn_attacker_ai_v3_phase2_final.pt"
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


def has_los(a: Position, b: Position, grid: np.ndarray) -> bool:
    """壁に遮られていない場合だけTrue。始点と終点は壁判定から除外する。"""
    cells = line_cells(a, b)
    for row, col in cells[1:-1]:
        if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
            return False
        if grid[row, col] == 1:
            return False
    return True


def epsilon_by_steps(steps: int) -> float:
    fraction = min(max(steps / EPSILON_DECAY_STEPS, 0.0), 1.0)
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


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

    def reset(self, pos: Position) -> None:
        self.pos = pos
        self.hp = MAX_HP
        self.alive = True
        self.moved = False
        self.is_defusing = False
        self.last_seen_enemy = None

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


class Phase2OneVsTwoEnv:
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
        return self.observation()

    def player_visible(self, observer: Player, target: Player) -> bool:
        return (
            observer.alive
            and target.alive
            and has_los(observer.pos, target.pos, self.grid)
        )

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
        out += [1.0, 0.0, 0.0, 0.0, 0.0]
        out += [0.0]
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
        out += [0.0] * 24

        alive_defenders = self.roster.alive(Team.DEFENDER)
        out.append(
            float(
                any(
                    has_los(self.attacker.pos, d.pos, self.grid)
                    for d in alive_defenders
                )
            )
        )
        for dr, dc in MOVE_DELTAS.values():
            probe = (r + dr, c + dc)
            out.append(
                float(
                    self.walkable(probe)
                    and any(has_los(probe, d.pos, self.grid) for d in alive_defenders)
                )
            )

        defuser = self.get_defuser()
        out += [
            float(defuser is not None and self.defuse_timer > 0),
            min(self.defuse_timer / DEFUSE_REQUIRED_TICKS, 1.0),
            ((defuser.pos[0] - r) / max(self.h - 1, 1) if defuser is not None else 0.0),
            ((defuser.pos[1] - c) / max(self.w - 1, 1) if defuser is not None else 0.0),
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

    def defender_action(self, defender: Player) -> Optional[int]:
        if not defender.alive:
            return None

        if self.bomb_planted and self.spike_pos is not None:
            if chebyshev(defender.pos, self.spike_pos) <= 1:
                return None
            if self.rng.random() < 0.25:
                return None
            target = self.spike_pos
        elif self.player_visible(defender, self.attacker):
            defender.last_seen_enemy = self.attacker.pos
            target = self.attacker.pos
        elif defender.last_seen_enemy is not None:
            target = defender.last_seen_enemy
            if defender.pos == target:
                defender.last_seen_enemy = None
                return None
        else:
            target = self.target_plant

        if not self.bomb_planted and self.rng.random() < 0.5:
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
        self.defuse_timer = 0

    def handle_defuse(self, add_reward) -> None:
        if not self.bomb_planted or self.spike_pos is None:
            self.clear_defuse()
            return

        candidates = [
            d
            for d in self.roster.alive(Team.DEFENDER)
            if chebyshev(d.pos, self.spike_pos) <= 1
        ]
        current = self.get_defuser()

        if current not in candidates:
            self.clear_defuse()
            current = None

        if current is None and candidates:
            current = min(candidates, key=lambda p: p.player_id)
            current.is_defusing = True
            self.defuser_id = current.player_id

        if current is None:
            return

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

    def shoot_player(
        self,
        shooter: Player,
        target: Player,
        add_reward,
    ) -> None:
        if not shooter.alive or not target.alive:
            return
        if not has_los(shooter.pos, target.pos, self.grid):
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
        actual = target.take_damage(damage)

        if shooter.team == Team.ATTACKER:
            add_reward("DAMAGE_DEALT", actual * R_DAMAGE_DEALT)
        else:
            add_reward("DAMAGE_TAKEN", actual * R_DAMAGE_TAKEN)

        if target.alive:
            return

        if target_was_defusing:
            self.clear_defuse()

        if shooter.team == Team.ATTACKER:
            add_reward("KILL", R_KILL)
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
            enemies = [
                enemy
                for enemy in self.roster.enemies_of(shooter)
                if has_los(shooter.pos, enemy.pos, self.grid)
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

        if action not in VALID_PHASE1_ACTIONS:
            action = ACTION_WAIT
            add_reward("INVALID_ACTION", -1.0)

        self.last_action = action
        self.action_counts[ACTION_NAMES[action]] += 1
        add_reward("STEP", R_STEP)

        if not self.attacker.alive:
            action = ACTION_WAIT

        if action in MOVE_DELTAS and self.attacker.alive:
            self.resolve_attacker_move(action, add_reward)
        elif action == ACTION_PLANT and self.attacker.alive:
            self.handle_plant(add_reward)
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

        if self.attacker.pos in self.plant_cells and self.attacker.alive:
            self.site_control = True
            add_reward("SITE_HOLD", R_SITE_HOLD)

        self.check_end(add_reward)

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
        return (
            value
            + advantage
            - advantage.mean(
                dim=1,
                keepdim=True,
            )
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
        policy_next_all = policy(next_states)
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
    env = Phase2OneVsTwoEnv(seed)
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
                        "hp": env.hp,
                        "defender_hp": [d.hp for d in env.defenders],
                        "defender_position": [d.pos for d in env.defenders],
                        "defuse_timer": env.defuse_timer,
                        "bomb_planted": env.bomb_planted,
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
            f"hp={item['hp']} "
            f"enemy_hp={item['defender_hp']} "
            f"enemy={item['defender_position']} "
            f"planted={item['bomb_planted']} "
            f"defuse={item['defuse_timer']} "
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
    phase1_model: Optional[Path] = None,
    tensorboard: bool = True,
    eval_q_trace_steps: int = 12,
    eval_only: bool = False,  # ←追加
) -> None:
    seed_everything(seed)

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "phase": "phase2_6_one_vs_two",
                "obs_dim": OBS_DIM,
                "n_actions": N_ACTIONS,
                "valid_phase1_actions": list(VALID_PHASE1_ACTIONS),
                "action_names": ACTION_NAMES,
                "plant_required_ticks": PLANT_REQUIRED_TICKS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    env = Phase2OneVsTwoEnv(seed)

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

    if resume is not None and phase1_model is not None:
        raise ValueError("--resumeと--phase1-modelは同時に指定できません")

    if phase1_model is not None:
        loaded = torch.load(
            phase1_model,
            map_location=device,
            weights_only=False,
        )
        if isinstance(loaded, dict) and "policy" in loaded:
            state_dict = loaded["policy"]
        elif isinstance(loaded, dict) and "model_state_dict" in loaded:
            state_dict = loaded["model_state_dict"]
        else:
            state_dict = loaded

        policy.load_state_dict(state_dict)
        target.load_state_dict(policy.state_dict())
        print(f"[PHASE1] loaded initial weights: {phase1_model}")

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
                    f"win100={np.mean(recent_wins):.3f} "
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
                        "best_win_rate": best_win_rate,
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
        "--phase1-model",
        type=Path,
        default=None,
        help=(
            "Phase1のモデルを初期重みとして読み込みます。"
            "--resumeと同時指定はできません"
        ),
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
        "--eval",
        action="store_true",
        help="学習せず評価のみ実行",
    )

    args = parser.parse_args()

    if args.resume is not None and not args.resume.is_file():
        raise SystemExit(f"resume file not found: {args.resume}")

    train(
        episodes=args.episodes,
        seed=args.seed,
        device_name=args.device,
        resume=args.resume,
        phase1_model=args.phase1_model,
        tensorboard=not args.no_tensorboard,
        eval_q_trace_steps=max(args.eval_q_trace_steps, 0),
        eval_only=args.eval,
    )


if __name__ == "__main__":
    main()
