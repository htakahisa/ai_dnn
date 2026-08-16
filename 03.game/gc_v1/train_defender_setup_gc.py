"""Ghost Champions Defender Setup -- setup-only trainer.

Stage 1:
- フル試合を実行しない。
- Opening / Search / Retake / combat / plant / round score を実行しない。
- 本番と同じDefender spawn(NEW_MAZE_STRの4)、Setup進入制限、
  GCDefenderSetupPlannerのBFS移動だけを DEFENDER_SETUP_TICKS 回す。
- Setup終了時点の配置品質だけで学習する。

新GC roles:
    Xdll      RECON
    SyouTa    SMOKE
    Absol     FLASH
    eKo       RECON
    SugarZ3ro SMOKE
"""

from __future__ import annotations

import argparse
import math
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from map_data import NEW_MAZE_STR
from map_data_defender_setup import (
    DEFENDER_SETUP_TICKS,
    is_setup_position_allowed,
)
from map_data_defender_setup import get_setup_mask
from gc_v1.map_data_defender_setup_positions_gc import (
    get_gc_setup_position_candidates,
)

from party_presets import all_preset_names, get_preset
from gc_v1.map_data_defender_opening_ability_gc import (
    OPENING_PATTERN_IDS,
    ABILITY_PATTERN_LAYERS,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "defender_setup_gc_data"
BEST_MODEL = DATA_DIR / "dqn_defender_setup_gc_best.pt"
LATEST_MODEL = DATA_DIR / "dqn_defender_setup_gc_latest.pt"
FINAL_MODEL = DATA_DIR / "dqn_defender_setup_gc_final.pt"
INTERRUPT_MODEL = DATA_DIR / "dqn_defender_setup_gc_interrupt.pt"
LOG_FILE = DATA_DIR / "training_log.jsonl"

GC_ROSTER_ORDER = [
    "Xdll",
    "SyouTa",
    "Absol",
    "eKo",
    "SugarZ3ro",
]

CANDIDATES = list(get_gc_setup_position_candidates())
ACTION_DIM = len(CANDIDATES)
PLAYER_COUNT = len(GC_ROSTER_ORDER)


@dataclass
class SetupAssignment:
    player_name: str
    target: tuple[int, int]


class GCDefenderSetupPlanner:
    """Setup-only training用の独立Planner。

    実戦GC controllerやlearning_defender_setup_gc.pyには依存しない。
    本番と同じSetup maskと、4方向BFSだけを使う。
    """

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.setup_mask = get_setup_mask()
        self.candidates = list(CANDIDATES)
        self.assignments = {}
        self.round_initialized = False

    def validate(self):
        errors = []

        if len(self.candidates) < len(GC_ROSTER_ORDER):
            errors.append(
                "GC Setup position candidates must contain at least "
                f"{len(GC_ROSTER_ORDER)} cells, found {len(self.candidates)}"
            )

        for pos in self.candidates:
            if not is_setup_position_allowed(*pos):
                errors.append(
                    f"GC Setup candidate forbidden by setup mask: {pos}"
                )

        return errors

    def reset_round(self):
        self.assignments.clear()
        self.round_initialized = False

    def _neighbors(self, pos):
        r, c = int(pos[0]), int(pos[1])
        h = len(self.setup_mask)
        w = len(self.setup_mask[0])

        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < h and 0 <= nc < w):
                continue

            if not is_setup_position_allowed(nr, nc):
                continue

            yield (nr, nc)

    def _bfs_next_step(self, start, goal, blocked=None):
        start = tuple(map(int, start))
        goal = tuple(map(int, goal))
        blocked = set(blocked or ())

        if start == goal:
            return start

        blocked.discard(start)
        blocked.discard(goal)

        q = deque([start])
        prev = {start: None}

        while q:
            cur = q.popleft()

            if cur == goal:
                break

            for nxt in self._neighbors(cur):
                if nxt in blocked:
                    continue
                if nxt in prev:
                    continue

                prev[nxt] = cur
                q.append(nxt)

        if goal not in prev:
            return start

        cur = goal
        while prev[cur] is not None and prev[cur] != start:
            cur = prev[cur]

        return cur

    def get_target(self, char):
        return self.assignments.get(char.name)

    def decide_setup_move(self, char, chars):
        target = self.get_target(char)
        if target is None:
            return list(char.pos)

        start = tuple(map(int, char.pos))

        occupied = {
            tuple(map(int, other.pos))
            for other in chars
            if getattr(other, "is_alive", True)
            and other is not char
        }

        nxt = self._bfs_next_step(
            start,
            target,
            blocked=occupied,
        )

        return [int(nxt[0]), int(nxt[1])]

# Setup v2 observation:
# - player one-hot
# - already selected candidate mask
# - own spawn/current position
# - all candidate coordinates
# - opponent preset one-hot
# - setup variation one-hot
#
# Opponent input is included now so the same checkpoint can later be fine-tuned
# end-to-end. In Setup-only Stage 1 we do NOT invent an arbitrary "team X must
# defend left" reward.
OPPONENT_NAMES = sorted(
    name
    for name in all_preset_names()
    if name != "Ghost Champions"
    and get_preset(name) is not None
    and len(tuple(get_preset(name).players)) == 5
)
OPPONENT_DIM = len(OPPONENT_NAMES)
OPPONENT_INDEX = {name: i for i, name in enumerate(OPPONENT_NAMES)}

SETUP_VARIATIONS = ("BALANCED", "LEFT_LEAN", "RIGHT_LEAN")
VARIATION_DIM = len(SETUP_VARIATIONS)

OBS_DIM = (
    PLAYER_COUNT
    + ACTION_DIM
    + 2
    + ACTION_DIM * 2
    + OPPONENT_DIM
    + VARIATION_DIM
)

REPLAY_CAPACITY = 50_000
BATCH_SIZE = 128
LEARNING_START = 500
LR = 2e-4
TARGET_SYNC_EVERY = 250

ARRIVAL_REWARD_PER_PLAYER = 0.04
ALL_ARRIVED_BONUS = 0.20

# Remaining BFS distance after Setup.
DISTANCE_PENALTY_PER_CELL = -0.012
DISTANCE_PENALTY_CAP_PER_PLAYER = -0.12

MIN_PROGRESS_FACTOR = 0.15

COVERAGE_TWO_ZONES_REWARD = 0.08
COVERAGE_THREE_ZONES_REWARD = 0.18
COVERAGE_ONE_ZONE_PENALTY = -0.10

CROWDING_DISTANCE = 2
CROWDING_PENALTY_PER_PAIR = -0.025
CROWDING_PENALTY_CAP = -0.10

ROLE_PAIR_FAR_DISTANCE = 10
ROLE_PAIR_CLOSE_DISTANCE = 4
ROLE_PAIR_FAR_REWARD = 0.06
ROLE_PAIR_CLOSE_PENALTY = -0.06

# Opening connectivity: only the NEAREST usable Origin matters.
# Far-away A/B patterns are not penalized just for being far away.
OPENING_CONNECTIVITY_PER_PLAYER = 0.07
OPENING_CONNECTIVITY_MAX_DISTANCE = {
    "RECON": 18.0,
    "FLASH": 22.0,
}

# Variation reward is deliberately mild. It creates several meaningful Setup
# families without overpowering arrival/coverage/opening-connectivity.
VARIATION_REWARD_MAX = 0.10
VARIATION_TARGET_OFFSET = 0.16  # normalized horizontal center shift

GC_SETUP_ROLES = {
    "Xdll": "RECON",
    "SyouTa": "SMOKE",
    "Absol": "FLASH",
    "eKo": "RECON",
    "SugarZ3ro": "SMOKE",
}

CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _maze_rows():
    rows = [row.strip() for row in str(NEW_MAZE_STR).strip().splitlines() if row.strip()]
    if not rows:
        raise RuntimeError("NEW_MAZE_STR is empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise RuntimeError("NEW_MAZE_STR rows have inconsistent widths")
    return rows


BASE_ROWS = _maze_rows()
HEIGHT = len(BASE_ROWS)
WIDTH = len(BASE_ROWS[0])


def _pattern_cells(text):
    result = {int(pid): [] for pid in OPENING_PATTERN_IDS}
    rows = [
        row.strip()
        for row in str(text).strip().splitlines()
        if row.strip()
    ]
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            if value.isdigit() and int(value) in result:
                result[int(value)].append((r, c))
    return {pid: cells for pid, cells in result.items() if cells}


OPENING_ORIGINS = {}
for _ability in ("RECON", "FLASH"):
    _layer = ABILITY_PATTERN_LAYERS.get(_ability, {})
    _origin_text = _layer.get("origin")
    OPENING_ORIGINS[_ability] = (
        _pattern_cells(_origin_text)
        if _origin_text
        else {}
    )


def _normal_map_bfs_distance(start, goal):
    """Opening uses the normal walkable map, not the Setup boundary mask."""
    start = tuple(map(int, start))
    goal = tuple(map(int, goal))
    if start == goal:
        return 0

    q = deque([(start, 0)])
    seen = {start}

    while q:
        (r, c), d = q.popleft()
        nd = d + 1
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if nxt in seen:
                continue
            if not (0 <= nr < HEIGHT and 0 <= nc < WIDTH):
                continue
            if BASE_ROWS[nr][nc] == "1":
                continue
            if nxt == goal:
                return nd
            seen.add(nxt)
            q.append((nxt, nd))

    return 999


def _nearest_opening_origin_distance(position, ability):
    """Minimum over ALL patterns/origins for the assigned ability.

    Deliberately ignores far-away alternatives: only the nearest usable
    Opening route matters for Setup quality.
    """
    patterns = OPENING_ORIGINS.get(str(ability).upper(), {})
    best = 999
    best_pid = None

    for pid, origins in patterns.items():
        for origin in origins:
            d = _normal_map_bfs_distance(position, origin)
            if d < best:
                best = d
                best_pid = int(pid)

    return best, best_pid


DEFENDER_SPAWNS = [
    (r, c)
    for r, row in enumerate(BASE_ROWS)
    for c, value in enumerate(row)
    if value == "4"
]

if len(DEFENDER_SPAWNS) < PLAYER_COUNT:
    raise RuntimeError(
        f"Defender spawn cells不足: {len(DEFENDER_SPAWNS)} < {PLAYER_COUNT}"
    )

# run_game uses roster i -> area_4[i], so row-major scan is intentional.

DEFENDER_SPAWNS = DEFENDER_SPAWNS[:PLAYER_COUNT]

# 20Tickを全部使い切る前提にはせず、2Tick分の渋滞余裕を持たせる。
# 本番Setup時間を変えた場合も自動追従する。
MAX_CANDIDATE_BFS_DISTANCE = max(1, int(DEFENDER_SETUP_TICKS) - 2)


def _static_setup_bfs_distance(start, goal):
    """他プレイヤーを無視したSetup mask上の静的BFS距離。"""
    start = tuple(map(int, start))
    goal = tuple(map(int, goal))

    if start == goal:
        return 0

    q = deque([(start, 0)])
    seen = {start}

    while q:
        (r, c), d = q.popleft()
        nd = d + 1

        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)

            if nxt in seen:
                continue
            if not is_setup_position_allowed(nr, nc):
                continue
            if nxt == goal:
                return nd

            seen.add(nxt)
            q.append((nxt, nd))

    return 999


# player_index x action_index の静的BFS距離。
PLAYER_CANDIDATE_BFS_DISTANCES = [
    [
        _static_setup_bfs_distance(spawn, candidate)
        for candidate in CANDIDATES
    ]
    for spawn in DEFENDER_SPAWNS
]

PLAYER_REACHABLE_MASKS = np.asarray(
    [
        [
            dist <= MAX_CANDIDATE_BFS_DISTANCE
            for dist in row
        ]
        for row in PLAYER_CANDIDATE_BFS_DISTANCES
    ],
    dtype=np.bool_,
)


def _remaining_players_have_distinct_candidates(
    next_player_index,
    unavailable_actions,
):
    """残り全員へ重複なしの到達可能候補を割り当てられるか。

    小規模(最大5人・現在7候補)なのでDFSの二部マッチングで十分高速。
    """
    unavailable = set(int(x) for x in unavailable_actions)
    remaining_players = list(range(int(next_player_index), PLAYER_COUNT))

    if not remaining_players:
        return True

    available_actions = [
        a for a in range(ACTION_DIM)
        if a not in unavailable
    ]
    if len(available_actions) < len(remaining_players):
        return False

    # 制約の強いプレイヤーから割り当てる。
    remaining_players.sort(
        key=lambda p: int(
            np.count_nonzero(
                PLAYER_REACHABLE_MASKS[p]
                & np.asarray(
                    [a not in unavailable for a in range(ACTION_DIM)],
                    dtype=np.bool_,
                )
            )
        )
    )

    matched_action_to_player = {}

    def augment(player_index, seen_actions):
        for action in range(ACTION_DIM):
            if action in unavailable:
                continue
            if action in seen_actions:
                continue
            if not PLAYER_REACHABLE_MASKS[player_index, action]:
                continue

            seen_actions.add(action)

            previous_player = matched_action_to_player.get(action)
            if previous_player is None or augment(previous_player, seen_actions):
                matched_action_to_player[action] = player_index
                return True

        return False

    for player_index in remaining_players:
        if not augment(player_index, set()):
            return False

    return True


def validate_reachability_assignment_space():
    errors = []

    for player_index, name in enumerate(GC_ROSTER_ORDER):
        count = int(np.count_nonzero(PLAYER_REACHABLE_MASKS[player_index]))
        if count == 0:
            errors.append(
                f"{name} has no candidate within "
                f"{MAX_CANDIDATE_BFS_DISTANCE} Setup BFS ticks"
            )

    if not _remaining_players_have_distinct_candidates(0, set()):
        errors.append(
            "No full 5-player distinct reachable assignment exists "
            f"within {MAX_CANDIDATE_BFS_DISTANCE} Setup BFS ticks"
        )

    return errors


@dataclass
class UnitStub:
    name: str
    team: str
    pos: list[int]
    is_alive: bool = True


class SetupOnlyGame:
    def __init__(self):
        # build_obs only needs shape; keep actual map values for diagnostics.
        self.grid = np.asarray(
            [[int(ch) if ch.isdigit() else 1 for ch in row] for row in BASE_ROWS],
            dtype=np.int16,
        )
        self.width = WIDTH
        self.height = HEIGHT
        self.chars = [
            UnitStub(name, "D", [spawn[0], spawn[1]])
            for name, spawn in zip(GC_ROSTER_ORDER, DEFENDER_SPAWNS)
        ]


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    valid_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.data = deque(maxlen=int(capacity))

    def __len__(self):
        return len(self.data)

    def add(self, transition):
        self.data.append(transition)

    def sample(self, batch_size, rng):
        idx = rng.sample(range(len(self.data)), batch_size)
        return [self.data[i] for i in idx]


class SetupQNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.Linear(192, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def _normalize_pos(pos):
    r, c = int(pos[0]), int(pos[1])
    return (
        r / max(1.0, float(HEIGHT - 1)),
        c / max(1.0, float(WIDTH - 1)),
    )


def build_obs(char, player_index, selected_indices, opponent_name, variation_index):
    player_one_hot = np.zeros(PLAYER_COUNT, dtype=np.float32)
    player_one_hot[player_index] = 1.0

    selected_mask = np.zeros(ACTION_DIM, dtype=np.float32)
    for idx in selected_indices:
        if 0 <= idx < ACTION_DIM:
            selected_mask[idx] = 1.0

    own_r, own_c = _normalize_pos(char.pos)

    candidate_features = []
    for pos in CANDIDATES:
        rr, cc = _normalize_pos(pos)
        candidate_features.extend((rr, cc))

    opponent_one_hot = np.zeros(OPPONENT_DIM, dtype=np.float32)
    opponent_idx = OPPONENT_INDEX.get(str(opponent_name))
    if opponent_idx is not None:
        opponent_one_hot[opponent_idx] = 1.0

    variation_one_hot = np.zeros(VARIATION_DIM, dtype=np.float32)
    variation_one_hot[int(variation_index)] = 1.0

    obs = np.concatenate([
        player_one_hot,
        selected_mask,
        np.asarray([own_r, own_c], dtype=np.float32),
        np.asarray(candidate_features, dtype=np.float32),
        opponent_one_hot,
        variation_one_hot,
    ]).astype(np.float32)

    if obs.shape != (OBS_DIM,):
        raise RuntimeError(f"Setup OBS mismatch: {obs.shape} != {(OBS_DIM,)}")
    return obs


def valid_action_mask(player_index, selected_indices):
    """現在選手が到達可能かつ、残り全員も配置可能なActionだけを許可。"""
    player_index = int(player_index)
    selected = {int(idx) for idx in selected_indices}

    mask = PLAYER_REACHABLE_MASKS[player_index].copy()

    # 既に選ばれた候補は重複禁止。
    for idx in selected:
        if 0 <= idx < ACTION_DIM:
            mask[idx] = False

    # この候補を取った後、残り選手にdistinct reachable assignmentが
    # 残らないActionは先読みで禁止する。
    for action in np.flatnonzero(mask):
        unavailable = selected | {int(action)}
        if not _remaining_players_have_distinct_candidates(
            player_index + 1,
            unavailable,
        ):
            mask[action] = False

    return mask


def choose_action(model, obs, mask, epsilon, device, rng):
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        raise RuntimeError("No valid Setup action remains")

    if rng.random() < epsilon:
        return int(rng.choice(valid.tolist()))

    with torch.no_grad():
        x = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        q = model(x)[0].detach().cpu().numpy()

    q = q.copy()
    q[~mask] = -1e30
    return int(np.argmax(q))


class TrainableSetupPlanner(GCDefenderSetupPlanner):
    def __init__(
        self,
        model,
        device,
        epsilon,
        rng,
        transition_sink,
        *,
        opponent_name,
        variation_index,
        greedy=False,
    ):
        super().__init__(seed=None)
        self.model = model
        self.device = device
        self.epsilon = float(epsilon)
        self.rng = rng
        self.transition_sink = transition_sink
        self.greedy = bool(greedy)
        self.opponent_name = str(opponent_name)
        self.variation_index = int(variation_index)
        self.pending_choices = []
        self.setup_end_positions = {}
        self.setup_end_reached = {}

    def initialize_round(self, chars):
        errors = self.validate()
        if errors:
            raise RuntimeError(
                "GC Defender Setup map validation failed:\n- "
                + "\n- ".join(errors)
            )

        char_by_name = {
            c.name: c for c in chars
            if c.team == "D" and c.is_alive and c.name in GC_ROSTER_ORDER
        }
        missing = [n for n in GC_ROSTER_ORDER if n not in char_by_name]
        if missing:
            raise RuntimeError("Missing GC players: " + ", ".join(missing))

        selected = []
        assignments = {}
        self.pending_choices = []

        for player_index, name in enumerate(GC_ROSTER_ORDER):
            char = char_by_name[name]
            obs = build_obs(char, player_index, selected, self.opponent_name, self.variation_index)
            mask = valid_action_mask(player_index, selected)
            eps = 0.0 if self.greedy else self.epsilon
            action = choose_action(
                self.model, obs, mask, eps, self.device, self.rng
            )
            target = CANDIDATES[action]
            assignments[name] = target
            selected.append(action)

            next_obs = build_obs(char, player_index, selected, self.opponent_name, self.variation_index)
            self.pending_choices.append({
                "obs": obs,
                "action": action,
                "next_obs": next_obs,
                "valid_mask": mask.copy(),
            })

        self.assignments = assignments
        self.round_initialized = True
        return [
            SetupAssignment(name, assignments[name])
            for name in GC_ROSTER_ORDER
        ]

    def capture_setup_end(self, chars):
        self.setup_end_positions = {
            c.name: tuple(map(int, c.pos))
            for c in chars
            if c.name in GC_ROSTER_ORDER
        }
        self.setup_end_reached = {
            name: self.setup_end_positions.get(name) == target
            for name, target in self.assignments.items()
        }

    def finalize_episode(self, reward):
        for item in self.pending_choices:
            self.transition_sink(
                Transition(
                    obs=item["obs"],
                    action=int(item["action"]),
                    reward=float(reward),
                    next_obs=item["next_obs"],
                    done=True,
                    valid_mask=item["valid_mask"].astype(np.bool_),
                )
            )
        self.pending_choices = []


def _setup_bfs_distance(start, goal):
    return _static_setup_bfs_distance(start, goal)


def _manhattan(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _horizontal_zone(col):
    x = float(col) / float(max(1, WIDTH - 1))
    if x < 1.0 / 3.0:
        return 0
    if x < 2.0 / 3.0:
        return 1
    return 2


def setup_quality_reward(planner, variation_index):
    positions = dict(planner.setup_end_positions)
    breakdown = {
        "arrival": 0.0,
        "distance": 0.0,
        "coverage": 0.0,
        "crowding": 0.0,
        "role_spread": 0.0,
        "opening_connectivity": 0.0,
        "variation": 0.0,
    }

    reached_count = 0
    remaining_distances = []
    progress_scores = []

    for name in GC_ROSTER_ORDER:
        target = planner.assignments[name]
        actual = positions[name]
        remaining = _setup_bfs_distance(actual, target)
        remaining_distances.append(remaining)

        if remaining == 0:
            reached_count += 1
            breakdown["arrival"] += ARRIVAL_REWARD_PER_PLAYER
            progress_scores.append(1.0)
        else:
            effective = 12 if remaining >= 999 else remaining
            breakdown["distance"] += max(
                DISTANCE_PENALTY_CAP_PER_PLAYER,
                effective * DISTANCE_PENALTY_PER_CELL,
            )
            progress_scores.append(max(0.0, 1.0 - (effective / 12.0)))

    if reached_count == PLAYER_COUNT:
        breakdown["arrival"] += ALL_ARRIVED_BONUS

    progress_factor = max(
        MIN_PROGRESS_FACTOR,
        min(1.0, float(np.mean(progress_scores))),
    )

    zones = {_horizontal_zone(pos[1]) for pos in positions.values()}
    if len(zones) >= 3:
        raw_coverage = COVERAGE_THREE_ZONES_REWARD
    elif len(zones) == 2:
        raw_coverage = COVERAGE_TWO_ZONES_REWARD
    elif len(zones) == 1:
        raw_coverage = COVERAGE_ONE_ZONE_PENALTY
    else:
        raw_coverage = 0.0
    breakdown["coverage"] = raw_coverage * progress_factor

    names = list(positions)
    close_pairs = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if _manhattan(positions[names[i]], positions[names[j]]) <= CROWDING_DISTANCE:
                close_pairs += 1
    breakdown["crowding"] = max(
        CROWDING_PENALTY_CAP,
        close_pairs * CROWDING_PENALTY_PER_PAIR,
    )

    for role in ("SMOKE", "RECON"):
        role_names = [
            n for n in GC_ROSTER_ORDER
            if GC_SETUP_ROLES[n] == role
        ]
        a, b = role_names
        dist = _manhattan(positions[a], positions[b])
        if dist >= ROLE_PAIR_FAR_DISTANCE:
            breakdown["role_spread"] += ROLE_PAIR_FAR_REWARD
        elif dist <= ROLE_PAIR_CLOSE_DISTANCE:
            breakdown["role_spread"] += ROLE_PAIR_CLOSE_PENALTY

    breakdown["role_spread"] *= progress_factor

    # ---------------------------------------------------------------
    # Opening connectivity
    # ---------------------------------------------------------------
    # Only Xdll/eKo/Absol have position-bound Opening Origins.
    # For each player, score ONLY the nearest origin among every pattern.
    for name in ("Xdll", "eKo", "Absol"):
        ability = GC_SETUP_ROLES[name]
        pos = positions[name]
        min_dist, _pid = _nearest_opening_origin_distance(pos, ability)

        max_dist = OPENING_CONNECTIVITY_MAX_DISTANCE[ability]
        if min_dist >= 999:
            score = 0.0
        else:
            score = max(0.0, 1.0 - (float(min_dist) / max_dist))

        breakdown["opening_connectivity"] += (
            score * OPENING_CONNECTIVITY_PER_PLAYER
        )

    breakdown["opening_connectivity"] *= progress_factor

    # ---------------------------------------------------------------
    # Setup variation
    # ---------------------------------------------------------------
    # BALANCED / LEFT_LEAN / RIGHT_LEAN create several meaningful
    # starting shapes. This is mild and never overrides reachability.
    cols = [float(pos[1]) for pos in positions.values()]
    mean_x = float(np.mean(cols)) / max(1.0, float(WIDTH - 1))
    center = 0.5

    variation = SETUP_VARIATIONS[int(variation_index)]
    if variation == "BALANCED":
        target = center
    elif variation == "LEFT_LEAN":
        target = center - VARIATION_TARGET_OFFSET
    else:
        target = center + VARIATION_TARGET_OFFSET

    error = abs(mean_x - target)
    # Full reward at target, linearly goes to zero at 0.22 normalized width.
    variation_score = max(0.0, 1.0 - error / 0.22)
    breakdown["variation"] = (
        VARIATION_REWARD_MAX * variation_score * progress_factor
    )

    total = float(sum(breakdown.values()))
    return total, breakdown, reached_count, remaining_distances


def simulate_setup(model, device, epsilon, rng, replay=None, greedy=False, opponent_name=None, variation_index=None):
    game = SetupOnlyGame()

    if opponent_name is None:
        opponent_name = rng.choice(OPPONENT_NAMES)
    if variation_index is None:
        variation_index = rng.randrange(VARIATION_DIM)

    sink = (lambda t: replay.add(t)) if replay is not None else (lambda t: None)
    planner = TrainableSetupPlanner(
        model=model,
        device=device,
        epsilon=epsilon,
        rng=rng,
        transition_sink=sink,
        opponent_name=opponent_name,
        variation_index=variation_index,
        greedy=greedy,
    )

    # First call chooses all five targets.
    planner.initialize_round(game.chars)

    # Same conceptual movement order as the game:
    # roster order, one movement decision per player per Setup Tick.
    for _tick in range(int(DEFENDER_SETUP_TICKS)):
        for char in game.chars:
            nxt = planner.decide_setup_move(char, game.chars)
            nxt = (int(nxt[0]), int(nxt[1]))
            current = tuple(map(int, char.pos))

            if nxt == current:
                continue
            if not is_setup_position_allowed(*nxt):
                continue

            occupied = {
                tuple(map(int, other.pos))
                for other in game.chars
                if other is not char and other.is_alive
            }
            if nxt in occupied:
                continue

            char.pos = [nxt[0], nxt[1]]

    planner.capture_setup_end(game.chars)
    reward, breakdown, reached_count, remaining = setup_quality_reward(planner, variation_index)
    planner.finalize_episode(reward)

    finite_remaining = [12 if x >= 999 else x for x in remaining]

    return {
        "setup_reward": reward,
        "reward_breakdown": breakdown,
        "reached_count": reached_count,
        "all_arrived": int(reached_count == PLAYER_COUNT),
        "avg_remaining_distance": float(np.mean(finite_remaining)),
        "max_remaining_distance": int(max(finite_remaining)),
        "assignments": dict(planner.assignments),
        "final_positions": dict(planner.setup_end_positions),
        "opponent": str(opponent_name),
        "variation": SETUP_VARIATIONS[int(variation_index)],
        "opening_min_distance": {
            name: _nearest_opening_origin_distance(
                planner.setup_end_positions[name],
                GC_SETUP_ROLES[name],
            )[0]
            for name in ("Xdll", "eKo", "Absol")
        },
    }


def optimize(model, optimizer, replay, device, rng, batch_size, learning_start):
    if len(replay) < max(batch_size, learning_start):
        return None

    batch = replay.sample(batch_size, rng)
    obs = torch.from_numpy(np.stack([x.obs for x in batch])).float().to(device)
    actions = torch.tensor([x.action for x in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([x.reward for x in batch], dtype=torch.float32, device=device)

    q = model(obs)
    q_sa = q.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Every assignment receives the terminal Setup quality directly.
    loss = F.smooth_l1_loss(q_sa, rewards)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    return float(loss.item())


def evaluate(model, device, episodes, seed):
    model.eval()
    rng = random.Random(seed)

    # Evaluate the Cartesian product:
    # every opponent x every Setup variation.
    # If episodes > number of combinations, repeat the whole matrix evenly.
    combinations = [
        (opponent, variation_index)
        for opponent in OPPONENT_NAMES
        for variation_index in range(VARIATION_DIM)
    ]

    if not combinations:
        raise RuntimeError("No opponent/variation combinations available for evaluation")

    requested = max(int(episodes), len(combinations))
    repeats = int(math.ceil(requested / len(combinations)))

    schedule = []
    for _ in range(repeats):
        schedule.extend(combinations)

    # Respect the requested count after guaranteeing at least one full matrix.
    schedule = schedule[:requested]

    results = []
    for opponent, variation_index in schedule:
        results.append(
            simulate_setup(
                model=model,
                device=device,
                epsilon=0.0,
                rng=rng,
                replay=None,
                greedy=True,
                opponent_name=opponent,
                variation_index=variation_index,
            )
        )

    keys = (
        "arrival",
        "distance",
        "coverage",
        "crowding",
        "role_spread",
        "opening_connectivity",
        "variation",
    )

    by_variation = {}
    for variation in SETUP_VARIATIONS:
        subset = [x for x in results if x["variation"] == variation]
        by_variation[variation] = {
            "n": len(subset),
            "avg_reward": float(np.mean([x["setup_reward"] for x in subset])) if subset else 0.0,
            "all_arrived_rate": float(np.mean([x["all_arrived"] for x in subset])) if subset else 0.0,
            "avg_columns": {
                name: float(np.mean([x["final_positions"][name][1] for x in subset]))
                if subset else 0.0
                for name in GC_ROSTER_ORDER
            },
        }

    # Full opponent x variation matrix.
    opponent_variation_assignments = {}
    opponent_variation_rewards = {}

    for x in results:
        key = f"{x['opponent']} | {x['variation']}"

        sig = " | ".join(
            f"{name}={x['assignments'][name]}"
            for name in GC_ROSTER_ORDER
        )

        opponent_variation_assignments.setdefault(key, {})
        opponent_variation_assignments[key][sig] = (
            opponent_variation_assignments[key].get(sig, 0) + 1
        )

        opponent_variation_rewards.setdefault(key, [])
        opponent_variation_rewards[key].append(float(x["setup_reward"]))

    opponent_variation_rewards = {
        key: {
            "n": len(values),
            "avg_reward": float(np.mean(values)),
        }
        for key, values in opponent_variation_rewards.items()
    }

    # Also make it easy to compare all three variations within one opponent.
    by_opponent = {}
    for opponent in OPPONENT_NAMES:
        by_opponent[opponent] = {}
        for variation in SETUP_VARIATIONS:
            subset = [
                x for x in results
                if x["opponent"] == opponent
                and x["variation"] == variation
            ]

            by_opponent[opponent][variation] = {
                "n": len(subset),
                "avg_reward": (
                    float(np.mean([x["setup_reward"] for x in subset]))
                    if subset else 0.0
                ),
                "assignments": {
                    " | ".join(
                        f"{name}={x['assignments'][name]}"
                        for name in GC_ROSTER_ORDER
                    ): sum(
                        1
                        for y in subset
                        if all(
                            y["assignments"][name] == x["assignments"][name]
                            for name in GC_ROSTER_ORDER
                        )
                    )
                    for x in subset
                } if subset else {},
            }

    opening_distance = {
        name: float(np.mean([
            min(60, int(x["opening_min_distance"][name]))
            for x in results
        ]))
        for name in ("Xdll", "eKo", "Absol")
    }

    metrics = {
        "eval_count": len(results),
        "eval_matrix_size": len(combinations),
        "avg_setup_reward": float(np.mean([x["setup_reward"] for x in results])),
        "all_arrived_rate": float(np.mean([x["all_arrived"] for x in results])),
        "avg_arrived_players": float(np.mean([x["reached_count"] for x in results])),
        "avg_remaining_distance": float(np.mean([x["avg_remaining_distance"] for x in results])),
        "max_remaining_distance": int(max(x["max_remaining_distance"] for x in results)),
        "reward_breakdown": {
            key: float(np.mean([x["reward_breakdown"][key] for x in results]))
            for key in keys
        },
        "avg_nearest_opening_origin_distance": opening_distance,
        "by_variation": by_variation,
        "by_opponent": by_opponent,
        "opponent_variation_rewards": opponent_variation_rewards,
        "opponent_variation_assignments": opponent_variation_assignments,
    }

    model.train()
    return metrics


def save_checkpoint(path, model, optimizer, *, episode, global_step, best_setup_reward, best_all_arrived_rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "gc_defender_setup_dqn_setup_v3_contextual",
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "candidate_positions": list(CANDIDATES),
        "roster_order": list(GC_ROSTER_ORDER),
        "setup_ticks": int(DEFENDER_SETUP_TICKS),
        "max_candidate_bfs_distance": int(MAX_CANDIDATE_BFS_DISTANCE),
        "player_candidate_bfs_distances": PLAYER_CANDIDATE_BFS_DISTANCES,
        "roles": dict(GC_SETUP_ROLES),
        "opponent_names": list(OPPONENT_NAMES),
        "setup_variations": list(SETUP_VARIATIONS),
        "opening_connectivity_max_distance": dict(OPENING_CONNECTIVITY_MAX_DISTANCE),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode": int(episode),
        "global_step": int(global_step),
        "best_setup_reward": float(best_setup_reward),
        "best_all_arrived_rate": float(best_all_arrived_rate),
    }, path)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def epsilon_by_episode(episode, start, end, decay_episodes):
    t = min(1.0, max(0.0, episode / max(1, decay_episodes)))
    return end + (start - end) * (1.0 - t)


def train(args):
    if ACTION_DIM < PLAYER_COUNT:
        raise RuntimeError("Not enough Setup candidates")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = SetupQNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    replay = ReplayBuffer(args.replay_capacity)
    losses = deque(maxlen=100)

    best_setup_reward = -1e9
    best_all_arrived_rate = 0.0
    global_step = 0

    print(f"device={device}")
    print(f"OBS_DIM={OBS_DIM}")
    print(f"ACTION_DIM={ACTION_DIM}")
    print(f"GC roster={GC_ROSTER_ORDER}")
    print(f"roles={GC_SETUP_ROLES}")
    print(f"defender spawns={DEFENDER_SPAWNS}")
    print(f"setup ticks={DEFENDER_SETUP_TICKS}")
    print(f"candidates={CANDIDATES}")
    print(f"opponents={len(OPPONENT_NAMES)} presets")
    print(f"setup variations={SETUP_VARIATIONS}")
    print("opening connectivity=min distance to ANY matching Origin only")
    print(
        f"candidate reachability: BFS<={MAX_CANDIDATE_BFS_DISTANCE} "
        f"(setup={DEFENDER_SETUP_TICKS} ticks)"
    )
    for player_index, name in enumerate(GC_ROSTER_ORDER):
        distances = PLAYER_CANDIDATE_BFS_DISTANCES[player_index]
        reachable = [
            CANDIDATES[a]
            for a, dist in enumerate(distances)
            if dist <= MAX_CANDIDATE_BFS_DISTANCE
        ]
        print(
            f"  {name:10s} spawn={DEFENDER_SPAWNS[player_index]} "
            f"dist={distances} reachable={reachable}"
        )

    reachability_errors = validate_reachability_assignment_space()
    if reachability_errors:
        raise RuntimeError(
            "GC Setup reachable assignment validation failed:\\n- "
            + "\\n- ".join(reachability_errors)
        )

    print("mode=SETUP_ONLY (no Opening/Search/Retake/combat/opponent AI)")
    print(f"learning starts={args.learning_start} batch={args.batch_size}")

    try:
        for episode in range(1, args.episodes + 1):
            epsilon = epsilon_by_episode(
                episode,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay,
            )

            before = len(replay)
            result = simulate_setup(
                model=model,
                device=device,
                epsilon=epsilon,
                rng=rng,
                replay=replay,
                greedy=False,
            )
            added = len(replay) - before

            # One Setup episode contributes 5 decisions -> one update is enough.
            if len(replay) >= max(args.learning_start, args.batch_size):
                loss = optimize(
                    model,
                    optimizer,
                    replay,
                    device,
                    rng,
                    args.batch_size,
                    args.learning_start,
                )
                if loss is not None:
                    losses.append(loss)
                    global_step += 1

            recent_loss = float(np.mean(losses)) if losses else 0.0

            if episode <= 10 or episode % args.log_every == 0:
                print(
                    f"[{episode:6d}/{args.episodes}] "
                    f"eps={epsilon:.3f} | "
                    f"R={result['setup_reward']:+.3f} | "
                    f"arrived={result['reached_count']}/5 | "
                    f"remain={result['avg_remaining_distance']:.2f} | "
                    f"replay={len(replay)} (+{added}) | "
                    f"loss={recent_loss:.4f}"
                )

            if episode % args.save_every == 0:
                save_checkpoint(
                    LATEST_MODEL,
                    model,
                    optimizer,
                    episode=episode,
                    global_step=global_step,
                    best_setup_reward=best_setup_reward,
                    best_all_arrived_rate=best_all_arrived_rate,
                )

            if episode % args.eval_every == 0:
                metrics = evaluate(
                    model,
                    device,
                    args.eval_episodes,
                    args.seed + episode * 1000,
                )
                print("[EVAL]", metrics)

                with LOG_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(
                        {
                            "episode": episode,
                            "global_step": global_step,
                            **metrics,
                        },
                        ensure_ascii=False,
                    ) + "\n")

                score = metrics["avg_setup_reward"]
                arrived = metrics["all_arrived_rate"]

                if (
                    score > best_setup_reward
                    or (
                        abs(score - best_setup_reward) < 1e-9
                        and arrived > best_all_arrived_rate
                    )
                ):
                    best_setup_reward = score
                    best_all_arrived_rate = arrived
                    save_checkpoint(
                        BEST_MODEL,
                        model,
                        optimizer,
                        episode=episode,
                        global_step=global_step,
                        best_setup_reward=best_setup_reward,
                        best_all_arrived_rate=best_all_arrived_rate,
                    )
                    print(
                        f"[BEST] setup_reward={best_setup_reward:+.4f} "
                        f"all_arrived={best_all_arrived_rate:.3f}"
                    )

        save_checkpoint(
            FINAL_MODEL,
            model,
            optimizer,
            episode=args.episodes,
            global_step=global_step,
            best_setup_reward=best_setup_reward,
            best_all_arrived_rate=best_all_arrived_rate,
        )

    except KeyboardInterrupt:
        episode_now = locals().get("episode", 0)
        save_checkpoint(
            INTERRUPT_MODEL,
            model,
            optimizer,
            episode=episode_now,
            global_step=global_step,
            best_setup_reward=best_setup_reward,
            best_all_arrived_rate=best_all_arrived_rate,
        )
        print(f"\n[INTERRUPT] saved: {INTERRUPT_MODEL}")
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")

    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=50)

    p.add_argument("--learning-rate", type=float, default=LR)
    p.add_argument("--replay-capacity", type=int, default=REPLAY_CAPACITY)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--learning-start", type=int, default=LEARNING_START)

    p.add_argument("--epsilon-start", type=float, default=0.90)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=int, default=3000)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
