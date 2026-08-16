"""Runtime for the setup-only Ghost Champions Defender Setup model.

Must match train_defender_setup_gc.py setup-only OBS_DIM.
No dependency on learning_defender_setup_gc.py to avoid circular imports.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from map_data import NEW_MAZE_STR
from map_data_defender_setup import (
    DEFENDER_SETUP_TICKS,
    get_setup_mask,
    is_setup_position_allowed,
)
from gc_v1.map_data_defender_setup_positions_gc import (
    get_gc_setup_position_candidates,
)


GC_ROSTER_ORDER = ["Xdll", "SyouTa", "Absol", "eKo", "SugarZ3ro"]
MOVES_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

CANDIDATES = list(get_gc_setup_position_candidates())
ACTION_DIM = len(CANDIDATES)
PLAYER_COUNT = len(GC_ROSTER_ORDER)
OBS_DIM = PLAYER_COUNT + ACTION_DIM + 2 + ACTION_DIM * 2

_ROWS = [row.strip() for row in str(NEW_MAZE_STR).strip().splitlines() if row.strip()]
HEIGHT = len(_ROWS)
WIDTH = len(_ROWS[0])

DEFENDER_SPAWNS = [
    (r, c)
    for r, row in enumerate(_ROWS)
    for c, value in enumerate(row)
    if value == "4"
][:PLAYER_COUNT]

if len(DEFENDER_SPAWNS) < PLAYER_COUNT:
    raise RuntimeError(
        f"Defender spawn cells不足: {len(DEFENDER_SPAWNS)} < {PLAYER_COUNT}"
    )

MAX_CANDIDATE_BFS_DISTANCE = max(1, int(DEFENDER_SETUP_TICKS) - 2)


def _static_setup_bfs_distance(start, goal):
    start = tuple(map(int, start))
    goal = tuple(map(int, goal))

    if start == goal:
        return 0

    q = deque([(start, 0)])
    seen = {start}

    while q:
        (r, c), d = q.popleft()
        nd = d + 1

        for dr, dc in MOVES_4:
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

    matched_action_to_player = {}

    def augment(player_index, seen_actions):
        for action in range(ACTION_DIM):
            if action in unavailable or action in seen_actions:
                continue
            if not PLAYER_REACHABLE_MASKS[player_index, action]:
                continue

            seen_actions.add(action)
            previous_player = matched_action_to_player.get(action)

            if previous_player is None or augment(previous_player, seen_actions):
                matched_action_to_player[action] = player_index
                return True

        return False

    # constrained players first
    remaining_players.sort(
        key=lambda p: int(np.count_nonzero(PLAYER_REACHABLE_MASKS[p]))
    )

    for player_index in remaining_players:
        if not augment(player_index, set()):
            return False

    return True


def _valid_action_mask(player_index, selected_indices):
    selected = {int(x) for x in selected_indices}
    mask = PLAYER_REACHABLE_MASKS[int(player_index)].copy()

    for idx in selected:
        if 0 <= idx < ACTION_DIM:
            mask[idx] = False

    for action in np.flatnonzero(mask):
        if not _remaining_players_have_distinct_candidates(
            int(player_index) + 1,
            selected | {int(action)},
        ):
            mask[action] = False

    return mask

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "data" / "defender_setup_gc_data" / "dqn_defender_setup_gc_best.pt"


@dataclass
class SetupAssignment:
    player_name: str
    target: tuple[int, int]


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


def _build_obs(char, player_index, selected_indices):
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

    return np.concatenate([
        player_one_hot,
        selected_mask,
        np.asarray([own_r, own_c], dtype=np.float32),
        np.asarray(candidate_features, dtype=np.float32),
    ]).astype(np.float32)


class LearningDefenderSetupGCRuntime:
    def __init__(self, model_path=None, *, device=None, verbose=True):
        self.verbose = bool(verbose)
        self.game = None
        self.setup_mask = get_setup_mask()
        self.candidates = list(CANDIDATES)
        self.assignments = {}
        self.round_initialized = False

        self.model_path = Path(model_path) if model_path else MODEL_PATH
        if not self.model_path.is_absolute():
            self.model_path = (Path.cwd() / self.model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        ck_obs = int(checkpoint.get("obs_dim", -1))
        ck_action = int(checkpoint.get("action_dim", -1))
        ck_max_dist = checkpoint.get("max_candidate_bfs_distance")
        if ck_max_dist is not None and int(ck_max_dist) != MAX_CANDIDATE_BFS_DISTANCE:
            raise RuntimeError(
                "GC Setup reachability threshold mismatch: "
                f"checkpoint={int(ck_max_dist)} current={MAX_CANDIDATE_BFS_DISTANCE}"
            )
        if ck_obs != OBS_DIM:
            raise RuntimeError(
                f"GC Setup OBS_DIM mismatch: checkpoint={ck_obs} current={OBS_DIM}"
            )
        if ck_action != ACTION_DIM:
            raise RuntimeError(
                f"GC Setup ACTION_DIM mismatch: checkpoint={ck_action} current={ACTION_DIM}"
            )

        ck_candidates = checkpoint.get("candidate_positions")
        if ck_candidates is not None:
            normalized = [tuple(map(int, p)) for p in ck_candidates]
            if normalized != self.candidates:
                raise RuntimeError("GC Setup candidate positions changed")

        self.model = SetupQNet().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.episode = int(checkpoint.get("episode", 0))
        self.best_setup_reward = float(
            checkpoint.get("best_setup_reward", float("nan"))
        )
        self.best_all_arrived_rate = float(
            checkpoint.get("best_all_arrived_rate", float("nan"))
        )

        if self.verbose:
            print(
                "[GC D-SETUP] loaded "
                f"{self.model_path} episode={self.episode} "
                f"bestSetup={self.best_setup_reward:+.4f} "
                f"allArrived={self.best_all_arrived_rate:.3f}"
            )

    def set_game(self, game):
        self.game = game

    def reset_round(self):
        self.assignments.clear()
        self.round_initialized = False

    def validate(self):
        errors = []
        if len(self.candidates) < PLAYER_COUNT:
            errors.append(
                f"GC Setup candidates must contain at least {PLAYER_COUNT}"
            )
        for pos in self.candidates:
            if not is_setup_position_allowed(*pos):
                errors.append(f"GC Setup candidate forbidden: {pos}")
        return errors

    def initialize_round(self, chars):
        errors = self.validate()
        if errors:
            raise RuntimeError("\n".join(errors))

        by_name = {
            c.name: c for c in chars
            if getattr(c, "team", None) == "D"
            and getattr(c, "is_alive", True)
            and c.name in GC_ROSTER_ORDER
        }
        missing = [n for n in GC_ROSTER_ORDER if n not in by_name]
        if missing:
            raise RuntimeError("Missing GC players: " + ", ".join(missing))

        selected = []
        assignments = {}

        for player_index, name in enumerate(GC_ROSTER_ORDER):
            char = by_name[name]
            obs = _build_obs(char, player_index, selected)

            valid = _valid_action_mask(
                player_index,
                selected,
            )

            if not np.any(valid):
                raise RuntimeError(
                    f"GC Setup runtime has no feasible action for {name}; "
                    f"selected={selected}"
                )

            with torch.no_grad():
                x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                q = self.model(x)[0].detach().cpu().numpy()

            q = q.copy()
            q[~valid] = -1e30
            action = int(np.argmax(q))

            selected.append(action)
            assignments[name] = self.candidates[action]

        self.assignments = assignments
        self.round_initialized = True

        return [
            SetupAssignment(name, assignments[name])
            for name in GC_ROSTER_ORDER
        ]

    def _neighbors(self, pos):
        r, c = pos
        h = len(self.setup_mask)
        w = len(self.setup_mask[0])
        for dr, dc in MOVES_4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and is_setup_position_allowed(nr, nc):
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
                if nxt in blocked or nxt in prev:
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
        if not self.round_initialized:
            self.initialize_round(chars)

        target = self.get_target(char)
        if target is None:
            return list(char.pos)

        occupied = {
            tuple(map(int, other.pos))
            for other in chars
            if getattr(other, "is_alive", True)
            and other is not char
        }

        nxt = self._bfs_next_step(
            tuple(map(int, char.pos)),
            target,
            blocked=occupied,
        )
        return [int(nxt[0]), int(nxt[1])]


DefenderSetupGCRuntime = LearningDefenderSetupGCRuntime
