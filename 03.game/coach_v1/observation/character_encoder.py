"""Actor-safe observation and legal action mask for one fixed-roster character."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coach_v1.common.constants import FIXED_ROSTER, MAP_COLUMNS, MAP_ROWS, MOVEMENT_DELTAS
from coach_v1.common.types import ModelFamily, MovementAction, ObjectiveAction, TacticalIntent
from coach_v1.common.versions import CHARACTER_ACTION_VERSION, CHARACTER_OBSERVATION_VERSION
from coach_v1.observation.coach_encoder import CoachObservationEncoder
from coach_v1.perception.belief_memory import BeliefSnapshot
from coach_v1.perception.team_perception import TeamPerceptionSnapshot


CHARACTER_GRID_CHANNELS = ("self_position", "coach_destination")
CHARACTER_VECTOR_FIELDS = (
    *(f"self_slot_{i}" for i in range(5)),
    *(f"coach_move_{action.value}" for action in MovementAction),
    *(f"coach_objective_{action.value}" for action in ObjectiveAction),
    *(f"intent_{intent.value}" for intent in TacticalIntent),
)


class CharacterInputError(ValueError):
    """Invalid character instruction, observation, or action."""


@dataclass(frozen=True)
class CoachInstruction:
    movement: MovementAction
    objective: ObjectiveAction
    intent: TacticalIntent


@dataclass(frozen=True)
class CharacterActionMask:
    facing: np.ndarray  # bool [8]
    ability_use: np.ndarray  # bool [2], indices 0=no, 1=yes
    ability_target: np.ndarray  # bool [26, 44]


@dataclass(frozen=True)
class CharacterObservation:
    grid: np.ndarray  # float32 [29, 26, 44]
    vector: np.ndarray  # float32 [106]
    mask: CharacterActionMask
    version: str
    map_hash: str
    watch_points_hash: str


class CharacterObservationEncoder:
    """Adds only own slot and coach's external instruction to shared actor data."""

    def __init__(self) -> None:
        self._coach_encoder = CoachObservationEncoder()

    def validate_checkpoint(self, metadata: object, *, slot: int) -> None:
        if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < len(FIXED_ROSTER):
            raise CharacterInputError("invalid character slot")
        if (getattr(metadata, "model_family", None) is not ModelFamily.CHARACTER
                or getattr(metadata, "target_id", None) != FIXED_ROSTER[slot].checkpoint_id
                or getattr(metadata, "observation_version", None) != CHARACTER_OBSERVATION_VERSION
                or getattr(metadata, "action_version", None) != CHARACTER_ACTION_VERSION
                or getattr(metadata, "map_hash", None) != self._coach_encoder.map_hash
                or getattr(metadata, "watch_points_hash", None) != self._coach_encoder.watch_points_hash
                or getattr(metadata, "roster", None) != tuple(item.character_name for item in FIXED_ROSTER)):
            raise CharacterInputError("checkpoint observation metadata mismatch")

    def encode(self, snapshot: TeamPerceptionSnapshot, belief: BeliefSnapshot,
               *, situation: str, slot: int, instruction: CoachInstruction) -> CharacterObservation:
        if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < len(FIXED_ROSTER):
            raise CharacterInputError("slot must be a fixed roster index")
        if not isinstance(instruction, CoachInstruction) or not all((
            isinstance(instruction.movement, MovementAction),
            isinstance(instruction.objective, ObjectiveAction),
            isinstance(instruction.intent, TacticalIntent),
        )):
            raise CharacterInputError("invalid coach instruction")
        shared = self._coach_encoder.encode(snapshot, belief, situation=situation)
        ally = snapshot.allies[slot]
        row, column = ally.position
        grid = np.zeros((shared.grid.shape[0] + 2, MAP_ROWS, MAP_COLUMNS), dtype=np.float32)
        grid[:shared.grid.shape[0]] = shared.grid
        if ally.is_alive:
            grid[-2, row, column] = 1.0
            dr, dc = MOVEMENT_DELTAS[instruction.movement.value]
            destination = row + dr, column + dc
            if (0 <= destination[0] < MAP_ROWS and 0 <= destination[1] < MAP_COLUMNS
                    and shared.grid[0, destination[0], destination[1]] == 1):
                grid[-1, destination[0], destination[1]] = 1.0
        vector = np.zeros(shared.vector.size + len(CHARACTER_VECTOR_FIELDS), dtype=np.float32)
        vector[:shared.vector.size] = shared.vector
        offset = shared.vector.size
        fields = {name: i for i, name in enumerate(CHARACTER_VECTOR_FIELDS)}
        vector[offset + fields[f"self_slot_{slot}"]] = 1.0
        vector[offset + fields[f"coach_move_{instruction.movement.value}"]] = 1.0
        vector[offset + fields[f"coach_objective_{instruction.objective.value}"]] = 1.0
        vector[offset + fields[f"intent_{instruction.intent.value}"]] = 1.0

        facing = np.full(8, ally.is_alive, dtype=np.bool_)
        target = np.zeros((MAP_ROWS, MAP_COLUMNS), dtype=np.bool_)
        ability = FIXED_ROSTER[slot].normal_ability
        if ally.is_alive and ally.normal_ability_charges > 0 and ability != "HUNT" and instruction.objective is ObjectiveAction.NONE:
            for tr in range(MAP_ROWS):
                for tc in range(MAP_COLUMNS):
                    if shared.grid[0, tr, tc] == 0:
                        continue
                    if ability in ("FLASH", "RECON") and not _projectile_can_launch(
                        (row, column), (tr, tc), shared.grid[0]
                    ):
                        continue
                    target[tr, tc] = True
        ability_use = np.array([ally.is_alive, bool(target.any())], dtype=np.bool_)
        for array in (grid, vector, facing, ability_use, target):
            array.setflags(write=False)
        return CharacterObservation(
            grid, vector, CharacterActionMask(facing, ability_use, target),
            CHARACTER_OBSERVATION_VERSION, shared.map_hash, shared.watch_points_hash,
        )


def _projectile_can_launch(start: tuple[int, int], target: tuple[int, int],
                           walkable: np.ndarray) -> bool:
    """Match the first step of the game's Bresenham projectile path."""
    if start == target:
        return False
    y0, x0 = start
    scale = max(MAP_ROWS, MAP_COLUMNS) * 3
    y1 = y0 + (target[0] - y0) * scale
    x1 = x0 + (target[1] - x0) * scale
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    e2 = 2 * err
    if e2 >= dy:
        err += dy
        x0 += sx
    if e2 <= dx:
        y0 += sy
    return 0 <= y0 < MAP_ROWS and 0 <= x0 < MAP_COLUMNS and bool(walkable[y0, x0])
