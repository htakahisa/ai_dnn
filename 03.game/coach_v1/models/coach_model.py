"""Five-slot recurrent coach actor and a training-only centralized critic."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from map_data_defender_setup import get_setup_mask

from coach_v1.common.constants import MAP_COLUMNS, MAP_ROWS, MOVEMENT_DELTAS, ROSTER_SIZE
from coach_v1.common.types import MovementAction, ObjectiveAction, TacticalIntent
from coach_v1.common.versions import COACH_OBSERVATION_VERSION
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.observation.coach_encoder import (
    COACH_GRID_CHANNELS, COACH_VECTOR_FIELDS, CoachObservation,
    CoachObservationEncoder, CoachObservationInputError,
)


MOVES = tuple(MovementAction)
INTENTS = tuple(TacticalIntent)
OBJECTIVES = tuple(ObjectiveAction)
_SETUP_ALLOWED = np.asarray(get_setup_mask(), dtype=np.int8) == 0


@dataclass(frozen=True)
class CoachModelConfig:
    grid_channels: int = len(COACH_GRID_CHANNELS)
    vector_features: int = len(COACH_VECTOR_FIELDS)
    hidden_channels: int = 32
    hidden_features: int = 96
    action_feedback: bool = False

    def to_dict(self) -> dict:
        result = asdict(self)
        if not self.action_feedback:
            result.pop("action_feedback")  # v1 checkpoint compatibility
        return result


class CoachActorModel(nn.Module):
    """One forward computes all five slots; recurrent state is round local."""

    def __init__(self, config: CoachModelConfig = CoachModelConfig()) -> None:
        super().__init__()
        if (config.grid_channels != len(COACH_GRID_CHANNELS)
                or config.vector_features != len(COACH_VECTOR_FIELDS)
                or min(config.hidden_channels, config.hidden_features) <= 0):
            raise ValueError("invalid coach model configuration")
        self.config = config
        width, features = config.hidden_channels, config.hidden_features
        self.spatial = nn.Sequential(
            nn.Conv2d(config.grid_channels, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
        )
        self.input = nn.Sequential(nn.Linear(width + config.vector_features, features), nn.ReLU())
        self.memory = nn.GRUCell(features, features)
        self.slot = nn.Sequential(nn.Linear(features + 14 + (7 if config.action_feedback else 0), features), nn.ReLU())
        self.movement = nn.Linear(features, len(MOVES))
        self.intent = nn.Linear(features, len(INTENTS))
        self.objective = nn.Linear(features, len(OBJECTIVES))

    def forward(self, grid: torch.Tensor, vector: torch.Tensor,
                hidden: torch.Tensor | None = None,
                feedback: torch.Tensor | None = None):
        if grid.ndim != 4 or grid.shape[1:] != (self.config.grid_channels, MAP_ROWS, MAP_COLUMNS):
            raise ValueError("invalid coach grid shape")
        if vector.shape != (grid.shape[0], self.config.vector_features):
            raise ValueError("invalid coach vector shape")
        if hidden is not None and hidden.shape != (grid.shape[0], self.config.hidden_features):
            raise ValueError("invalid recurrent state shape")
        spatial = self.spatial(grid).mean((2, 3))
        state = self.memory(self.input(torch.cat((spatial, vector), 1)), hidden)
        # The fixed encoder puts fourteen fields per slot at the end of vector.
        slots = vector[:, -ROSTER_SIZE * 14:].reshape(-1, ROSTER_SIZE, 14)
        shared = state[:, None, :].expand(-1, ROSTER_SIZE, -1)
        if self.config.action_feedback:
            if feedback is None:
                feedback = vector.new_zeros((grid.shape[0], ROSTER_SIZE, 7))
            if feedback.shape != (grid.shape[0], ROSTER_SIZE, 7):
                raise ValueError("invalid previous action feedback shape")
            features = self.slot(torch.cat((shared, slots, feedback), -1))
        else:
            if feedback is not None:
                raise ValueError("legacy coach model does not accept action feedback")
            features = self.slot(torch.cat((shared, slots), -1))
        return self.movement(features), self.intent(features), self.objective(features), state


class CoachCritic(nn.Module):
    """Privileged training network; its weights are excluded from actor files."""

    def __init__(self, config: CoachModelConfig = CoachModelConfig()) -> None:
        super().__init__()
        width = config.hidden_channels
        self.spatial = nn.Sequential(
            nn.Conv2d(config.grid_channels + 1, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(width + config.vector_features, config.hidden_features),
                                   nn.ReLU(), nn.Linear(config.hidden_features, 1))
        self.config = config

    def forward(self, grid: torch.Tensor, vector: torch.Tensor,
                enemy_truth: torch.Tensor) -> torch.Tensor:
        if enemy_truth.shape != (grid.shape[0], 1, MAP_ROWS, MAP_COLUMNS):
            raise ValueError("invalid critic-only enemy truth shape")
        encoded = self.spatial(torch.cat((grid, enemy_truth), 1)).mean((2, 3))
        return self.value(torch.cat((encoded, vector), 1)).squeeze(-1)


@dataclass(frozen=True)
class CoachActionMask:
    movement: np.ndarray  # bool [5, 5]
    objective: np.ndarray  # bool [5, 3]


def action_feedback(current: CoachObservation,
                    previous: CoachObservation | None,
                    previous_actions: tuple[CoachInstruction, ...] | None) -> np.ndarray:
    """Encode only own previous instructions and publicly observed movement."""
    if not isinstance(current, CoachObservation):
        raise TypeError("current coach observation required")
    result = np.zeros((ROSTER_SIZE, 7), dtype=np.float32)
    if previous is None and previous_actions is None:
        return result
    if (not isinstance(previous, CoachObservation) or previous_actions is None
            or len(previous_actions) != ROSTER_SIZE):
        raise ValueError("previous observation and five actions must be paired")
    if (current.version != previous.version or current.map_hash != previous.map_hash
            or current.watch_points_hash != previous.watch_points_hash):
        raise ValueError("previous coach observation contract mismatch")
    old_slots = previous.vector[-ROSTER_SIZE * 14:].reshape(ROSTER_SIZE, 14)
    new_slots = current.vector[-ROSTER_SIZE * 14:].reshape(ROSTER_SIZE, 14)
    for slot, instruction in enumerate(previous_actions):
        if not isinstance(instruction, CoachInstruction) or not isinstance(instruction.movement, MovementAction):
            raise ValueError("invalid previous coach instruction")
        if not old_slots[slot, 0] or not new_slots[slot, 0]:
            continue
        movement = instruction.movement
        result[slot, MOVES.index(movement)] = 1.0
        if movement is not MovementAction.STAY:
            changed = not np.array_equal(old_slots[slot, 4:6], new_slots[slot, 4:6])
            result[slot, 5 if changed else 6] = 1.0
    return result


def legal_action_mask(observation: CoachObservation) -> CoachActionMask:
    """Mask walls, allied occupancy, bounds and impossible actions."""
    if not isinstance(observation, CoachObservation):
        raise TypeError("coach action mask requires actor observation")
    grid, vector = observation.grid, observation.vector
    if grid.shape != (len(COACH_GRID_CHANNELS), MAP_ROWS, MAP_COLUMNS) or vector.shape != (len(COACH_VECTOR_FIELDS),):
        raise CoachObservationInputError("coach observation shape mismatch")
    fields = {name: i for i, name in enumerate(COACH_VECTOR_FIELDS)}
    movement = np.zeros((ROSTER_SIZE, len(MOVES)), dtype=np.bool_)
    objective = np.zeros((ROSTER_SIZE, len(OBJECTIVES)), dtype=np.bool_)
    planted = bool(vector[fields["spike_planted"]])
    attacker = bool(vector[fields["side_attacker"]])
    setup = bool(vector[fields["defender_setup"]])
    ally_occupied = np.zeros((MAP_ROWS, MAP_COLUMNS), dtype=np.bool_)
    for ally_slot in range(ROSTER_SIZE):
        ally_occupied |= grid[COACH_GRID_CHANNELS.index(f"ally_slot_{ally_slot}")] > 0
    for slot in range(ROSTER_SIZE):
        prefix = f"slot_{slot}_"
        movement[slot, MOVES.index(MovementAction.STAY)] = True
        objective[slot, OBJECTIVES.index(ObjectiveAction.NONE)] = True
        if not vector[fields[prefix + "alive"]]:
            continue
        row = int(round(float(vector[fields[prefix + "row"]]) * (MAP_ROWS - 1)))
        col = int(round(float(vector[fields[prefix + "column"]]) * (MAP_COLUMNS - 1)))
        if grid[COACH_GRID_CHANNELS.index(f"ally_slot_{slot}"), row, col] != 1:
            raise CoachObservationInputError("slot vector and grid position disagree")
        for index, move in enumerate(MOVES):
            dr, dc = MOVEMENT_DELTAS[move.value]
            nr, nc = row + dr, col + dc
            if (0 <= nr < MAP_ROWS and 0 <= nc < MAP_COLUMNS
                    and grid[COACH_GRID_CHANNELS.index("walkable"), nr, nc] == 1
                    and not ally_occupied[nr, nc]
                    and (not setup or not attacker and _SETUP_ALLOWED[nr, nc])):
                movement[slot, index] = True
        if (attacker and not planted and vector[fields[prefix + "has_spike"]]
                and grid[COACH_GRID_CHANNELS.index("plantable"), row, col]):
            objective[slot, OBJECTIVES.index(ObjectiveAction.PLANT)] = True
        if not attacker and planted:
            planted_cells = np.argwhere(grid[COACH_GRID_CHANNELS.index("spike_planted")] > 0)
            if len(planted_cells) == 1 and max(abs(int(planted_cells[0, 0]) - row),
                                                abs(int(planted_cells[0, 1]) - col)) <= 1:
                objective[slot, OBJECTIVES.index(ObjectiveAction.DEFUSE)] = True
    movement.setflags(write=False)
    objective.setflags(write=False)
    return CoachActionMask(movement, objective)


class CoachPolicy:
    """Actor-only checkpoint policy for TeamExecutionCoordinator."""

    def __init__(self, model: CoachActorModel, encoder: CoachObservationEncoder,
                 *, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.encoder = encoder
        self.device = device
        self.hidden: torch.Tensor | None = None
        self._previous_observation: CoachObservation | None = None
        self._previous_actions: tuple[CoachInstruction, ...] | None = None

    def reset_round(self) -> None:
        self.hidden = None
        self._previous_observation = None
        self._previous_actions = None

    def act(self, observation: CoachObservation) -> tuple[CoachInstruction, ...]:
        if (not isinstance(observation, CoachObservation)
                or observation.version != COACH_OBSERVATION_VERSION
                or observation.map_hash != self.encoder.map_hash
                or observation.watch_points_hash != self.encoder.watch_points_hash):
            raise CoachObservationInputError("coach policy observation mismatch")
        mask = legal_action_mask(observation)
        feedback = action_feedback(observation, self._previous_observation,
                                   self._previous_actions) if self.model.config.action_feedback else None
        with torch.no_grad():
            grid = torch.as_tensor(observation.grid.copy(), device=self.device)[None]
            vector = torch.as_tensor(observation.vector.copy(), device=self.device)[None]
            feedback_tensor = (torch.as_tensor(feedback, device=self.device)[None]
                               if feedback is not None else None)
            move, intent, objective, self.hidden = self.model(
                grid, vector, self.hidden, feedback=feedback_tensor)
            move = move[0].masked_fill(~torch.as_tensor(mask.movement.copy(), device=self.device), -torch.inf)
            objective = objective[0].masked_fill(~torch.as_tensor(mask.objective.copy(), device=self.device), -torch.inf)
            instructions = tuple(CoachInstruction(
                MOVES[int(move[slot].argmax())], OBJECTIVES[int(objective[slot].argmax())],
                INTENTS[int(intent[0, slot].argmax())],
            ) for slot in range(ROSTER_SIZE))
            self._previous_observation = observation
            self._previous_actions = instructions
            return instructions
