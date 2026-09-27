"""Character action contract shared by curriculum training and later runtime wiring.

This module does not simulate a round or choose movement. The caller supplies
the coach instruction and safe team snapshots for each step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from coach_v1.common.constants import FIXED_ROSTER, MOVEMENT_DELTAS
from coach_v1.common.types import Facing, GridPosition, ObjectiveAction
from coach_v1.observation.character_encoder import (
    CharacterInputError, CharacterObservation, CharacterObservationEncoder,
    CoachInstruction,
)
from coach_v1.perception.belief_memory import BeliefSnapshot
from coach_v1.perception.team_perception import TeamPerceptionSnapshot


class CurriculumStage(str, Enum):
    ONE_V_ONE = "1v1"
    TWO_V_ONE = "2v1"


@dataclass(frozen=True)
class CharacterAction:
    facing: Facing
    use_ability: bool = False
    target: Optional[GridPosition] = None


@dataclass(frozen=True)
class CharacterStep:
    observation: CharacterObservation
    instruction: CoachInstruction
    position: GridPosition
    ability_name: str
    is_alive: bool


class CharacterEnvironment:
    """Build one actor step and translate its legal action to the game API."""

    def __init__(self) -> None:
        self.encoder = CharacterObservationEncoder()

    def prepare(self, snapshot: TeamPerceptionSnapshot, belief: BeliefSnapshot,
                *, situation: str, slot: int, instruction: CoachInstruction,
                curriculum: Optional[CurriculumStage] = None) -> CharacterStep:
        if curriculum is not None:
            if not isinstance(curriculum, CurriculumStage):
                raise CharacterInputError("unknown curriculum stage")
            expected_allies = 1 if curriculum is CurriculumStage.ONE_V_ONE else 2
            if (sum(ally.is_alive for ally in snapshot.allies) != expected_allies
                    or sum(enemy.is_alive for enemy in snapshot.enemies) != 1):
                raise CharacterInputError("snapshot does not match curriculum stage")
        observation = self.encoder.encode(
            snapshot, belief, situation=situation, slot=slot, instruction=instruction,
        )
        ally = snapshot.allies[slot]
        if curriculum is not None and not ally.is_alive:
            raise CharacterInputError("curriculum actor must be alive")
        return CharacterStep(observation, instruction, ally.position,
                             FIXED_ROSTER[slot].normal_ability, ally.is_alive)

    def resolve(self, step: CharacterStep, action: CharacterAction) -> tuple:
        """Return the existing controller format; reject masked actions."""
        if not isinstance(step, CharacterStep) or not isinstance(action, CharacterAction):
            raise CharacterInputError("expected prepared step and character action")
        if not step.is_alive:
            raise CharacterInputError("dead character cannot act")
        if not isinstance(action.facing, Facing):
            raise CharacterInputError("facing must be one of eight directions")
        if not isinstance(action.use_ability, bool):
            raise CharacterInputError("use_ability must be bool")
        if action.use_ability:
            if not bool(step.observation.mask.ability_use[1]):
                raise CharacterInputError("ability use is masked")
            if (not isinstance(action.target, tuple) or len(action.target) != 2
                    or any(not isinstance(value, int) or isinstance(value, bool)
                           for value in action.target)):
                raise CharacterInputError("ability target must be an integer map cell")
            row, column = action.target
            if (not 0 <= row < step.observation.mask.ability_target.shape[0]
                    or not 0 <= column < step.observation.mask.ability_target.shape[1]
                    or not bool(step.observation.mask.ability_target[row, column])):
                raise CharacterInputError("ability target is masked")
        elif action.target is not None:
            raise CharacterInputError("target requires ability use")

        if step.instruction.objective is not ObjectiveAction.NONE:
            return step.position, step.instruction.objective.value
        if action.use_ability:
            return step.position, {
                "ability": step.ability_name, "target": action.target,
                "facing": action.facing.value,
            }
        dr, dc = MOVEMENT_DELTAS[step.instruction.movement.value]
        position = step.position[0] + dr, step.position[1] + dc
        return position, {"facing": action.facing.value}
