"""Small immutable types shared by coach_v1 modules.

This module intentionally contains no game objects.  In particular, actor-side
code must not receive a live game, Character, or perceived-object wrapper.
The perception DTOs themselves belong to Task 03 and are not defined here.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


GridPosition = Tuple[int, int]


class Side(str, Enum):
    ATTACKER = "attacker"
    DEFENDER = "defender"


class ModelFamily(str, Enum):
    COACH = "coach"
    CHARACTER = "character"


class Facing(str, Enum):
    N = "N"
    NE = "NE"
    E = "E"
    SE = "SE"
    S = "S"
    SW = "SW"
    W = "W"
    NW = "NW"


class MovementAction(str, Enum):
    STAY = "STAY"
    MOVE_N = "MOVE_N"
    MOVE_E = "MOVE_E"
    MOVE_S = "MOVE_S"
    MOVE_W = "MOVE_W"


class ObjectiveAction(str, Enum):
    NONE = "NONE"
    PLANT = "PLANT"
    DEFUSE = "DEFUSE"


class TacticalIntent(str, Enum):
    HOLD = "HOLD"
    ADVANCE = "ADVANCE"
    ENTRY = "ENTRY"
    SUPPORT_ENTRY = "SUPPORT_ENTRY"
    WAIT_TEAM = "WAIT_TEAM"
    CLEAR_AREA = "CLEAR_AREA"
    MULTI_PEEK = "MULTI_PEEK"
    RETREAT = "RETREAT"
    UTILITY_REQUEST = "UTILITY_REQUEST"


@dataclass(frozen=True)
class RosterSlot:
    slot: int
    character_name: str
    checkpoint_id: str
    normal_ability: str


@dataclass(frozen=True)
class ModelTarget:
    """Identifies the owner of one checkpoint without importing model code."""

    family: ModelFamily
    target_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target_id, str)
            or not self.target_id
            or self.target_id.strip() != self.target_id
        ):
            raise ValueError("target_id must be a non-empty, trimmed string")

    @classmethod
    def coach(cls, side: Side) -> "ModelTarget":
        return cls(ModelFamily.COACH, side.value)

    @classmethod
    def character(cls, checkpoint_id: str) -> "ModelTarget":
        return cls(ModelFamily.CHARACTER, checkpoint_id)
