"""Fixed v1 constants and output locations.

The current map and roster are deliberately fixed.  Changing either is a
checkpoint-incompatible design change and requires retraining.
"""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Tuple

from .types import Facing, GridPosition, RosterSlot


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_DIR / "config"
CHECKPOINTS_DIR = PACKAGE_DIR / "checkpoints"
LOGS_DIR = PACKAGE_DIR / "logs"
REPORTS_DIR = PACKAGE_DIR / "reports"

MAP_ROWS = 26
MAP_COLUMNS = 44
ROSTER_SIZE = 5

FIXED_ROSTER: Tuple[RosterSlot, ...] = (
    RosterSlot(0, "ごりまる", "gorimaru", "SMOKE"),
    RosterSlot(1, "ごんごん", "gongon", "HUNT"),
    RosterSlot(2, "ごんた", "gonta", "RECON"),
    RosterSlot(3, "くんた", "kunta", "FLASH"),
    RosterSlot(4, "くりまる", "kurimaru", "FLASH"),
)
FIXED_ROSTER_NAMES = tuple(slot.character_name for slot in FIXED_ROSTER)
CHARACTER_CHECKPOINT_IDS = tuple(slot.checkpoint_id for slot in FIXED_ROSTER)

MOVEMENT_DELTAS: Mapping[str, GridPosition] = MappingProxyType(
    {
        "MOVE_N": (-1, 0),
        "MOVE_E": (0, 1),
        "MOVE_S": (1, 0),
        "MOVE_W": (0, -1),
        "STAY": (0, 0),
    }
)
FACING_DELTAS: Mapping[Facing, GridPosition] = MappingProxyType(
    {
        Facing.N: (-1, 0),
        Facing.NE: (-1, 1),
        Facing.E: (0, 1),
        Facing.SE: (1, 1),
        Facing.S: (1, 0),
        Facing.SW: (1, -1),
        Facing.W: (0, -1),
        Facing.NW: (-1, -1),
    }
)


@dataclass(frozen=True)
class RuntimePaths:
    checkpoints: Path
    logs: Path
    reports: Path

    def ensure_exists(self) -> None:
        """Create writable output directories when a trainer explicitly starts."""

        for path in (self.checkpoints, self.logs, self.reports):
            path.mkdir(parents=True, exist_ok=True)


DEFAULT_RUNTIME_PATHS = RuntimePaths(CHECKPOINTS_DIR, LOGS_DIR, REPORTS_DIR)

COACH_CHECKPOINT_PATHS = MappingProxyType(
    {
        "attacker": CHECKPOINTS_DIR / "coach" / "attacker",
        "defender": CHECKPOINTS_DIR / "coach" / "defender",
    }
)
CHARACTER_CHECKPOINT_PATHS = MappingProxyType(
    {
        checkpoint_id: CHECKPOINTS_DIR / "characters" / checkpoint_id
        for checkpoint_id in CHARACTER_CHECKPOINT_IDS
    }
)
