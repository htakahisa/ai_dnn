"""Legal masks and translation to the existing single-action game API."""

from dataclasses import dataclass
import numpy as np

from frc_v1 import ABILITIES, ULTIMATES, FACING, FACING_STEPS
from game_core import ASH_RANGE_CELLS, DANCE_MAX_HP

KINDS = ("STAY", "N", "E", "S", "W", "PLANT", "DEFUSE", "COLLECT_ORB", "ABILITY", "ULTIMATE")
MOVE_STEPS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
PHASES = ("PREPARE", "ENTRY", "CLEAR", "PLANT", "POSTPLANT", "REGROUP", "DEFEND", "RETAKE")
INTENTS = ("HOLD", "ADVANCE", "ENTRY", "SUPPORT_ENTRY", "WAIT_TEAM", "CLEAR_AREA", "MULTI_PEEK", "RETREAT", "UTILITY_REQUEST")


@dataclass(frozen=True)
class FrcAction:
    kind: str = "STAY"
    facing: str = "N"
    target: tuple[int, int] | None = None
    ally_slot: int | None = None


@dataclass(frozen=True)
class TeamDecision:
    actions: tuple[FrcAction, ...]
    site: int = 0
    phase: str = "PREPARE"
    entry: int = 2
    follow: int = 4
    objective: int = 0
    intents: tuple[int, ...] = (3, 3, 2, 3, 3)


@dataclass(frozen=True)
class ActionMasks:
    kind: np.ndarray
    facing: np.ndarray
    ultimate_facing: np.ndarray
    target: np.ndarray  # [slot, ability/ultimate, flattened cell]; DANCE uses first 5 entries.


def build_masks(snapshot):
    grid = np.asarray(snapshot.grid)
    rows, columns = grid.shape
    kinds = np.zeros((5, len(KINDS)), dtype=bool)
    facings = np.ones((5, 8), dtype=bool)
    ultimate_facings = np.ones((5, 8), dtype=bool)
    targets = np.zeros((5, 2, rows * columns), dtype=bool)
    setup_cells = set(snapshot.setup_cells)
    occupied_allies = {a.position for a in snapshot.allies if a.alive}
    sightings = {s.position for s in snapshot.sightings}
    walkable = grid != 1
    target_rows, target_columns = np.indices(grid.shape)
    for a in snapshot.allies:
        kinds[a.slot, 0] = True
        if a.forced_facing or not a.alive:
            facings[a.slot] = False
            facings[a.slot, FACING.index(a.facing)] = True
        if not a.alive or a.movement_disabled > 0 or a.warping:
            continue
        r, c = a.position
        for kind, (dr, dc) in MOVE_STEPS.items():
            pos = r + dr, c + dc
            kinds[a.slot, KINDS.index(kind)] = (0 <= pos[0] < rows and 0 <= pos[1] < columns
                and grid[pos] != 1 and pos not in occupied_allies and not a.ramp_blocked
                and (snapshot.phase != "setup" or pos in setup_cells))
        if snapshot.phase == "setup":
            continue
        kinds[a.slot, 5] = snapshot.side == "A" and a.has_spike and not snapshot.is_planted and grid[r, c] == 2
        active_defuser = next((v.slot for v in snapshot.allies if v.defuse_progress > 0), None)
        kinds[a.slot, 6] = (snapshot.side == "D" and snapshot.is_planted and snapshot.spike_planted is not None
            and max(abs(r - snapshot.spike_planted[0]), abs(c - snapshot.spike_planted[1])) <= 1
            and active_defuser in (None, a.slot))
        # The game permits collection at full points; leave that legal choice available.
        kinds[a.slot, 7] = a.position in snapshot.orbs
        if a.charges > 0 and ABILITIES[a.slot] == "DANCE":
            for ally in snapshot.allies:
                cap = min(DANCE_MAX_HP, ally.max_hp) if ally.max_hp_lost > 0 else DANCE_MAX_HP
                targets[a.slot, 0, ally.slot] = ally.slot != a.slot and ally.alive and ally.hp < cap
        elif a.charges > 0 and ABILITIES[a.slot] != "HUNT":
            legal = walkable
            dr, dc = target_rows - r, target_columns - c
            if ABILITIES[a.slot] == "ASH":
                legal = legal & (dr ** 2 + dc ** 2 <= ASH_RANGE_CELLS ** 2)
            elif ABILITIES[a.slot] == "RECON":
                # Evaluate the same first Bresenham step for all targets at once.
                dx, dy = abs(dc), -abs(dr)
                e2 = 2 * (dx + dy)
                nr = r + np.where((dr != 0) & (e2 <= dx), np.where(dr > 0, 1, -1), 0)
                nc = c + np.where((dc != 0) & (e2 >= dy), np.where(dc > 0, 1, -1), 0)
                legal = legal & ((nr != r) | (nc != c)) & (grid[nr, nc] != 1)
            targets[a.slot, 0] = legal.ravel()
        kinds[a.slot, 8] = targets[a.slot, 0].any()
        if a.points >= a.cost and a.cost > 0 and ULTIMATES[a.slot] != "SERENADE":
            if a.ramp_blocked and ULTIMATES[a.slot] in ("ESCAPE", "RAID"):
                continue
            if ULTIMATES[a.slot] == "ESCAPE":
                legal = walkable.copy()
                for rr, cc in occupied_allies | sightings:
                    if 0 <= rr < rows and 0 <= cc < columns:
                        legal[rr, cc] = False
                targets[a.slot, 1] = legal.ravel()
                kinds[a.slot, 9] = targets[a.slot, 1].any()
            elif ULTIMATES[a.slot] == "RAID":
                # Do not mask using unseen enemy occupancy.
                possible = [a.facing] if a.forced_facing else FACING
                ultimate_facings[a.slot] = False
                for f in possible:
                    dr, dc = FACING_STEPS[FACING.index(f)]
                    pos = r + dr, c + dc
                    ultimate_facings[a.slot, FACING.index(f)] = (0 <= pos[0] < rows and 0 <= pos[1] < columns
                        and grid[pos] != 1 and pos not in occupied_allies and pos not in sightings)
                kinds[a.slot, 9] = ultimate_facings[a.slot].any()
            else:
                kinds[a.slot, 9] = True
    ultimate_facings &= facings
    for value in (kinds, facings, ultimate_facings, targets):
        value.setflags(write=False)
    return ActionMasks(kinds, facings, ultimate_facings, targets)


def target_required(slot, kind):
    return kind == "ABILITY" or (kind == "ULTIMATE" and ULTIMATES[slot] == "ESCAPE")


def validate_action(snapshot, masks, slot, action):
    if action.kind not in KINDS or not masks.kind[slot, KINDS.index(action.kind)]:
        raise ValueError(f"masked FRC action: slot={slot}, {action.kind}")
    if action.facing not in FACING or not masks.facing[slot, FACING.index(action.facing)]:
        raise ValueError("masked FRC facing")
    if action.kind == "ULTIMATE" and not masks.ultimate_facing[slot, FACING.index(action.facing)]:
        raise ValueError("masked ultimate facing")
    if target_required(slot, action.kind):
        if slot == 0 and action.kind == "ABILITY":
            index = action.ally_slot
            if index is None or not 0 <= index < 5:
                raise ValueError("DANCE requires a teammate slot")
        else:
            if action.target is None:
                raise ValueError("ability requires a target")
            rows, columns = len(snapshot.grid), len(snapshot.grid[0])
            r, c = action.target
            if not (0 <= r < rows and 0 <= c < columns):
                raise ValueError("FRC target out of bounds")
            index = r * columns + c
        if not masks.target[slot, int(action.kind == "ULTIMATE"), index]:
            raise ValueError("masked FRC target")
    elif action.target is not None or action.ally_slot is not None:
        raise ValueError("unexpected target on FRC action")
    if action.kind == "ULTIMATE" and slot == 2:
        a = snapshot.allies[slot]
        dr, dc = FACING_STEPS[FACING.index(action.facing)]
        pos = a.position[0] + dr, a.position[1] + dc
        if not (0 <= pos[0] < len(snapshot.grid) and 0 <= pos[1] < len(snapshot.grid[0])) or snapshot.grid[pos[0]][pos[1]] == 1:
            raise ValueError("RAID cannot move in this direction")


def to_game_action(snapshot, slot, action, runtime_names):
    a = snapshot.allies[slot]
    pos = a.position
    if action.kind in MOVE_STEPS:
        dr, dc = MOVE_STEPS[action.kind]
        return (pos[0] + dr, pos[1] + dc), {"facing": action.facing}
    if action.kind == "ABILITY":
        payload = {"ability": ABILITIES[slot], "facing": action.facing}
        payload.update({"target_name": runtime_names[action.ally_slot]} if slot == 0 else {"target": action.target})
        return pos, payload
    if action.kind == "ULTIMATE":
        payload = {"ultimate": ULTIMATES[slot], "facing": action.facing}
        if action.target is not None:
            payload["target"] = action.target
        return pos, payload
    if action.kind in ("PLANT", "DEFUSE", "COLLECT_ORB"):
        return pos, action.kind, {"facing": action.facing}
    return pos, {"facing": action.facing}
