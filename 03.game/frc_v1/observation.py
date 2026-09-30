"""Versioned actor tensors; no live game objects cross this API."""

from dataclasses import dataclass
import hashlib
import numpy as np

from frc_v1 import ROSTER, FACING
from frc_v1.actions import PHASES, INTENTS, KINDS, build_masks
import game_core

VERSION = 2
EFFECT_KINDS = ("FLASH", "RECON", "ASH", "TUNNEL", "NEON", "BALEMOON", "SMOKE", "DESTRUCTION", "MONITOR", "ESCAPE", "RAID")
EFFECT_PHASES = ("flight", "warning", "active", "fresh", "fading", "faint")
TOKEN_LIMIT = 64
GRID_FIELDS = ("walkable", "wall", "plantable", "attacker_spawn", "defender_spawn", "orb_available",
    "visible", "clear_known", "clear_age", "current_enemy", "last_seen_known", "last_seen_age", "smoke",
    "spike_dropped", "spike_planted", "setup_allowed") + tuple(f"ally_{i}" for i in range(5)) + tuple(
    f"{kind.lower()}_{phase}" for kind in EFFECT_KINDS for phase in ("flight", "warning", "active"))
ALLY_FIELDS = ("alive", "hp", "max_hp", "charges", "points", "cost", "has_spike", "plant", "defuse", "orb",
    "blind", "reveal", "contract", "max_hp_lost", "disabled", "warping", "iq", "accuracy", "hs_rate", "dodge", "reaction", "forced", "ramp_blocked") + tuple(f"facing_{f}" for f in FACING)
GLOBAL_FIELDS = ("attacker", "setup", "round_time", "detonation_time", "planted", "defuse_notified",
    "enemies_alive", "sighted", "idol_death_warning", "effects_overflow")
VECTOR_FIELDS = GLOBAL_FIELDS + tuple(f"slot_{i}_{field}" for i in range(5) for field in ALLY_FIELDS)
TOKEN_FIELDS = tuple(f"kind_{k}" for k in EFFECT_KINDS) + tuple(f"phase_{p}" for p in EFFECT_PHASES) + (
    "own", "enemy", "unknown", "row", "column", "position_known", "direction_row", "direction_column",
    "age", "start_known", "remaining", "remaining_known", "level", "blink", "drawn", "display_hp", "coverage")
ABILITY_CONSTANTS = ("DANCE_HEAL_HP", "DANCE_MAX_HP", "SERENADE_REVEAL_TICKS", "ASH_RANGE_CELLS",
    "BALEMOON_WARNING_TICKS", "DESTRUCTION_AREA_TICKS", "CONTRACT_DAMAGE_PER_TICK", "FLASH_SPEED_CELLS_PER_TICK",
    "RECON_SPEED_CELLS_PER_TICK", "FLASH_MAX_FLIGHT_TICKS", "TUNNEL_WARNING_TICKS", "NEON_WARNING_TICKS",
    "RAID_DISTANCE_CELLS", "ESCAPE_WARP_DELAY_TICKS", "PLANT_REQUIRED_TICKS", "DEFUSE_REQUIRED_TICKS",
    "ORB_COLLECT_REQUIRED_TICKS", "ORB_ULTIMATE_POINTS")


@dataclass(frozen=True)
class FrcObservation:
    grid: np.ndarray
    vector: np.ndarray
    tokens: np.ndarray
    token_mask: np.ndarray
    masks: object
    source_key: tuple
    overflow: int


def metadata(grid, side):
    return {"version": VERSION, "side": side, "roster": list(ROSTER),
        "shape": [len(grid), len(grid[0])], "map_hash": hashlib.sha256(np.asarray(grid, dtype=np.int32).tobytes()).hexdigest(),
        "grid_fields": list(GRID_FIELDS), "vector_fields": list(VECTOR_FIELDS), "token_fields": list(TOKEN_FIELDS),
        "token_limit": TOKEN_LIMIT, "action_kinds": list(KINDS), "facing": list(FACING),
        "phases": list(PHASES), "intents": list(INTENTS), "public_effect_scope": "display_global_v1",
        "normalization": "hp/100,time/256,resources/10,iq/200,reaction/200,coordinates/max_index",
        "ability_constants": {name: getattr(game_core, name) for name in ABILITY_CONSTANTS},
        "ultimate_costs": dict(game_core.ULTIMATE_COSTS)}


def validate_metadata(saved, grid, side):
    expected = metadata(grid, side)
    if saved != expected:
        changed = [key for key in expected if saved.get(key) != expected[key]]
        raise ValueError("FRC checkpoint/schema mismatch: " + ", ".join(changed))


class FrcObservationEncoder:
    def encode(self, snapshot, belief, *, effects_mode="all"):
        if belief.source_key != snapshot.key:
            raise ValueError("FRC snapshot/history ticks differ")
        if effects_mode not in ("all", "none", "flight", "warning"):
            raise ValueError("unknown effect observation mode")
        rows, columns = len(snapshot.grid), len(snapshot.grid[0])
        grid = np.zeros((len(GRID_FIELDS), rows, columns), np.float32)
        fields = {name: grid[i] for i, name in enumerate(GRID_FIELDS)}
        board = np.asarray(snapshot.grid)
        for name, value in (("wall", 1), ("plantable", 2), ("attacker_spawn", 3), ("defender_spawn", 4)):
            fields[name][:] = board == value
        fields["walkable"][:] = board != 1
        def mark(name, cells, value=1.0):
            for r, c in cells:
                if 0 <= r < rows and 0 <= c < columns:
                    fields[name][r, c] = value
        mark("orb_available", snapshot.orbs)
        mark("visible", snapshot.visible_cells)
        mark("setup_allowed", snapshot.setup_cells)
        mark("smoke", snapshot.smoke_cells)  # smoke remains part of the baseline sensor.
        for pos, age in belief.clear:
            mark("clear_known", (pos,))
            mark("clear_age", (pos,), min(1.0, age / 256))
        mark("current_enemy", (s.position for s in snapshot.sightings))
        for _, pos, age in belief.last_seen:
            mark("last_seen_known", (pos,))
            mark("last_seen_age", (pos,), min(1.0, age / 256))
        for name, pos in (("spike_dropped", snapshot.spike_dropped), ("spike_planted", snapshot.spike_planted)):
            if pos is not None:
                mark(name, (pos,))
        for a in snapshot.allies:
            if a.alive:
                mark(f"ally_{a.slot}", (a.position,))
        effects = [e for e in snapshot.effects if effects_mode == "all" or
                   effects_mode == "flight" and e.phase == "flight" or
                   effects_mode == "warning" and e.phase == "warning"]
        for e in effects:
            phase = e.phase if e.phase in ("flight", "warning") else "active"
            cells = e.cells or (() if e.position is None else (e.position,))
            mark(f"{e.kind.lower()}_{phase}", cells)
        histories = {e.handle: e for e in belief.effects}
        effects.sort(key=lambda e: (0 if e.phase == "warning" else 1 if e.phase != "flight" else 2, e.handle))
        overflow = max(0, len(effects) - TOKEN_LIMIT)
        vector = np.zeros(len(VECTOR_FIELDS), np.float32)
        vector[:len(GLOBAL_FIELDS)] = (snapshot.side == "A", snapshot.phase == "setup", snapshot.round_timer / 256,
            snapshot.detonate_timer / 256, snapshot.is_planted, snapshot.defuse_notified,
            sum(e.alive for e in snapshot.enemies) / 5, len(snapshot.sightings) / 5,
            any(e.kind == "BALEMOON" and e.phase == "warning" for e in effects), overflow / TOKEN_LIMIT)
        for a in snapshot.allies:
            values = (a.alive, a.hp / 100, a.max_hp / 100, a.charges / 10, a.points / 10, a.cost / 10,
                a.has_spike, a.plant_progress / 10, a.defuse_progress / 10, a.orb_progress / 10,
                a.blind / 256, a.reveal / 256, a.contract / 10, a.max_hp_lost / 100, a.movement_disabled / 256,
                a.warping, a.effective_iq / 200, a.accuracy, a.hs_rate, a.dodge, a.reaction / 200, a.forced_facing, a.ramp_blocked)
            offset = len(GLOBAL_FIELDS) + a.slot * len(ALLY_FIELDS)
            vector[offset:offset + len(values)] = values
            vector[offset + len(values):offset + len(ALLY_FIELDS)] = [a.facing == f for f in FACING]
        tokens = np.zeros((TOKEN_LIMIT, len(TOKEN_FIELDS)), np.float32)
        token_mask = np.zeros(TOKEN_LIMIT, bool)
        for i, e in enumerate(effects[:TOKEN_LIMIT]):
            h = histories[e.handle]
            token_mask[i] = True
            tokens[i, EFFECT_KINDS.index(e.kind)] = 1
            tokens[i, len(EFFECT_KINDS) + EFFECT_PHASES.index(e.phase)] = 1
            offset = len(EFFECT_KINDS) + len(EFFECT_PHASES)
            r, c = e.position or (0, 0)
            tokens[i, offset:] = (h.affiliation == "own", h.affiliation == "enemy", h.affiliation == "unknown",
                r / max(1, rows - 1), c / max(1, columns - 1), e.position is not None, *e.direction,
                min(1, h.age / 256), h.start_known, (h.predicted_remaining or 0) / 256,
                h.predicted_remaining is not None, e.level / 10, e.blinking, e.drawn_this_frame,
                e.displayed_hp / 200, len(e.cells) / (rows * columns))
        for array in (grid, vector, tokens, token_mask):
            array.setflags(write=False)
        return FrcObservation(grid, vector, tokens, token_mask, build_masks(snapshot), snapshot.key, overflow)
