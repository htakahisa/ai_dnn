"""Standalone public observations, independent of earlier GC controllers and encoders."""
import numpy as np
from map_data import NEW_MAZE_STR as ARENA_TEXT
from map_data_defender_setup import get_setup_mask
from .gc3_engine_actions import ABILITIES, ULTIMATES, FACING_DIRS
from .gc3_spatial_policy import parse_map, hint_fields, phase_settings
from . import gc3_map_attacker_postplant as postplant_map
from . import gc3_map_defender_initial as defender_map

ARENA_SHAPE = (len(ARENA_TEXT.strip().splitlines()), len(ARENA_TEXT.strip().splitlines()[0]))
CELLS = int(np.prod(ARENA_SHAPE))
BOARD_CHANNELS = 10
CONTEXT_WIDTH = 20
NUMERIC_FIELDS = (
    ("hp", 100), ("max_hp", 100), ("shield_hp", 100), ("ultimate_points", 10), ("ultimate_cost", 10),
    *((name+"_charges", 3) for name in ("recon", "flash", "smoke", "ramp", "dance", "ash")),
    *((name+"_remaining", 10) for name in ("blind", "reveal", "electric", "ability_seal", "fate_loom", "movement_disabled")),
    ("move_steps_per_tick", 3), ("accuracy", 2), ("hs_rate", 2), ("dodge_rate", 1),
    ("reaction", 200), ("effective_iq", 200), ("sees_through_smoke", 1), ("shield_abilities_enabled", 1),
)
UNIT_WIDTH = 6+len(NUMERIC_FIELDS)+len(ABILITIES)+len(ULTIMATES)+len(FACING_DIRS)
OBS_DIM = CONTEXT_WIDTH+BOARD_CHANNELS*CELLS+10*UNIT_WIDTH


def sampling_phase(char, state):
    if char.team == "D" and state.get("defender_setup_active"):
        return "defender"
    if char.team == "A" and state.get("is_planted"):
        return "postplant"
    return "standard"


def map_hints(char, state):
    phase = sampling_phase(char, state)
    text = (defender_map.DEFENDER_SETUP_POSITION_MAP_GC_STR if phase == "defender" else
            postplant_map.NEW_MAZE_STR if phase == "postplant" else None)
    if text is None:
        return np.zeros((2, CELLS), dtype=np.float32)
    markers = parse_map(text)
    if markers.shape != np.asarray(state["grid"]).shape or not np.array_equal(markers == 1, np.asarray(state["grid"]) == 1):
        raise ValueError("Candidate map walls/dimensions do not match the arena")
    return np.stack(hint_fields(text, phase_settings(phase)[3])).reshape(2, CELLS).astype(np.float32)


def _unit_features(unit, char, setup):
    if unit is None:
        return np.zeros(UNIT_WIDTH, dtype=np.float32)
    own = unit.team == char.team
    known = own or (not setup and getattr(unit, "position_known", False))
    # Setup can receive raw engine objects; no enemy live field is read there.
    values = [1., own and unit.name == char.name, bool(unit.is_alive) if own or not setup else 0., known]
    values += [unit.pos[0]/ARENA_SHAPE[0], unit.pos[1]/ARENA_SHAPE[1]] if known else [0., 0.]
    values += [float(getattr(unit, key, 0))/scale for key, scale in NUMERIC_FIELDS] if known else [0.]*len(NUMERIC_FIELDS)
    for key, vocabulary in (("ability_name", ABILITIES), ("ultimate_name", ULTIMATES), ("facing", FACING_DIRS)):
        label = getattr(unit, key, None) if known else None
        values.extend(float(label == entry) for entry in vocabulary)
    return np.asarray(values, dtype=np.float32)


def encode(char, state, hints):
    grid = np.asarray(state["grid"])
    if grid.shape != ARENA_SHAPE:
        raise ValueError("GC v3 observation arena shape mismatch")
    setup = bool(state.get("defender_setup_active"))
    planted = bool(state.get("is_planted"))
    spike = state.get("planted_pos") if planted else state.get("spike_pos")
    own = sorted((c for c in state.get("chars", ()) if c.team == char.team), key=lambda c: str(c.name))[:5]
    enemies = sorted((c for c in state.get("chars", ()) if c.team != char.team), key=lambda c: str(c.name))[:5]
    taps = list((state.get("defender_defuse_info") or {}).values()) if planted else []
    context = [char.pos[0]/grid.shape[0], char.pos[1]/grid.shape[1], planted, setup,
        state.get("battle_tick", 0)/100, state.get("defender_setup_ticks_remaining", 0)/20,
        state.get("round_timer", 0)/100, state.get("detonate_timer", 0)/55,
        spike is not None, spike[0]/grid.shape[0] if spike else 0., spike[1]/grid.shape[1] if spike else 0.,
        bool(getattr(char, "has_spike", False)), getattr(char, "plant_timer", 0)/4,
        getattr(char, "defuse_timer", 0)/6, getattr(char, "orb_collect_timer", 0)/5,
        sum(c.is_alive for c in own)/5, sum(c.is_alive for c in enemies)/5 if not setup else 0.,
        len(taps)/5, max((p/max(1, required) for p, required in taps), default=0.), char.team == "D"]
    board = np.zeros((BOARD_CHANNELS, *grid.shape), dtype=np.float32)
    board[0], board[1] = grid == 1, grid == 2
    for channel, points in ((2, state.get("smoke_cells", ())), (3, state.get("available_orbs", ()))):
        for r, c in points:
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                board[channel, r, c] = 1.
    board[4:6] = hints.reshape(2, *grid.shape)
    if not setup:
        for effect in state.get("public_effects", ()):
            channel = 6 if effect.phase == "flight" else 7 if effect.phase == "warning" else 8
            points = list(effect.cells)
            if effect.position is not None:
                points.append(effect.position)
            for r, c in points:
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    board[channel, r, c] = 1.
    board[9] = (np.asarray(get_setup_mask()) == 0) if setup else (grid != 1)
    units = [_unit_features(team[i] if i < len(team) else None, char, setup)
             for team in (own, enemies) for i in range(5)]
    observation = np.concatenate((np.asarray(context, dtype=np.float32), board.reshape(-1), *units)).astype(np.float32)
    if observation.shape != (OBS_DIM,) or not np.isfinite(observation).all():
        raise ValueError("Invalid GC v3 observation")
    return observation
