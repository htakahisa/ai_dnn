"""Shared policy features from displayed areas and own public status only."""
import numpy as np

from frc_v1.actions import MOVE_STEPS

HAZARD_RADIUS = 3
HAZARD_CHANNELS = ("neon_warning", "neon_active", "destruction", "balemoon_warning")
HAZARD_MOVES = ("STAY", "N", "E", "S", "W")
HAZARD_OBS_DIM = ((2 * HAZARD_RADIUS + 1) ** 2 + len(HAZARD_MOVES) + 2) * len(HAZARD_CHANNELS) + 3


def hazard_schema():
    return dict(version=1, channels=HAZARD_CHANNELS, radius=HAZARD_RADIUS,
                moves=HAZARD_MOVES, targets=("goal", "spike"), obs_dim=HAZARD_OBS_DIM,
                status=("contract_active", "contract_remaining_div10", "max_hp_lost_div100"),
                destruction="displayed_level_div10_min_0.1_max_1",
                information="displayed_cells_phase_level_and_own_status",
                affiliation="unknown_potential_hazard", selection="learned_legal_actions")


def hazard_features(snapshot, ally, goal):
    """Do not infer owners, hidden traps, expiry timers or future cast targets.

    DisplayEffect intentionally lacks allegiance, so these are potential hazards,
    not claims that every displayed area will damage this player. Actual HP loss
    remains the training signal; neither legal actions nor routes are changed.
    """
    height, width = len(snapshot.grid), len(snapshot.grid[0])
    areas = np.zeros((height, width, len(HAZARD_CHANNELS)), np.float32)
    for effect in snapshot.effects:
        value = 1.
        if effect.kind == "NEON" and effect.phase in ("warning", "active"):
            channel = 0 if effect.phase == "warning" else 1
        elif effect.kind == "DESTRUCTION" and effect.phase == "active":
            channel = 2
            value = float(np.clip(effect.level / 10, .1, 1.))
        elif effect.kind == "BALEMOON" and effect.phase == "warning":
            channel = 3
        else:
            continue
        for row, column in effect.cells:
            if 0 <= row < height and 0 <= column < width:
                areas[row, column, channel] = max(areas[row, column, channel], value)

    def cell(position):
        if position is None:
            return [0.] * len(HAZARD_CHANNELS)
        row, column = position
        return areas[row, column].tolist() if 0 <= row < height and 0 <= column < width else [0.] * len(HAZARD_CHANNELS)

    row, column = ally.position
    features = []
    for dr in range(-HAZARD_RADIUS, HAZARD_RADIUS + 1):
        for dc in range(-HAZARD_RADIUS, HAZARD_RADIUS + 1):
            features.extend(cell((row + dr, column + dc)))
    for move in HAZARD_MOVES:
        dr, dc = MOVE_STEPS.get(move, (0, 0))
        features.extend(cell((row + dr, column + dc)))
    features.extend(cell(goal))
    features.extend(cell(snapshot.spike_planted or snapshot.spike_dropped))
    features.extend((float(ally.contract > 0), float(np.clip(ally.contract / 10, 0, 1)),
                     float(np.clip(ally.max_hp_lost / 100, 0, 1))))
    return np.asarray(features, np.float32)
