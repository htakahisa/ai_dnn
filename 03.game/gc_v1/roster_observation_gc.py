"""Versioned public opponent-lineup suffix shared by GC training and inference.

Five slots sorted by base name: presence, 32 stable identity bits, eight role
indicators (including unknown), and twelve normalized base-stat/ULT features.
Unused slots are zero. Death, visibility, positions, HP and charges are excluded.
Existing observation indices remain unchanged; this block is appended last.
"""

from hashlib import blake2s
import math

import numpy as np

from game_core import get_character_combat_stats, ULTIMATE_COSTS

ROSTER_OBSERVATION_VERSION = 1
MAX_ROSTER_SIZE = 5
IDENTITY_BITS = 32
ROLES = ("フラッシュ", "スモーカー", "シーカー", "タイガー", "エンジニア",
         "アイドル", "コントラクター", "UNKNOWN")
STAT_SCALES = (("accuracy", 2.0), ("hs_rate", 2.0), ("dodge_rate", 1.0),
               ("iq", 200.0), ("reaction", 200.0), ("influence", 100.0),
               ("form_variance", 10.0), ("mental", 20.0), ("shield_hp", 100.0),
               ("shield_piercer", 1.0), ("shield_crash", 100.0), ("ultimate_cost", 10.0))
PLAYER_FEATURE_DIM = 1 + IDENTITY_BITS + len(ROLES) + len(STAT_SCALES)
ENEMY_ROSTER_DIM = MAX_ROSTER_SIZE * PLAYER_FEATURE_DIM
ROSTER_METADATA = {"enemy_roster_version": ROSTER_OBSERVATION_VERSION,
                   "enemy_roster_dim": ENEMY_ROSTER_DIM}


def _entry(player):
    if isinstance(player, dict):
        return player
    name = getattr(player, "base_name", getattr(player, "name", player))
    # Preserve season-specific string subclasses until stats have been read.
    return {"base_name": str(name), "stats": get_character_combat_stats(name)}


def enemy_roster_features(*, game_state=None, game=None, chars=(), viewer_team="A", roster=None):
    """Prefer the explicit public lineup, then a full game roster, then units.

    ``roster`` also accepts a preset's player names for synthetic environments.
    Never infer missing lineup members from a visible-enemy list when a full
    public roster is available.
    """
    if roster is None and game_state is not None and "enemy_roster" in game_state:
        roster = game_state["enemy_roster"]
    if roster is None and game is not None:
        roster = getattr(game, "defender_roster" if viewer_team == "A" else "attacker_roster", None)
    if roster is None:
        units = getattr(game, "chars", chars) if game is not None else chars
        roster = [c for c in units if getattr(c, "team", None) != viewer_team]
    entries = sorted((_entry(p) for p in roster),
                     key=lambda p: str(p.get("base_name", p.get("name", ""))))
    result = np.zeros((MAX_ROSTER_SIZE, PLAYER_FEATURE_DIM), dtype=np.float32)
    for row, entry in zip(result, entries):
        name = str(entry.get("base_name", entry.get("name", "")))
        stats = entry.get("stats")
        if stats is None:
            stats = get_character_combat_stats(name)
        role = entry.get("role", stats.get("role", "UNKNOWN"))
        row[0] = 1.0
        digest = np.frombuffer(blake2s(name.encode("utf-8"), digest_size=4).digest(), dtype=np.uint8)
        row[1:1 + IDENTITY_BITS] = np.unpackbits(digest)
        role_index = ROLES.index(role) if role in ROLES else len(ROLES) - 1
        row[1 + IDENTITY_BITS + role_index] = 1.0
        for i, (key, scale) in enumerate(STAT_SCALES):
            value = (entry.get(key, ULTIMATE_COSTS.get(role, 0))
                     if key == "ultimate_cost" else stats.get(key, 0))
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0
            row[1 + IDENTITY_BITS + len(ROLES) + i] = (
                np.clip(value / scale, 0.0, 1.0) if math.isfinite(value) else 0.0)
    return result.reshape(-1)


def append_enemy_roster(observation, **context):
    return np.concatenate((np.asarray(observation, dtype=np.float32),
                           enemy_roster_features(**context)))


def base_checkpoint_dim(checkpoint, dimension):
    """Checkpoint obs_dim includes the suffix only in the versioned format."""
    version = int(checkpoint.get("enemy_roster_version", 0))
    if version not in (0, ROSTER_OBSERVATION_VERSION):
        raise ValueError(f"Unsupported enemy roster observation version: {version}")
    if version and int(checkpoint.get("enemy_roster_dim", -1)) != ENEMY_ROSTER_DIM:
        raise ValueError("Enemy roster feature dimension mismatch")
    return int(dimension) - (ENEMY_ROSTER_DIM if version else 0)


def expand_roster_state(model, state):
    """Zero-initialize appended input weights, preserving legacy predictions.

    Applies to both action and independent facing input layers. Other shape
    mismatches remain errors in load_state_dict; no learned weights are lost.
    """
    expanded = dict(state)
    target = model.state_dict()
    for key, value in state.items():
        wanted = target.get(key)
        if (wanted is not None and value.ndim == 2 and wanted.ndim == 2
                and wanted.shape[0] == value.shape[0]
                and wanted.shape[1] == value.shape[1] + ENEMY_ROSTER_DIM):
            weights = wanted.new_zeros(wanted.shape)
            weights[:, :value.shape[1]] = value
            expanded[key] = weights
    return expanded
