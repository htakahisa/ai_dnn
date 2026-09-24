"""Fixed-map, actor-only coach observation contract (coach-observation-v1).

The only dynamic inputs are copied team perception and belief DTOs. This
module never accepts a game, Character, critic tensor, or hidden enemy state.
Channel order, normalization and slot order are checkpoint compatibility data;
changes require a new observation version and retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from map_data import NEW_MAZE_STR

from coach_v1.common.constants import (
    FIXED_ROSTER,
    MAP_COLUMNS,
    MAP_ROWS,
    ROSTER_SIZE,
    WATCH_POINTS_CONFIG_PATH,
)
from coach_v1.common.hashing import map_sha256, normalize_map_text
from coach_v1.common.types import Facing, ModelFamily, Side
from coach_v1.common.versions import COACH_OBSERVATION_VERSION
from coach_v1.common.watch_points import (
    ATTACKER_SITUATIONS,
    DEFENDER_SITUATIONS,
    WatchPointConfig,
    load_watch_points,
)
from coach_v1.perception.belief_memory import BeliefSnapshot, normalize_age
from coach_v1.perception.team_perception import TeamPerceptionSnapshot


AGE_CAP_TICKS = 256
FACING_ORDER = tuple(Facing)
SITUATION_ORDER = ("carry", "retrieve", "guard", "search", "retake")
COACH_GRID_CHANNELS = (
    "walkable", "wall", "plantable", "attacker_spawn", "defender_spawn",
    "orb", "watch_importance", "watch_facing_row", "watch_facing_column",
    "watch_confirmed", "watch_confirmation_age", "currently_visible",
    "visible_viewer_count", "clear_known", "clear_age", "smoke",
    "current_enemy_sighting", "last_seen_enemy_count", "last_seen_known",
    "last_seen_age", "spike_dropped", "spike_planted",
    *(f"ally_slot_{slot}" for slot in range(ROSTER_SIZE)),
)
_GLOBAL_VECTOR_FIELDS = (
    "side_attacker", "side_defender",
    *(f"situation_{name}" for name in SITUATION_ORDER),
    "defender_setup", "spike_carried_by_ally", "spike_dropped",
    "spike_planted", "allies_alive", "enemies_alive", "enemies_sighted",
)
_SLOT_VECTOR_FIELDS = (
    "alive", "hp", "normal_ability_available", "has_spike", "row", "column",
    *(f"facing_{facing.value}" for facing in FACING_ORDER),
)
COACH_VECTOR_FIELDS = _GLOBAL_VECTOR_FIELDS + tuple(
    f"slot_{slot}_{field}"
    for slot in range(ROSTER_SIZE)
    for field in _SLOT_VECTOR_FIELDS
)


class CoachObservationInputError(ValueError):
    """The safe DTOs or fixed-map metadata do not match this observation."""


@dataclass(frozen=True)
class CoachObservation:
    grid: np.ndarray  # float32, [channel, 26, 44]
    vector: np.ndarray  # float32, fixed field order
    version: str
    map_hash: str
    watch_points_hash: str


class CoachObservationEncoder:
    """Convert safe snapshots to copied CNN and vector inputs for all 5 slots."""

    def __init__(self) -> None:
        map_text = normalize_map_text(NEW_MAZE_STR)
        map_rows = tuple(map_text.split("\n"))
        if len(map_rows) != MAP_ROWS or any(len(row) != MAP_COLUMNS for row in map_rows):
            raise CoachObservationInputError("fixed map shape changed")
        if any(cell not in "012345" for row in map_rows for cell in row):
            raise CoachObservationInputError("fixed map has unsupported cells")
        config = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
        self._map_hash = map_sha256(NEW_MAZE_STR)
        self._config: WatchPointConfig = config
        self._map_rows = map_rows
        self._static = _static_grid(map_rows)

    @property
    def map_hash(self) -> str:
        return self._map_hash

    @property
    def watch_points_hash(self) -> str:
        return self._config.config_hash

    def validate_checkpoint(self, metadata: object, *, side: Side) -> None:
        """Reject a checkpoint whose actor observation contract differs."""

        if (
            getattr(metadata, "model_family", None) is not ModelFamily.COACH
            or getattr(metadata, "target_id", None) != side.value
            or getattr(metadata, "observation_version", None) != COACH_OBSERVATION_VERSION
            or getattr(metadata, "map_hash", None) != self._map_hash
            or getattr(metadata, "watch_points_hash", None) != self.watch_points_hash
            or getattr(metadata, "roster", None) != tuple(slot.character_name for slot in FIXED_ROSTER)
        ):
            raise CoachObservationInputError("checkpoint observation metadata mismatch")

    def encode(
        self,
        snapshot: TeamPerceptionSnapshot,
        belief: BeliefSnapshot,
        *,
        situation: str,
    ) -> CoachObservation:
        if not isinstance(snapshot, TeamPerceptionSnapshot) or not isinstance(belief, BeliefSnapshot):
            raise TypeError("encode requires team perception and belief snapshots")
        _validate_inputs(snapshot, belief, situation, self._config)
        grid = np.zeros((len(COACH_GRID_CHANNELS), MAP_ROWS, MAP_COLUMNS), dtype=np.float32)
        grid[:6] = self._static
        channels = {name: grid[index] for index, name in enumerate(COACH_GRID_CHANNELS)}

        for point in self._config.for_side(snapshot.side):
            if not point.supports_situation(situation):
                continue
            row, column = point.position
            channels["watch_importance"][row, column] = point.importance / 5.0
            delta = _FACING_DELTAS[point.facing]
            channels["watch_facing_row"][row, column] = delta[0]
            channels["watch_facing_column"][row, column] = delta[1]
            point_belief = belief.watch_point_for(point.point_id)
            if point_belief.confirmation_age is not None:
                channels["watch_confirmed"][row, column] = 1.0
                channels["watch_confirmation_age"][row, column] = normalize_age(
                    point_belief.confirmation_age, AGE_CAP_TICKS
                )

        channels["currently_visible"][:] = np.asarray(belief.currently_visible, dtype=np.float32)
        channels["visible_viewer_count"][:] = np.asarray(
            belief.visible_viewer_count, dtype=np.float32
        ) / ROSTER_SIZE
        for row in range(MAP_ROWS):
            for column in range(MAP_COLUMNS):
                clear_age = belief.clear_age[row][column]
                if clear_age is not None:
                    channels["clear_known"][row, column] = 1.0
                    channels["clear_age"][row, column] = normalize_age(clear_age, AGE_CAP_TICKS)
                count = belief.last_seen_enemy_count[row][column]
                channels["last_seen_enemy_count"][row, column] = count / ROSTER_SIZE
                seen_age = belief.last_seen_age[row][column]
                if seen_age is not None:
                    channels["last_seen_known"][row, column] = 1.0
                    channels["last_seen_age"][row, column] = normalize_age(seen_age, AGE_CAP_TICKS)
        for row, column in snapshot.smoke_cells:
            channels["smoke"][row, column] = 1.0
        for sighting in snapshot.sightings:
            row, column = sighting.reported_position
            channels["current_enemy_sighting"][row, column] += 1.0 / ROSTER_SIZE
        for name, position in (
            ("spike_dropped", snapshot.spike.dropped_position),
            ("spike_planted", snapshot.spike.planted_position),
        ):
            if position is not None:
                channels[name][position] = 1.0

        vector = np.zeros(len(COACH_VECTOR_FIELDS), dtype=np.float32)
        fields = {name: index for index, name in enumerate(COACH_VECTOR_FIELDS)}
        vector[fields[f"side_{snapshot.side.value}"]] = 1.0
        vector[fields[f"situation_{situation}"]] = 1.0
        vector[fields["defender_setup"]] = float(snapshot.tick.phase == "defender_setup")
        vector[fields["spike_carried_by_ally"]] = float(snapshot.spike.own_carrier_slot is not None)
        vector[fields["spike_dropped"]] = float(snapshot.spike.dropped_position is not None)
        vector[fields["spike_planted"]] = float(snapshot.spike.is_planted)
        vector[fields["allies_alive"]] = sum(ally.is_alive for ally in snapshot.allies) / ROSTER_SIZE
        vector[fields["enemies_alive"]] = sum(enemy.is_alive for enemy in snapshot.enemies) / ROSTER_SIZE
        vector[fields["enemies_sighted"]] = len(snapshot.sightings) / ROSTER_SIZE
        for ally in snapshot.allies:
            if not ally.is_alive:
                continue  # Dead slots retain their fixed offset and remain all zero.
            prefix = f"slot_{ally.slot}_"
            vector[fields[prefix + "alive"]] = 1.0
            vector[fields[prefix + "hp"]] = min(ally.hp, 100) / 100.0
            vector[fields[prefix + "normal_ability_available"]] = float(ally.normal_ability_charges > 0)
            vector[fields[prefix + "has_spike"]] = float(ally.has_spike)
            vector[fields[prefix + "row"]] = ally.position[0] / (MAP_ROWS - 1)
            vector[fields[prefix + "column"]] = ally.position[1] / (MAP_COLUMNS - 1)
            vector[fields[prefix + f"facing_{ally.facing.value}"]] = 1.0
            channels[f"ally_slot_{ally.slot}"][ally.position] = 1.0

        if not np.isfinite(grid).all() or not np.isfinite(vector).all():
            raise CoachObservationInputError("observation contains non-finite values")
        grid.setflags(write=False)
        vector.setflags(write=False)
        return CoachObservation(
            grid, vector, COACH_OBSERVATION_VERSION,
            self._map_hash, self.watch_points_hash,
        )


_FACING_DELTAS = {
    Facing.N: (-1, 0), Facing.NE: (-1, 1), Facing.E: (0, 1),
    Facing.SE: (1, 1), Facing.S: (1, 0), Facing.SW: (1, -1),
    Facing.W: (0, -1), Facing.NW: (-1, -1),
}


def _static_grid(map_rows: Tuple[str, ...]) -> np.ndarray:
    grid = np.zeros((6, MAP_ROWS, MAP_COLUMNS), dtype=np.float32)
    for row, cells in enumerate(map_rows):
        for column, cell in enumerate(cells):
            grid[0, row, column] = float(cell != "1")
            for channel, symbol in enumerate("12345", start=1):
                grid[channel, row, column] = float(cell == symbol)
    return grid


def _validate_inputs(
    snapshot: TeamPerceptionSnapshot,
    belief: BeliefSnapshot,
    situation: str,
    config: WatchPointConfig,
) -> None:
    if not isinstance(snapshot.side, Side):
        raise CoachObservationInputError("invalid side")
    allowed = ATTACKER_SITUATIONS if snapshot.side is Side.ATTACKER else DEFENDER_SITUATIONS
    if situation not in allowed:
        raise CoachObservationInputError("situation does not match side")
    if belief.side is not snapshot.side or belief.source_tick != snapshot.tick:
        raise CoachObservationInputError("belief and perception refer to different ticks or sides")
    if belief.round_number != snapshot.tick.round_number:
        raise CoachObservationInputError("belief round and perception round disagree")
    if belief.currently_visible != snapshot.currently_visible or belief.visible_viewer_count != snapshot.visible_viewer_count:
        raise CoachObservationInputError("belief and perception visibility disagree")
    for name, field in (
        ("currently_visible", snapshot.currently_visible),
        ("visible_viewer_count", snapshot.visible_viewer_count),
        ("clear_age", belief.clear_age),
        ("last_seen_enemy_count", belief.last_seen_enemy_count),
        ("last_seen_age", belief.last_seen_age),
    ):
        if len(field) != MAP_ROWS or any(len(row) != MAP_COLUMNS for row in field):
            raise CoachObservationInputError(f"{name} has wrong shape")
    if len(snapshot.allies) != ROSTER_SIZE or tuple(ally.slot for ally in snapshot.allies) != tuple(range(ROSTER_SIZE)):
        raise CoachObservationInputError("allies must use fixed roster slot order")
    if tuple(ally.character_id for ally in snapshot.allies) != tuple(slot.checkpoint_id for slot in FIXED_ROSTER):
        raise CoachObservationInputError("ally character ids must match the fixed roster")
    if len(snapshot.enemies) > ROSTER_SIZE or len(snapshot.sightings) > ROSTER_SIZE:
        raise CoachObservationInputError("too many enemies")
    if len({enemy.enemy_id for enemy in snapshot.enemies}) != len(snapshot.enemies):
        raise CoachObservationInputError("duplicate enemy ids")
    if len({s.enemy_id for s in snapshot.sightings}) != len(snapshot.sightings):
        raise CoachObservationInputError("duplicate sightings")
    public_ids = {enemy.enemy_id for enemy in snapshot.enemies if enemy.is_alive}
    if any(s.enemy_id not in public_ids for s in snapshot.sightings):
        raise CoachObservationInputError("sighting of unknown or dead enemy")
    belief_ids = {enemy.enemy_id for enemy in belief.enemies}
    if belief_ids != {enemy.enemy_id for enemy in snapshot.enemies}:
        raise CoachObservationInputError("belief enemy ids disagree")
    expected_points = {point.point_id: point.position for point in config.for_side(snapshot.side)}
    actual_points = {point.point_id: point.position for point in belief.watch_points}
    if expected_points != actual_points or len(actual_points) != len(belief.watch_points):
        raise CoachObservationInputError("belief watch points do not match configuration")
    for ally in snapshot.allies:
        if not isinstance(ally.facing, Facing) or not _in_bounds(ally.position):
            raise CoachObservationInputError("invalid ally position or facing")
        if not isinstance(ally.hp, int) or isinstance(ally.hp, bool) or ally.hp < 0:
            raise CoachObservationInputError("invalid ally HP")
        if not isinstance(ally.normal_ability_charges, int) or isinstance(ally.normal_ability_charges, bool) or ally.normal_ability_charges < 0:
            raise CoachObservationInputError("invalid ability charges")
    for position in snapshot.smoke_cells:
        if not _in_bounds(position):
            raise CoachObservationInputError("smoke position outside fixed map")
    for sighting in snapshot.sightings:
        if not _in_bounds(sighting.reported_position):
            raise CoachObservationInputError("sighting outside fixed map")
    for position in (snapshot.spike.dropped_position, snapshot.spike.planted_position):
        if position is not None and not _in_bounds(position):
            raise CoachObservationInputError("spike outside fixed map")
    if snapshot.spike.is_planted != (snapshot.spike.planted_position is not None):
        raise CoachObservationInputError("planted spike fields disagree")
    if snapshot.spike.own_carrier_slot is not None and not 0 <= snapshot.spike.own_carrier_slot < ROSTER_SIZE:
        raise CoachObservationInputError("invalid spike carrier slot")
    for row in range(MAP_ROWS):
        for column in range(MAP_COLUMNS):
            count = belief.visible_viewer_count[row][column]
            enemy_count = belief.last_seen_enemy_count[row][column]
            if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= ROSTER_SIZE:
                raise CoachObservationInputError("invalid viewer count")
            if not isinstance(enemy_count, int) or isinstance(enemy_count, bool) or not 0 <= enemy_count <= ROSTER_SIZE:
                raise CoachObservationInputError("invalid last-seen count")
            if belief.currently_visible[row][column] != (count > 0):
                raise CoachObservationInputError("visible mask and count disagree")
            for age in (belief.clear_age[row][column], belief.last_seen_age[row][column]):
                if age is not None and (not isinstance(age, int) or isinstance(age, bool) or age < 0):
                    raise CoachObservationInputError("invalid age")
            if (enemy_count > 0) != (belief.last_seen_age[row][column] is not None):
                raise CoachObservationInputError("last-seen count and age disagree")
    for point in belief.watch_points:
        age = point.confirmation_age
        if age is not None and (not isinstance(age, int) or isinstance(age, bool) or age < 0):
            raise CoachObservationInputError("invalid watch confirmation age")


def _in_bounds(position: object) -> bool:
    return (
        isinstance(position, tuple) and len(position) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in position)
        and 0 <= position[0] < MAP_ROWS and 0 <= position[1] < MAP_COLUMNS
    )
