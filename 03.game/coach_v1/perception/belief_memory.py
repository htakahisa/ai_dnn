"""Round-scoped, actor-safe belief history for the coach_v1 team.

The memory consumes only :class:`TeamPerceptionSnapshot` values produced by
the sensor boundary.  It deliberately has no API that accepts a game or a
Character, so an unseen enemy cannot be followed through live state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from coach_v1.common.constants import FACING_DELTAS
from coach_v1.common.types import Facing, GridPosition, Side
from coach_v1.common.watch_points import WatchPoint

from .team_perception import (
    BoolGrid,
    CountGrid,
    EnemySighting,
    PerceptionTick,
    TeamPerceptionSnapshot,
)


OptionalTickGrid = Tuple[Tuple[Optional[int], ...], ...]


class BeliefInputError(ValueError):
    """Raised when snapshots cannot form one consistent round history."""


@dataclass(frozen=True)
class EnemyBelief:
    """The latest legal report for one enemy, never its hidden live state."""

    enemy_id: str
    is_alive: bool
    last_seen_position: Optional[GridPosition]
    last_seen_tick: Optional[int]
    last_seen_age: Optional[int]
    last_seen_direction: Optional[Facing]


@dataclass(frozen=True)
class WatchPointBelief:
    point_id: str
    position: GridPosition
    last_confirmed_tick: Optional[int]
    confirmation_age: Optional[int]


@dataclass(frozen=True)
class BeliefSnapshot:
    """Immutable history state suitable for a later actor encoder."""

    side: Side
    round_number: int
    memory_tick: int
    source_tick: PerceptionTick
    currently_visible: BoolGrid
    visible_viewer_count: CountGrid
    last_clear_tick: OptionalTickGrid
    clear_age: OptionalTickGrid
    enemies: Tuple[EnemyBelief, ...]
    last_seen_enemy_count: CountGrid
    last_seen_age: OptionalTickGrid
    watch_points: Tuple[WatchPointBelief, ...]

    def enemy_for(self, enemy_id: str) -> Optional[EnemyBelief]:
        return next(
            (enemy for enemy in self.enemies if enemy.enemy_id == enemy_id),
            None,
        )

    def watch_point_for(self, point_id: str) -> Optional[WatchPointBelief]:
        return next(
            (point for point in self.watch_points if point.point_id == point_id),
            None,
        )


@dataclass(frozen=True)
class _WatchPointRef:
    point_id: str
    position: GridPosition


class BeliefMemory:
    """Update clear and sighting history from legal team snapshots only."""

    def __init__(self, watch_points: Sequence[WatchPoint] = ()) -> None:
        refs = tuple(
            _WatchPointRef(str(point.point_id), tuple(point.position))
            for point in watch_points
        )
        point_ids = tuple(point.point_id for point in refs)
        if any(not point_id for point_id in point_ids):
            raise BeliefInputError("watch-point ids must be non-empty")
        if len(set(point_ids)) != len(point_ids):
            raise BeliefInputError("duplicate watch-point id")
        self._watch_points = refs
        self._state: Optional[BeliefSnapshot] = None
        self._last_input: Optional[TeamPerceptionSnapshot] = None

    @property
    def state(self) -> Optional[BeliefSnapshot]:
        return self._state

    def reset(self) -> None:
        """Explicitly discard all round history."""

        self._state = None
        self._last_input = None

    def update(self, snapshot: TeamPerceptionSnapshot) -> BeliefSnapshot:
        if not isinstance(snapshot, TeamPerceptionSnapshot):
            raise TypeError("belief memory accepts TeamPerceptionSnapshot only")
        rows, columns = _validate_snapshot(snapshot)
        self._validate_watch_points(rows, columns)
        if self._state is not None and snapshot.side is not self._state.side:
            raise BeliefInputError("one belief memory cannot mix team sides")

        if self._state is None:
            memory_tick = 0
            previous_clear = _empty_optional_grid(rows, columns)
            previous_enemies: Dict[str, EnemyBelief] = {}
            previous_watch_ticks: Dict[str, Optional[int]] = {}
        elif snapshot.tick.round_number != self._state.round_number:
            if snapshot.tick.round_number < self._state.round_number:
                raise BeliefInputError("round number moved backwards")
            memory_tick = 0
            previous_clear = _empty_optional_grid(rows, columns)
            previous_enemies = {}
            previous_watch_ticks = {}
        else:
            if self._last_input is not None and snapshot.tick == self._last_input.tick:
                if snapshot == self._last_input:
                    return self._state
                raise BeliefInputError("same perception tick has different content")
            memory_tick = self._state.memory_tick + _elapsed_ticks(
                self._state.source_tick, snapshot.tick
            )
            if (rows, columns) != _grid_shape(self._state.currently_visible):
                raise BeliefInputError("perception grid shape changed within a round")
            previous_clear = self._state.last_clear_tick
            previous_enemies = {
                enemy.enemy_id: enemy for enemy in self._state.enemies
            }
            previous_watch_ticks = {
                point.point_id: point.last_confirmed_tick
                for point in self._state.watch_points
            }

        current_enemy_ids = tuple(enemy.enemy_id for enemy in snapshot.enemies)
        if previous_enemies and set(previous_enemies) != set(current_enemy_ids):
            raise BeliefInputError("enemy ids changed within a round")

        last_clear = tuple(
            tuple(
                memory_tick if snapshot.currently_visible[row][column]
                else previous_clear[row][column]
                for column in range(columns)
            )
            for row in range(rows)
        )
        clear_age = _age_grid(last_clear, memory_tick)

        sightings = {sighting.enemy_id: sighting for sighting in snapshot.sightings}
        enemy_beliefs = tuple(
            self._updated_enemy(
                public.enemy_id,
                public.is_alive,
                sightings.get(public.enemy_id),
                previous_enemies.get(public.enemy_id),
                snapshot,
                memory_tick,
            )
            for public in snapshot.enemies
        )
        enemy_count, last_seen_age = _enemy_grids(
            enemy_beliefs, rows, columns
        )

        watch_beliefs = tuple(
            WatchPointBelief(
                point_id=point.point_id,
                position=point.position,
                last_confirmed_tick=(
                    memory_tick
                    if snapshot.currently_visible[point.position[0]][point.position[1]]
                    else previous_watch_ticks.get(point.point_id)
                ),
                confirmation_age=None,
            )
            for point in self._watch_points
        )
        watch_beliefs = tuple(
            WatchPointBelief(
                point_id=point.point_id,
                position=point.position,
                last_confirmed_tick=point.last_confirmed_tick,
                confirmation_age=(
                    None
                    if point.last_confirmed_tick is None
                    else memory_tick - point.last_confirmed_tick
                ),
            )
            for point in watch_beliefs
        )

        result = BeliefSnapshot(
            side=snapshot.side,
            round_number=snapshot.tick.round_number,
            memory_tick=memory_tick,
            source_tick=snapshot.tick,
            currently_visible=snapshot.currently_visible,
            visible_viewer_count=snapshot.visible_viewer_count,
            last_clear_tick=last_clear,
            clear_age=clear_age,
            enemies=enemy_beliefs,
            last_seen_enemy_count=enemy_count,
            last_seen_age=last_seen_age,
            watch_points=watch_beliefs,
        )
        self._state = result
        self._last_input = snapshot
        return result

    def _validate_watch_points(self, rows: int, columns: int) -> None:
        for point in self._watch_points:
            row, column = point.position
            if not (0 <= row < rows and 0 <= column < columns):
                raise BeliefInputError(
                    f"watch point {point.point_id!r} is outside perception grid"
                )

    @staticmethod
    def _updated_enemy(
        enemy_id: str,
        is_alive: bool,
        sighting: Optional[EnemySighting],
        previous: Optional[EnemyBelief],
        snapshot: TeamPerceptionSnapshot,
        memory_tick: int,
    ) -> EnemyBelief:
        if not is_alive:
            if sighting is not None:
                raise BeliefInputError("dead enemy cannot have a current sighting")
            return EnemyBelief(enemy_id, False, None, None, None, None)

        if sighting is not None:
            viewer = snapshot.allies[sighting.viewer_slot]
            direction = _direction_from(viewer.position, sighting.reported_position)
            return EnemyBelief(
                enemy_id=enemy_id,
                is_alive=True,
                last_seen_position=sighting.reported_position,
                last_seen_tick=memory_tick,
                last_seen_age=0,
                last_seen_direction=direction,
            )

        if previous is None or previous.last_seen_tick is None:
            return EnemyBelief(enemy_id, True, None, None, None, None)
        return EnemyBelief(
            enemy_id=enemy_id,
            is_alive=True,
            last_seen_position=previous.last_seen_position,
            last_seen_tick=previous.last_seen_tick,
            last_seen_age=memory_tick - previous.last_seen_tick,
            last_seen_direction=previous.last_seen_direction,
        )


def normalize_age(age: Optional[int], maximum_age_ticks: int) -> float:
    """Map an age to [0, 1]; unknown is represented as maximally old.

    The caller supplies the cap because Task 05 owns the final observation
    normalization horizon.  A separate presence map distinguishes ``None``
    from a genuinely old value.
    """

    if (
        isinstance(maximum_age_ticks, bool)
        or not isinstance(maximum_age_ticks, int)
        or maximum_age_ticks <= 0
    ):
        raise ValueError("maximum_age_ticks must be a positive integer")
    if age is None:
        return 1.0
    if isinstance(age, bool) or not isinstance(age, int) or age < 0:
        raise ValueError("age must be a non-negative integer or None")
    return min(age, maximum_age_ticks) / maximum_age_ticks


def _validate_snapshot(snapshot: TeamPerceptionSnapshot) -> Tuple[int, int]:
    if not isinstance(snapshot.side, Side):
        raise BeliefInputError("snapshot side must be a Side value")
    rows, columns = _grid_shape(snapshot.currently_visible)
    if _grid_shape(snapshot.visible_viewer_count) != (rows, columns):
        raise BeliefInputError("visibility grids must have the same shape")
    for row in range(rows):
        for column in range(columns):
            visible = snapshot.currently_visible[row][column]
            count = snapshot.visible_viewer_count[row][column]
            if not isinstance(visible, bool):
                raise BeliefInputError("currently_visible must contain bool values")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise BeliefInputError(
                    "visible_viewer_count must contain non-negative integers"
                )
            if visible != (count > 0):
                raise BeliefInputError("visible mask and viewer count disagree")

    enemy_ids = tuple(enemy.enemy_id for enemy in snapshot.enemies)
    if len(set(enemy_ids)) != len(enemy_ids):
        raise BeliefInputError("duplicate enemy id")
    sighting_ids = tuple(sighting.enemy_id for sighting in snapshot.sightings)
    if len(set(sighting_ids)) != len(sighting_ids):
        raise BeliefInputError("duplicate sighting for one enemy")
    if not set(sighting_ids).issubset(enemy_ids):
        raise BeliefInputError("sighting references an unknown enemy")
    for sighting in snapshot.sightings:
        row, column = sighting.reported_position
        if not (0 <= row < rows and 0 <= column < columns):
            raise BeliefInputError("reported enemy position is outside the grid")
        if not 0 <= sighting.viewer_slot < len(snapshot.allies):
            raise BeliefInputError("sighting references an invalid viewer slot")
    return rows, columns


def _grid_shape(grid: Sequence[Sequence[object]]) -> Tuple[int, int]:
    rows = len(grid)
    if rows == 0:
        raise BeliefInputError("perception grid must not be empty")
    columns = len(grid[0])
    if columns == 0 or any(len(row) != columns for row in grid):
        raise BeliefInputError("perception grid must be rectangular")
    return rows, columns


def _empty_optional_grid(rows: int, columns: int) -> OptionalTickGrid:
    return tuple(tuple(None for _ in range(columns)) for _ in range(rows))


def _age_grid(last_ticks: OptionalTickGrid, memory_tick: int) -> OptionalTickGrid:
    return tuple(
        tuple(None if tick is None else memory_tick - tick for tick in row)
        for row in last_ticks
    )


def _enemy_grids(
    enemies: Sequence[EnemyBelief], rows: int, columns: int
) -> Tuple[CountGrid, OptionalTickGrid]:
    count = [[0 for _ in range(columns)] for _ in range(rows)]
    age: list[list[Optional[int]]] = [
        [None for _ in range(columns)] for _ in range(rows)
    ]
    for enemy in enemies:
        if enemy.last_seen_position is None or enemy.last_seen_age is None:
            continue
        row, column = enemy.last_seen_position
        count[row][column] += 1
        current = age[row][column]
        age[row][column] = (
            enemy.last_seen_age
            if current is None
            else min(current, enemy.last_seen_age)
        )
    return (
        tuple(tuple(row) for row in count),
        tuple(tuple(row) for row in age),
    )


def _elapsed_ticks(previous: PerceptionTick, current: PerceptionTick) -> int:
    if current.round_number != previous.round_number:
        raise BeliefInputError("elapsed tick comparison crossed a round")
    if previous.phase == current.phase == "live":
        elapsed = current.tick - previous.tick
    elif previous.phase == current.phase == "defender_setup":
        elapsed = previous.tick - current.tick
    elif previous.phase == "defender_setup" and current.phase == "live":
        elapsed = previous.tick + current.tick
    else:
        raise BeliefInputError("perception phase moved backwards")
    if elapsed <= 0:
        raise BeliefInputError("perception tick must move forwards")
    return elapsed


_DIRECTION_BY_DELTA = {delta: facing for facing, delta in FACING_DELTAS.items()}


def _direction_from(
    origin: GridPosition, target: GridPosition
) -> Optional[Facing]:
    row_delta = (target[0] > origin[0]) - (target[0] < origin[0])
    column_delta = (target[1] > origin[1]) - (target[1] < origin[1])
    if row_delta == 0 and column_delta == 0:
        return None
    return _DIRECTION_BY_DELTA[(row_delta, column_delta)]
