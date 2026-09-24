"""Build immutable, actor-safe perception shared by one coach-controlled team.

Only :class:`TeamPerceptionBuilder` may inspect live game and Character objects.
Its result contains copied primitive values, enums and tuples; it never retains
the game, a Character, or one of the legacy perceived-object proxies.

This module intentionally implements only current-tick sensing.  Clear ages,
last-seen ages and other round memory belong to Task 04's belief memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Optional, Sequence, Tuple, Union

from iq_perception import ENEMY_POSITION_MAX_ERROR, IQPerceptionEngine

from coach_v1.common.constants import FACING_DELTAS, FIXED_ROSTER
from coach_v1.common.types import Facing, GridPosition, Side


BoolGrid = Tuple[Tuple[bool, ...], ...]
CountGrid = Tuple[Tuple[int, ...], ...]


class PerceptionInputError(ValueError):
    """Raised when live state cannot satisfy the fixed coach_v1 contract."""


class SightingSource(str, Enum):
    NORMAL = "normal"
    RECON = "recon"


@dataclass(frozen=True)
class PerceptionTick:
    round_number: int
    phase: str
    tick: int

    def __post_init__(self) -> None:
        if self.phase not in {"live", "defender_setup"}:
            raise ValueError(f"unsupported perception phase: {self.phase!r}")


@dataclass(frozen=True)
class AllyPerception:
    slot: int
    character_id: str
    position: GridPosition
    facing: Facing
    is_alive: bool
    hp: int
    normal_ability_charges: int
    has_spike: bool


@dataclass(frozen=True)
class EnemyPublicState:
    """Enemy information that remains legal even while the enemy is unseen."""

    enemy_id: str
    is_alive: bool


@dataclass(frozen=True)
class EnemySighting:
    """A current-tick report produced only after a legal sighting check."""

    enemy_id: str
    reported_position: GridPosition
    viewer_slot: int
    source: SightingSource


@dataclass(frozen=True)
class SpikeSharedInfo:
    """Public spike state without enemy-carrier identity or position."""

    own_carrier_slot: Optional[int]
    dropped_position: Optional[GridPosition]
    is_planted: bool
    planted_position: Optional[GridPosition]


@dataclass(frozen=True)
class TeamPerceptionSnapshot:
    """Immutable current-tick input boundary shared by all five actor slots."""

    side: Side
    tick: PerceptionTick
    allies: Tuple[AllyPerception, ...]
    enemies: Tuple[EnemyPublicState, ...]
    sightings: Tuple[EnemySighting, ...]
    currently_visible: BoolGrid
    visible_viewer_count: CountGrid
    smoke_cells: Tuple[GridPosition, ...]
    spike: SpikeSharedInfo

    def sighting_for(self, enemy_id: str) -> Optional[EnemySighting]:
        return next(
            (item for item in self.sightings if item.enemy_id == enemy_id), None
        )


@dataclass(frozen=True)
class _Viewer:
    slot: int
    character: Any
    position: GridPosition
    facing: Facing
    effective_iq: float


class _IQEnemyPositionReporter:
    """Narrow adapter around the existing deterministic IQ position error."""

    def __init__(self, engine: Optional[IQPerceptionEngine] = None) -> None:
        self._engine = engine or IQPerceptionEngine()

    def report(
        self,
        *,
        game: Any,
        viewer: Any,
        enemy_id: str,
        actual_position: GridPosition,
    ) -> GridPosition:
        # Qualification has already happened in TeamPerceptionBuilder.  In
        # particular this method is never called for an unseen enemy.
        reported = self._engine._blur_pos(
            game,
            viewer,
            actual_position,
            ENEMY_POSITION_MAX_ERROR,
            "coach_enemy_sighting",
            enemy_id,
        )
        if reported is None:
            raise PerceptionInputError("IQ reporter discarded a qualified sighting")
        return _copy_position(reported, "reported enemy position")


class TeamPerceptionBuilder:
    """Sensor boundary that converts live state into an actor-safe snapshot."""

    _TEAM_CODE = {Side.ATTACKER: "A", Side.DEFENDER: "D"}

    def __init__(self, *, iq_engine: Optional[IQPerceptionEngine] = None) -> None:
        self._reporter = _IQEnemyPositionReporter(iq_engine)

    def build(
        self, *, game: Any, side: Union[Side, str]
    ) -> TeamPerceptionSnapshot:
        parsed_side = _parse_side(side)
        grid = getattr(game, "grid", None)
        rows, columns = _grid_shape(grid)
        smoke_cells = _smoke_cells(game, rows, columns)
        characters = tuple(getattr(game, "chars", ()))
        team_code = self._TEAM_CODE[parsed_side]

        raw_allies = tuple(
            character
            for character in characters
            if str(getattr(character, "team", "")) == team_code
        )
        slotted_allies = _slot_allies(raw_allies)
        viewers = tuple(
            _Viewer(
                slot=slot,
                character=character,
                position=_bounded_position(
                    getattr(character, "pos", None),
                    rows,
                    columns,
                    f"ally slot {slot} position",
                ),
                facing=_parse_facing(getattr(character, "facing", None), slot),
                effective_iq=_effective_iq(character),
            )
            for slot, character in enumerate(slotted_allies)
        )

        allies = tuple(
            _copy_ally(viewer, slotted_allies[viewer.slot]) for viewer in viewers
        )
        visible_by_viewer = tuple(
            self._visible_cells(
                game=game,
                grid=grid,
                rows=rows,
                columns=columns,
                smoke_cells=smoke_cells,
                viewer=viewer,
            )
            for viewer in viewers
        )
        visible, viewer_count = _merge_visibility(
            visible_by_viewer, rows=rows, columns=columns
        )

        raw_enemies = tuple(
            character
            for character in characters
            if str(getattr(character, "team", "")) != team_code
        )
        identified_enemies = _identify_enemies(raw_enemies)
        enemies = tuple(
            EnemyPublicState(enemy_id=enemy_id, is_alive=_is_alive(enemy))
            for enemy_id, enemy in identified_enemies
        )
        sightings = tuple(
            sighting
            for enemy_id, enemy in identified_enemies
            if (
                sighting := self._sighting_for(
                    game=game,
                    enemy_id=enemy_id,
                    enemy=enemy,
                    viewers=viewers,
                )
            )
            is not None
        )

        return TeamPerceptionSnapshot(
            side=parsed_side,
            tick=_tick_key(game),
            allies=allies,
            enemies=enemies,
            sightings=sightings,
            currently_visible=visible,
            visible_viewer_count=viewer_count,
            smoke_cells=tuple(sorted(smoke_cells)),
            spike=_copy_spike(game, parsed_side, allies, rows, columns),
        )

    def _visible_cells(
        self,
        *,
        game: Any,
        grid: Any,
        rows: int,
        columns: int,
        smoke_cells: frozenset[GridPosition],
        viewer: _Viewer,
    ) -> frozenset[GridPosition]:
        character = viewer.character
        if not _is_alive(character) or _is_blind(character):
            return frozenset()

        visible = set()
        for row in range(rows):
            for column in range(columns):
                position = (row, column)
                if _grid_value(grid, row, column) == 1:
                    continue
                if position in smoke_cells:
                    continue
                if not _in_forward_half_plane(
                    viewer.position, position, viewer.facing
                ):
                    continue
                if _has_normal_los(game, viewer.position, position):
                    visible.add(position)
        return frozenset(visible)

    def _sighting_for(
        self,
        *,
        game: Any,
        enemy_id: str,
        enemy: Any,
        viewers: Sequence[_Viewer],
    ) -> Optional[EnemySighting]:
        if not _is_alive(enemy):
            return None
        rows, columns = _grid_shape(getattr(game, "grid", None))
        enemy_position = _bounded_position(
            getattr(enemy, "pos", None),
            rows,
            columns,
            f"enemy {enemy_id!r} sensor position",
        )

        candidates = []
        for viewer in viewers:
            if (
                _is_alive(viewer.character)
                and not _is_blind(viewer.character)
                and _in_forward_half_plane(
                    viewer.position, enemy_position, viewer.facing
                )
                and _has_normal_los(game, viewer.position, enemy_position)
            ):
                candidates.append((viewer, SightingSource.NORMAL))

        if _is_recon_revealed(enemy):
            recon_viewer = next(
                (
                    viewer
                    for viewer in viewers
                    if FIXED_ROSTER[viewer.slot].normal_ability == "RECON"
                ),
                None,
            )
            if recon_viewer is not None and all(
                candidate.slot != recon_viewer.slot for candidate, _ in candidates
            ):
                # A live reveal remains team-shared even if its owner is blind
                # or died after launching it.
                candidates.append((recon_viewer, SightingSource.RECON))

        if not candidates:
            return None

        selected, source = min(
            candidates,
            key=lambda item: (-item[0].effective_iq, item[0].slot),
        )
        reported_position = self._reporter.report(
            game=game,
            viewer=selected.character,
            enemy_id=enemy_id,
            actual_position=enemy_position,
        )
        return EnemySighting(
            enemy_id=enemy_id,
            reported_position=reported_position,
            viewer_slot=selected.slot,
            source=source,
        )


def _parse_side(value: Union[Side, str]) -> Side:
    try:
        return value if isinstance(value, Side) else Side(value)
    except (TypeError, ValueError) as exc:
        raise PerceptionInputError("side must be 'attacker' or 'defender'") from exc


def _grid_shape(grid: Any) -> Tuple[int, int]:
    shape = getattr(grid, "shape", None)
    if shape is not None and len(shape) == 2:
        rows, columns = int(shape[0]), int(shape[1])
    else:
        try:
            rows = len(grid)
            columns = len(grid[0])
        except (TypeError, IndexError) as exc:
            raise PerceptionInputError("game grid must be a non-empty 2D grid") from exc
    if rows <= 0 or columns <= 0:
        raise PerceptionInputError("game grid must be a non-empty 2D grid")
    return rows, columns


def _grid_value(grid: Any, row: int, column: int) -> int:
    try:
        return int(grid[row, column])
    except (TypeError, IndexError):
        return int(grid[row][column])


def _copy_position(value: Any, name: str) -> GridPosition:
    try:
        if len(value) != 2:
            raise ValueError
        row, column = value
        if isinstance(row, bool) or isinstance(column, bool):
            raise ValueError
        return int(row), int(column)
    except (TypeError, ValueError, IndexError) as exc:
        raise PerceptionInputError(f"{name} must be a two-integer position") from exc


def _bounded_position(
    value: Any, rows: int, columns: int, name: str
) -> GridPosition:
    position = _copy_position(value, name)
    if not (0 <= position[0] < rows and 0 <= position[1] < columns):
        raise PerceptionInputError(f"{name} is outside the map: {position}")
    return position


def _optional_bounded_position(
    value: Any, rows: int, columns: int, name: str
) -> Optional[GridPosition]:
    if value is None:
        return None
    return _bounded_position(value, rows, columns, name)


def _parse_facing(value: Any, slot: int) -> Facing:
    try:
        return value if isinstance(value, Facing) else Facing(value)
    except (TypeError, ValueError) as exc:
        raise PerceptionInputError(
            f"ally slot {slot} has unsupported facing: {value!r}"
        ) from exc


def _slot_allies(allies: Sequence[Any]) -> Tuple[Any, ...]:
    by_name = {}
    for character in allies:
        name = str(getattr(character, "name", ""))
        if name in by_name:
            raise PerceptionInputError(f"duplicate allied character id: {name!r}")
        by_name[name] = character
    expected = {slot.character_name for slot in FIXED_ROSTER}
    actual = set(by_name)
    if actual != expected:
        raise PerceptionInputError(
            "coach_v1 side must use the fixed Gorigons roster: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return tuple(by_name[slot.character_name] for slot in FIXED_ROSTER)


def _identify_enemies(enemies: Sequence[Any]) -> Tuple[Tuple[str, Any], ...]:
    identified = []
    seen = set()
    for enemy in enemies:
        enemy_id = str(getattr(enemy, "name", ""))
        if not enemy_id:
            raise PerceptionInputError("enemy character id must be non-empty")
        if enemy_id in seen:
            raise PerceptionInputError(f"duplicate enemy character id: {enemy_id!r}")
        seen.add(enemy_id)
        identified.append((enemy_id, enemy))
    return tuple(sorted(identified, key=lambda item: item[0]))


def _is_alive(character: Any) -> bool:
    return bool(getattr(character, "is_alive", True))


def _is_blind(character: Any) -> bool:
    try:
        return float(getattr(character, "blind_remaining", 0.0)) > 0.0
    except (TypeError, ValueError) as exc:
        raise PerceptionInputError("blind_remaining must be numeric") from exc


def _is_recon_revealed(character: Any) -> bool:
    try:
        return float(getattr(character, "reveal_remaining", 0.0)) > 0.0
    except (TypeError, ValueError) as exc:
        raise PerceptionInputError("reveal_remaining must be numeric") from exc


def _effective_iq(character: Any) -> float:
    try:
        value = float(
            getattr(character, "effective_iq", getattr(character, "iq", 50.0))
        )
    except (TypeError, ValueError):
        value = 50.0
    if not math.isfinite(value):
        value = 50.0
    return max(0.0, value)


def _normal_ability_charges(character: Any, slot: int) -> int:
    ability = FIXED_ROSTER[slot].normal_ability
    attribute = {
        "SMOKE": "smoke_charges",
        "RECON": "recon_charges",
        "FLASH": "flash_charges",
        "HUNT": None,
    }[ability]
    if attribute is None:
        return 0
    try:
        return max(0, int(getattr(character, attribute, 0)))
    except (TypeError, ValueError) as exc:
        raise PerceptionInputError(f"{attribute} must be an integer") from exc


def _copy_ally(viewer: _Viewer, character: Any) -> AllyPerception:
    try:
        hp = max(0, int(getattr(character, "hp", 0)))
    except (TypeError, ValueError) as exc:
        raise PerceptionInputError("ally hp must be an integer") from exc
    return AllyPerception(
        slot=viewer.slot,
        character_id=FIXED_ROSTER[viewer.slot].checkpoint_id,
        position=viewer.position,
        facing=viewer.facing,
        is_alive=_is_alive(character),
        hp=hp,
        normal_ability_charges=_normal_ability_charges(character, viewer.slot),
        has_spike=bool(getattr(character, "has_spike", False)),
    )


def _smoke_cells(
    game: Any, rows: int, columns: int
) -> frozenset[GridPosition]:
    getter = getattr(game, "_smoke_cells", None)
    if callable(getter):
        raw_cells = getter()
    else:
        raw_cells = (
            cell
            for smoke in getattr(game, "smokes", ())
            for cell in smoke.get("cells", ())
        )
    return frozenset(
        _bounded_position(cell, rows, columns, "smoke cell") for cell in raw_cells
    )


def _in_forward_half_plane(
    origin: GridPosition, target: GridPosition, facing: Facing
) -> bool:
    delta_row = target[0] - origin[0]
    delta_column = target[1] - origin[1]
    facing_row, facing_column = FACING_DELTAS[facing]
    return delta_row * facing_row + delta_column * facing_column >= 0


def _has_normal_los(
    game: Any, origin: GridPosition, target: GridPosition
) -> bool:
    checker = getattr(game, "check_cell_line_of_sight", None)
    if not callable(checker):
        raise PerceptionInputError(
            "game must provide check_cell_line_of_sight for the existing LOS rule"
        )
    return bool(checker(origin, target, block_smoke=True))


def _merge_visibility(
    visible_by_viewer: Sequence[frozenset[GridPosition]],
    *,
    rows: int,
    columns: int,
) -> Tuple[BoolGrid, CountGrid]:
    count_rows = []
    visible_rows = []
    for row in range(rows):
        counts = tuple(
            sum((row, column) in cells for cells in visible_by_viewer)
            for column in range(columns)
        )
        count_rows.append(counts)
        visible_rows.append(tuple(count > 0 for count in counts))
    return tuple(visible_rows), tuple(count_rows)


def _copy_spike(
    game: Any,
    side: Side,
    allies: Sequence[AllyPerception],
    rows: int,
    columns: int,
) -> SpikeSharedInfo:
    carriers = [ally.slot for ally in allies if ally.has_spike]
    if len(carriers) > 1:
        raise PerceptionInputError("multiple allied spike carriers")
    own_carrier_slot = carriers[0] if carriers else None
    if side is Side.DEFENDER and own_carrier_slot is not None:
        raise PerceptionInputError("defender cannot be the spike carrier")

    is_planted = bool(getattr(game, "is_planted", False))
    planted_position = _optional_bounded_position(
        getattr(game, "planted_pos", None), rows, columns, "planted spike position"
    )
    if is_planted != (planted_position is not None):
        raise PerceptionInputError(
            "is_planted and planted spike position must agree"
        )
    dropped_position = _optional_bounded_position(
        getattr(game, "spike_pos", None), rows, columns, "dropped spike position"
    )
    if is_planted and dropped_position is not None:
        raise PerceptionInputError("planted spike cannot also be dropped")
    if is_planted and own_carrier_slot is not None:
        raise PerceptionInputError("planted spike cannot also have a carrier")
    return SpikeSharedInfo(
        own_carrier_slot=own_carrier_slot,
        dropped_position=dropped_position,
        is_planted=is_planted,
        planted_position=planted_position,
    )


def _tick_key(game: Any) -> PerceptionTick:
    try:
        round_number = int(getattr(game, "current_round", 0))
        setup_phase = getattr(game, "defender_setup_phase", None)
        setup_active = bool(
            getattr(game, "defender_setup_active", False)
            or getattr(setup_phase, "active", False)
        )
        if setup_active:
            remaining = getattr(
                game,
                "defender_setup_ticks_remaining",
                getattr(setup_phase, "ticks_remaining", 0),
            )
            return PerceptionTick(round_number, "defender_setup", int(remaining))
        return PerceptionTick(
            round_number, "live", int(getattr(game, "battle_tick", 0))
        )
    except (TypeError, ValueError) as exc:
        raise PerceptionInputError("round and tick values must be integers") from exc
