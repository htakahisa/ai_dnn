"""Strict fixed-map watch-point configuration and text visualization.

Watch points are training data and observation metadata.  This module does not
read live game objects and deliberately contains no enemy-state or path-finding
logic.
"""

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from .hashing import load_hashed_json_config, map_sha256, normalize_map_text
from .constants import FACING_DELTAS
from .types import Facing, GridPosition, Side


WATCH_POINTS_SCHEMA_VERSION = "watch-points-v1"
IMPORTANCE_MIN = 1
IMPORTANCE_MAX = 5
RANDOM_RADIUS_MIN = 1
RANDOM_RADIUS_MAX = 3

ATTACKER_SITUATIONS = frozenset({"carry", "retrieve", "guard"})
DEFENDER_SITUATIONS = frozenset({"search", "retake"})
ALLOWED_SITUATIONS = ATTACKER_SITUATIONS | DEFENDER_SITUATIONS
ALLOWED_TAGS = frozenset(
    {
        "choke",
        "corner",
        "flank",
        "long_angle",
        "off_angle",
        "orb",
        "post_plant",
        "retake",
        "site_entry",
    }
)

_ROOT_KEYS = frozenset({"schema_version", "map", "points"})
_MAP_KEYS = frozenset({"rows", "columns", "sha256"})
_POINT_KEYS = frozenset(
    {
        "id",
        "position",
        "importance",
        "facing",
        "sides",
        "situations",
        "random_radius",
        "tags",
    }
)
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class WatchPointConfigError(ValueError):
    """Raised when watch-point data does not match the fixed v1 contract."""


@dataclass(frozen=True)
class WatchPoint:
    point_id: str
    position: GridPosition
    importance: int
    facing: Facing
    sides: Tuple[Side, ...]
    situations: Tuple[str, ...]
    random_radius: int
    tags: Tuple[str, ...]

    def supports_side(self, side: Side) -> bool:
        return side in self.sides

    def supports_situation(self, situation: str) -> bool:
        return situation in self.situations


@dataclass(frozen=True)
class WatchPointConfig:
    schema_version: str
    map_rows: int
    map_columns: int
    map_hash: str
    config_hash: str
    points: Tuple[WatchPoint, ...]

    def for_side(self, side: Union[Side, str]) -> Tuple[WatchPoint, ...]:
        parsed_side = _parse_side(side, "side")
        return tuple(point for point in self.points if point.supports_side(parsed_side))

    def for_situation(self, situation: str) -> Tuple[WatchPoint, ...]:
        if situation not in ALLOWED_SITUATIONS:
            raise ValueError(f"unsupported situation: {situation!r}")
        return tuple(
            point for point in self.points if point.supports_situation(situation)
        )


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset, location: str
) -> None:
    keys = frozenset(value.keys())
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise WatchPointConfigError(
            f"{location} has invalid keys: missing={missing}, extra={extra}"
        )


def _require_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WatchPointConfigError(f"{name} must be int")
    if not minimum <= value <= maximum:
        raise WatchPointConfigError(
            f"{name} must be in range {minimum}..{maximum}: {value}"
        )
    return value


def _require_string_list(
    value: Any, name: str, allowed: Iterable[str]
) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise WatchPointConfigError(f"{name} must be a non-empty list")
    if any(not isinstance(item, str) for item in value):
        raise WatchPointConfigError(f"{name} must contain only strings")
    if len(set(value)) != len(value):
        raise WatchPointConfigError(f"{name} contains duplicate values")
    allowed_values = frozenset(allowed)
    invalid = sorted(set(value) - allowed_values)
    if invalid:
        raise WatchPointConfigError(f"{name} has unsupported values: {invalid}")
    return tuple(value)


def _parse_side(value: Union[Side, str], name: str) -> Side:
    try:
        return value if isinstance(value, Side) else Side(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be 'attacker' or 'defender'") from exc


def _parse_map_rows(map_text: str) -> Tuple[str, ...]:
    rows = tuple(normalize_map_text(map_text).split("\n"))
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise WatchPointConfigError("map rows must be non-empty and rectangular")
    return rows


def _parse_point(value: Any, index: int, map_rows: Sequence[str]) -> WatchPoint:
    location = f"points[{index}]"
    if not isinstance(value, Mapping):
        raise WatchPointConfigError(f"{location} must be an object")
    _require_exact_keys(value, _POINT_KEYS, location)

    point_id = value["id"]
    if not isinstance(point_id, str) or _ID_PATTERN.fullmatch(point_id) is None:
        raise WatchPointConfigError(
            f"{location}.id must match {_ID_PATTERN.pattern!r}"
        )

    raw_position = value["position"]
    if not isinstance(raw_position, list) or len(raw_position) != 2:
        raise WatchPointConfigError(f"{location}.position must be [row, column]")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw_position):
        raise WatchPointConfigError(f"{location}.position values must be integers")
    row, column = raw_position
    if not (0 <= row < len(map_rows) and 0 <= column < len(map_rows[0])):
        raise WatchPointConfigError(
            f"{location} position is outside map: ({row}, {column})"
        )
    if map_rows[row][column] == "1":
        raise WatchPointConfigError(
            f"{location} position is a wall: ({row}, {column})"
        )

    try:
        facing = Facing(value["facing"])
    except (TypeError, ValueError) as exc:
        raise WatchPointConfigError(
            f"{location}.facing must be one of {[item.value for item in Facing]}"
        ) from exc
    facing_delta = FACING_DELTAS[facing]
    facing_row = row + facing_delta[0]
    facing_column = column + facing_delta[1]
    if not (
        0 <= facing_row < len(map_rows)
        and 0 <= facing_column < len(map_rows[0])
    ) or map_rows[facing_row][facing_column] == "1":
        raise WatchPointConfigError(
            f"{location}.facing points immediately outside the walkable map"
        )

    side_values = _require_string_list(
        value["sides"], f"{location}.sides", (side.value for side in Side)
    )
    sides = tuple(Side(side) for side in side_values)
    situations = _require_string_list(
        value["situations"], f"{location}.situations", ALLOWED_SITUATIONS
    )
    allowed_for_sides = set()
    if Side.ATTACKER in sides:
        allowed_for_sides.update(ATTACKER_SITUATIONS)
        if ATTACKER_SITUATIONS.isdisjoint(situations):
            raise WatchPointConfigError(
                f"{location} has attacker side without attacker situation"
            )
    if Side.DEFENDER in sides:
        allowed_for_sides.update(DEFENDER_SITUATIONS)
        if DEFENDER_SITUATIONS.isdisjoint(situations):
            raise WatchPointConfigError(
                f"{location} has defender side without defender situation"
            )
    invalid_situations = sorted(set(situations) - allowed_for_sides)
    if invalid_situations:
        raise WatchPointConfigError(
            f"{location} situations do not match sides: {invalid_situations}"
        )

    return WatchPoint(
        point_id=point_id,
        position=(row, column),
        importance=_require_int(
            value["importance"],
            f"{location}.importance",
            IMPORTANCE_MIN,
            IMPORTANCE_MAX,
        ),
        facing=facing,
        sides=sides,
        situations=situations,
        random_radius=_require_int(
            value["random_radius"],
            f"{location}.random_radius",
            RANDOM_RADIUS_MIN,
            RANDOM_RADIUS_MAX,
        ),
        tags=_require_string_list(value["tags"], f"{location}.tags", ALLOWED_TAGS),
    )


def validate_watch_points(
    value: Mapping[str, Any], map_text: str, config_hash: str = ""
) -> WatchPointConfig:
    """Validate raw config against the supplied fixed map and return typed data."""

    if not isinstance(value, Mapping):
        raise WatchPointConfigError("configuration root must be an object")
    _require_exact_keys(value, _ROOT_KEYS, "configuration root")
    if value["schema_version"] != WATCH_POINTS_SCHEMA_VERSION:
        raise WatchPointConfigError(
            f"unsupported schema_version: {value['schema_version']!r}"
        )

    map_metadata = value["map"]
    if not isinstance(map_metadata, Mapping):
        raise WatchPointConfigError("map must be an object")
    _require_exact_keys(map_metadata, _MAP_KEYS, "map")
    rows = _parse_map_rows(map_text)
    actual_hash = map_sha256(map_text)
    expected_metadata = {
        "rows": len(rows),
        "columns": len(rows[0]),
        "sha256": actual_hash,
    }
    if dict(map_metadata) != expected_metadata:
        raise WatchPointConfigError(
            "map metadata does not match supplied map: "
            f"expected={expected_metadata}, actual={dict(map_metadata)}"
        )

    raw_points = value["points"]
    if not isinstance(raw_points, list) or not raw_points:
        raise WatchPointConfigError("points must be a non-empty list")
    points = tuple(
        _parse_point(point, index, rows) for index, point in enumerate(raw_points)
    )
    point_ids = [point.point_id for point in points]
    positions = [point.position for point in points]
    if len(set(point_ids)) != len(point_ids):
        raise WatchPointConfigError("duplicate watch-point id")
    if len(set(positions)) != len(positions):
        raise WatchPointConfigError("duplicate watch-point position")
    for side in Side:
        if not any(point.supports_side(side) for point in points):
            raise WatchPointConfigError(f"no watch points registered for {side.value}")
    for situation in ALLOWED_SITUATIONS:
        if not any(point.supports_situation(situation) for point in points):
            raise WatchPointConfigError(
                f"no watch points registered for situation {situation!r}"
            )

    return WatchPointConfig(
        schema_version=WATCH_POINTS_SCHEMA_VERSION,
        map_rows=len(rows),
        map_columns=len(rows[0]),
        map_hash=actual_hash,
        config_hash=config_hash,
        points=points,
    )


def load_watch_points(
    path: Union[str, Path], map_text: str
) -> WatchPointConfig:
    raw_config, config_hash = load_hashed_json_config(path)
    return validate_watch_points(raw_config, map_text, config_hash)


_FACING_ARROWS = {
    Facing.N: "↑",
    Facing.NE: "↗",
    Facing.E: "→",
    Facing.SE: "↘",
    Facing.S: "↓",
    Facing.SW: "↙",
    Facing.W: "←",
    Facing.NW: "↖",
}
_MAP_GLYPHS = {"0": ".", "1": "#", "2": "P", "3": "A", "4": "D", "5": "O"}


def render_watch_points(
    config: WatchPointConfig,
    map_text: str,
    *,
    side: Optional[Union[Side, str]] = None,
    situation: Optional[str] = None,
) -> str:
    """Render a coordinate grid whose arrows show recommended facing."""

    map_rows = _parse_map_rows(map_text)
    points = config.points
    if side is not None:
        parsed_side = _parse_side(side, "side")
        points = tuple(point for point in points if point.supports_side(parsed_side))
    if situation is not None:
        if situation not in ALLOWED_SITUATIONS:
            raise ValueError(f"unsupported situation: {situation!r}")
        points = tuple(
            point for point in points if point.supports_situation(situation)
        )

    canvas = [[_MAP_GLYPHS.get(cell, "?") for cell in row] for row in map_rows]
    for point in points:
        row, column = point.position
        canvas[row][column] = _FACING_ARROWS[point.facing]

    lines = [
        "    " + "".join(str(column // 10 or " ") for column in range(config.map_columns)),
        "    " + "".join(str(column % 10) for column in range(config.map_columns)),
    ]
    lines.extend(f"{row:02}: {''.join(cells)}" for row, cells in enumerate(canvas))
    lines.append("legend: #=wall .=floor P=plant A=attacker-start D=defender-start O=orb")
    lines.append("facing: ↑=N ↗=NE →=E ↘=SE ↓=S ↙=SW ←=W ↖=NW")
    lines.append(f"points: {len(points)} / config_sha256: {config.config_hash}")
    for point in points:
        sides = ",".join(side.value for side in point.sides)
        situations = ",".join(point.situations)
        tags = ",".join(point.tags)
        lines.append(
            f"- {point.point_id} {point.position} {_FACING_ARROWS[point.facing]} "
            f"importance={point.importance} radius={point.random_radius} "
            f"sides={sides} situations={situations} tags={tags}"
        )
    return "\n".join(lines)
