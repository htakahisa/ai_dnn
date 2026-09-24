"""Convert the editable ``watch_points_map.py`` overlay into JSON.

The overlay is intentionally a tiny authoring format: cells containing ``4``
are watch-point coordinates.  Metadata not representable by the overlay is
filled deterministically, while matching coordinates from the previous JSON
retain their reviewed metadata.
"""

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from map_data import NEW_MAZE_STR as BASE_MAP

from coach_v1.common.constants import WATCH_POINTS_CONFIG_PATH
from coach_v1.common.hashing import (
    canonical_json_sha256,
    load_json_config,
    map_sha256,
    normalize_map_text,
)
from coach_v1.common.constants import FACING_DELTAS
from coach_v1.common.types import Facing
from coach_v1.common.watch_points import (
    WATCH_POINTS_SCHEMA_VERSION,
    validate_watch_points,
)


DEFAULT_MARKED_MAP_PATH = Path(__file__).resolve().parent / "config" / "watch_points_map.py"


def _rows(map_text: str, label: str) -> Tuple[str, ...]:
    rows = tuple(normalize_map_text(map_text).split("\n"))
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{label} must be a non-empty rectangular map")
    invalid = sorted({cell for row in rows for cell in row} - set("012345"))
    if invalid:
        raise ValueError(f"{label} contains invalid cells: {invalid}")
    return rows


def load_marked_map(path: Path) -> str:
    """Read NEW_MAZE_STR from a Python authoring file without executing it."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        assignment = node if isinstance(node, ast.Assign) else node if isinstance(node, ast.AnnAssign) else None
        if assignment is None:
            continue
        targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
        if not any(isinstance(target, ast.Name) and target.id == "NEW_MAZE_STR" for target in targets):
            continue
        try:
            value = ast.literal_eval(assignment.value)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise ValueError(f"{path} NEW_MAZE_STR must be a string literal") from exc
        if not isinstance(value, str):
            raise ValueError(f"{path} NEW_MAZE_STR must be a string literal")
        _rows(value, "marked map")
        return value
    raise ValueError(f"{path} does not define NEW_MAZE_STR")


def extract_watch_positions(marked_map: str, base_map: str) -> Tuple[Tuple[int, int], ...]:
    marked_rows = _rows(marked_map, "marked map")
    base_rows = _rows(base_map, "base map")
    if (len(marked_rows), len(marked_rows[0])) != (len(base_rows), len(base_rows[0])):
        raise ValueError("marked map dimensions do not match base map")
    positions = []
    for row, marked_line in enumerate(marked_rows):
        for column, cell in enumerate(marked_line):
            if cell != "4":
                continue
            if base_rows[row][column] == "1":
                raise ValueError(f"watch point ({row}, {column}) is on a wall")
            positions.append((row, column))
    if not positions:
        raise ValueError("marked map contains no watch-point cells ('4')")
    return tuple(positions)


def _walkable(base_rows: Tuple[str, ...], row: int, column: int) -> bool:
    return 0 <= row < len(base_rows) and 0 <= column < len(base_rows[0]) and base_rows[row][column] != "1"


def _facing_for_position(position: Tuple[int, int], base_rows: Tuple[str, ...]) -> str:
    """Choose the longest open straight ray, with a stable direction tie-break."""

    row, column = position
    best = None
    for order, facing in enumerate(Facing):
        delta_row, delta_column = FACING_DELTAS[facing]
        next_row, next_column = row + delta_row, column + delta_column
        if not _walkable(base_rows, next_row, next_column):
            continue
        length = 0
        while _walkable(base_rows, next_row, next_column):
            length += 1
            next_row += delta_row
            next_column += delta_column
        candidate = (length, -order, facing.value)
        if best is None or candidate > best[0]:
            best = (candidate, facing.value)
    if best is None:
        raise ValueError(f"watch point {position} has no walkable facing direction")
    return best[1]


def _default_point(position: Tuple[int, int], base_rows: Tuple[str, ...]) -> Dict[str, Any]:
    row, column = position
    plant_cells = [
        (r, c)
        for r, line in enumerate(base_rows)
        for c, cell in enumerate(line)
        if cell == "2"
    ]
    near_plant = any(max(abs(row - r), abs(column - c)) <= 1 for r, c in plant_cells)
    on_plant = base_rows[row][column] == "2"
    if on_plant:
        importance, tags, radius = 5, ["post_plant", "retake"], 1
    elif near_plant:
        importance, tags, radius = 4, ["site_entry"], 2
    else:
        importance, tags, radius = 3, ["off_angle"], 2
    return {
        "id": f"watch_r{row:02d}_c{column:02d}",
        "position": [row, column],
        "importance": importance,
        "facing": _facing_for_position(position, base_rows),
        "sides": ["attacker", "defender"],
        "situations": ["carry", "retrieve", "guard", "search", "retake"],
        "random_radius": radius,
        "tags": tags,
    }


def convert_map_to_config(
    marked_map: str,
    base_map: str = BASE_MAP,
    previous_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    base_rows = _rows(base_map, "base map")
    positions = extract_watch_positions(marked_map, base_map)
    previous_by_position = {
        tuple(point["position"]): point
        for point in (previous_config or {}).get("points", [])
        if isinstance(point, Mapping) and isinstance(point.get("position"), list)
    }
    points = []
    for position in positions:
        previous = previous_by_position.get(position)
        points.append(dict(previous) if previous is not None else _default_point(position, base_rows))
    result = {
        "schema_version": WATCH_POINTS_SCHEMA_VERSION,
        "map": {
            "rows": len(base_rows),
            "columns": len(base_rows[0]),
            "sha256": map_sha256(base_map),
        },
        "points": points,
    }
    validate_watch_points(result, base_map)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-map", type=Path, default=DEFAULT_MARKED_MAP_PATH)
    parser.add_argument("--output", type=Path, default=WATCH_POINTS_CONFIG_PATH)
    parser.add_argument("--write", action="store_true", help="write the generated JSON")
    args = parser.parse_args(argv)

    marked_map = load_marked_map(args.input_map)
    previous = load_json_config(args.output) if args.output.is_file() else None
    config = convert_map_to_config(marked_map, BASE_MAP, previous)
    rendered = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    print(f"watch points: {len(config['points'])}")
    print(f"watch_points_sha256: {canonical_json_sha256(config)}")
    if args.write:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote: {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
