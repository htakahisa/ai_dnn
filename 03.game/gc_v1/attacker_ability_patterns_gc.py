"""Parser for attacker ability locations defined in map-data form."""

try:
    from .map_data_attacker_ability_gc import (
        ABILITY_PATTERN_LAYERS,
        ATTACKER_PATTERN_IDS,
    )
except ImportError:
    from map_data_attacker_ability_gc import (
        ABILITY_PATTERN_LAYERS,
        ATTACKER_PATTERN_IDS,
    )


def _rows(maze_str):
    rows = [row.strip() for row in str(maze_str).strip().splitlines() if row.strip()]
    if not rows:
        raise ValueError("Attacker ability map is empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Attacker ability map rows have inconsistent widths")
    return rows


def _collect_pattern_cells(maze_str):
    result = {pid: [] for pid in ATTACKER_PATTERN_IDS}
    if isinstance(maze_str, dict):
        for pid, cells in maze_str.items():
            if int(pid) in result:
                result[int(pid)].extend(
                    (int(pos[0]), int(pos[1])) for pos in cells
                )
        return {pid: cells for pid, cells in result.items() if cells}
    for row_idx, row in enumerate(_rows(maze_str)):
        for col_idx, cell in enumerate(row):
            if cell.isdigit() and int(cell) in result:
                result[int(cell)].append((row_idx, col_idx))
    return {pid: cells for pid, cells in result.items() if cells}


def load_attacker_ability_patterns():
    patterns = {}
    for ability, layers in ABILITY_PATTERN_LAYERS.items():
        layer_cells = {
            name: _collect_pattern_cells(maze_str)
            for name, maze_str in layers.items()
        }
        patterns[ability] = {
            pid: {"id": pid, "ability": ability, "targets": targets}
            for pid in ATTACKER_PATTERN_IDS
            if (targets := layer_cells.get("target", {}).get(pid, []))
        }
    return patterns


ATTACKER_ABILITY_PATTERNS = load_attacker_ability_patterns()


def get_available_pattern_ids(ability):
    return tuple(sorted(ATTACKER_ABILITY_PATTERNS.get(str(ability).upper(), {})))


def get_pattern_targets(ability):
    result = []
    for pattern in ATTACKER_ABILITY_PATTERNS.get(str(ability).upper(), {}).values():
        result.extend(pattern.get("targets", ()))
    return tuple(result)


def get_pattern_target(
    ability, source, grid, *, anchor=None, max_range=None, blocked_cells=()
):
    """Return the best configured target that can currently be thrown to.

    The map data contains the intended landing cells; the controller still has
    to wait until the cell is reachable by a clear throw line.  ``anchor`` is
    normally the active site or the last known enemy position and keeps a
    lineup on the relevant side of the map.
    """
    source = (int(source[0]), int(source[1]))
    anchor = tuple(map(int, anchor)) if anchor is not None else None
    candidates = []
    for target in get_pattern_targets(ability):
        target = (int(target[0]), int(target[1]))
        r, c = target
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            continue
        if int(grid[r, c]) == 1:
            continue
        distance = max(abs(target[0] - source[0]), abs(target[1] - source[1]))
        if max_range is not None and distance > int(max_range):
            continue

        # Local import avoids a circular import: tactical_ability owns the
        # Bresenham/LOS helper, while this module owns map-data parsing.
        try:
            from .tactical_ability import has_los
        except ImportError:
            from tactical_ability import has_los
        if not has_los(grid, source, target, blocked_cells):
            continue

        anchor_distance = (
            max(abs(target[0] - anchor[0]), abs(target[1] - anchor[1]))
            if anchor is not None else 0
        )
        candidates.append((anchor_distance, distance, target))

    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1], item[2]))[2]
