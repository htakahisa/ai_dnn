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
