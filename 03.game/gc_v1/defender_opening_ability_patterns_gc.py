"""Ghost Champions Defender Opening Ability pattern parser.

map_data_defender_opening_ability_gc.py の各レイヤーを読み取り、
Pattern IDごとの座標を構造化する。
座標形式はゲーム内グリッドに合わせて (row, col)。
"""

from map_data_defender_opening_ability_gc import (
    OPENING_PATTERN_IDS,
    ABILITY_PATTERN_LAYERS,
)


def _rows(maze_str):
    rows = [row.strip() for row in maze_str.strip().splitlines() if row.strip()]
    if not rows:
        raise ValueError("Opening ability map is empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Opening ability map rows have inconsistent widths")
    return rows


def _collect_pattern_cells(maze_str):
    """Return {pattern_id: [(row, col), ...]} for IDs 2-9."""
    result = {pid: [] for pid in OPENING_PATTERN_IDS}
    for row_idx, row in enumerate(_rows(maze_str)):
        for col_idx, cell in enumerate(row):
            if cell.isdigit():
                value = int(cell)
                if value in result:
                    # ゲーム本体の char.pos / grid と同じ (row, col)
                    result[value].append((row_idx, col_idx))
    return {pid: cells for pid, cells in result.items() if cells}


def load_opening_ability_patterns():
    patterns = {}

    for ability, layers in ABILITY_PATTERN_LAYERS.items():
        layer_cells = {
            layer_name: _collect_pattern_cells(maze_str)
            for layer_name, maze_str in layers.items()
        }

        ability_patterns = {}
        for pid in OPENING_PATTERN_IDS:
            entry = {"id": pid, "ability": ability}

            if "origin" in layer_cells:
                origins = layer_cells["origin"].get(pid, [])
                if origins:
                    entry["origins"] = origins

            if "target" in layer_cells:
                targets = layer_cells["target"].get(pid, [])
                if targets:
                    entry["targets"] = targets

            # SmokeはTargetのみで成立。
            # Flash/ReconはOriginとTargetの両方が揃ったPatternだけ有効。
            if ability == "SMOKE":
                valid = bool(entry.get("targets"))
            else:
                valid = bool(entry.get("origins")) and bool(entry.get("targets"))

            if valid:
                ability_patterns[pid] = entry

        patterns[ability] = ability_patterns

    return patterns


OPENING_ABILITY_PATTERNS = load_opening_ability_patterns()


def get_pattern(ability, pattern_id):
    return OPENING_ABILITY_PATTERNS.get(str(ability).upper(), {}).get(int(pattern_id))


def get_available_pattern_ids(ability):
    return tuple(sorted(OPENING_ABILITY_PATTERNS.get(str(ability).upper(), {})))


def validate_opening_ability_patterns():
    """Return human-readable warnings for incomplete pattern definitions."""
    warnings = []

    for ability, layers in ABILITY_PATTERN_LAYERS.items():
        cells = {
            name: _collect_pattern_cells(maze)
            for name, maze in layers.items()
        }
        for pid in OPENING_PATTERN_IDS:
            if ability == "SMOKE":
                continue
            has_origin = bool(cells.get("origin", {}).get(pid))
            has_target = bool(cells.get("target", {}).get(pid))
            if has_origin != has_target:
                missing = "target" if has_origin else "origin"
                warnings.append(
                    f"{ability} Pattern {pid}: {missing} is missing"
                )

    return warnings


if __name__ == "__main__":
    from pprint import pprint

    pprint(OPENING_ABILITY_PATTERNS)
    warnings = validate_opening_ability_patterns()
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(" -", warning)
    else:
        print("\nAll opening ability patterns are valid.")
