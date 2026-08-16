"""Ghost Champions Defender Setup 終了位置候補マップ。

0 = 通常床（候補ではない）
1 = 壁
2 = Setup終了時の配置候補
"""

DEFENDER_SETUP_POSITION_MAP_GC_STR = """
11111111111111111111111111111111111111111111
11111111111111111100000111111111111111111111
11111111111111111000000000000000000200000001
11111111111111111001111110111110111111000111
11111100000000000011111110111110000111000001
11111101111101111011111110111111000111000001
10001121011121111200000000000000000111000001
10000001010001110011100000000002000000000001
10000001010000110111100011111110001111000001
11100001000000000001120211111000001111000001
10001101000001110001100011111000001111000001
10000000000001110011100011111100011111110111
10000000111001110001100001100000111111000111
10001100111111110001111000001100111111110111
11101111111111110000000000111110000000000111
11100001111100000001100000111110101111110111
11100000001100110111100000000110111111110111
11101111001100111110000000000000111111110111
11101111001100111110000000000000111111110111
11101111001100111110000000000000111111110111
11100000000000000000000000000000000000000111
11111111111111111111111111111111111111111111
""".strip()


def _rows(text):
    rows = [
        row.strip()
        for row in str(text).strip().splitlines()
        if row.strip()
    ]

    if not rows:
        raise ValueError("GC Defender Setup position map is empty")

    width = len(rows[0])

    if any(len(row) != width for row in rows):
        raise ValueError(
            "GC Defender Setup position map rows have inconsistent widths"
        )

    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            if value not in {"0", "1", "2"}:
                raise ValueError(
                    "GC Defender Setup position map accepts only 0/1/2: "
                    f"row={r} col={c} value={value!r}"
                )

    return rows


def get_gc_setup_position_map():
    return [
        [int(value) for value in row]
        for row in _rows(DEFENDER_SETUP_POSITION_MAP_GC_STR)
    ]


def get_gc_setup_position_candidates():
    rows = _rows(DEFENDER_SETUP_POSITION_MAP_GC_STR)
    return [
        (r, c)
        for r, row in enumerate(rows)
        for c, value in enumerate(row)
        if value == "2"
    ]


def is_gc_setup_position_candidate(row, col):
    rows = _rows(DEFENDER_SETUP_POSITION_MAP_GC_STR)
    row = int(row)
    col = int(col)

    if not (
        0 <= row < len(rows)
        and 0 <= col < len(rows[0])
    ):
        return False

    return rows[row][col] == "2"


def validate_against_map(base_maze_str):
    setup_rows = _rows(DEFENDER_SETUP_POSITION_MAP_GC_STR)
    base_rows = [
        row.strip()
        for row in str(base_maze_str).strip().splitlines()
        if row.strip()
    ]

    errors = []

    if not base_rows:
        return ["base map is empty"]

    if len(setup_rows) != len(base_rows):
        errors.append(
            f"height mismatch: setup={len(setup_rows)} "
            f"base={len(base_rows)}"
        )
        return errors

    if len(setup_rows[0]) != len(base_rows[0]):
        errors.append(
            f"width mismatch: setup={len(setup_rows[0])} "
            f"base={len(base_rows[0])}"
        )
        return errors

    for r in range(len(setup_rows)):
        for c in range(len(setup_rows[0])):
            if setup_rows[r][c] == "2" and base_rows[r][c] == "1":
                errors.append(
                    f"candidate is on wall: row={r} col={c}"
                )

    return errors


if __name__ == "__main__":
    rows = _rows(DEFENDER_SETUP_POSITION_MAP_GC_STR)
    candidates = get_gc_setup_position_candidates()
    print(
        f"GC Defender Setup Position Map: "
        f"{len(rows)}x{len(rows[0])}"
    )
    print(f"candidates={len(candidates)}")
    print(f"candidate cells={candidates}")
    print("0=normal / 1=wall / 2=setup position candidate")
    print()
    print(DEFENDER_SETUP_POSITION_MAP_GC_STR)
