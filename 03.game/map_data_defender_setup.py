"""Defender Setup Phase 用の進入許可マップ。

このファイルで編集するのは DEFENDER_SETUP_MASK_STR だけです。

数字の意味
----------
0 = Defender が Setup Phase 中に進入できる
1 = Defender が Setup Phase 中に進入できない

指定方法はこの0/1グリッドだけです。
 や  などの
別指定は使用しません。

通常ラウンド開始後は、このSetup制限を無効にして
通常の map_data.py の移動判定へ戻します。
"""

DEFENDER_SETUP_TICKS = 20

# 0 = Setup中に進入可能
# 1 = Setup中に進入禁止
DEFENDER_SETUP_MASK_STR = """
11111111111111111111111111111111111111111111
11111111111111111100000111111111111111111111
11111111111111111000000000000000000010000001
11111111111111110001111110111110111111000111
11111100000000000011111110111110000111000001
11111101111101111011111110111111000111000101
10001101011101111000000000000000000111000101
10000011010011110111100000000000000000000101
10000101010000110111100011111111111111000101
11101101000000000001100011111000001111000101
10001101000001110001111111111000001111000001
10000000000001110011100011111100011111110111
10000000111001110001100001100000111111000111
10001100111111110001111000001100111111110111
11101111111111110000000000111110000000000111
11100001111100000001100000111110101111110111
11100000001100110111100000000110111111110111
11101111001100111110000111110110111100000111
11111111000000111110000111110000000001110001
11111111001111111110000111111111111111110001
11111111001111111110000111111111111111110001
11111111100000000000000111111111011111110111
11111111100000000000000000000000000000000111
11111111111111111000000111111111111111111111
11111111111111111111111111111111111111111111
11111111111111111111111111111111111111111111
""".strip()

# Setup Phase中の進入禁止床を薄い黄色で表示するための設定。
SETUP_FORBIDDEN_OVERLAY_COLOR = "#F6E7A1"
SETUP_FORBIDDEN_OVERLAY_STIPPLE = "gray50"
SETUP_FORBIDDEN_OUTLINE_COLOR = "#E6CF69"


def _raw_rows(text):
    """任意の数字マップを行単位で読む。値の種類は制限しない。"""
    rows = [row.strip() for row in str(text).strip().splitlines() if row.strip()]

    if not rows:
        raise ValueError("map is empty")

    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("map rows have inconsistent widths")

    return rows


def _rows(text):
    """Setupマップ専用。0/1だけを許可する。"""
    rows = _raw_rows(text)

    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            if value not in {"0", "1"}:
                raise ValueError(
                    "Defender Setup map accepts only 0/1: "
                    f"row={r} col={c} value={value!r}"
                )

    return rows


def get_setup_mask():
    """Setupマップを list[list[int]] で返す。"""
    return [[int(value) for value in row] for row in _rows(DEFENDER_SETUP_MASK_STR)]


def is_setup_position_allowed(row, col):
    """Setup Phase中にそのマスへ進入できるか。"""
    rows = _rows(DEFENDER_SETUP_MASK_STR)

    row = int(row)
    col = int(col)

    if not (0 <= row < len(rows) and 0 <= col < len(rows[0])):
        return False

    return rows[row][col] == "0"


def validate_against_map(base_maze_str):
    """通常マップとSetupマップの縦横サイズが一致するか確認する。"""
    setup_rows = _rows(DEFENDER_SETUP_MASK_STR)
    base_rows = _raw_rows(base_maze_str)

    errors = []

    if len(setup_rows) != len(base_rows):
        errors.append(
            f"height mismatch: setup={len(setup_rows)} " f"base={len(base_rows)}"
        )
        return errors

    if len(setup_rows[0]) != len(base_rows[0]):
        errors.append(
            f"width mismatch: setup={len(setup_rows[0])} " f"base={len(base_rows[0])}"
        )

    return errors


def get_setup_forbidden_floor_cells(base_maze_str):
    """黄色表示すべきSetup専用進入禁止床を(row, col)で返す。

    Setupマップでは1だが、通常マップでは壁(1)ではないセルだけを返す。
    そのため通常の壁まで黄色く塗られない。
    """
    setup_rows = _rows(DEFENDER_SETUP_MASK_STR)
    base_rows = _raw_rows(base_maze_str)

    if len(setup_rows) != len(base_rows) or len(setup_rows[0]) != len(base_rows[0]):
        raise ValueError("Setup map size does not match base map")

    return [
        (r, c)
        for r in range(len(setup_rows))
        for c in range(len(setup_rows[0]))
        if setup_rows[r][c] == "1" and base_rows[r][c] != "1"
    ]


if __name__ == "__main__":
    rows = _rows(DEFENDER_SETUP_MASK_STR)
    print(
        f"Defender Setup Map: {len(rows)}x{len(rows[0])} "
        f"/ ticks={DEFENDER_SETUP_TICKS}"
    )
    print("0=allowed / 1=forbidden")
    print(DEFENDER_SETUP_MASK_STR)
