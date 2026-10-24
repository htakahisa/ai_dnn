"""GC v3 ディフェンダー初期配置候補マップ。この文字列を編集する。

0=道 / 1=壁 / 2・3・4=表示用（AIは道として扱う）
5=配置候補 / 6=警戒位置候補
候補は柔らかい指標。候補点への完全一致や固定の組み合わせを強制しない。
ポストプラントよりランダム性を残してサンプリングする。学習はここでは起動しない。
"""

DEFENDER_SETUP_POSITION_MAP_GC_STR = """
11111111111111111111111111111111111111111111
11111111111111111144444111111111111111111111
11111111111111111000000000000000000065000551
11111111111111110001111116111116111111000111
11111100000000000011111116111116000111005521
11111161111161111611111115111111000111505121
15051161011161111555555555555550000111005121
10022501010001115511155555555550000065002121
15022101010000115111100011111110001111002121
11121101000000665001100011111000001111002121
15061151000001110001100011111000001111552251
10000066000001110011100011111100011111110111
10000000111001110001100001155500111111550111
15561155111111110001111005551155111111110111
11151111111111116665555005111115555000060111
11155551111100000001100000111110151111110111
11155555551100110111166660000110111111116111
11151111551155111110000111110110111155556111
11111111556555111110000111110000000051110551
11111111551111111110000111111111111111110551
11111111551111111110000111111111111111116551
11111111156000000000000111111111011111115111
11111111156000000066666000000000000000005111
11111111111111111033333111111111111111111111
11111111111111111111111111111111111111111111
11111111111111111111111111111111111111111111
""".strip()


def _rows():
    rows = [
        row.strip()
        for row in DEFENDER_SETUP_POSITION_MAP_GC_STR.strip().splitlines()
        if row.strip()
    ]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("GC v3 candidate map must be a nonempty rectangle")
    if any(ch not in "0123456" for row in rows for ch in row):
        raise ValueError("GC v3 candidate map accepts only 0/1/2/3/4/5/6")
    return rows


def _candidates(marker):
    return [
        (r, c)
        for r, row in enumerate(_rows())
        for c, ch in enumerate(row)
        if ch == marker
    ]


def get_gc_setup_position_map():
    return [[int(ch) for ch in row] for row in _rows()]


def get_gc_setup_position_candidates():
    return _candidates("5")


def get_gc_setup_watch_candidates():
    return _candidates("6")


def get_gc_setup_watch_map():
    return get_gc_setup_position_map()


def is_gc_setup_position_candidate(row, col):
    return (int(row), int(col)) in get_gc_setup_position_candidates()


def validate_against_map(base_maze_str):
    base = [row.strip() for row in base_maze_str.strip().splitlines() if row.strip()]
    rows = _rows()
    if len(rows) != len(base) or any(len(a) != len(b) for a, b in zip(rows, base)):
        return ["Defender candidate map dimensions differ from arena"]
    return [
        f"wall mismatch: row={r} col={c}"
        for r, (a, b) in enumerate(zip(rows, base))
        for c, (x, y) in enumerate(zip(a, b))
        if (x == "1") != (y == "1")
    ]
