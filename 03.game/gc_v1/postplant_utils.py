"""ポストプラントの監視対象セル。設置マスと周囲8マスを含む。"""


def postplant_watch_cells(grid, planted_pos):
    r0, c0 = map(int, planted_pos)
    cells = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] != 1:
                cells.append((r, c))
    return tuple(cells)

