"""共有する攻撃アビリティの戦術判定。"""


def line_cells(start, end):
    r0, c0 = int(start[0]), int(start[1])
    r1, c1 = int(end[0]), int(end[1])
    dr, dc = abs(r1 - r0), -abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr + dc
    cells = []
    while True:
        cells.append((r0, c0))
        if (r0, c0) == (r1, c1):
            return cells
        e2 = 2 * err
        if e2 >= dc:
            err += dc
            r0 += sr
        if e2 <= dr:
            err += dr
            c0 += sc


def has_los(grid, start, end, smoke_cells=()):
    return all(grid[r, c] != 1 and (r, c) not in smoke_cells
               for r, c in line_cells(start, end))


def nearest_visible(char, chars, grid, smoke_cells=()):
    enemies = [
        other for other in chars
        if getattr(other, "is_alive", True)
        and getattr(other, "team", None) != getattr(char, "team", None)
        and has_los(grid, char.pos, other.pos, smoke_cells)
    ]
    return sorted(
        enemies,
        key=lambda other: max(abs(other.pos[0] - char.pos[0]),
                              abs(other.pos[1] - char.pos[1])),
    )


def _walkable(grid, pos):
    r, c = int(pos[0]), int(pos[1])
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] != 1


def smoke_target(source, threat, grid):
    cells = [cell for cell in line_cells(source, threat) if _walkable(grid, cell)]
    if not cells:
        return None
    usable = cells[1:] or cells
    return tuple(map(int, usable[max(0, (len(usable) - 1) // 2)]))


def choose_pre_entry_ability(char, chars, grid, smoke_cells=(), destination=None,
                             last_seen=None, max_range=8):
    """接敵前に使うべき能力と標的を返す。不要ならNone。"""
    ability = str(getattr(char, "ability_name", "")).upper()
    charges = {"SMOKE": getattr(char, "smoke_charges", 0),
               "FLASH": getattr(char, "flash_charges", 0),
               "RECON": getattr(char, "recon_charges", 0)}.get(ability, 0)
    if charges <= 0:
        return None

    visible = nearest_visible(char, chars, grid, smoke_cells)
    threat_pos = tuple(map(int, visible[0].pos)) if visible else None
    if threat_pos is None and last_seen is not None:
        threat_pos = tuple(map(int, last_seen[:2]))

    if ability == "SMOKE":
        # 敵位置がない状態で味方位置/サイトへ投げるのは、射線遮断にならないため避ける。
        target = smoke_target(char.pos, threat_pos, grid) if threat_pos else None
        if target is not None and _walkable(grid, target):
            return ability, tuple(map(int, target))
    elif ability == "FLASH":
        if threat_pos is not None:
            dist = max(abs(threat_pos[0] - char.pos[0]), abs(threat_pos[1] - char.pos[1]))
            if dist <= max_range and _walkable(grid, threat_pos):
                return ability, threat_pos
    elif ability == "RECON":
        target = threat_pos or destination
        if target is not None and _walkable(grid, target):
            return ability, tuple(map(int, target))
    return None
