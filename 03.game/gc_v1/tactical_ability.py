"""共有する攻撃アビリティの戦術判定。"""


try:
    from .attacker_ability_patterns_gc import get_pattern_target, get_pattern_targets
except ImportError:
    try:
        from attacker_ability_patterns_gc import get_pattern_target, get_pattern_targets
    except ImportError:
        def get_pattern_targets(_ability):
            return ()
        def get_pattern_target(*_args, **_kwargs):
            return None


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


def _custom_smoke_target(source, threat, grid):
    """Choose a configured smoke point that is near the current threat line."""
    if threat is None:
        return None
    candidates = [
        tuple(map(int, pos))
        for pos in get_pattern_targets("SMOKE")
        if _walkable(grid, pos)
    ]
    if not candidates:
        return None

    threat_line = line_cells(source, threat)

    def score(pos):
        line_distance = min(
            max(abs(pos[0] - cell[0]), abs(pos[1] - cell[1]))
            for cell in threat_line
        )
        source_distance = max(
            abs(pos[0] - source[0]), abs(pos[1] - source[1])
        )
        return line_distance, source_distance

    target = min(candidates, key=score)
    # Do not use a pattern from an unrelated part of the map.
    return target if score(target)[0] <= 3 else None


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
        target = _custom_smoke_target(char.pos, threat_pos, grid)
        if target is None and threat_pos:
            target = smoke_target(char.pos, threat_pos, grid)
        if target is not None and _walkable(grid, target):
            return ability, tuple(map(int, target))
    elif ability == "FLASH":
        # Touyama-style lineup: use the configured landing cell as soon as
        # the throw line opens, even before an enemy is directly visible.
        target = get_pattern_target(
            ability, char.pos, grid, anchor=threat_pos or destination,
            max_range=max(15, max_range), blocked_cells=smoke_cells,
        )
        if target is not None:
            return ability, target
        if threat_pos is not None:
            dist = max(abs(threat_pos[0] - char.pos[0]), abs(threat_pos[1] - char.pos[1]))
            if dist <= max_range and _walkable(grid, threat_pos):
                return ability, threat_pos
    elif ability == "RECON":
        # A carrier's position is passed as ``destination`` by the escort
        # controller.  Using it as a blind fallback makes RECON fire at the
        # round-start spawn/feet when no enemy has been found yet.  Recon is
        # information-gathering, so require an actual threat/last-known
        # location; the learned policy can still choose when to use it.
        # Recon must not fire at the round-start spawn/feet.  When a threat
        # or last-known position exists, prefer a configured map cell near it
        # if its throw line is open; otherwise retain the known enemy target.
        target = get_pattern_target(
            ability, char.pos, grid, anchor=threat_pos or destination,
            max_range=max_range, blocked_cells=smoke_cells,
        ) if threat_pos is not None else None
        if target is not None:
            return ability, target
        if threat_pos is not None and _walkable(grid, threat_pos):
            return ability, tuple(map(int, threat_pos))
    return None
