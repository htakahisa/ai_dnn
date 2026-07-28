"""学習AI用のアビリティ判断・衝突回避ヘルパー。"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Iterable, Optional, Sequence, Tuple

MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _pos(value) -> Tuple[int, int]:
    return int(value[0]), int(value[1])


def occupied_cells(chars, exclude=None):
    return {
        _pos(ch.pos)
        for ch in chars
        if getattr(ch, "is_alive", True) and ch is not exclude
    }


def _walkable(cell, grid, blocked=frozenset()):
    r, c = cell
    return (
        0 <= r < grid.shape[0]
        and 0 <= c < grid.shape[1]
        and grid[r, c] != 1
        and cell not in blocked
    )


def _bfs_distance(start, target, grid, blocked=frozenset()):
    start, target = _pos(start), _pos(target)
    if start == target:
        return 0
    q = deque([(start, 0)])
    seen = {start}
    while q:
        (r, c), dist = q.popleft()
        for dr, dc in MOVES:
            nxt = (r + dr, c + dc)
            if nxt in seen or not _walkable(nxt, grid, blocked):
                continue
            if nxt == target:
                return dist + 1
            seen.add(nxt)
            q.append((nxt, dist + 1))
    return math.inf


def collision_safe_step(char, desired, target, grid, chars):
    """希望マスが埋まっていたら、目的地へ近づく空き隣接マスへ迂回する。

    他キャラの現在地は一時的な壁として扱う。完全停止が続く場合は、
    目的地への距離・周囲の空き・直前位置からの変化を使って横へ逃がす。
    """
    current = _pos(char.pos)
    desired = _pos(desired)
    target = _pos(target) if target is not None else desired
    blocked = occupied_cells(chars, exclude=char)

    if _walkable(desired, grid, blocked):
        return [desired[0], desired[1]]

    candidates = []
    for dr, dc in MOVES:
        nxt = (current[0] + dr, current[1] + dc)
        if not _walkable(nxt, grid, blocked):
            continue
        distance = _bfs_distance(nxt, target, grid, blocked)
        free_neighbors = sum(
            _walkable((nxt[0] + rr, nxt[1] + cc), grid, blocked)
            for rr, cc in MOVES
        )
        # 距離を最優先し、袋小路を避け、同点はランダム化。
        candidates.append((distance, -free_neighbors, random.random(), nxt))

    if not candidates:
        return [current[0], current[1]]

    candidates.sort(key=lambda item: item[:3])
    chosen = candidates[0][3]
    return [chosen[0], chosen[1]]


def line_of_sight(start, end, grid):
    y0, x0 = _pos(start)
    y1, x1 = _pos(end)
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        if grid[y0, x0] == 1:
            return False
        if (y0, x0) == (y1, x1):
            return True
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _nearest_enemy(char, chars, grid, require_los=False):
    enemies = [
        other for other in chars
        if getattr(other, "is_alive", False) and other.team != char.team
    ]
    if require_los:
        enemies = [
            other for other in enemies
            if line_of_sight(char.pos, other.pos, grid)
        ]
    if not enemies:
        return None
    return min(
        enemies,
        key=lambda other: max(
            abs(other.pos[0] - char.pos[0]),
            abs(other.pos[1] - char.pos[1]),
        ),
    )


def decide_ability(char, game_state):
    """既存DQNを作り直さずに使える、ロール別の保守的なアビリティ判断。

    戻り値:
        None
        または {"ability": "SMOKE"|"FLASH"|"RECON", "target": (r, c)}
    """
    grid = game_state["grid"]
    chars = game_state["chars"]
    ability = getattr(char, "ability_name", "HUNT")

    if ability == "HUNT" or not getattr(char, "is_alive", True):
        return None
    if getattr(char, "blind_remaining", 0) > 0:
        return None

    charge_attr = {
        "SMOKE": "smoke_charges",
        "FLASH": "flash_charges",
        "RECON": "recon_charges",
    }.get(ability)
    if not charge_attr or getattr(char, charge_attr, 0) <= 0:
        return None

    nearest_visible = _nearest_enemy(char, chars, grid, require_los=True)
    nearest_any = _nearest_enemy(char, chars, grid, require_los=False)
    is_planted = bool(game_state.get("is_planted"))
    objective = game_state.get("planted_pos") if is_planted else game_state.get("target_plant_pos")

    # 同一ラウンドで毎Tick判定しても、低確率かつ有効局面だけで使用する。
    if ability == "FLASH":
        if nearest_visible is None:
            return None
        distance = max(
            abs(nearest_visible.pos[0] - char.pos[0]),
            abs(nearest_visible.pos[1] - char.pos[1]),
        )
        if 2 <= distance <= 12 and random.random() < 0.55:
            return {"ability": "FLASH", "target": _pos(nearest_visible.pos)}

    elif ability == "RECON":
        target = _pos(nearest_any.pos) if nearest_any is not None else (
            _pos(objective) if objective is not None else None
        )
        if target is not None:
            distance = max(abs(target[0] - char.pos[0]), abs(target[1] - char.pos[1]))
            # 敵をまだ直視できていない時、またはリテイク時に優先。
            if (nearest_visible is None or is_planted) and distance >= 3 and random.random() < 0.35:
                return {"ability": "RECON", "target": target}

    elif ability == "SMOKE":
        target = None
        if nearest_visible is not None:
            # 敵と自分の中間付近に置き、射線を切る。
            target = (
                int(round((char.pos[0] + nearest_visible.pos[0]) / 2)),
                int(round((char.pos[1] + nearest_visible.pos[1]) / 2)),
            )
        elif objective is not None and is_planted:
            target = _pos(objective)

        if target is not None and grid[target[0], target[1]] != 1 and random.random() < 0.40:
            return {"ability": "SMOKE", "target": target}

    return None
