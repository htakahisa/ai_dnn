"""Observable context and target construction for GC learned ultimate actions."""

import numpy as np


ULTIMATE_CONTEXT_DIM = 4

FACING_STEPS = {
    "N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1),
    "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1),
}


def ultimate_ready(char):
    return (getattr(char, "ultimate_cost", 0) > 0
            and getattr(char, "ultimate_points", 0) >= char.ultimate_cost)


def ultimate_context_features(char, engaged=False, objective_window=False, urgent=False):
    """Return learned cast context without exposing hidden opponent state.

    Context order is ready, direct combat, objective timing, and urgency.  The
    action mask still owns executability; these values let one shared output
    distinguish a useful cast from merely having enough points.
    """
    return np.asarray((
        float(ultimate_ready(char)),
        float(bool(engaged)),
        float(bool(objective_window)),
        float(bool(urgent)),
    ), dtype=np.float32)


def tactical_ultimate_window(char, context):
    """Whether an observable context is a suitable training label for the ult."""
    if context is None or len(context) < ULTIMATE_CONTEXT_DIM or context[0] <= 0:
        return False
    engaged, objective, urgent = (bool(context[1]), bool(context[2]), bool(context[3]))
    name = str(getattr(char, "ultimate_name", "")).upper()
    if name == "RAID":
        return engaged or objective
    if name == "ESCAPE":
        return objective or (engaged and urgent)
    if name == "MONITOR":
        return objective and not engaged
    if name == "TUNNEL":
        return engaged or objective
    return False


def _occupied(chars, owner):
    return {tuple(map(int, other.pos)) for other in chars
            if other is not owner and getattr(other, "is_alive", True)}


def _escape_target(grid, char, chars, destination):
    if destination is None:
        return None
    destination = tuple(map(int, destination))
    occupied = _occupied(chars, char)
    candidates = []
    for radius in range(4):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                cell = (destination[0] + dr, destination[1] + dc)
                if (0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]
                        and grid[cell] != 1 and cell not in occupied
                        and cell != tuple(map(int, char.pos))):
                    candidates.append((abs(dr) + abs(dc), cell))
        if candidates:
            return min(candidates)[1]
    return None


def build_ultimate_action(grid, char, chars, destination=None):
    """Return an executable payload; the policy still decides whether to cast."""
    if not ultimate_ready(char):
        return None
    name = str(getattr(char, "ultimate_name", "")).upper()
    if name == "ESCAPE":
        target = _escape_target(grid, char, chars, destination)
        return None if target is None else {"ultimate": name, "target": target}
    if name == "RAID":
        step = FACING_STEPS.get(getattr(char, "facing", None))
        if step is None:
            return None
        pos = tuple(map(int, char.pos))
        nxt = (pos[0] + step[0], pos[1] + step[1])
        occupied = _occupied(chars, char)
        if (not (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1])
                or grid[nxt] == 1 or nxt in occupied):
            return None
        return {"ultimate": name}
    if name in {"MONITOR", "TUNNEL"}:
        return {"ultimate": name}
    return None
