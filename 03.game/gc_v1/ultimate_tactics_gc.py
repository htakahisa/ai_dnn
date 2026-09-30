"""Observable context and target construction for GC learned ultimate actions."""

import numpy as np


ULTIMATE_CONTEXT_DIM = 4
ORB_CONTEXT_DIM = 4
ATTACKER_ORB_APPROACH_RADIUS = 8
ESCAPE_MIN_TRAVEL_CELLS = 6

FACING_STEPS = {
    "N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1),
    "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1),
}


def ultimate_ready(char):
    return (getattr(char, "ultimate_cost", 0) > 0
            and getattr(char, "ultimate_points", 0) >= char.ultimate_cost
            and (not getattr(char, "is_alive", True) if getattr(char, "ultimate_name", None) == "SERENADE"
                 else getattr(char, "is_alive", True)))


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


def orb_context_features(char, available_orbs, collect_required_ticks=5):
    """Observable orb state: on-orb, progress, and direction to nearest orb."""
    orbs = {tuple(map(int, cell)) for cell in (available_orbs or ())}
    pos = tuple(map(int, char.pos))
    on_orb = pos in orbs
    progress = min(
        1.0,
        max(0, int(getattr(char, "orb_collect_timer", 0)))
        / max(1, int(collect_required_ticks)),
    )
    dr = dc = 0.0
    if orbs:
        nearest = min(
            orbs,
            key=lambda cell: (
                abs(cell[0] - pos[0]) + abs(cell[1] - pos[1]),
                cell,
            ),
        )
        dr = float(np.sign(nearest[0] - pos[0]))
        dc = float(np.sign(nearest[1] - pos[1]))
    return np.asarray((float(on_orb), progress, dr, dc), dtype=np.float32)


def attacker_orb_context_features(char, available_orbs, collect_required_ticks=5):
    """Add nearby-orb distance without changing Defender/Guard observations."""
    base = orb_context_features(char, available_orbs, collect_required_ticks)
    pos = tuple(map(int, char.pos))
    distances = [
        abs(int(cell[0]) - pos[0]) + abs(int(cell[1]) - pos[1])
        for cell in (available_orbs or ())
    ]
    proximity = (
        max(0.0, 1.0 - min(distances) / ATTACKER_ORB_APPROACH_RADIUS)
        if distances else 0.0
    )
    return np.append(base, np.float32(proximity))


def can_collect_orb(char, available_orbs):
    return (
        tuple(map(int, char.pos)) in {tuple(map(int, cell)) for cell in (available_orbs or ())}
        and getattr(char, "ultimate_cost", 0) > 0
        and getattr(char, "ultimate_points", 0) < getattr(char, "ultimate_cost", 0)
    )


def tactical_ultimate_window(char, context):
    """Whether an observable context is a suitable training label for the ult."""
    if context is None or len(context) < ULTIMATE_CONTEXT_DIM or context[0] <= 0:
        return False
    engaged, objective, urgent = (bool(context[1]), bool(context[2]), bool(context[3]))
    name = str(getattr(char, "ultimate_name", "")).upper()
    if name == "SERENADE":
        return True
    if name == "BALEMOON":
        return engaged or objective or urgent
    if name == "RAID":
        return engaged or objective
    if name == "ESCAPE":
        return objective or (engaged and urgent)
    if name == "MONITOR":
        return objective and not engaged
    if name in {"TUNNEL", "NEON"}:
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
    char_pos = tuple(map(int, char.pos))
    # ESCAPE is a global teleport with a long wind-up.  Treat a destination
    # near the caster as an objective anchor and select the closest meaningful
    # landing around it instead of spending the ultimate for a one-cell warp.
    # A far, legal requested destination is still selected exactly.
    for radius in range(max(grid.shape)):
        candidates = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                cell = (destination[0] + dr, destination[1] + dc)
                if (0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]
                        and grid[cell] != 1 and cell not in occupied
                        and max(abs(cell[0] - char_pos[0]),
                                abs(cell[1] - char_pos[1]))
                        >= ESCAPE_MIN_TRAVEL_CELLS):
                    travel = max(
                        abs(cell[0] - char_pos[0]),
                        abs(cell[1] - char_pos[1]),
                    )
                    candidates.append((-travel, abs(dr) + abs(dc), cell))
        if candidates:
            return min(candidates)[2]
    return None


def build_ultimate_action(grid, char, chars, destination=None):
    """Return an executable payload; the policy still decides whether to cast."""
    if not ultimate_ready(char):
        return None
    name = str(getattr(char, "ultimate_name", "")).upper()
    if name in {"SERENADE", "BALEMOON"}:
        return {"ultimate": name}
    if name == "NEON":
        target = destination
        if target is None:
            enemies = [enemy for enemy in chars
                       if getattr(enemy, "is_alive", True) and enemy.team != char.team]
            if not enemies:
                return None
            target = max(enemies, key=lambda enemy: sum(
                max(abs(enemy.pos[0] - other.pos[0]), abs(enemy.pos[1] - other.pos[1])) <= 3
                for other in enemies
            )).pos
        target = tuple(map(int, target))
        if not (0 <= target[0] < grid.shape[0] and 0 <= target[1] < grid.shape[1]) or grid[target] == 1:
            return None
        return {"ultimate": name, "target": target}
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
