"""Public-information post-plant positioning and smoke-defuse response."""

from collections import deque
from dataclasses import replace

from frc_v1.actions import FrcAction, KINDS, MOVE_STEPS
from frc_v1.baseline import plant_sites, route_step
from frc_v1.navigation import _face_team


def _chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _distances(grid, origin):
    distances = {origin: 0}
    queue = deque((origin,))
    while queue:
        r, c = queue.popleft()
        for dr, dc in MOVE_STEPS.values():
            pos = r + dr, c + dc
            if (0 <= pos[0] < len(grid) and 0 <= pos[1] < len(grid[0]) and
                    grid[pos[0]][pos[1]] != 1 and pos not in distances):
                distances[pos] = distances[r, c] + 1
                queue.append(pos)
    return distances


def _clear_line(grid, start, end):
    r, c = start
    end_r, end_c = end
    dx, dy = abs(end_c - c), -abs(end_r - r)
    sx, sy = (1 if c < end_c else -1), (1 if r < end_r else -1)
    error = dx + dy
    while True:
        if grid[r][c] == 1:
            return False
        if (r, c) == end:
            return True
        twice = 2 * error
        if twice >= dy:
            error += dy
            c += sx
        if twice <= dx:
            error += dx
            r += sy


def postplant_approaches(grid, spike):
    """Find distinct map entrances from the defender side of the planted site."""
    site = next((set(cells) for cells in plant_sites(grid) if spike in cells), {spike})
    boundary = set()
    for r, c in site:
        for dr, dc in MOVE_STEPS.values():
            pos = r + dr, c + dc
            if (0 <= pos[0] < len(grid) and 0 <= pos[1] < len(grid[0]) and
                    grid[pos[0]][pos[1]] != 1 and pos not in site):
                boundary.add(pos)
    defender_spawn = tuple((r, c) for r, row in enumerate(grid) for c, value in enumerate(row)
                           if value == 4)
    origin = defender_spawn[len(defender_spawn) // 2]
    defender_distances = _distances(grid, origin)
    ranked = sorted(boundary, key=lambda pos: (defender_distances.get(pos, 10000), pos))
    if not ranked:
        return (spike,)
    first = ranked[0]
    second = next((pos for pos in ranked[1:] if _chebyshev(pos, first) >= 3), None)
    return (first, second) if second is not None else (first,)


def postplant_positions(snapshot):
    """Choose spaced positions with map sightlines to the main retake angles."""
    spike = snapshot.spike_planted
    if spike is None:
        raise ValueError("post-plant plan requires a visible spike")
    approaches = postplant_approaches(snapshot.grid, spike)
    spike_distances = _distances(snapshot.grid, spike)
    candidates = []
    for r in range(max(0, spike[0] - 6), min(len(snapshot.grid), spike[0] + 7)):
        for c in range(max(0, spike[1] - 6), min(len(snapshot.grid[0]), spike[1] + 7)):
            pos = (r, c)
            if snapshot.grid[r][c] == 1 or pos == spike:
                continue
            distance = spike_distances.get(pos, 10000)
            if 2 <= distance <= 5:
                candidates.append((pos, distance))
    if not candidates:
        candidates = [(spike, 0)]
    anchors = [ally.position for ally in snapshot.allies]
    chosen = []
    for ally in snapshot.allies:
        if not ally.alive:
            continue
        angle = approaches[ally.slot % len(approaches)]
        ally_distances = _distances(snapshot.grid, ally.position)
        anchor = min(candidates, key=lambda item: (
            ally_distances.get(item[0], 10000) +
            abs(item[1] - 3) * 2 +
            (0 if _clear_line(snapshot.grid, item[0], angle) else 7) +
            sum(10 if item[0] == other else 5 if _chebyshev(item[0], other) <= 1 else 0
                for other in chosen),
            item[0]))[0]
        anchors[ally.slot] = anchor
        chosen.append(anchor)
    return tuple(anchors), approaches


def guard_attack_postplant(decision, snapshot, masks, *, plan=None, last_recon_tick=-100):
    if snapshot.side != "A" or not snapshot.is_planted or snapshot.spike_planted is None:
        return decision
    spike = snapshot.spike_planted
    anchors, approaches = plan if plan is not None else postplant_positions(snapshot)
    actions = list(decision.actions)
    smoke_near_spike = any(_chebyshev(cell, spike) <= 1 for cell in snapshot.smoke_cells)
    urgent = smoke_near_spike or snapshot.defuse_notified
    alive = [ally for ally in snapshot.allies if ally.alive]
    reserved = {ally.position for ally in alive}
    danger = {cell for effect in snapshot.effects if effect.phase in ("warning", "active")
              and effect.kind != "SMOKE" for cell in effect.cells}

    recon_cast = False
    recon_slot = next((a.slot for a in snapshot.allies if a.alive and masks.abilities[a.slot] == "RECON"
                       and masks.kind[a.slot, KINDS.index("ABILITY")]), None)
    if urgent and recon_slot is not None and snapshot.tick - last_recon_tick >= 5:
        columns = len(snapshot.grid[0])
        targets = [(r, c) for r in range(max(0, spike[0] - 2), min(len(snapshot.grid), spike[0] + 3))
                   for c in range(max(0, spike[1] - 2), min(columns, spike[1] + 3))
                   if masks.target[recon_slot, 0, r * columns + c]]
        if targets:
            target = min(targets, key=lambda pos: (_chebyshev(pos, spike), pos))
            actions[recon_slot] = FrcAction("ABILITY", actions[recon_slot].facing, target=target)
            recon_cast = True

    # A friendly smoke on the spike would make the common defuse tactic easier.
    for slot, ability in enumerate(masks.abilities):
        if (ability == "SMOKE" and actions[slot].kind == "ABILITY" and actions[slot].target is not None and
                _chebyshev(actions[slot].target, spike) <= 2):
            actions[slot] = FrcAction("STAY", actions[slot].facing)

    rushers = set()
    if urgent:
        eligible = [ally for ally in alive if ally.slot != recon_slot or not recon_cast]
        eligible.sort(key=lambda ally: (route_step(snapshot.grid, ally.position, (spike,))[1], ally.slot))
        rushers = {ally.slot for ally in eligible[:2 if snapshot.defuse_notified else 1]}
    for ally in alive:
        slot = ally.slot
        if recon_cast and slot == recon_slot:
            continue
        if actions[slot].kind not in ("STAY", *MOVE_STEPS):
            continue
        goals = (spike,) if slot in rushers else (anchors[slot],)
        if (slot in rushers and any(_chebyshev(seen.position, spike) <= 1 and
                                    _chebyshev(ally.position, seen.position) <= 3
                                    for seen in snapshot.sightings)):
            actions[slot] = FrcAction("STAY", actions[slot].facing)
            continue
        if ally.position in goals:
            actions[slot] = FrcAction("STAY", actions[slot].facing)
            continue
        kind, _ = route_step(snapshot.grid, ally.position, goals, danger,
                             first_step_blocked=reserved - {ally.position})
        if kind in MOVE_STEPS and masks.kind[slot, KINDS.index(kind)]:
            actions[slot] = FrcAction(kind, actions[slot].facing)
            dr, dc = MOVE_STEPS[kind]
            reserved.add((ally.position[0] + dr, ally.position[1] + dc))
        else:
            actions[slot] = FrcAction("STAY", actions[slot].facing)

    def aim(ally):
        if urgent and (ally.slot in rushers or snapshot.defuse_notified):
            return spike
        return approaches[ally.slot % len(approaches)]

    aim_snapshot = (replace(snapshot, sightings=tuple(seen for seen in snapshot.sightings
                    if _chebyshev(seen.position, spike) <= 2)) if snapshot.defuse_notified else snapshot)
    _face_team(actions, aim_snapshot, masks, aim)
    return replace(decision, actions=tuple(actions), phase="POSTPLANT",
                   site=next((i for i, cells in enumerate(plant_sites(snapshot.grid)) if spike in cells),
                             decision.site), objective=min(rushers) if rushers else decision.objective)
