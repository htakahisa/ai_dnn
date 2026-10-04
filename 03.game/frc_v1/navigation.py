"""Explicit entry/carrier navigation guarantees for the learned attacker."""

from collections import deque
from dataclasses import replace
from functools import lru_cache

from frc_v1 import FACING
from frc_v1.actions import KINDS, MOVE_STEPS, FrcAction
from frc_v1.baseline import _grid_key, facing_to, plant_sites, route_step

# The east site has a second approach along the outer right corridor.
EAST_LONG_WAYPOINT = (18, 40)


def _danger_cells(snapshot, belief=None):
    own = {effect.handle for effect in getattr(belief, "effects", ())
           if effect.affiliation == "own"}
    return {cell for effect in snapshot.effects
            if effect.phase in ("warning", "active") and
            effect.kind in ("NEON", "TUNNEL", "BALEMOON", "DESTRUCTION") and
            effect.handle not in own for cell in effect.cells}


def _attack_goals(ally, site_goals, route_waypoint):
    if (route_waypoint is not None and
            (ally.position[0] > route_waypoint[0] or ally.position[1] < route_waypoint[1] - 2)):
        return (route_waypoint,)
    return site_goals


def _face_team(actions, snapshot, masks, default_target):
    for ally in snapshot.allies:
        if not ally.alive or ally.forced_facing:
            continue
        seen = min(snapshot.sightings, key=lambda e: abs(e.position[0] - ally.position[0]) +
                   abs(e.position[1] - ally.position[1])) if snapshot.sightings else None
        target = seen.position if seen else (default_target(ally) if callable(default_target)
                                            else default_target)
        face = facing_to(ally.position, target)
        index = FACING.index(face)
        if (masks.facing[ally.slot, index] and
                (actions[ally.slot].kind != "ULTIMATE" or masks.ultimate_facing[ally.slot, index])):
            actions[ally.slot] = replace(actions[ally.slot], facing=face)


def guard_attack_navigation(decision, snapshot, masks, *, site_index=None, setup_positions=None,
                            route_waypoint=None, belief=None):
    """Keep the entry first and give the spike carrier a walkable route to site.

    This is a visible tactical constraint on inference, not a hidden training
    observation. Abilities and combat decisions remain with the learned actor.
    """
    if snapshot.side != "A" or snapshot.is_planted:
        return decision
    if snapshot.phase == "setup":
        return guard_attack_setup(decision, snapshot, masks, site_index=site_index,
                                  anchors=setup_positions, route_waypoint=route_waypoint)
    carrier = next((a.slot for a in snapshot.allies if a.alive and a.has_spike), None)
    if carrier is None:
        if snapshot.spike_dropped is not None:
            alive = [ally for ally in snapshot.allies if ally.alive]
            if alive:
                retriever = min(alive, key=lambda ally: route_step(
                    snapshot.grid, ally.position, (snapshot.spike_dropped,))[1])
                occupied = {ally.position for ally in alive}
                danger = _danger_cells(snapshot, belief)
                kind, _ = route_step(snapshot.grid, retriever.position, (snapshot.spike_dropped,),
                                     danger, first_step_blocked=occupied - {retriever.position})
                actions = list(decision.actions)
                if kind in MOVE_STEPS and masks.kind[retriever.slot, KINDS.index(kind)]:
                    actions[retriever.slot] = FrcAction(kind, actions[retriever.slot].facing)
                return replace(decision, actions=tuple(actions), objective=retriever.slot,
                               phase="REGROUP")
        return decision
    entry = 2 if snapshot.allies[2].alive else carrier
    sites = plant_sites(snapshot.grid)
    if not sites:
        return decision
    site = (site_index if site_index is not None and 0 <= site_index < len(sites) else
            min(range(len(sites)), key=lambda i: route_step(
                snapshot.grid, snapshot.allies[entry].position, sites[i])[1]))
    goals = sites[site]
    spawn_cells = tuple((r, c) for r, row in enumerate(snapshot.grid) for c, cell in enumerate(row) if cell == 3)
    spawn_distances = _walk_distances(snapshot.grid, spawn_cells)
    entrance = min(goals, key=lambda pos: (spawn_distances.get(pos, 10000), pos))
    entrance_distances = _walk_distances(snapshot.grid, (entrance,))
    actions = list(decision.actions)
    occupied = {a.position for a in snapshot.allies if a.alive}
    reserved = set(occupied)
    danger = _danger_cells(snapshot, belief)

    def route(slot, target_goals):
        ally = snapshot.allies[slot]
        # Teammates only obstruct the next step. Treating their current cells
        # as permanent walls can make a narrow corridor look unreachable and
        # leave the learned movement to retreat away from the execute.
        distance = route_step(snapshot.grid, ally.position, target_goals, danger)[1]
        kind, _ = route_step(snapshot.grid, ally.position, target_goals, danger,
                             first_step_blocked=reserved - {ally.position})
        if kind not in MOVE_STEPS or not masks.kind[slot, KINDS.index(kind)]:
            return None, distance
        dr, dc = MOVE_STEPS[kind]
        next_pos = (ally.position[0] + dr, ally.position[1] + dc)
        if route_step(snapshot.grid, next_pos, target_goals, danger)[1] >= distance:
            return None, distance
        return kind, distance

    def force(slot, kind):
        actions[slot] = FrcAction(kind, actions[slot].facing)
        if kind in MOVE_STEPS:
            dr, dc = MOVE_STEPS[kind]
            pos = snapshot.allies[slot].position
            reserved.add((pos[0] + dr, pos[1] + dc))

    def needs_route(slot, kind, distance, target_goals):
        if kind == "STAY":
            return True
        if kind not in MOVE_STEPS:
            return False
        ally = snapshot.allies[slot]
        dr, dc = MOVE_STEPS[kind]
        next_pos = (ally.position[0] + dr, ally.position[1] + dc)
        if next_pos in reserved or next_pos in danger:
            return True
        return route_step(snapshot.grid, next_pos, target_goals)[1] >= distance

    # The carrier moves first in the engine. Hold the team until Lohen has
    # opened the spawn exit, then keep the carrier and support behind him.
    entry_spawned = entry != carrier and snapshot.grid[snapshot.allies[entry].position[0]][
        snapshot.allies[entry].position[1]] == 3
    if entry_spawned:
        kind, _ = route(entry, _attack_goals(snapshot.allies[entry], goals, route_waypoint))
        if kind:
            force(entry, kind)
        if snapshot.grid[snapshot.allies[carrier].position[0]][snapshot.allies[carrier].position[1]] == 3:
            force(carrier, "STAY")
            for slot in (1, 3, 4):
                if snapshot.allies[slot].alive and actions[slot].kind in MOVE_STEPS:
                    force(slot, "STAY")
            _face_team(actions, snapshot, masks, goals[0])
            return replace(decision, actions=tuple(actions), site=site, entry=entry,
                           follow=4, objective=carrier)

    entry_distance = route_step(snapshot.grid, snapshot.allies[entry].position, goals)[1]
    entry_origin = spawn_cells[len(spawn_cells) // 2]
    entry_travel = route_step(snapshot.grid, entry_origin, (snapshot.allies[entry].position,))[1]
    for slot in dict.fromkeys((entry, carrier, 1, 3, 4)):
        if slot == entry and entry_spawned:
            continue
        ally = snapshot.allies[slot]
        if not ally.alive:
            continue
        if masks.kind[slot, KINDS.index("PLANT")]:
            force(slot, "PLANT")
            continue
        if slot != carrier and ally.position in goals and snapshot.allies[carrier].position not in goals:
            exits = []
            for move, (dr, dc) in MOVE_STEPS.items():
                pos = (ally.position[0] + dr, ally.position[1] + dc)
                if (pos in goals and entrance_distances.get(pos, -1) >
                        entrance_distances.get(ally.position, -1) and
                        pos not in danger and pos not in reserved and
                        masks.kind[slot, KINDS.index(move)]):
                    exits.append((pos[1] == entrance[1], move))
            if exits:
                force(slot, min(exits)[1])
                continue
        if (slot == entry and slot != carrier and ally.position in goals and
                snapshot.allies[carrier].position not in goals and
                route_step(snapshot.grid, snapshot.allies[carrier].position, goals)[1] <= 2):
            exits = []
            for move, (dr, dc) in MOVE_STEPS.items():
                pos = ally.position[0] + dr, ally.position[1] + dc
                if masks.kind[slot, KINDS.index(move)] and pos not in danger and pos not in reserved:
                    exits.append((pos not in goals, move))
            if exits:
                force(slot, min(exits)[1])
                continue
        target_goals = _attack_goals(ally, goals, route_waypoint)
        if (slot == carrier and snapshot.allies[entry].alive and
                snapshot.round_timer > 40):
            if entry_travel < 3:
                if actions[slot].kind in ("STAY", *MOVE_STEPS):
                    force(slot, "STAY")
                    continue
            leader = snapshot.allies[entry]
            leader_gap = route_step(snapshot.grid, ally.position,
                                    (leader.position,))[1]
            if entry_distance > 2:
                if entry_travel < 6:
                    if leader_gap <= 2:
                        if actions[slot].kind in ("STAY", *MOVE_STEPS):
                            force(slot, "STAY")
                            continue
                    target_goals = (leader.position,)
                elif (route_step(snapshot.grid, ally.position, goals)[1] <=
                      route_step(snapshot.grid, leader.position, goals)[1] + 1):
                    if actions[slot].kind in ("STAY", *MOVE_STEPS):
                        force(slot, "STAY")
                        continue
        kind, distance = route(slot, target_goals)
        nearby_enemy = any(max(abs(ally.position[0] - seen.position[0]),
                               abs(ally.position[1] - seen.position[1])) <= 2
                           for seen in snapshot.sightings)
        urgent_execute = snapshot.round_timer <= 70 and actions[slot].kind in ("STAY", *MOVE_STEPS)
        if kind and (urgent_execute or
                     snapshot.grid[ally.position[0]][ally.position[1]] == 3 or
                     not nearby_enemy and slot == entry and actions[slot].kind in ("STAY", *MOVE_STEPS) or
                     not nearby_enemy and needs_route(slot, actions[slot].kind, distance, target_goals)):
            force(slot, kind)
        elif actions[slot].kind in MOVE_STEPS:
            dr, dc = MOVE_STEPS[actions[slot].kind]
            next_pos = (ally.position[0] + dr, ally.position[1] + dc)
            if (next_pos in reserved or next_pos in danger or
                    route_step(snapshot.grid, next_pos, target_goals)[1] >= distance):
                force(slot, "STAY")
            else:
                reserved.add(next_pos)
    _face_team(actions, snapshot, masks, goals[0])
    return replace(decision, actions=tuple(actions), site=site, entry=entry,
                   follow=4, objective=carrier)


def defense_site_assignments(snapshot, *, variation=None):
    """Vary the 2/3 site split while keeping travel distances reasonable."""
    sites = plant_sites(snapshot.grid)
    if len(sites) != 2:
        raise ValueError("FRC defense requires two sites")
    round_index = snapshot.round_number if variation is None else variation
    order = sorted(range(len(snapshot.allies)), key=lambda slot: (
        route_step(snapshot.grid, snapshot.allies[slot].position, sites[0])[1] -
        route_step(snapshot.grid, snapshot.allies[slot].position, sites[1])[1],
        (slot + round_index) % len(snapshot.allies)))
    site_zero = set(order[:2 + round_index % 2])
    return tuple(0 if slot in site_zero else 1 for slot in range(len(snapshot.allies)))


def _site_ring(grid, cells, depth=2):
    visited = set(cells)
    frontier = set(cells)
    for _ in range(depth):
        following = set()
        for r, c in frontier:
            for dr, dc in MOVE_STEPS.values():
                pos = r + dr, c + dc
                if (0 <= pos[0] < len(grid) and 0 <= pos[1] < len(grid[0]) and
                        grid[pos[0]][pos[1]] != 1 and pos not in visited):
                    following.add(pos)
        visited.update(following)
        frontier = following
    return tuple(sorted(frontier))


def _walk_distances(grid, origins):
    return _cached_walk_distances(_grid_key(grid), tuple(origins), tuple(MOVE_STEPS.values())).copy()


@lru_cache(maxsize=512)
def _cached_walk_distances(grid, origins, moves):
    distances = {pos: 0 for pos in origins if grid[pos[0]][pos[1]] != 1}
    queue = deque(distances)
    while queue:
        r, c = queue.popleft()
        for dr, dc in moves:
            pos = (r + dr, c + dc)
            if (0 <= pos[0] < len(grid) and 0 <= pos[1] < len(grid[0]) and
                    grid[pos[0]][pos[1]] != 1 and pos not in distances):
                distances[pos] = distances[r, c] + 1
                queue.append(pos)
    return distances


def _line_clear(grid, start, end):
    r, c = start
    er, ec = end
    dx, dy = abs(ec - c), -abs(er - r)
    sx, sy = (1 if c < ec else -1), (1 if r < er else -1)
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


def site_approaches(grid, site, spawn_value):
    return _cached_site_approaches(_grid_key(grid), frozenset(site), spawn_value, tuple(MOVE_STEPS.items()))


@lru_cache(maxsize=128)
def _cached_site_approaches(grid, site, spawn_value, moves):
    """Public-map entrances on the side from which a team approaches a site."""
    site = set(site)
    boundary = set()
    for r, c in site:
        for dr, dc in MOVE_STEPS.values():
            pos = r + dr, c + dc
            if (0 <= pos[0] < len(grid) and 0 <= pos[1] < len(grid[0]) and
                    grid[pos[0]][pos[1]] != 1 and pos not in site):
                boundary.add(pos)
    spawns = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row)
              if value == spawn_value]
    distances = _walk_distances(grid, spawns)
    ranked = sorted(boundary, key=lambda pos: (distances.get(pos, 10000), pos))
    if not ranked:
        return (min(site),)
    first = ranked[0]
    second = next((pos for pos in ranked[1:]
                   if max(abs(pos[0] - first[0]), abs(pos[1] - first[1])) >= 3), None)
    return (first, second) if second is not None else (first,)


def site_interdiction_cells(grid, site, spawn_value):
    return _cached_site_interdiction(_grid_key(grid), frozenset(site), spawn_value, tuple(MOVE_STEPS.items()))


@lru_cache(maxsize=128)
def _cached_site_interdiction(grid, site, spawn_value, moves):
    """Cells just outside entrances, so a 3x3 smoke does not cover the site."""
    site = set(site)
    spawns = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row)
              if value == spawn_value]
    distances = _walk_distances(grid, spawns)
    results = []
    for entrance in site_approaches(grid, site, spawn_value):
        r, c = entrance
        outer = [(r + dr, c + dc) for dr, dc in MOVE_STEPS.values()]
        outer = [pos for pos in outer if pos in distances and pos not in site and
                 distances[pos] < distances.get(entrance, 10000)]
        results.append(min(outer, key=lambda pos: (distances[pos], pos)) if outer else entrance)
    return tuple(dict.fromkeys(results))


def site_watch_points(grid, site):
    return _cached_site_watch_points(_grid_key(grid), frozenset(site), tuple(MOVE_STEPS.items()))


@lru_cache(maxsize=128)
def _cached_site_watch_points(grid, site, moves):
    """Look beyond the attacker-side doorway, toward the lane enemies use."""
    site = set(site)
    points = []
    for entrance in site_approaches(grid, site, 3):
        adjacent = [cell for cell in site if sum(abs(a - b) for a, b in zip(cell, entrance)) == 1]
        if not adjacent:
            points.append(entrance)
            continue
        inside = min(adjacent)
        dr, dc = entrance[0] - inside[0], entrance[1] - inside[1]
        point = entrance
        for distance in range(1, 9):
            next_point = entrance[0] + dr * distance, entrance[1] + dc * distance
            if (not 0 <= next_point[0] < len(grid) or
                    not 0 <= next_point[1] < len(grid[0]) or
                    grid[next_point[0]][next_point[1]] == 1):
                break
            point = next_point
        points.append(point)
    return tuple(points)


def _setup_grid(snapshot):
    allowed = set(snapshot.setup_cells)
    return tuple(tuple(0 if (r, c) in allowed else 1 for c in range(len(row)))
                 for r, row in enumerate(snapshot.grid))


def attack_setup_positions(snapshot, site_index, *, route_waypoint=None):
    """Choose stable, spaced setup goals with Lohen at the front."""
    sites = plant_sites(snapshot.grid)
    if not sites or not snapshot.setup_cells:
        return tuple(ally.position for ally in snapshot.allies)
    site = site_index if site_index is not None and 0 <= site_index < len(sites) else 1
    setup_grid = _setup_grid(snapshot)
    site_distances = _walk_distances(snapshot.grid,
                                     (route_waypoint,) if route_waypoint is not None else sites[site])
    entry = 2 if snapshot.allies[2].alive else next((a.slot for a in snapshot.allies if a.alive), 0)
    carrier = next((a.slot for a in snapshot.allies if a.alive and a.has_spike), entry)
    order = list(dict.fromkeys((entry, carrier, 1, 3, 4, 0)))
    anchors = [a.position for a in snapshot.allies]
    chosen = []
    for slot in order:
        ally = snapshot.allies[slot]
        if not ally.alive:
            continue
        reachable = _walk_distances(setup_grid, (ally.position,))
        candidates = [cell for cell, travel in reachable.items() if travel <= snapshot.tick]
        if not candidates:
            continue
        front = min(site_distances.get(cell, 10000) for cell in candidates)
        gap = 0 if slot == entry else 2 if slot == carrier else 3 + slot % 2
        target_distance = front if slot == entry else site_distances.get(anchors[entry], front) + gap
        anchors[slot] = min(candidates, key=lambda cell: (
            3 * abs(site_distances.get(cell, 10000) - target_distance) +
            15 * max(0, site_distances.get(anchors[entry], front) + 1 -
                     site_distances.get(cell, 10000)) * (slot != entry) +
            12 * sum(cell == other for other in chosen) +
            3 * sum(max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) <= 1
                    for other in chosen) -
            0.3 * min(reachable[cell], 6),
            cell))
        chosen.append(anchors[slot])
    return tuple(anchors)


def guard_attack_setup(decision, snapshot, masks, *, site_index=None, anchors=None,
                       route_waypoint=None):
    """Move to the setup plan without changing its destinations each tick."""
    sites = plant_sites(snapshot.grid)
    if not sites or not snapshot.setup_cells:
        return decision
    site = site_index if site_index is not None and 0 <= site_index < len(sites) else 1
    anchors = anchors if anchors is not None else attack_setup_positions(
        snapshot, site, route_waypoint=route_waypoint)
    setup_grid = _setup_grid(snapshot)
    entry = 2 if snapshot.allies[2].alive else next((a.slot for a in snapshot.allies if a.alive), 0)
    carrier = next((a.slot for a in snapshot.allies if a.alive and a.has_spike), entry)
    order = list(dict.fromkeys((entry, carrier, 1, 3, 4, 0)))
    actions = list(decision.actions)
    reserved = {ally.position for ally in snapshot.allies if ally.alive}
    for slot in order:
        ally = snapshot.allies[slot]
        if not ally.alive or ally.position == anchors[slot]:
            actions[slot] = FrcAction("STAY", actions[slot].facing)
            continue
        kind, _ = route_step(setup_grid, ally.position, (anchors[slot],),
                             blocked=reserved - {ally.position})
        if kind in MOVE_STEPS and masks.kind[slot, KINDS.index(kind)]:
            actions[slot] = FrcAction(kind, actions[slot].facing)
            dr, dc = MOVE_STEPS[kind]
            reserved.add((ally.position[0] + dr, ally.position[1] + dc))
        else:
            actions[slot] = FrcAction("STAY", actions[slot].facing)
    _face_team(actions, snapshot, masks, sites[site][0])
    return replace(decision, actions=tuple(actions), site=site, phase="PREPARE",
                   entry=entry, objective=carrier)


def defense_anchor_positions(snapshot, assignments, *, variation=None):
    """Pick distinct positions reachable during setup, as near each site as allowed."""
    sites = plant_sites(snapshot.grid)
    round_index = snapshot.round_number if variation is None else variation
    setup_cells = set(snapshot.setup_cells)
    setup_grid = (tuple(tuple(0 if (r, c) in setup_cells else 1
                              for c in range(len(row))) for r, row in enumerate(snapshot.grid))
                  if snapshot.phase == "setup" else snapshot.grid)
    site_distances = [_walk_distances(snapshot.grid, site) for site in sites]
    approaches = [site_watch_points(snapshot.grid, site) for site in sites]
    anchors = [None] * len(snapshot.allies)
    chosen = []
    for site in range(len(sites)):
        ring = _site_ring(snapshot.grid, sites[site])
        slots = [slot for slot, assigned in enumerate(assignments) if assigned == site]
        slots.sort(key=lambda slot: (slot + round_index) % len(snapshot.allies))
        for slot in slots:
            origin = snapshot.allies[slot].position
            reachable = _walk_distances(setup_grid, (origin,))
            candidates = [cell for cell in ring if cell in reachable and
                          (snapshot.phase != "setup" or reachable[cell] <= snapshot.tick)]
            if not candidates:
                candidates = [cell for cell, travel in reachable.items()
                              if snapshot.phase != "setup" or travel <= max(1, snapshot.tick)]
            if not candidates:
                candidates = [origin]
            candidates.sort()
            offset = (round_index + slot * 2) % len(candidates)
            anchor = min(candidates, key=lambda cell: (
                site_distances[site].get(cell, 10000) * (3 if cell not in ring else 0) +
                reachable.get(cell, 0) +
                (0 if any(_line_clear(snapshot.grid, cell, angle)
                          for angle in approaches[site]) else 18) +
                sum(8 if cell == other else
                    3 * max(0, 3 - max(abs(cell[0] - other[0]), abs(cell[1] - other[1])))
                    for other in chosen) +
                (candidates.index(cell) - offset) % len(candidates) * 0.5,
                cell))
            anchors[slot] = anchor
            chosen.append(anchor)
    return tuple(anchors)


def guard_defense_navigation(decision, snapshot, masks, *, site_assignments=None,
                             anchor_positions=None):
    """Hold both sites, retake planted spike and restrict defensive smoke."""
    if snapshot.side != "D":
        return decision
    actions = list(decision.actions)
    sites = plant_sites(snapshot.grid)
    assignments = site_assignments or defense_site_assignments(snapshot)
    anchors = anchor_positions or defense_anchor_positions(snapshot, assignments)
    occupied = {a.position for a in snapshot.allies if a.alive}
    danger = {cell for effect in snapshot.effects if effect.phase in ("warning", "active")
              for cell in effect.cells}
    if snapshot.phase == "setup":
        allowed = set(snapshot.setup_cells)
        danger.update((r, c) for r, row in enumerate(snapshot.grid) for c, value in enumerate(row)
                      if value != 1 and (r, c) not in allowed)
    planted = snapshot.is_planted and snapshot.spike_planted is not None
    spike = snapshot.spike_planted
    retake = (() if not planted else tuple((r, c) for r in range(len(snapshot.grid))
        for c in range(len(snapshot.grid[0])) if snapshot.grid[r][c] != 1 and
        max(abs(r - spike[0]), abs(c - spike[1])) <= 1))
    rally = ()
    group_ready = False
    if planted:
        planted_site = next((site for site in sites if spike in site), (spike,))
        defender_entry = site_approaches(snapshot.grid, planted_site, 4)[0]
        spike_distances = _walk_distances(snapshot.grid, (spike,))
        entry_distances = _walk_distances(snapshot.grid, (defender_entry,))
        rally = tuple(sorted((pos for pos, distance in spike_distances.items()
                              if 3 <= distance <= 5 and entry_distances.get(pos, 10000) <= 3),
                             key=lambda pos: (entry_distances[pos], pos)))
        if not rally:
            rally = tuple(pos for pos, distance in spike_distances.items() if 3 <= distance <= 5)
        alive = [ally for ally in snapshot.allies if ally.alive]
        close = sum(spike_distances.get(ally.position, 10000) <= 5 for ally in alive)
        group_ready = (len(alive) <= 1 or close >= 2 or snapshot.detonate_timer <= 12)
    ready = [a for a in snapshot.allies if a.alive and masks.kind[a.slot, KINDS.index("DEFUSE")]]
    defuser = (min(ready, key=lambda a: (-a.defuse_progress, a.slot)).slot if ready else None)

    def force(slot, kind, *, target=None, ally_slot=None):
        actions[slot] = FrcAction(kind, actions[slot].facing, target, ally_slot)

    def target_legal(slot, pos):
        r, c = pos
        return bool(masks.kind[slot, KINDS.index("ABILITY")] and
                    masks.target[slot, 0, r * len(snapshot.grid[0]) + c])

    for ally in snapshot.allies:
        slot = ally.slot
        if not ally.alive:
            continue
        if slot == defuser:
            force(slot, "DEFUSE")
            continue
        if slot == 0 and masks.kind[0, KINDS.index("ABILITY")]:
            hurt = [a for a in snapshot.allies if masks.target[0, 0, a.slot] and a.hp <= 60]
            if hurt:
                force(0, "ABILITY", ally_slot=min(hurt, key=lambda a: a.hp).slot)
                continue
        if slot == 3 and masks.kind[slot, KINDS.index("ABILITY")]:
            usable = [seen.position for seen in snapshot.sightings if target_legal(slot, seen.position)]
            if usable and (ally.charges == 2 or snapshot.tick % 15 == 4):
                target = min(usable, key=lambda pos: route_step(snapshot.grid, ally.position, (pos,))[1])
                force(slot, "ABILITY", target=target)
                continue
        if actions[slot].kind not in ("STAY", *MOVE_STEPS):
            continue
        goals = ((retake if group_ready and defuser is None else rally) if planted
                 else (anchors[slot],))
        if ally.position in goals:
            if actions[slot].kind in MOVE_STEPS:
                force(slot, "STAY")
            continue
        kind, _ = route_step(snapshot.grid, ally.position, goals, danger,
                             first_step_blocked=occupied - {ally.position})
        if kind in MOVE_STEPS and masks.kind[slot, KINDS.index(kind)]:
            force(slot, kind)
        elif actions[slot].kind in MOVE_STEPS:
            force(slot, "STAY")
    approaches = [site_watch_points(snapshot.grid, site) for site in sites]
    defended_site = next((index for index, site in enumerate(sites) if planted and spike in site), None)
    def held_angle(ally):
        index = defended_site if defended_site is not None else assignments[ally.slot]
        visible = [point for point in approaches[index]
                   if _line_clear(snapshot.grid, ally.position, point)]
        if visible:
            return min(visible, key=lambda point: (
                route_step(snapshot.grid, ally.position, (point,))[1], point))
        return approaches[index][ally.slot % len(approaches[index])]
    _face_team(actions, snapshot, masks, held_angle)
    return replace(decision, actions=tuple(actions),
                   phase="PREPARE" if snapshot.phase == "setup" else "RETAKE" if planted else "DEFEND",
                   objective=defuser if defuser is not None else decision.objective)


def guard_tactical_utility(decision, snapshot, masks, *, last_cast=None, cast_history=None):
    """Use Lisa and Arlecchino at a threatened entrance without wasting charges."""
    if snapshot.phase == "setup":
        return decision
    last_cast = last_cast or {}
    cast_history = cast_history or {}
    sites = plant_sites(snapshot.grid)
    if not sites:
        return decision
    actions = list(decision.actions)
    site_distances = [_walk_distances(snapshot.grid, site) for site in sites]

    def cheb(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def legal(slot, target):
        if target is None or not masks.kind[slot, KINDS.index("ABILITY")]:
            return False
        r, c = target
        return bool(masks.target[slot, 0, r * len(snapshot.grid[0]) + c])

    def safe(slot, target, site):
        if not legal(slot, target):
            return False
        if slot == 4 and any(cheb(target, previous) <= 2 for previous in cast_history.get(4, ())):
            return False
        allies = (ally for ally in snapshot.allies if ally.alive and ally.slot != slot)
        if slot == 1:
            return (all(cheb(target, ally.position) >= 3 for ally in allies) and
                    min(cheb(target, cell) for cell in sites[site]) >= 2 and
                    all(cheb(target, smoke) >= 2 for smoke in snapshot.smoke_cells))
        # ASH applies its contract only to enemies, including when allies are
        # already crossing its area.
        return _line_clear(snapshot.grid, snapshot.allies[slot].position, target)

    threatened = []
    for index, distances in enumerate(site_distances):
        seen = [s.position for s in snapshot.sightings if
                distances.get(s.position, 10000) <= (20 if snapshot.side == "D" else 8)]
        if seen:
            threatened.append((min(distances[pos] for pos in seen), index, seen))
        if snapshot.side == "D" and not snapshot.is_planted:
            casualties = [ally.position for ally in snapshot.allies
                          if (not ally.alive or ally.hp < ally.max_hp * 0.6) and
                          distances.get(ally.position, 10000) <= 4]
            if casualties:
                threatened.append((min(distances[pos] for pos in casualties), index, seen))
    if snapshot.side == "D":
        site = min(threatened)[1] if threatened else None
    else:
        site = decision.site if 0 <= decision.site < len(sites) else None
    if site is None or snapshot.side == "D" and snapshot.is_planted:
        if snapshot.side == "D" and actions[1].kind == "ABILITY":
            actions[1] = FrcAction("STAY", actions[1].facing)
            return replace(decision, actions=tuple(actions))
        return decision
    seen = [s.position for s in snapshot.sightings if
            site_distances[site].get(s.position, 10000) <= (20 if snapshot.side == "D" else 8)]
    close_contact = any(site_distances[site].get(pos, 10000) <= 8 for pos in seen)
    local_casualty = snapshot.side == "D" and any(
        (not ally.alive or ally.hp < ally.max_hp * 0.6) and
        site_distances[site].get(ally.position, 10000) <= 4
        for ally in snapshot.allies)
    entry = snapshot.allies[decision.entry] if snapshot.side == "A" else None
    near_execute = (entry is not None and entry.alive and
                    site_distances[site].get(entry.position, 10000) <= 5)
    if snapshot.side == "A" and not seen and not near_execute:
        return decision
    ingress = site_interdiction_cells(snapshot.grid, sites[site],
                                      3 if snapshot.side == "D" else 4)
    ingress = sorted(ingress, key=lambda pos: (min((cheb(pos, enemy) for enemy in seen), default=0), pos))
    nearby_ingress = sorted((pos for pos in site_distances[site]
                            if pos not in ingress and
                            min(cheb(pos, entrance) for entrance in ingress) <= 2 and
                            site_distances[site][pos] <= 5),
                           key=lambda pos: (min(cheb(pos, entrance) for entrance in ingress),
                                            site_distances[site][pos], pos))
    for slot, cooldown in ((1, 10), (4, 8)):
        ally = snapshot.allies[slot]
        if not ally.alive or ally.charges <= 0:
            continue
        if slot == 4 and snapshot.side == "D" and not (close_contact or local_casualty):
            continue
        if snapshot.tick - last_cast.get(slot, -100) < cooldown:
            if actions[slot].kind == "ABILITY":
                actions[slot] = FrcAction("STAY", actions[slot].facing)
            continue
        candidates = ((sorted(seen, key=lambda pos: (site_distances[site].get(pos, 10000), pos),
                              reverse=True) + ingress)
                      if slot == 1 and snapshot.side == "D" and not close_contact else
                      ingress if slot == 1 else
                      sorted(seen, key=lambda pos: (cheb(pos, ally.position), pos)) +
                      ingress + nearby_ingress)
        target = next((pos for pos in candidates if safe(slot, pos, site) and
                       (snapshot.side == "A" or not seen or local_casualty or
                        min(cheb(pos, enemy) for enemy in seen) <= 4)),
                      None)
        if target is not None:
            actions[slot] = FrcAction("ABILITY", actions[slot].facing, target=target)
        elif actions[slot].kind == "ABILITY" and not safe(slot, actions[slot].target, site):
            actions[slot] = FrcAction("STAY", actions[slot].facing)
    return replace(decision, actions=tuple(actions))
