"""Shared site-level observations for the Omoko defender search policy."""

import numpy as np


SITE_CONTEXT_DIM = 12
SITE_SUPPORT_RADIUS = 6
LOCAL_SUPPORT_RADIUS = 5


def site_index(pos, site_dist_maps):
    """Find the closest reachable plant site by walking distance."""
    if pos is None or not site_dist_maps:
        return None
    cell = tuple(map(int, pos))
    distances = [int(dist_map[cell]) for dist_map in site_dist_maps]
    reachable = [(distance, index) for index, distance in enumerate(distances) if distance >= 0]
    return min(reachable)[1] if reachable else None


def walking_distance(pos, dist_map):
    if pos is None or dist_map is None:
        return -1
    return int(dist_map[tuple(map(int, pos))])


def regroup_pressure(unit, allies, visible_enemies):
    """Return a small regroup signal for an isolated, engaged defender."""
    nearby_allies = [
        ally for ally in allies
        if max(abs(ally.pos[0] - unit.pos[0]),
               abs(ally.pos[1] - unit.pos[1])) <= LOCAL_SUPPORT_RADIUS
    ]
    nearby_enemies = [
        enemy for enemy in visible_enemies
        if max(abs(enemy.pos[0] - unit.pos[0]),
               abs(enemy.pos[1] - unit.pos[1])) <= LOCAL_SUPPORT_RADIUS
    ]
    if nearby_allies or not nearby_enemies or not allies:
        return 0.0
    return 1.0 if len(nearby_enemies) >= 2 or unit.hp <= unit.max_hp * 0.5 else 0.4


def site_context(unit, defenders, visible_enemies, site_dist_maps, assigned_pos,
                 tactical_pos, spike_held, height, width):
    """Describe whether this defender should hold a site, rotate, or regroup.

    Every input is observable at inference time. The caller supplies only enemies
    visible to the team and a known spike or sighting position.
    """
    result = np.zeros(SITE_CONTEXT_DIM, dtype=np.float32)
    result[0] = float(bool(spike_held and tactical_pos is not None))
    target_site = site_index(tactical_pos, site_dist_maps)
    allies = [d for d in defenders if d is not unit and d.is_alive]
    if target_site is not None:
        target_map = site_dist_maps[target_site]
        current_distance = walking_distance(unit.pos, target_map)
        assigned_site = site_index(assigned_pos, site_dist_maps)
        result[1] = 1.0
        result[2] = float(assigned_site == target_site)
        scale = max(1, height + width)
        result[3] = min(max(current_distance, 0), scale) / scale
        assigned_distance = walking_distance(assigned_pos, target_map)
        result[4] = min(max(assigned_distance, 0), scale) / scale
        result[5] = sum(
            0 <= walking_distance(d.pos, target_map) <= SITE_SUPPORT_RADIUS
            for d in allies
        ) / 4.0
    result[6] = sum(
        max(abs(d.pos[0] - unit.pos[0]), abs(d.pos[1] - unit.pos[1]))
        <= LOCAL_SUPPORT_RADIUS for d in allies
    ) / 4.0
    result[7] = min(len(visible_enemies), 5) / 5.0
    nearby_enemies = [
        enemy for enemy in visible_enemies
        if max(abs(enemy.pos[0] - unit.pos[0]), abs(enemy.pos[1] - unit.pos[1]))
        <= LOCAL_SUPPORT_RADIUS
    ]
    result[8] = min(len(nearby_enemies), 5) / 5.0
    if allies:
        nearest = min(
            allies,
            key=lambda d: max(abs(d.pos[0] - unit.pos[0]),
                              abs(d.pos[1] - unit.pos[1])),
        )
        result[9] = (nearest.pos[0] - unit.pos[0]) / height
        result[10] = (nearest.pos[1] - unit.pos[1]) / width
        result[11] = regroup_pressure(unit, allies, nearby_enemies)
    return result
