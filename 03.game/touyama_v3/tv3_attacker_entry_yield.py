"""Public goal repair when a helper blocks the carrier's site entry."""
from grid_paths import distance_map
from touyama_v3.tv3_attacker_site_entry import traffic_distances,MAX_TRAFFIC_DETOUR_STEPS


def yielding_entry_goal(scenario,snapshot,ally,goal,carrier_goal,site):
    if ally.has_spike or ally.position!=carrier_goal or scenario.grid[ally.position]!=2:
        return None
    traffic=traffic_distances(snapshot,ally,goal)[ally.position]
    static=distance_map(scenario.grid,goal)[ally.position]
    if traffic>=0 and (static<0 or traffic-static<=MAX_TRAFFIC_DETOUR_STEPS):
        return None
    occupied={a.position for a in snapshot.allies if a.alive and a.slot!=ally.slot}
    alternatives=[]
    for position in scenario.sites[site]:
        if position==ally.position or position in occupied:
            continue
        distance=traffic_distances(snapshot,ally,position)[ally.position]
        if 0<=distance<=MAX_TRAFFIC_DETOUR_STEPS:
            alternatives.append((int(distance),position))
    return min(alternatives)[1] if alternatives else None


def entry_yield_schema():
    return dict(version=1,trigger='inside_helper_blocks_current_carrier_goal_and_own_goal_excessive_detour',
        replacement='nearest_unoccupied_reachable_selected_site_cell',max_steps=MAX_TRAFFIC_DETOUR_STEPS,
        persistence='update_helper_endpoint_until_route_changes',selection='learned_actions_no_forced_move')
