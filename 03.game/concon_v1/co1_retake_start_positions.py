"""Foundation starts around the five live defender-search posts (row, col)."""

START_CELLS = {
    "a": ((11, 12), (10, 12), (11, 11), (12, 12), (9, 12), (10, 11), (11, 10), (12, 11)),
    "b": ((11, 16), (10, 16), (11, 17), (12, 16), (9, 16), (10, 17), (12, 17), (13, 16)),
    "c": ((10, 21), (9, 21), (10, 22), (11, 21), (8, 21), (9, 22), (10, 23), (11, 22), (12, 21)),
    "d": ((11, 32), (10, 32), (11, 31), (9, 32), (10, 31), (10, 33), (11, 30), (12, 31)),
    "e": ((14, 31), (13, 31), (14, 32), (15, 31), (12, 31), (13, 30), (14, 33), (16, 31)),
}
MAX_START_BFS_DISTANCE = 2


def validate_starts():
    from concon_v1.co1_defender_scenario import get_scenario
    from concon_v1.co1_attacker_common import bfs_distance_map
    scenario = get_scenario()
    if set(START_CELLS) != set(scenario.positions):
        raise ValueError("foundation starts must cover all five search posts")
    for slot, candidates in START_CELLS.items():
        if scenario.positions[slot] not in candidates or len(set(candidates)) != len(candidates):
            raise ValueError(f"{slot}: include the search post and unique surrounding starts")
        distances = bfs_distance_map(scenario.grid, scenario.positions[slot])
        for r, c in candidates:
            if not (0 <= r < scenario.grid.shape[0] and 0 <= c < scenario.grid.shape[1]
                    and 0 <= distances[r, c] <= MAX_START_BFS_DISTANCE):
                raise ValueError(f"{slot}: start {(r, c)} must be within BFS distance {MAX_START_BFS_DISTANCE} of its search post")
    return START_CELLS
