"""GC v3 postplant deployment map. Coordinates are (row, column).

The arena is unchanged; these are new guard positions, sight lines and retake
lanes. Every plant tile in either site is covered by the same site plan.
"""
from dataclasses import dataclass
from ghost_champions_v2.geometry import valid, los, route, watch_cells


@dataclass(frozen=True)
class Post:
    name: str
    position: tuple
    aim: tuple


GUARD_MAP = {
    "A": {
        "plants": ((7, 3), (7, 4), (8, 3), (8, 4), (9, 3)),
        "posts": (
            Post("entry_crossfire", (7, 1), (7, 6)),
            Post("upper_cover", (6, 3), (7, 4)),
            Post("lower_trade", (11, 3), (11, 6)),
            Post("entry_trade", (7, 5), (7, 6)),
            Post("defuse_watch", (11, 2), (8, 3)),
        ),
        "lanes": ((7, 6), (11, 6), (4, 6)),
        "retake_spawns": ((4, 8), (4, 9), (4, 10), (11, 8), (11, 9)),
    },
    "B": {
        "plants": ((4, 40), (4, 41), (5, 40), (6, 40), (7, 40),
                   (8, 40), (9, 40), (10, 40), (10, 41)),
        "posts": (
            Post("west_crossfire", (7, 38), (7, 34)),
            Post("upper_trade", (5, 39), (2, 39)),
            Post("lower_trade", (9, 39), (12, 40)),
            Post("defuse_watch", (6, 40), (7, 40)),
            Post("upper_crossfire", (3, 40), (2, 38)),
        ),
        "lanes": ((7, 34), (2, 38), (12, 40)),
        "retake_spawns": ((7, 33), (7, 34), (2, 35), (2, 36), (12, 40)),
    },
    "B_outer": {
        "plants": tuple((r, 42) for r in range(4, 11)),
        "posts": (
            Post("upper_exit", (4, 41), (4, 38)),
            Post("upper_defuse", (5, 42), (4, 42)),
            Post("central_defuse", (7, 42), (5, 42)),
            Post("lower_defuse", (9, 42), (10, 42)),
            Post("lower_exit", (10, 41), (10, 38)),
        ),
        "lanes": ((4, 38), (10, 38), (12, 40)),
        "retake_spawns": ((7, 33), (7, 34), (2, 35), (2, 36), (12, 40)),
    },
}


def site_plan(spike):
    return GUARD_MAP["A" if spike[1] < 20 else "B_outer" if spike[1] == 42 else "B"]


def validate_map(grid):
    for site, plan in GUARD_MAP.items():
        points = [*plan["plants"], *plan["lanes"], *plan["retake_spawns"]]
        points += [p for post in plan["posts"] for p in (post.position, post.aim)]
        if not all(valid(grid, point) for point in points):
            raise ValueError(f"GC v3 {site} map contains a wall or off-map cell")
        for post in plan["posts"]:
            if not los(grid, post.position, post.aim):
                raise ValueError(f"GC v3 {site}/{post.name} preaim is behind a wall")
        for plant in plan["plants"]:
            if grid[plant] != 2:
                raise ValueError(f"GC v3 {site} plant is not a plant tile: {plant}")
            if not any(los(grid, post.position, q) for post in plan["posts"] for q in watch_cells(grid, plant)):
                raise ValueError(f"GC v3 {site} has no defuse coverage at {plant}")
            if not all(len(route(grid, spawn, [plant])) > 1 for spawn in plan["retake_spawns"]):
                raise ValueError(f"GC v3 {site} retake route is disconnected")
