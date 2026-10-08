"""Public map geometry and editable observation posts (coordinates: row, column)."""
from dataclasses import dataclass
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np

from map_data import NEW_MAZE_STR
from map_data_defender_setup import get_setup_mask
from grid_paths import distance_map
from abilities_los import AbilityLosMixin
from toruAI_v4.tv4_map_attacker_branch import AT_MAP_STR

OPPONENTS = {
    "gc_v1": ("gc_v1", "Ghost Champions"),
    "touyama_v2": ("touyama_gaming_v2", "Touyama Gaming"),
    "omoko_v1": ("omoko_gaming_v1", "Omoko Gaming"),
    "fnatic_v3": ("fnatic_v3", "Fnatic2023"),
    "frc_v1": ("frc_v1", "Furina Classic"),
    "toru_ai_v3": ("toru_ai_v3.1", "Team Elites"),
}
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class Post:
    role: str
    watch: tuple
    retreat: tuple
    alternate: tuple
    look: tuple
    site: str


class Scenario(AbilityLosMixin):
    def __init__(self, config=None):
        # The leading '=' is a heading, not a map row. Letters overlay the
        # production board, preserving the orb tile hidden by the 'f' marker.
        rows = [s.strip() for s in AT_MAP_STR.splitlines()
                if s.strip() and set(s.strip()) <= set("012345abcdefghijklmn")]
        self.grid = np.array([[int(c) for c in r] for r in NEW_MAZE_STR.strip().splitlines()])
        if len(rows) != len(self.grid) or any(len(r) != self.grid.shape[1] for r in rows):
            raise ValueError("Branch overlay size differs from the production map")
        self.branches = {}
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                if value.isalpha():
                    if self.grid[r, c] == 1:
                        raise ValueError(f"Branch {value} is on a wall at {(r, c)}")
                    self.branches.setdefault(value, []).append((r, c))
                elif int(value) != self.grid[r, c]:
                    raise ValueError(f"Branch overlay changes the production board at {(r, c)}")
        self.smokes = []
        self.names = tuple(sorted(self.branches))
        self.setup = np.asarray(get_setup_mask())
        self.spawn = tuple(map(int, np.argwhere(self.grid == 4)[2]))
        self.spawn_dist = distance_map(self.grid, self.spawn)
        fields = [np.minimum.reduce([distance_map(self.grid, p) for p in self.branches[n]])
                  for n in self.names]
        self.region = np.argmin(np.stack([np.where(f < 0, 9999, f) for f in fields]), axis=0)
        self.sites = {side: tuple(map(tuple, np.argwhere((self.grid == 2) &
                      ((np.indices(self.grid.shape)[1] < self.grid.shape[1] // 2) if side == "L"
                       else (np.indices(self.grid.shape)[1] >= self.grid.shape[1] // 2)))))
                      for side in ("L", "R")}
        self.site_dist = {side: np.minimum.reduce([distance_map(self.grid, p) for p in cells])
                          for side, cells in self.sites.items()}
        # Rally overlays supply geometry only; uppercase utility markers are ignored.
        from toruAI_v4.tv4_map_retake_L import MAZE_STR as LEFT_RALLY
        from toruAI_v4.tv4_map_retake_R import MAZE_STR as RIGHT_RALLY
        self.rally_points, self.rally_dist, self.retake_entries = {}, {}, {}
        for side, overlay in (("L", LEFT_RALLY), ("R", RIGHT_RALLY)):
            lines = overlay.strip().splitlines()
            if len(lines) != len(self.grid) or any(len(line) != self.grid.shape[1] for line in lines):
                raise ValueError(f"{side} rally overlay size differs from the production map")
            groups = {marker: tuple((r, c) for r, line in enumerate(lines)
                                   for c, value in enumerate(line) if value == marker)
                      for marker in ("a", "b", "c") if marker in overlay}
            if any(not groups.get(marker) for marker in ("a", "b")):
                raise ValueError(f"{side} rally overlay requires both a and b")
            points = tuple(p for group in groups.values() for p in group)
            if len(points) < 5 or any(self.grid[p] == 1 for p in points):
                raise ValueError(f"{side} rally cells must provide five walkable positions")
            self.rally_points[side] = groups
            self.retake_entries[side] = {marker.lower(): tuple((r, c) for r, line in enumerate(lines)
                                         for c, value in enumerate(line) if value == marker)
                                         for marker in ("A", "B", "C") if marker in overlay}
            maps = [distance_map(self.grid, p) for p in points]
            self.rally_dist[side] = np.minimum.reduce([np.where(m >= 0, m, np.inf) for m in maps])
        self.posts = self.default_posts() if config is None else self.read_posts(config)
        self.validate_posts()
        self.maze = NEW_MAZE_STR
        self.signature = hashlib.sha256(json.dumps(self.metadata(), sort_keys=True).encode()).hexdigest()

    def clear(self, origin, target):
        return self.check_cell_line_of_sight(origin, target, block_smoke=False)

    def neighbors(self, pos):
        for dr, dc in MOVES:
            p = pos[0] + dr, pos[1] + dc
            if 0 <= p[0] < self.grid.shape[0] and 0 <= p[1] < self.grid.shape[1] and self.grid[p] != 1:
                yield p

    def local(self, origin, limit):
        q, seen = deque([(origin, 0)]), {origin}
        while q:
            p, d = q.popleft()
            yield p, d
            if d < limit:
                for n in self.neighbors(p):
                    if n not in seen:
                        seen.add(n)
                        q.append((n, d + 1))

    def default_posts(self):
        # Candidate posts have a corner within three legal steps. This is a
        # geometry heuristic, not a guarantee of survival against enemy fire.
        posts, used = [], set()
        for role, branch, side in (("left_scout", "e", "L"),
                                    ("mid_scout", "f", "L"),
                                    ("right_scout", "g", "R")):
            cells = self.branches[branch]
            look = cells[len(cells) // 2]
            candidates = []
            for p, d in self.local(look, 5):
                if p in used or self.setup[p] != 0 or self.spawn_dist[p] < 0 or not self.clear(p, look):
                    continue
                covers = [(q, steps) for q, steps in self.local(p, 3)
                          if self.setup[q] == 0 and self.spawn_dist[q] <= self.spawn_dist[p]
                          and not self.clear(q, look)]
                if covers:
                    cover, steps = min(covers, key=lambda item: (item[1], self.spawn_dist[item[0]], item[0]))
                    candidates.append((d + 2 * steps + max(0, self.spawn_dist[p] - 20), p, cover))
            if not candidates:
                raise ValueError(f"No observation post with cover near {branch}; supply --posts")
            _, watch, cover = min(candidates)
            used.add(watch)
            posts.append(Post(role, watch, cover, watch, look, side))
        for role, branch, side in (("left_anchor", "i", "L"), ("right_anchor", "n", "R")):
            look = self.branches[branch][0]
            candidates = [(p, d) for p, d in self.local(look, 5)
                          if p not in used and self.setup[p] == 0 and not self.clear(p, look)]
            if not candidates:
                raise ValueError(f"No anchor cover near {branch}; supply --posts")
            watch, _ = min(candidates, key=lambda item: (item[1], self.spawn_dist[item[0]], item[0]))
            used.add(watch)
            posts.append(Post(role, watch, watch, watch, look, side))
        return tuple(posts)

    def read_posts(self, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return tuple(Post(p["role"], tuple(p["watch"]), tuple(p["retreat"]),
                          tuple(p.get("alternate", p["watch"])), tuple(p["look"]), p["site"])
                     for p in values)

    def validate_posts(self):
        expected = {"left_scout", "mid_scout", "right_scout", "left_anchor", "right_anchor"}
        if len(self.posts) != 5 or {p.role for p in self.posts} != expected:
            raise ValueError("Posts must contain the five distinct scout/anchor roles")
        for post in self.posts:
            if post.site not in self.sites:
                raise ValueError("Post site must be L or R")
            for field in ("watch", "retreat", "alternate", "look"):
                p = getattr(post, field)
                if len(p) != 2 or any(not isinstance(v, int) for v in p) or not (
                        0 <= p[0] < self.grid.shape[0] and 0 <= p[1] < self.grid.shape[1]) or self.grid[p] == 1:
                    raise ValueError(f"Invalid {field} for {post.role}: {p}")
                if field != "look" and (self.setup[p] != 0 or self.spawn_dist[p] < 0):
                    raise ValueError(f"{post.role} {field} is not a reachable setup position")

    def metadata(self):
        from dataclasses import asdict
        return {"board": self.grid.tolist(), "branches": self.branches,
                "posts": [asdict(p) for p in self.posts], "sensor": "frc_public_team_v1"}

    def site_of(self, position):
        return "L" if position[1] < self.grid.shape[1] // 2 else "R"
