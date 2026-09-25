"""Coordinate the first retake entry using ov1_map_data_retake annotations."""

from collections import deque

try:
    from .ov1_map_data_retake import retake_entry_pairs
except ImportError:  # run_game adds omoko_v1 to sys.path for top-level imports
    from ov1_map_data_retake import retake_entry_pairs


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _path(grid, start, goal, blocked=()):
    start, goal = tuple(start), tuple(goal)
    blocked = set(blocked) - {start}
    queue = deque([start])
    parents = {start: None}
    while queue:
        pos = queue.popleft()
        if pos == goal:
            result = []
            while pos is not None:
                result.append(pos)
                pos = parents[pos]
            return result[::-1]
        for dr, dc in STEPS:
            nxt = (pos[0] + dr, pos[1] + dc)
            if (
                0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]
                and grid[nxt] != 1 and nxt not in blocked and nxt not in parents
            ):
                parents[nxt] = pos
                queue.append(nxt)
    return None


class RetakeEntryCoordinator:
    """Stage two lead defenders at paired entries, then release the team."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.assignments = None
        self.finished = False
        self.released = False
        self.marker_pairs = None

    def _assign(self, grid, allies, planted_pos, timer, defuse_ticks):
        if len(allies) < 2:
            self.finished = True
            return
        if any(max(abs(a.pos[0] - planted_pos[0]), abs(a.pos[1] - planted_pos[1])) <= 6 for a in allies):
            # A teammate has already entered the site; do not pull them back.
            self.finished = True
            return

        if self.marker_pairs is None:
            self.marker_pairs = retake_entry_pairs(grid)
        side = "left" if planted_pos[1] < grid.shape[1] // 2 else "right"
        pairs = self.marker_pairs[side]
        plans = []
        # Each letter marks a pair of simultaneous entries. Choose the pair
        # that can assemble while leaving enough time to reach and defuse.
        for group in (pairs[:2], pairs[2:]):
            options = []
            for entry, watch in group:
                spike_route = _path(grid, entry, planted_pos)
                if spike_route is None:
                    break
                candidates = []
                for ally in allies:
                    route = _path(grid, tuple(ally.pos), entry)
                    if route is not None and len(route) >= 3 and route[-2] not in (group[0][0], group[1][0]):
                        candidates.append((len(route), ally.name, route))
                if not candidates:
                    break
                options.append((entry, watch, spike_route, candidates))
            if len(options) != 2:
                continue

            # The farther entry leads through the choke point. Put the
            # nearest defender there and the farthest remaining defender at
            # the upstream entry so it cannot park ahead of the leader.
            options.sort(key=lambda item: min(c[0] for c in item[3]), reverse=True)
            deep, shallow = options
            first = min(deep[3])
            others = [candidate for candidate in shallow[3] if candidate[1] != first[1]]
            if not others:
                continue
            second = max(others)
            leads = (
                (first[1], first[2][-2], deep[0], deep[1]),
                (second[1], second[2][-2], shallow[0], shallow[1]),
            )
            if leads[0][1] == leads[1][1]:
                continue
            latest_stage = max(first[0], second[0]) - 2
            entry_to_spike = max(len(deep[2]), len(shallow[2])) - 1
            needed = latest_stage + 1 + entry_to_spike + defuse_ticks + 8
            if timer > needed:
                plans.append((needed, leads))

        if not plans:
            self.finished = True
            return
        _, leads = min(plans, key=lambda item: item[0])
        assignments = {
            name: (stage, entry, watch, False)
            for name, stage, entry, watch in leads
        }
        for ally in allies:
            if ally.name in assignments:
                continue
            # Supports hold their current positions until both leads are
            # ready, then start advancing on the release tick.
            entry, watch = min(
                ((entry, watch) for _, _, entry, watch in leads),
                key=lambda pair: len(_path(grid, tuple(ally.pos), pair[0])),
            )
            assignments[ally.name] = (tuple(ally.pos), entry, watch, True)
        self.assignments = assignments

    def actions(self, grid, chars, planted_pos, timer, defuse_ticks):
        """Return coordinated (next position, watch target) decisions by name."""
        if self.finished:
            return {}
        allies = [c for c in chars if c.team == "D" and c.is_alive]
        if self.assignments is None:
            self._assign(grid, allies, planted_pos, timer, defuse_ticks)
        if self.finished:
            return {}

        active = [a for a in allies if a.name in self.assignments]
        leads = [a for a in active if not self.assignments[a.name][3]]
        if len(leads) < 2:
            self.finished = True
            return {}
        enemies = [c for c in chars if c.team != "D" and c.is_alive]
        if any(_visible(grid, tuple(a.pos), tuple(e.pos)) for a in active for e in enemies):
            self.finished = True
            return {}
        if any(tuple(a.pos) == self.assignments[a.name][1] for a in active):
            self.finished = True
            return {}

        routes = [_path(grid, tuple(a.pos), planted_pos) for a in leads]
        if any(route is None for route in routes):
            self.finished = True
            return {}
        remaining = max(len(route) - 1 for route in routes)
        if timer <= remaining + defuse_ticks + 4:
            self.finished = True
            return {}

        occupied = {tuple(c.pos) for c in chars if c.is_alive}
        ready = all(tuple(a.pos) == self.assignments[a.name][0] for a in leads)
        if ready and any(
            self.assignments[a.name][1] in occupied
            for a in active if not self.assignments[a.name][3]
        ):
            ready = False
        results = {}
        for ally in active:
            stage, entry, watch, follower = self.assignments[ally.name]
            pos = tuple(ally.pos)
            if ready:
                target = entry
            else:
                target = pos if follower else stage
            if pos == target:
                step = pos
            else:
                route = _path(grid, pos, target, occupied - {pos, target})
                step = route[1] if route is not None and len(route) > 1 else pos
                if step in occupied and step != pos:
                    step = pos
            results[ally.name] = (step, watch)
        if ready:
            self.released = True
        elif self.released:
            self.finished = True
        return results


def _visible(grid, start, end):
    """Wall-only LOS for deciding when the trained combat policy takes over."""
    r0, c0 = start
    r1, c1 = end
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r0 < r1 else -1), (1 if c0 < c1 else -1)
    error = dc - dr
    while True:
        if grid[r0, c0] == 1:
            return False
        if (r0, c0) == end:
            return True
        double = 2 * error
        if double > -dr:
            error -= dr
            c0 += sc
        if double < dc:
            error += dc
            r0 += sr
