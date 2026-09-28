"""Shared retake slots and a round-wide stop-before-entry barrier."""

from itertools import permutations

import numpy as np

from fnatic_v1_rules import pos

from .map_data_retake_fnatic import RETAKE_POSITION_STR
from .positions import distances
from .retake_utility import FnaticRetakeUtility

GATHERING_SLOT_COUNT = 3
URGENT_RETAKE_TICKS = 25


class FnaticRetake:
    def __init__(self, map_text=None):
        text = RETAKE_POSITION_STR if map_text is None else map_text
        rows = [line.strip() for line in text.splitlines() if line.strip()]
        if rows and (len({len(row) for row in rows}) != 1
                     or any(c not in '0123456789ab' for row in rows for c in row)):
            raise ValueError('Fnatic retake map requires equal row widths and digits/a/b')
        self.grid = (np.array([[10 if c == 'a' else 11 if c == 'b' else int(c)
                               for c in row] for row in rows], dtype=np.int32) if rows else None)
        self.candidates = {site: tuple(tuple(map(int, p)) for p in zip(*np.where(self.grid == marker)))
                           if self.grid is not None else ()
                           for site, marker in (('A', 10), ('B', 11))}
        if any(0 < len(cells) < GATHERING_SLOT_COUNT for cells in self.candidates.values()):
            raise ValueError('Fnatic retake map needs at least three a/b cells for each enabled site')
        self.reset_round()

    def reset_round(self):
        self.site = None
        self.anchor = None
        self.selected = ()
        self.targets = {}
        self.stopped_at = {}
        self.launched = False
        self.phase_tick = None
        self.preparing = False
        self.urgent = False
        self.utility = FnaticRetakeUtility()

    @property
    def gathering(self):
        return self.site is not None and not self.launched

    def validate(self, grid):
        if not any(self.candidates.values()):
            return
        if self.grid.shape != grid.shape:
            raise ValueError('Fnatic retake map must match game map dimensions')
        if not np.array_equal(self.grid == 1, grid == 1):
            raise ValueError('Fnatic retake map walls must match the game map')

    def result(self, ctrl, char, planted, grid, allies, visible, remaining_ticks=None, state=None):
        site = 'A' if planted[1] < grid.shape[1] // 2 else 'B'
        candidates = self.candidates[site]
        owner = getattr(ctrl.game, 'real_game', ctrl.game)
        actual = sorted((getattr(c, 'real_character', c) for c in allies if c.is_alive), key=lambda c: c.name)
        if (not candidates and self.site is None and not self.utility.needed(owner, actual)
                and (remaining_ticks is None or remaining_ticks >= URGENT_RETAKE_TICKS)):
            return None
        if site != self.site:
            self.reset_round()
            self.site = site
            self.anchor = tuple(map(int, getattr(owner, 'planted_pos', None) or planted))
        if self.launched:
            return None
        self.urgent = self.urgent or (remaining_ticks is not None and remaining_ticks < URGENT_RETAKE_TICKS)
        if (self.preparing or not candidates
                or self.urgent):
            return self._prepare(ctrl, char, grid, actual, visible, state, remaining_ticks)
        if not self.selected:
            lengths = [distances(pos(c), grid) for c in actual]
            reachable = [p for p in candidates if all(p in d for d in lengths)]
            if len(reachable) < GATHERING_SLOT_COUNT:
                raise ValueError('Fnatic retake needs three marked positions reachable by all surviving defenders')
            self.selected = tuple(sorted(reachable, key=lambda p: (sum(d[p] for d in lengths), p))
                                  [:GATHERING_SLOT_COUNT])
        base = self.selected
        slots = list(base)
        # Extra survivors use adjacent floors, not more marked candidates.
        extra = max(0, len(actual) - len(base))
        if extra:
            adjacent = sorted({(p[0] + dr, p[1] + dc) for p in base
                               for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                               if 0 <= p[0] + dr < grid.shape[0]
                               and 0 <= p[1] + dc < grid.shape[1]
                               and grid[p[0] + dr, p[1] + dc] != 1
                               and grid[p[0] + dr, p[1] + dc] != 2} - set(candidates))
            previous = sorted(set(self.targets.values()) - set(base))
            slots.extend((previous + [p for p in adjacent if p not in previous])[:extra])
        self._assign(actual, slots, grid, ctrl)
        for c in actual:
            if pos(c) != self.targets[c.name]:
                self.stopped_at.pop(c.name, None)
        tick = int(getattr(owner, 'battle_tick', 0))
        # Evaluate once at the beginning of a decision tick. A last arrival
        # cannot let only the players processed later in that tick charge.
        if self.phase_tick != tick:
            self.phase_tick = tick
            ready = all(pos(c) == self.targets[c.name]
                        and self.stopped_at.get(c.name, tick) < tick for c in actual)
            if ready:
                return self._prepare(ctrl, char, grid, actual, visible, state, remaining_ticks)
        goal = self.targets[char.name]
        if pos(char) == goal:
            self.stopped_at.setdefault(char.name, tick)
            return ctrl._result(char, pos(char), pos(visible[0]) if visible else self.anchor)
        return self._move(ctrl, char, goal, grid, actual, visible)

    def _prepare(self, ctrl, char, grid, actual, visible, state, remaining_ticks):
        if not self.preparing:
            self.preparing = True
            ctrl.navigation.yields.clear()
        state = state if state is not None else {'chars': visible}
        result = self.utility.result(ctrl, char, self.anchor, grid, actual, state,
                                     urgent=self.urgent, remaining_ticks=remaining_ticks)
        if self.utility.phase == 'DONE':
            self.preparing = False
            self.launched = True
            ctrl.navigation.yields.clear()
        return result

    def _assign(self, actual, slots, grid, ctrl):
        names = {c.name for c in actual}
        self.targets = {name: p for name, p in self.targets.items() if name in names and p in slots}
        self.stopped_at = {name: tick for name, tick in self.stopped_at.items() if name in self.targets}
        unused = [p for p in slots if p not in self.targets.values()]
        missing = [c for c in actual if c.name not in self.targets]
        if not missing:
            return
        lengths = {c.name: distances(pos(c), grid) for c in missing}
        preparation = {(c.name, p): self.utility.preparation_cost(ctrl, c, p, self.anchor, grid)
                       for c in missing for p in unused}
        choices = [(sum(lengths[c.name].get(p, float('inf')) + 4 * preparation[c.name, p]
                        for c, p in zip(missing, cells)), cells)
                   for cells in permutations(unused, len(missing))]
        best = min(choices, default=(float('inf'), ()))
        if best[0] == float('inf'):
            raise ValueError('Fnatic retake gathering positions must be reachable by all surviving defenders')
        self.targets.update((c.name, p) for c, p in zip(missing, best[1]))

    def _move(self, ctrl, char, goal, grid, actual, visible):
        occupied = {pos(c) for c in actual if c.name != char.name}
        occupied.update(pos(c) for c in visible)
        nxt, direct_length, _ = ctrl._route(pos(char), [goal], grid)
        if nxt in occupied:
            direct_next = nxt
            nxt, length, reachable = ctrl._route(pos(char), [goal], grid, occupied)
            if reachable is None or length > direct_length + 2:
                blocker = next((c for c in actual if pos(c) == direct_next
                                and self.targets.get(c.name) == pos(c)), None)
                if blocker is not None:
                    self.targets[blocker.name] = goal
                    self.targets[char.name] = pos(blocker)
                    self.stopped_at.pop(blocker.name, None)
                    self.stopped_at.pop(char.name, None)
                    nxt = pos(char)
        return ctrl._result(char, nxt, pos(visible[0]) if visible else self.anchor)

    def push(self, ctrl, char, planted, grid, allies, visible):
        anchor = tuple(map(int, planted))
        actual = [getattr(c, 'real_character', c) for c in allies]
        occupied = {pos(c) for c in actual if c.name != char.name}
        occupied.update(pos(c) for c in visible)
        goals = [(anchor[0] + dr, anchor[1] + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                 if 0 <= anchor[0] + dr < grid.shape[0] and 0 <= anchor[1] + dc < grid.shape[1]
                 and grid[anchor[0] + dr, anchor[1] + dc] != 1]
        nxt, _, _ = ctrl._route(pos(char), goals, grid, occupied)
        return ctrl._result(char, nxt, pos(visible[0]) if visible else anchor)
