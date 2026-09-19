"""Fnatic v1: CPU-only rule controller; no learned policies or GC dependencies.

Positions are (row, column). Only observed enemies enter tactical memory.
"""
from collections import deque
import heapq
import math

import numpy as np

from controllers import BaseController
from game_core import PLANT_REQUIRED_TICKS
from map_data_defender_setup import is_setup_position_allowed


COVER_RADIUS = 3
DEADLINE_MARGIN = 10
ENEMY_MEMORY_TICKS = 6


def pos(char):
    return tuple(map(int, char.pos))


def distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class FnaticRulesController(BaseController):
    def __init__(self, side):
        self.side = side
        self.game = None
        self.round_number = 0
        self.reset_round()

    def set_game(self, game):
        self.game = game

    def reset_round(self):
        self.round_number += 1
        self.target = None
        self.memory = {}
        self.hold_targets = {}
        self.retriever = None
        self.strategy = ('RUSH', 'DEFAULT')[self.round_number % 2]
        self.cover_names = ()

    def _los(self, a, b, grid, smoke=True):
        if self.game is not None and hasattr(self.game, 'check_cell_line_of_sight'):
            return self.game.check_cell_line_of_sight(a, b, block_smoke=smoke)
        return self.has_line_of_sight(a, b, grid)

    def _route(self, start, goals, grid, blocked=(), risks=(), setup=False):
        """Stable multi-goal shortest path; no random fallback on obstruction."""
        goals = set(goals)
        blocked = set(blocked) - {start}
        queue = [(0, 0, start)]
        costs = {start: 0}
        parent = {start: None}
        while queue:
            cost, steps, cur = heapq.heappop(queue)
            if cost != costs[cur]:
                continue
            if cur in goals:
                nxt = cur
                while parent[nxt] is not None and parent[nxt] != start:
                    nxt = parent[nxt]
                return nxt, steps, cur
            for dr, dc in self.CARDINAL_MOVES:
                cell = (cur[0] + dr, cur[1] + dc)
                r, c = cell
                if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                    continue
                if grid[cell] == 1 or cell in blocked:
                    continue
                if setup and not is_setup_position_allowed(r, c):
                    continue
                exposure = sum(self._los(cell, enemy, grid) for enemy in risks)
                new_cost = cost + 1 + exposure * 4
                if new_cost < costs.get(cell, float('inf')):
                    costs[cell] = new_cost
                    parent[cell] = cur
                    heapq.heappush(queue, (new_cost, steps + 1, cell))
        return start, float('inf'), None

    def _result(self, char, destination, aim=None):
        aim = aim or destination
        dr, dc = aim[0] - char.pos[0], aim[1] - char.pos[1]
        if dr or dc:
            index = round(math.atan2(dc, -dr) / (math.pi / 4)) % 8
            facing = ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')[index]
        else:
            facing = getattr(char, 'facing', 'N')
        return list(destination), {'facing': facing}

    def _ability(self, char, target, kind):
        if getattr(char, 'ability_name', '') != kind:
            return None
        if getattr(char, kind.lower() + '_charges', 0) <= 0:
            return None
        return list(char.pos), {'ability': kind, 'target': list(target)}

    def _cover_engagement(self, char, allies, risks, grid, blocked):
        for enemy in risks:
            if not any(c.name != char.name and distance(pos(c), pos(char)) <= 5
                       and self._los(pos(c), enemy, grid) for c in allies):
                continue
            goals = [(r, c) for r in range(max(0, char.pos[0] - 5), min(grid.shape[0], char.pos[0] + 6))
                     for c in range(max(0, char.pos[1] - 5), min(grid.shape[1], char.pos[1] + 6))
                     if grid[r, c] != 1 and self._los((r, c), enemy, grid)]
            nxt, length, _ = self._route(pos(char), goals, grid, blocked)
            if length <= 5:
                return self._result(char, nxt, enemy)
        return None

    def _guard_position(self, char, anchor, allies, grid, setup=False):
        key = (char.name, anchor, setup)
        if key not in self.hold_targets:
            watch = [(anchor[0] + dr, anchor[1] + dc)
                     for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                     if 0 <= anchor[0] + dr < grid.shape[0]
                     and 0 <= anchor[1] + dc < grid.shape[1]
                     and grid[anchor[0] + dr, anchor[1] + dc] != 1]
            candidates = []
            assigned = list(self.hold_targets.values())
            lengths = {pos(char): 0}
            queue = deque([pos(char)])
            while queue:
                cur = queue.popleft()
                for dr, dc in self.CARDINAL_MOVES:
                    cell = (cur[0] + dr, cur[1] + dc)
                    r, c = cell
                    if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                        continue
                    if grid[cell] == 1 or cell in lengths:
                        continue
                    if setup and not is_setup_position_allowed(r, c):
                        continue
                    lengths[cell] = lengths[cur] + 1
                    queue.append(cell)
            for r in range(max(0, anchor[0] - 9), min(grid.shape[0], anchor[0] + 10)):
                for c in range(max(0, anchor[1] - 9), min(grid.shape[1], anchor[1] + 10)):
                    cell = (r, c)
                    if grid[cell] == 1 or not 2 <= distance(cell, anchor) <= 8:
                        continue
                    if setup and not is_setup_position_allowed(r, c):
                        continue
                    coverage = sum(self._los(cell, w, grid, smoke=False) for w in watch)
                    if not coverage:
                        continue
                    length = lengths.get(cell)
                    if length is None:
                        continue
                    crowd = sum(max(0, 3 - distance(cell, p)) for p in assigned)
                    candidates.append((length + crowd * 8 - coverage * 2, cell))
            self.hold_targets[key] = min(candidates)[1] if candidates else anchor
        return self.hold_targets[key]

    def decide_move(self, char, state):
        grid = np.asarray(state['grid'])
        allies = sorted((c for c in state.get('chars', [])
                         if c.team == char.team and getattr(c, 'is_alive', True)),
                        key=lambda c: c.name)
        if not allies:
            return list(char.pos)
        tick = int(getattr(self.game, 'battle_tick', 0))
        visible = [c for c in state.get('chars', [])
                   if not state.get('defender_setup_active')
                   and c.team != char.team and getattr(c, 'is_alive', True)
                   and (self._los(pos(char), pos(c), grid)
                        or getattr(c, 'reveal_remaining', 0) > 0)]
        for enemy in visible:
            self.memory[enemy.name] = (pos(enemy), tick)
        self.memory = {name: value for name, value in self.memory.items()
                       if tick - value[1] <= ENEMY_MEMORY_TICKS}
        risks = [value[0] for value in self.memory.values()]
        blocked = {pos(c) for c in allies if c.name != char.name}
        plants = [tuple(map(int, p)) for p in zip(*np.where(grid == 2))]
        planted = state.get('planted_pos') if state.get('is_planted') else None
        if not plants:
            return list(char.pos)
        if state.get('defender_setup_active'):
            # Two A, one Mid, two B, chosen from legal setup cells.
            index = next(i for i, c in enumerate(allies) if c.name == char.name)
            ordered = sorted(plants, key=lambda p: p[1])
            anchor = ordered[0] if index < 2 else ordered[-1]
            if index == 2:
                anchor = (grid.shape[0] // 2, grid.shape[1] // 2)
            goal = self._guard_position(char, anchor, allies, grid, setup=True)
            nxt, _, _ = self._route(pos(char), [goal], grid, blocked, setup=True)
            return self._result(char, nxt, anchor)
        if self.side == 'A':
            return self._attack(char, state, grid, allies, blocked, plants, planted, visible, risks)
        return self._defend(char, state, grid, allies, blocked, plants, planted, visible, risks)

    def _attack(self, char, state, grid, allies, blocked, plants, planted, visible, risks):
        if planted is not None:
            anchor = tuple(map(int, planted))
            # Prioritize contesting observed defusers over holding the assigned angle.
            defuser = next((e for e in visible if getattr(e, 'defuse_timer', 0) > 0), None)
            if defuser is not None:
                return self._result(char, pos(char), pos(defuser))
            if visible:
                return self._result(char, pos(char), pos(visible[0]))
            goal = self._guard_position(char, anchor, allies, grid)
            nxt, _, _ = self._route(pos(char), [goal], grid, blocked)
            return self._result(char, nxt, anchor)
        holder = next((c for c in allies if getattr(c, 'has_spike', False)), None)
        dropped = state.get('spike_pos')
        if holder is None and dropped is not None:
            anchor = tuple(map(int, dropped))
            if self.retriever not in {c.name for c in allies}:
                self.retriever = min(allies, key=lambda c: self._route(pos(c), [anchor], grid)[1]).name
            holder = next(c for c in allies if c.name == self.retriever)
            if char.name == holder.name:
                nxt, _, _ = self._route(pos(char), [anchor], grid, blocked)
                return self._result(char, nxt, risks[0] if risks else anchor)
        if holder is None:
            return list(char.pos)
        if self.target is None:
            _, _, self.target = self._route(pos(holder), plants, grid, risks=risks)
        target = self.target
        if target is None:
            return list(char.pos)
        _, shortest, _ = self._route(pos(holder), plants, grid)
        urgent = float(state.get('round_timer', 100)) <= shortest + PLANT_REQUIRED_TICKS + DEADLINE_MARGIN
        if char.name == holder.name:
            if grid[pos(char)] == 2 and getattr(char, 'has_spike', False):
                return list(char.pos), 'PLANT'
            goals = plants if urgent else [target]
            nxt, _, goal = self._route(pos(char), goals, grid, blocked, risks=() if urgent else risks)
            if goal is not None and urgent:
                self.target = goal
            return self._result(char, nxt, visible[0].pos if visible else target)
        # Nearest two teammates form the carrier group, irrespective of roster.
        escorts = sorted((c for c in allies if c.name != holder.name),
                         key=lambda c: (self._route(pos(c), [pos(holder)], grid)[1], c.name))
        if not self.cover_names or any(name not in {c.name for c in escorts} for name in self.cover_names):
            self.cover_names = tuple(c.name for c in escorts[:2])
        covering = char.name in self.cover_names
        if visible:
            ability = self._ability(char, pos(visible[0]), 'FLASH')
            if ability:
                return ability
            return self._result(char, pos(char), pos(visible[0]))
        if not urgent:
            support = self._cover_engagement(char, allies, risks, grid, blocked)
            if support:
                return support
        # Prepare recon/utility on entry, never on spawn or on our own feet.
        if distance(pos(char), target) <= 8 and self._los(pos(char), target, grid, smoke=False):
            for kind in ('RECON', 'SMOKE'):
                ability_target = target
                if kind == 'SMOKE':
                    spawns = [tuple(map(int, p)) for p in zip(*np.where(grid == 4))]
                    for _ in range(3):
                        if spawns:
                            ability_target, _, _ = self._route(ability_target, spawns, grid)
                ability = self._ability(char, ability_target, kind)
                if ability and distance(pos(char), target) >= 3:
                    return ability
        if covering:
            goals = [(r, c) for r in range(max(0, holder.pos[0] - 2), min(grid.shape[0], holder.pos[0] + 3))
                     for c in range(max(0, holder.pos[1] - 2), min(grid.shape[1], holder.pos[1] + 3))
                     if grid[r, c] != 1 and (r, c) != pos(holder)]
            nxt, _, _ = self._route(pos(char), goals, grid, blocked)
            # If already covering, move with the carrier instead of remaining behind.
            if nxt == pos(char):
                nxt, _, _ = self._route(pos(char), [target], grid, blocked, risks=risks)
        else:
            nxt, _, _ = self._route(pos(char), [target], grid, blocked, risks=risks)
            if self.strategy == 'DEFAULT' and int(getattr(self.game, 'battle_tick', 0)) % 2:
                nxt = pos(char)
        return self._result(char, nxt, risks[0] if risks else target)

    def _defend(self, char, state, grid, allies, blocked, plants, planted, visible, risks):
        if planted is not None:
            anchor = tuple(map(int, planted))
            if distance(pos(char), anchor) <= 3:
                ability = self._ability(char, anchor, 'SMOKE')
                if ability:
                    return ability
            if distance(pos(char), anchor) <= 1:
                return list(char.pos), 'DEFUSE'
            if visible:
                ability = self._ability(char, pos(visible[0]), 'FLASH')
                if ability:
                    return ability
                return self._result(char, pos(char), pos(visible[0]))
            goals = [(anchor[0] + dr, anchor[1] + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                     if 0 <= anchor[0] + dr < grid.shape[0] and 0 <= anchor[1] + dc < grid.shape[1]
                     and grid[anchor[0] + dr, anchor[1] + dc] != 1]
            nxt, _, _ = self._route(pos(char), goals, grid, blocked)
            return self._result(char, nxt, anchor)
        if visible:
            for kind in ('SMOKE', 'FLASH', 'RECON'):
                ability = self._ability(char, pos(visible[0]), kind)
                if ability:
                    return ability
            return self._result(char, pos(char), pos(visible[0]))
        support = self._cover_engagement(char, allies, risks, grid, blocked)
        if support:
            return support
        dropped = state.get('spike_pos')
        if dropped is not None:
            anchor = tuple(map(int, dropped))
        elif len(risks) >= 2:
            anchor = min(plants, key=lambda p: sum(distance(p, e) for e in risks))
        else:
            index = next(i for i, c in enumerate(allies) if c.name == char.name)
            ordered = sorted(plants, key=lambda p: p[1])
            anchor = ordered[0] if index < (len(allies) + 1) // 2 else ordered[-1]
        goal = self._guard_position(char, anchor, allies, grid)
        nxt, _, _ = self._route(pos(char), [goal], grid, blocked)
        return self._result(char, nxt, risks[0] if risks else anchor)


class FnaticV1AttackerController(FnaticRulesController):
    def __init__(self):
        super().__init__('A')


class FnaticV1DefenderController(FnaticRulesController):
    def __init__(self):
        super().__init__('D')
