"""Observable, rule-based Fnatic ultimate timing and target selection."""

from collections import deque
import random

import numpy as np

from fnatic_v1_rules import distance, pos
from game_core import (FACING_DIRECTIONS, FACING_VECTORS, NEON_RADIUS_CELLS,
                       RAID_DISTANCE_CELLS, TUNNEL_HALF_WIDTH)

from .formation import SITE_ENTRY_RADIUS
from .positions import distances, region


ESCAPE_ROTATE_DISTANCE = 20
ESCAPE_MAIN_COUNT = 3
ESCAPE_NEAR_ENEMY_RADIUS = 7
MONITOR_START_TICK = 20
FACING_STEPS = {'N': (-1, 0), 'NE': (-1, 1), 'E': (0, 1), 'SE': (1, 1),
                'S': (1, 0), 'SW': (1, -1), 'W': (0, -1), 'NW': (-1, -1)}


class FnaticUltimates:
    def __init__(self):
        self.kill_counts = {}
        self.kill_ticks = {}
        self.relocations = {}

    def sync(self, ctrl, char):
        pending = self.relocations.pop(char.name, None)
        if pending is None or char.ultimate_points >= pending[1]:
            return
        positions = getattr(ctrl, 'defender_positions', None)
        if positions is None:
            return
        cell = pending[0]
        old = positions.targets.get(char.name)
        for name, target in list(positions.targets.items()):
            if name != char.name and target == cell:
                if old is not None:
                    positions.targets[name] = old
                else:
                    del positions.targets[name]
        positions.targets[char.name] = cell

    def result(self, ctrl, char, state):
        owner = getattr(ctrl.game, 'real_game', ctrl.game)
        tick = int(getattr(owner, 'battle_tick', state.get('battle_tick', 0)))
        kills = int(getattr(char, 'round_kills', 0))
        if kills > self.kill_counts.get(char.name, 0):
            self.kill_ticks[char.name] = tick
        self.kill_counts[char.name] = kills
        kind = getattr(char, 'ultimate_name', '')
        if (not getattr(char, 'is_alive', False)
                or kind not in {'TUNNEL', 'NEON', 'RAID', 'ESCAPE', 'MONITOR'}
                or getattr(char, 'ultimate_cost', 0) <= 0
                or getattr(char, 'ultimate_points', 0) < char.ultimate_cost):
            return None
        if kind == 'MONITOR':
            return self._action(char, kind) if tick >= MONITOR_START_TICK else None
        if kind == 'NEON':
            revealed = self.revealed_neon(ctrl, char, state)
            if revealed is not None:
                return revealed
        grid = np.asarray(state['grid'])
        allies = [getattr(c, 'real_character', c) for c in state.get('chars', ())
                  if c.team == char.team and c.is_alive]
        seen = [c for c in state.get('chars', ())
                if c.team != char.team and c.is_alive and self._seen(ctrl, char, c, grid)]
        if kind in {'RAID', 'ESCAPE'}:
            rooted = getattr(owner, '_ramp_blocks_movement', None)
            if rooted is not None and rooted(getattr(char, 'real_character', char)):
                return None
        planted = bool(state.get('is_planted'))
        anchor = None
        entry = False
        if char.team == 'A' and not planted:
            holder = next((c for c in allies if getattr(c, 'has_spike', False)), None)
            anchor = getattr(ctrl, 'target', None)
            if holder is not None and anchor is not None:
                lengths = distances(anchor, grid)
                formation = getattr(ctrl, 'formation', None)
                entry = (lengths.get(pos(holder), float('inf')) <= SITE_ENTRY_RADIUS
                         or getattr(formation, 'rush_goal', None) is not None)
        elif planted:
            anchor = state.get('planted_pos')
            if char.team == 'A':
                anchor = getattr(owner, 'planted_pos', anchor)
        site = self._site_cells(grid, anchor) if anchor is not None else ()

        if kind == 'RAID':
            recently_killed = tick - self.kill_ticks.get(char.name, -100) <= 1
            if recently_killed and len(seen) >= 2:
                # Do not turn a failed attempt to find cover into an entry dash.
                return self._raid_escape(ctrl, char, grid, allies, seen, owner)
            if entry and site and char.name not in getattr(ctrl.formation, 'detached_names', ()):
                return self._raid_entry(ctrl, char, grid, allies, seen, site, owner)
            return None

        if kind in {'TUNNEL', 'NEON'}:
            if char.team == 'A' and planted:
                if not seen:
                    return None
                nearest = min(seen, key=lambda c: (distance(pos(char), pos(c)), c.name))
                targets = (pos(nearest),)
            else:
                retaking = (char.team == 'D' and planted and site
                            and (getattr(getattr(ctrl, 'retake', None), 'launched', False)
                                 or min((self._site_distance(pos(c), site, grid) for c in allies),
                                        default=float('inf')) <= SITE_ENTRY_RADIUS))
                if not site or not (entry or retaking):
                    return None
                targets = site
            if kind == 'TUNNEL':
                return self._tunnel(ctrl, char, targets)
            target = min(targets, key=lambda p: (
                -sum(distance(p, other) <= NEON_RADIUS_CELLS for other in targets),
                distance(p, pos(char)), p))
            return self._action(char, kind, target=target)

        # ESCAPE has a wind-up; reserve one destination rather than requesting
        # a second portal if points happen to refill while the first is active.
        if any(p.get('owner') == char.name and p.get('team', char.team) == char.team
               for p in getattr(owner, 'escape_portals', ())):
            return None
        reports = self._team_reports(ctrl, char, state, allies, grid, owner)
        if char.team == 'D' and not planted:
            target = self._reinforce(ctrl, char, reports, grid, owner)
            if target is not None:
                self.relocations[char.name] = (target, char.ultimate_points)
                return self._action(char, kind, target=target)
        if entry and site:
            legal = [p for p in site if self._landing_free(char, p, owner, allies, seen)]
            holder = next((c for c in allies if getattr(c, 'has_spike', False)), None)
            if holder is not None and holder.name != char.name:
                legal = [p for p in legal if p != tuple(anchor)]
            if legal:
                target = min(legal, key=lambda p: (-distance(p, pos(char)), p))
                return self._action(char, kind, target=target)
        if any(observer != char.name for _, observers in reports.values() for observer in observers):
            marker = 4 if char.team == 'A' else 3
            spawns = [tuple(map(int, p)) for p in zip(*np.where(grid == marker))]
            legal = [p for p in spawns if self._landing_free(char, p, owner, allies, seen)]
            if legal:
                return self._action(char, kind, target=random.choice(legal))
        return None

    def revealed_neon(self, ctrl, char, state):
        """Aim at the largest revealed group, including enemies behind walls."""
        if (not char.is_alive or getattr(char, 'ultimate_name', '') != 'NEON'
                or getattr(char, 'ultimate_cost', 0) <= 0
                or char.ultimate_points < char.ultimate_cost):
            return None
        targets = [pos(c) for c in state.get('chars', ()) if c.team != char.team
                   and c.is_alive and getattr(c, 'reveal_remaining', 0) > 0]
        if not targets:
            return None
        grid = np.asarray(state['grid'])
        radius = NEON_RADIUS_CELLS
        candidates = {(r, c) for target in targets
                      for r in range(max(0, target[0] - radius), min(grid.shape[0], target[0] + radius + 1))
                      for c in range(max(0, target[1] - radius), min(grid.shape[1], target[1] + radius + 1))
                      if grid[r, c] != 1}
        if not candidates:
            return None
        # The runtime blast is a square of floor cells; walls do not shield
        # enemies inside it. A midpoint can hit more enemies than any one
        # enemy's cell, so consider every legal center in their blast ranges.
        target = min(candidates, key=lambda p: (
            -sum(distance(p, enemy) <= radius for enemy in targets),
            sum(distance(p, enemy) for enemy in targets if distance(p, enemy) <= radius),
            p not in targets, distance(p, pos(char)), p))
        return self._action(char, 'NEON', target=target)

    @staticmethod
    def _action(char, kind, **fields):
        return list(char.pos), {'ultimate': kind, **fields}

    @staticmethod
    def _seen(ctrl, observer, enemy, grid):
        ignore_smoke = (getattr(observer, 'sees_through_smoke', False)
                        or getattr(enemy, 'reveal_remaining', 0) > 0)
        return ctrl._los(pos(observer), pos(enemy), grid, smoke=not ignore_smoke)

    @staticmethod
    def _site_cells(grid, anchor):
        anchor = tuple(map(int, anchor))
        plants = [tuple(map(int, p)) for p in zip(*np.where(grid == 2))]
        if not plants:
            return ()
        lengths = distances(anchor, grid)
        start = min(plants, key=lambda p: (lengths.get(p, float('inf')), distance(anchor, p), p))
        found = {start}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (cell[0] + dr, cell[1] + dc)
                if nxt not in found and nxt in plants:
                    found.add(nxt)
                    queue.append(nxt)
        return tuple(sorted(found))

    @staticmethod
    def _site_distance(start, site, grid):
        lengths = distances(start, grid)
        return min((lengths.get(p, float('inf')) for p in site), default=float('inf'))

    def _tunnel(self, ctrl, char, targets):
        def coverage(facing):
            fx, fy = FACING_VECTORS[facing]
            return sum((p[1] - char.pos[1]) * fx + (p[0] - char.pos[0]) * fy > 0
                       and abs((p[1] - char.pos[1]) * -fy + (p[0] - char.pos[0]) * fx)
                       <= TUNNEL_HALF_WIDTH for p in targets)
        facing = max(FACING_DIRECTIONS, key=coverage)
        if not coverage(facing):
            return None
        return self._action(char, 'TUNNEL', facing=facing)

    @staticmethod
    def _raid_paths(char, grid, allies, seen, owner):
        occupied = {pos(c) for c in (*allies, *seen) if c.name != char.name or c.team != char.team}
        occupied.update(tuple(p['pos']) for p in getattr(owner, 'escape_portals', ()))
        for facing, (dr, dc) in FACING_STEPS.items():
            endpoint = pos(char)
            length = 0
            for step in range(1, RAID_DISTANCE_CELLS + 1):
                p = (char.pos[0] + dr * step, char.pos[1] + dc * step)
                if (not (0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1])
                        or grid[p] == 1 or p in occupied):
                    break
                endpoint, length = p, step
            if length:
                yield facing, endpoint, length

    def _raid_entry(self, ctrl, char, grid, allies, seen, site, owner):
        before = self._site_distance(pos(char), site, grid)
        choices = [(self._site_distance(endpoint, site, grid), -length, facing)
                   for facing, endpoint, length in self._raid_paths(char, grid, allies, seen, owner)
                   if self._site_distance(endpoint, site, grid) < before]
        if not choices:
            return None
        return self._action(char, 'RAID', facing=min(choices)[2])

    def _raid_escape(self, ctrl, char, grid, allies, seen, owner):
        choices = []
        for facing, endpoint, length in self._raid_paths(char, grid, allies, seen, owner):
            exposed = any(ctrl._los(endpoint, pos(enemy), grid, smoke=not (
                getattr(enemy, 'sees_through_smoke', False)
                or getattr(char, 'reveal_remaining', 0) > 0)) for enemy in seen)
            if not exposed:
                choices.append((-min(distance(endpoint, pos(e)) for e in seen), -length, facing))
        if not choices:
            return None
        return self._action(char, 'RAID', facing=min(choices)[2])

    def _team_reports(self, ctrl, char, state, allies, grid, owner):
        reports = {}
        team_ctrl = getattr(owner, 'attacker_controller' if char.team == 'A' else 'defender_controller', None)
        engine = getattr(team_ctrl, 'perception_engine', None)
        for ally in allies:
            chars = state.get('chars', ())
            if engine is not None:
                view = engine.build_game_view(viewer=ally, game=owner)
                chars = view.chars
            for enemy in chars:
                if (enemy.team == char.team or not enemy.is_alive
                        or not self._seen(ctrl, ally, enemy, grid)):
                    continue
                key = (enemy.team, enemy.name)
                if key not in reports:
                    reports[key] = (pos(enemy), set())
                reports[key][1].add(ally.name)
        return reports

    @staticmethod
    def _landing_free(char, cell, owner, allies, seen):
        if cell == pos(char):
            return False
        occupied = getattr(owner, '_is_position_occupied', None)
        if occupied is not None:
            return not occupied(getattr(char, 'real_character', char), cell, pos(char))
        return cell not in {pos(c) for c in (*allies, *seen)}

    def _reinforce(self, ctrl, char, reports, grid, owner):
        groups = {label: [cell for cell, _ in reports.values() if region(cell, grid) == label]
                  for label in ('A', 'MID', 'B')}
        label = max(groups, key=lambda name: len(groups[name]))
        enemies = groups[label]
        if len(enemies) < ESCAPE_MAIN_COUNT:
            return None
        plants = [tuple(map(int, p)) for p in zip(*np.where(grid == 2)) if region(p, grid) == label]
        objectives = plants or enemies
        if self._site_distance(pos(char), objectives, grid) < ESCAPE_ROTATE_DISTANCE:
            return None
        positions = getattr(ctrl, 'defender_positions', None)
        if positions is None:
            return None
        ally_chars = [c for c in getattr(owner, 'chars', ()) if c.team == char.team and c.is_alive]
        lengths = [distances(p, grid) for p in enemies]
        candidates = [p for cells in positions.candidates.values() for p in cells
                      if region(p, grid) == label
                      and self._landing_free(char, p, owner, ally_chars, ())]
        if not candidates:
            return None
        scores = {p: min(d.get(p, float('inf')) for d in lengths) for p in candidates}
        near = [p for p in candidates if scores[p] <= ESCAPE_NEAR_ENEMY_RADIUS]
        if not near:
            nearest = min(scores.values())
            near = [p for p in candidates if scores[p] < float('inf') and scores[p] <= nearest + 2]
        if not near:
            return None
        return random.choice(near)
