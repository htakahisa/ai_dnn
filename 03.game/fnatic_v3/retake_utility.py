"""A shared recon -> flash -> smoke barrier before the team retakes."""

from types import SimpleNamespace

from fnatic_v1_rules import distance, pos
from game_core import DEFUSE_REQUIRED_TICKS, FLASH_MAX_FLIGHT_TICKS, FLASH_SPEED_CELLS_PER_TICK

from .positions import distances
from .ultimate_tactics import FnaticUltimates


class FnaticRetakeUtility:
    def __init__(self):
        self.phase = 'RECON'
        self.phase_tick = None
        self.pending = {}
        self.recon_targets = {}
        self.recon_impacts = {}
        self.flash_targets = []
        self.revealed = {}
        self.unavailable = set()
        self.cast_any = False
        self.launch_tick = None
        self.preparation_costs = {}
        self.emergency_tick = None

    def preparation_cost(self, ctrl, char, cell, anchor, grid):
        kind = getattr(char, 'ability_name', '')
        if kind not in ('RECON', 'FLASH') or getattr(char, kind.lower() + '_charges', 0) <= 0:
            return 0
        key = (kind, cell, tuple(anchor))
        if key not in self.preparation_costs:
            proxy = SimpleNamespace(name=char.name, pos=cell)
            owner = getattr(ctrl.game, 'real_game', ctrl.game)
            plan = self._throw_plan(ctrl, proxy, kind, anchor, grid, owner)
            self.preparation_costs[key] = (distances(cell, grid).get(plan[0], float('inf'))
                                          if plan is not None else grid.size)
        return self.preparation_costs[key]

    @staticmethod
    def _casters(allies, kind):
        return [c for c in allies if c.is_alive and c.ability_name == kind
                and getattr(c, kind.lower() + '_charges', 0) > 0
                and not getattr(c, 'defuse_timer', 0)]

    @staticmethod
    def _flying(owner, kind, team):
        return any(p.get('team') == team for p in getattr(owner, kind.lower() + '_projectiles', ()))

    @classmethod
    def needed(cls, owner, allies):
        return any(cls._casters(allies, kind) or cls._flying(owner, kind, 'D')
                   for kind in ('RECON', 'FLASH', 'SMOKE'))

    def _observe(self, state, team, grid):
        for enemy in state.get('chars', ()):
            if enemy.team == team:
                continue
            if not enemy.is_alive:
                self.revealed.pop(enemy.name, None)
            elif getattr(enemy, 'reveal_remaining', 0) > 0:
                cell = pos(enemy)
                if 0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1] and grid[cell] != 1:
                    self.revealed[enemy.name] = cell

    def _advance(self, ctrl, anchor, grid, owner, allies, tick, team, urgent, remaining_ticks):
        if self.phase_tick == tick:
            return
        self.phase_tick = tick
        actual = {c.name: c for c in getattr(owner, 'chars', ()) if c.team == team}
        actual.update((c.name, c) for c in allies)
        for name, (kind, charges, aim, impact) in self.pending.items():
            char = actual.get(name)
            if char is None or getattr(char, kind.lower() + '_charges', charges) >= charges:
                continue
            self.cast_any = True
            if kind == 'RECON':
                self.recon_targets.setdefault(name, []).append(aim)
                self.recon_impacts.setdefault(name, []).append(impact)
            elif kind == 'FLASH':
                self.flash_targets.append(aim)
        self.pending.clear()
        if urgent and remaining_ticks is not None and self.emergency_tick is None:
            lengths = distances(anchor, grid)
            travel = max(0, min((lengths.get(pos(c), float('inf')) for c in allies), default=0) - 2)
            # Reserve movement, the defuse channel and one cast/confirmation tick.
            if remaining_ticks <= travel + DEFUSE_REQUIRED_TICKS + 2:
                self.emergency_tick = tick
        if self.emergency_tick is not None:
            if tick > self.emergency_tick:
                self.phase = 'DONE'
                self.launch_tick = tick
            elif not any(self._ready_plan(ctrl, c, c.ability_name, anchor, grid, owner)
                         for c in allies if c in self._casters([c], c.ability_name)
                         and c.ability_name in ('RECON', 'FLASH', 'SMOKE')):
                self.phase = 'DONE'
                self.launch_tick = tick
            return
        while self.phase != 'DONE':
            if self.phase in ('RECON', 'FLASH', 'SMOKE'):
                if urgent:
                    for caster in self._casters(allies, self.phase):
                        if self._ready_plan(ctrl, caster, self.phase, anchor, grid, owner) is None:
                            self.unavailable.add((caster.name, self.phase))
                if any((c.name, self.phase) not in self.unavailable
                       for c in self._casters(allies, self.phase)):
                    return
                self.phase = {'RECON': 'RECON_WAIT', 'FLASH': 'FLASH_WAIT', 'SMOKE': 'DONE'}[self.phase]
            elif self.phase == 'RECON_WAIT':
                if self._flying(owner, 'RECON', team):
                    return
                self.phase = 'FLASH'
            elif self.phase == 'FLASH_WAIT':
                if self._flying(owner, 'FLASH', team):
                    return
                self.phase = 'SMOKE'
        self.launch_tick = tick

    def result(self, ctrl, char, anchor, grid, allies, state, urgent=False, remaining_ticks=None):
        owner = getattr(ctrl.game, 'real_game', ctrl.game)
        tick = int(getattr(owner, 'battle_tick', 0))
        self._observe(state, char.team, grid)
        self._advance(ctrl, anchor, grid, owner, allies, tick, char.team, urgent, remaining_ticks)
        if self.phase == 'DONE':
            return None
        hold = ctrl._result(char, pos(char), anchor)
        kind = char.ability_name if self.emergency_tick is not None else self.phase
        if kind not in ('RECON', 'FLASH', 'SMOKE') or char not in self._casters([char], kind):
            return hold
        if self.emergency_tick is None and (char.name, kind) in self.unavailable:
            return hold
        pending = self.pending.get(char.name)
        if pending is not None:
            # A second decision in the same tick cannot spend another charge.
            if getattr(char, kind.lower() + '_charges', 0) < pending[1]:
                return hold
            return ctrl._retake_ability(char, pending[2], kind)
        if kind == 'SMOKE':
            aim, impact = anchor, anchor
        else:
            plan = self._throw_plan(ctrl, char, kind, anchor, grid, owner,
                                    current_only=urgent or self.emergency_tick is not None)
            if plan is None:
                # An unreachable throwing direction must not hold the team forever.
                self.unavailable.add((char.name, kind))
                return hold
            cell, aim, impact = plan
            if cell != pos(char):
                occupied = {pos(c) for c in allies if c.name != char.name}
                occupied.update(pos(c) for c in state.get('chars', ()) if c.team != char.team and c.is_alive)
                nxt, _, _ = ctrl._route(pos(char), [cell], grid, occupied)
                return ctrl._result(char, nxt, aim)
        action = ctrl._retake_ability(char, aim, kind)
        if action is not None:
            self.pending[char.name] = (kind, getattr(char, kind.lower() + '_charges'), aim, impact)
        return action if action is not None else hold

    def _ready_plan(self, ctrl, char, kind, anchor, grid, owner):
        if kind == 'SMOKE':
            return pos(char), anchor, anchor
        return self._throw_plan(ctrl, char, kind, anchor, grid, owner, current_only=True)

    def _throw_plan(self, ctrl, char, kind, anchor, grid, owner, current_only=False):
        site = FnaticUltimates._site_cells(grid, anchor)
        site = site or (anchor,)
        lengths = distances(anchor, grid)
        area = {p for p in lengths if (p[1] < grid.shape[1] // 2) == (anchor[1] < grid.shape[1] // 2)
                and min(distance(p, cell) for cell in site) <= 4}
        # Recon goes into the site; a tiny plant area can also use nearby site floor.
        aims = tuple(site) + tuple(sorted(area - set(site))) if kind == 'RECON' else tuple(self.revealed.values()) or tuple(site)
        previous_aims = set(self.recon_targets.get(char.name, ()))
        previous_impacts = set(self.recon_impacts.get(char.name, ()))
        all_impacts = [p for cells in self.recon_impacts.values() for p in cells]

        def choices(origin):
            found = []
            for aim in aims:
                if aim == origin or (kind == 'RECON' and aim in previous_aims):
                    continue
                if not ctrl._los(origin, aim, grid, smoke=False):
                    continue
                path = owner._projectile_path(origin, aim)
                if len(path) <= 1:
                    continue
                impact = tuple(path[-1] if kind == 'RECON' else
                               path[min(len(path) - 1, FLASH_SPEED_CELLS_PER_TICK * FLASH_MAX_FLIGHT_TICKS)])
                if kind == 'RECON':
                    if impact in previous_impacts or impact not in area:
                        continue
                    spread = min((distance(impact, p) for p in all_impacts), default=0)
                    score = (aim not in site, impact not in site, -spread, distance(impact, anchor), aim)
                else:
                    hits = sum(ctrl._los(impact, p, grid) for p in self.revealed.values())
                    score = (-hits, self.flash_targets.count(aim), distance(impact, aim), aim)
                found.append((score, aim, impact))
            return min(found, default=None)

        start = pos(char)
        found = choices(start)
        if current_only:
            return ((start, found[1], found[2])
                    if found is not None and (kind != 'RECON' or found[1] in site) else None)
        fallback = found
        if found is not None and (kind != 'RECON' or found[1] in site):
            return start, found[1], found[2]
        walking = distances(start, grid)
        for cell in sorted(walking, key=lambda p: (walking[p], distance(p, anchor), p)):
            found = choices(cell)
            if found is not None and (kind != 'RECON' or found[1] in site):
                return cell, found[1], found[2]
            if fallback is None and found is not None:
                fallback = found
                start = cell
        if fallback is not None:
            return start, fallback[1], fallback[2]
        return None
