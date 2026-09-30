"""Confirm one recon cast per active contact, entry or smoke condition."""

from fnatic_v1_rules import pos
from game_core import REVEAL_DURATION_TICKS

from .positions import distances


class FnaticReconGate:
    def __init__(self):
        self.context = None
        self.used = {}
        self.pending = {}
        self.last_cast = {}

    def begin(self, ctrl, char, state):
        self.context = state
        self._sync(char)
        active = self._conditions(ctrl, char, state)
        self.used.setdefault(char.name, set()).intersection_update(active)

    def _sync(self, char):
        pending = self.pending.pop(char.name, None)
        if pending is not None and char.recon_charges < pending[0]:
            self.used.setdefault(char.name, set()).update(pending[2])
            self.last_cast[char.name] = pending[1]

    def request(self, ctrl, char, state=None, smoke=None):
        self._sync(char)
        state = state if state is not None else self.context
        if state is None and ctrl.game is not None:
            state = {key: getattr(ctrl.game, key, None) for key in
                     ('grid', 'chars', 'is_planted', 'planted_pos')}
        if state is None or state.get('grid') is None:
            return True
        owner = getattr(ctrl.game, 'real_game', ctrl.game)
        tick = int(getattr(owner, 'battle_tick', 0))
        active = self._conditions(ctrl, char, state)
        used = self.used.setdefault(char.name, set())
        used.intersection_update(active)
        normal = {key for key in active if key[0] != 'smoke'}
        condition = ('smoke', smoke) if smoke is not None else None
        fresh = condition not in used if condition is not None else bool(normal - used)
        in_flight = any(p.get('owner') == char.name and p.get('team', char.team) == char.team
                        for p in getattr(owner, 'recon_projectiles', ()))
        last = self.last_cast.get(char.name)
        if (not fresh or in_flight
                or (last is not None and tick - last < REVEAL_DURATION_TICKS + 3)):
            return False
        consumed = normal | ({condition} if condition is not None else set())
        self.pending[char.name] = (char.recon_charges, tick, consumed)
        return True

    @staticmethod
    def _conditions(ctrl, char, state):
        grid = state['grid']
        keys = {('enemy', c.team, c.name) for c in state.get('chars', ())
                if c.team != char.team and c.is_alive and (
                    getattr(c, 'reveal_remaining', 0) > 0
                    or ctrl._los(pos(char), pos(c), grid,
                                 smoke=not getattr(char, 'sees_through_smoke', False)))}
        if char.team == 'A':
            planted = bool(state.get('is_planted'))
            anchor = state.get('planted_pos') if planted else getattr(ctrl, 'target', None)
            if anchor is not None and distances(tuple(anchor), grid).get(pos(char), float('inf')) <= 10:
                keys.add(('postplant' if planted else 'entry', tuple(anchor)))
        owner = getattr(ctrl.game, 'real_game', ctrl.game)
        for smoke in getattr(owner, 'smokes', ()):
            if smoke.get('remaining_ticks', 0) > 0:
                keys.add(('smoke', id(smoke)))
        return keys
