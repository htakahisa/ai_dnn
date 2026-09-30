"""Observed opponent tendencies shared by both roles for one whole series."""

from collections import Counter
import random

from fnatic_v1_rules import pos

from .positions import region


SIMILAR_DEFENDER_COUNT = 0.25


class FnaticOpponentHistory:
    def __init__(self):
        self.data = {'attack_rounds': [], 'defence_rounds': []}
        self.scope = None
        self.round_key = None
        self.current = None

    def bind(self, ctrl):
        owner = getattr(ctrl.game, 'real_game', ctrl.game)
        if owner is None:
            return
        root = getattr(owner, 'fnatic_series_memory', None)
        if root is None:
            root = getattr(owner, 'series_context', {}).get('fnatic_memory')
            if root is None:
                root = {}
            owner.fnatic_series_memory = root
        label = getattr(owner, 'attacker_team_name' if ctrl.side == 'A' else 'defender_team_name', None)
        if label is None:
            roster = getattr(owner, 'attacker_roster' if ctrl.side == 'A' else 'defender_roster', None)
            roster = roster or [c.name for c in getattr(owner, 'chars', ()) if c.team == ctrl.side]
            label = '|'.join(sorted(map(str, roster))) or ctrl.side
        scope = (id(owner), label)
        if scope == self.scope:
            return
        self.finish()
        self.data = root.setdefault('teams', {}).setdefault(label, {'attack_rounds': [], 'defence_rounds': []})
        self.scope = scope
        self.current = None
        self.round_key = None

    def observe(self, ctrl, char, state):
        self.bind(ctrl)
        owner = getattr(ctrl.game, 'real_game', ctrl.game)
        key = (id(owner), getattr(owner, 'current_round', ctrl.round_number), ctrl.round_number)
        if key != self.round_key:
            self.finish()
            self.round_key = key
            self.current = {'side': char.team, 'round': key[1],
                            'map': getattr(owner, 'series_context', {}).get('maps_played', 0) + 1,
                            'contacts': {}, 'last': {}, 'site': None, 'finished': False}
        if self.current['finished'] or state.get('defender_setup_active'):
            return
        grid = state['grid']
        for enemy in state.get('chars', ()):
            if enemy.team == char.team or not enemy.is_alive:
                continue
            if not (getattr(enemy, 'reveal_remaining', 0) > 0 or getattr(enemy, 'los_revealed', False)
                    or ctrl._los(pos(char), pos(enemy), grid,
                                 smoke=not getattr(char, 'sees_through_smoke', False))):
                continue
            if char.team == 'A' and not state.get('is_planted'):
                self.current['contacts'].setdefault(enemy.name, {'pos': pos(enemy), 'region': region(pos(enemy), grid)})
            if char.team == 'D':
                self.current['last'][enemy.name] = region(pos(enemy), grid)
        if char.team == 'D' and state.get('is_planted') and state.get('planted_pos') is not None:
            self.current['site'] = 'A' if state['planted_pos'][1] < grid.shape[1] // 2 else 'B'

    def finish(self, ctrl=None):
        row = self.current
        if row is None or row['finished']:
            return
        if row['side'] == 'A':
            self.data['attack_rounds'].append({'map': row['map'], 'round': row['round'],
                                               'contacts': dict(row['contacts'])})
        else:
            owner = getattr(ctrl.game, 'real_game', ctrl.game) if ctrl is not None else None
            if owner is not None and getattr(owner, 'is_planted', False) and getattr(owner, 'planted_pos', None) is not None:
                row['site'] = 'A' if owner.planted_pos[1] < owner.grid.shape[1] // 2 else 'B'
            if row['site'] is None:
                counts = Counter(label for label in row['last'].values() if label in {'A', 'B'})
                if counts and counts['A'] != counts['B']:
                    row['site'] = max(('A', 'B'), key=lambda label: counts[label])
            self.data['defence_rounds'].append({'map': row['map'], 'round': row['round'], 'site': row['site']})
        row['finished'] = True

    def reset_round(self):
        self.finish()
        self.current = None
        self.round_key = None

    def attack_region(self, available):
        rows = [row for row in self.data['attack_rounds'] if row['contacts']]
        if not rows or not available:
            return None
        counts = Counter(contact['region'] for row in rows for contact in row['contacts'].values())
        scores = {label: counts[label] / len(rows) for label in available}
        minimum = min(scores.values())
        return random.choice([label for label, score in scores.items() if score <= minimum + SIMILAR_DEFENDER_COUNT])

    def defender_site(self):
        counts = Counter(row['site'] for row in self.data['defence_rounds'] if row['site'] in {'A', 'B'})
        return max(('A', 'B'), key=lambda label: counts[label]) if counts['A'] != counts['B'] else None
