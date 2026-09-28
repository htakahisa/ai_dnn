"""Nearby orb collection with shared, stable Fnatic priorities."""

from fnatic_v1_rules import pos

from .positions import distances

ORB_RADIUS = 5
ORB_PRIORITY = ('Alfajer', 'Chronicle', 'Leo', 'Boaster', 'Derke')


class FnaticOrbCollection:
    def __init__(self):
        self.targets = {}

    def result(self, ctrl, char, state):
        if char.team == 'D' and state.get('is_planted'):
            self.targets.clear()
            return None
        owner = getattr(ctrl.game, 'real_game', ctrl.game)
        orbs = {tuple(map(int, cell)) for cell in
                state.get('available_orbs', getattr(owner, 'available_orbs', ())) or ()}
        if not orbs:
            self.targets.clear()
            return None
        grid = state['grid']
        allies = [getattr(c, 'real_character', c) for c in state.get('chars', ())
                  if c.team == char.team and c.is_alive]
        eligible = {c.name: c for c in allies if c.name in ORB_PRIORITY
                    and getattr(c, 'ultimate_cost', 0) > 0
                    and getattr(c, 'ultimate_points', 0) < c.ultimate_cost
                    and not getattr(c, 'is_planting', False)
                    and getattr(c, 'plant_timer', 0) <= 0
                    and getattr(c, 'defuse_timer', 0) <= 0
                    and not (getattr(c, 'has_spike', False) and pos(c) == getattr(ctrl, 'target', None))
                    and not (getattr(ctrl, 'defuse_names', ())
                             and c.name == getattr(ctrl, 'defuse_responder', None))}
        lengths = {cell: distances(cell, grid) for cell in orbs}
        self.targets = {name: cell for name, cell in self.targets.items()
                        if name in eligible and cell in lengths
                        and lengths[cell].get(pos(eligible[name]), float('inf')) <= ORB_RADIUS}
        used = set(self.targets.values())
        for name in ORB_PRIORITY:
            if name not in eligible or name in self.targets:
                continue
            nearby = [cell for cell in orbs - used
                      if lengths[cell].get(pos(eligible[name]), float('inf')) <= ORB_RADIUS]
            if nearby:
                target = min(nearby, key=lambda cell: (lengths[cell][pos(eligible[name])], cell))
                self.targets[name] = target
                used.add(target)
        target = self.targets.get(char.name)
        if target is None:
            return None
        # Observed combat keeps the normal aim/ability action. The assignment
        # remains available when this player can safely resume collection.
        visible = [c for c in state.get('chars', ()) if c.team != char.team and c.is_alive
                   and (ctrl._los(pos(char), pos(c), grid) or getattr(c, 'reveal_remaining', 0) > 0)]
        if visible:
            return None
        if pos(char) == target:
            return list(char.pos), 'COLLECT_ORB'
        occupied = {pos(c) for c in allies if c.name != char.name}
        nxt, _, _ = ctrl._route(pos(char), [target], grid)
        if nxt in occupied:
            nxt, _, _ = ctrl._route(pos(char), [target], grid, occupied)
        return ctrl._result(char, nxt, target)
