"""Fnatic v3 rule controllers using independent tactical maps."""

from functools import lru_cache

import numpy as np

from fnatic_v1_rules import FnaticV1AttackerController, FnaticV1DefenderController, pos

from .positions import TacticalPositions, distances
from .formation import FnaticFormation, MID_PRIORITY


class FnaticV3FacingMixin:
    """Keep tactical facing; also enable the runtime clear-shot enemy aim."""

    auto_face_visible_enemy = True


class FnaticV3AttackerController(FnaticV3FacingMixin, FnaticV1AttackerController):
    def __init__(self, positions=None):
        self.positions = positions if positions is not None else TacticalPositions()
        super().__init__()

    def reset_round(self):
        super().reset_round()
        self.guard_anchor = None
        self.guard_targets = {}
        self.formation = FnaticFormation()

    def decide_move(self, char, state):
        self.positions.validate(state['grid'])
        return super().decide_move(char, state)

    def _publish_target(self):
        if self.game is not None:
            # Our own tactical intent is shared, irrespective of IQ coordinate noise.
            owner = getattr(self.game, 'real_game', self.game)
            owner.target_plant_pos = self.target
            self.game.target_plant_pos = self.target

    def _attack(self, char, state, grid, allies, blocked, plants, planted, visible, risks):
        if planted is not None:
            # The planting team knows where it planted its own spike.
            owner = getattr(self.game, 'real_game', self.game)
            anchor = getattr(owner, 'planted_pos', None) if owner is not None else None
            anchor = tuple(map(int, anchor if anchor is not None else planted))
            return self._postplant(char, anchor, grid, allies, blocked, visible)

        holder = next((c for c in allies if getattr(c, 'has_spike', False)), None)
        if holder is not None:
            registered = self.positions.plant_cells
            if self.target not in registered:
                _, _, self.target = self._route(pos(holder), registered, grid)
            elif self._route(pos(holder), [self.target], grid)[2] is None:
                _, _, self.target = self._route(pos(holder), registered, grid)
            self._publish_target()
            if char.name == holder.name:
                if pos(char) in registered:
                    self.target = pos(char)
                    self._publish_target()
                    # Repeat PLANT until the game completes its required ticks.
                    return list(char.pos), 'PLANT'
                if self.target is None:
                    return self._result(char, pos(char))
            result = self.formation.result(self, char, holder, allies, grid, blocked, visible)
            return self._formation_utility(char, holder, grid, visible, result)
        dropped = state.get('spike_pos')
        if dropped is not None:
            # Recover with the main group; the separate Mid player stays Mid.
            anchor = tuple(map(int, dropped))
            eligible_mid = self.formation.mid_name
            if eligible_mid not in {c.name for c in allies}:
                eligible_mid = next((name for name in MID_PRIORITY
                                     if len(allies) > 1 and any(c.name == name for c in allies)), None)
            main = [c for c in allies if c.name != eligible_mid] or allies
            if self.retriever not in {c.name for c in main}:
                self.retriever = min(main, key=lambda c: (self._route(pos(c), [anchor], grid)[1], c.name)).name
            retriever = next(c for c in main if c.name == self.retriever)
            return self.formation.result(self, char, retriever, allies, grid, blocked,
                                         visible, destination=anchor)
        return super()._attack(char, state, grid, allies, blocked,
                               self.positions.plant_cells, planted, visible, risks)

    def _formation_utility(self, char, holder, grid, visible, result):
        # Formation movement takes priority, while the existing utility rules
        # remain available when a main-group player is holding its position.
        if (char.name in {holder.name, self.formation.mid_name}
                or tuple(result[0]) != pos(char)):
            return result
        if visible:
            return self._ability(char, pos(visible[0]), 'FLASH') or result
        if (self.target is not None and 3 <= max(abs(char.pos[0] - self.target[0]),
                                                abs(char.pos[1] - self.target[1])) <= 8
                and self._los(pos(char), self.target, grid, smoke=False)):
            for kind in ('RECON', 'SMOKE'):
                target = self.target
                if kind == 'SMOKE':
                    spawns = [tuple(map(int, p)) for p in zip(*np.where(grid == 4))]
                    for _ in range(3):
                        if spawns:
                            target, _, _ = self._route(target, spawns, grid)
                ability = self._ability(char, target, kind)
                if ability:
                    return ability
        return result

    def _assign_guards(self, anchor, grid, allies):
        if self.guard_anchor != anchor:
            self.guard_anchor = anchor
            self.guard_targets = {}
        alive = {c.name for c in allies}
        self.guard_targets = {name: cell for name, cell in self.guard_targets.items()
                              if name in alive}
        unassigned = [c for c in allies if c.name not in self.guard_targets]
        if not unassigned:
            return
        marker = self.positions.pattern_for(anchor, grid)
        if marker is None:
            return
        used = set(self.guard_targets.values())
        candidates = [p for p in self.positions.guard_candidates(marker, len(allies), grid)
                      if p not in used]
        lengths = [distances(pos(c), grid) for c in unassigned]
        registered = set(self.positions.guards[marker])

        @lru_cache(None)
        def match(index, mask):
            if index == len(unassigned):
                return (0, 0, ())
            choices = []
            for slot, cell in enumerate(candidates):
                if mask & (1 << slot) or cell not in lengths[index]:
                    continue
                rest = match(index + 1, mask | (1 << slot))
                if rest is not None:
                    choices.append((rest[0] + (cell not in registered),
                                    rest[1] + lengths[index][cell], (cell,) + rest[2]))
            return min(choices) if choices else None

        assignment = match(0, 0)
        if assignment is not None:
            self.guard_targets.update((c.name, cell)
                                      for c, cell in zip(unassigned, assignment[2]))

    def _postplant(self, char, anchor, grid, allies, blocked, visible):
        self._assign_guards(anchor, grid, allies)
        target = self.guard_targets.get(char.name, pos(char))
        nxt, _, _ = self._route(pos(char), [target], grid, blocked)
        if nxt == pos(char) and pos(char) != target:
            # A guard already holding a narrow passage can trap teammates.
            # Exchange their mapped destinations so the front player proceeds
            # beyond the passage and the following player takes over its guard.
            for ally in allies:
                cell = pos(ally)
                if (ally.name == char.name
                        or self.guard_targets.get(ally.name) != cell):
                    continue
                if self._route(pos(char), [target], grid, blocked - {cell})[2] is not None:
                    self.guard_targets[ally.name] = target
                    self.guard_targets[char.name] = cell
                    target = cell
                    nxt, _, _ = self._route(pos(char), [target], grid, blocked)
                    break
        defuser = next((c for c in visible if getattr(c, 'defuse_timer', 0) > 0), None)
        aim = pos(defuser) if defuser is not None else pos(visible[0]) if visible else anchor
        return self._result(char, nxt, aim)


class FnaticV3DefenderController(FnaticV3FacingMixin, FnaticV1DefenderController):
    """Use the basic rule-based defense until Fnatic v3 defense is specified."""
