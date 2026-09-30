"""Fnatic v3 rule controllers using independent tactical maps."""

from functools import lru_cache
import random

import numpy as np

from fnatic_v1_rules import FnaticV1AttackerController, FnaticV1DefenderController, pos
from game_core import PLANT_REQUIRED_TICKS, REVEAL_DURATION_TICKS
from party_presets import get_preset

from .positions import TacticalPositions, distances, region
from .formation import FnaticFormation, MID_PRIORITY, REGROUP_RADIUS, MID_CONTROL_POSITIONS
from .attacker_ability import FnaticEngineerRoute
from .smoke_recon import FnaticSmokeRecon
from .defender_positions import FnaticDefenderPositions
from .ultimate_tactics import FnaticUltimates
from .retake import FnaticRetake
from .orb_collection import FnaticOrbCollection
from .navigation import FnaticNavigation
from .reveal_peek import FnaticRevealPeek
from .recon_gate import FnaticReconGate
from .opponent_history import FnaticOpponentHistory


class FnaticV3FacingMixin:
    """Keep tactical facing outside the game's rule-owned combat aim."""

    def reset_round(self):
        if hasattr(self, 'opponent_history'):
            self.opponent_history.reset_round()
        else:
            self.opponent_history = FnaticOpponentHistory()
        super().reset_round()
        self.smoke_recon = FnaticSmokeRecon()
        self.ultimates = FnaticUltimates()
        self.orbs = FnaticOrbCollection()
        self.navigation = FnaticNavigation()
        self.reveal_peek = FnaticRevealPeek()
        self.recon_gate = FnaticReconGate()

    def set_game(self, game):
        super().set_game(game)
        self.opponent_history.bind(self)

    def record_opponent_round_end(self):
        self.opponent_history.finish(self)

    def decide_move(self, char, state):
        if self.side == 'A':
            self.opponent_history.observe(self, char, state)
        if getattr(char, 'ability_name', '') == 'RECON':
            self.recon_gate.begin(self, char, state)
        self.navigation.begin(char, state)
        try:
            result = self._decide_tactics(char, state)
            return self.navigation.finish(self, char, state, result)
        finally:
            self.navigation.context = None
            self.recon_gate.context = None

    def _ability(self, char, target, kind):
        if (kind == 'RECON' and getattr(char, 'ability_name', '') == 'RECON'
                and getattr(char, 'recon_charges', 0) > 0
                and not self.recon_gate.request(self, char)):
            return None
        return super()._ability(char, target, kind)

    def _recon_at_smoke(self, char, target, state, smoke):
        if not self.recon_gate.request(self, char, state, smoke):
            return None
        return super()._ability(char, target, 'RECON')

    def _retake_ability(self, char, target, kind):
        # The ordered retake explicitly spends the reserve at distinct sites;
        # normal contact/entry debouncing must not interrupt this sequence.
        return super()._ability(char, target, kind)

    def _route(self, start, goals, grid, blocked=(), risks=(), setup=False):
        goals = tuple(goals)
        result = super()._route(start, goals, grid, blocked, risks, setup)
        self.navigation.record(self, start, goals, blocked, result)
        return result

    def _decide_tactics(self, char, state):
        self.ultimates.sync(self, char)
        result = super().decide_move(char, state)
        action = result[1] if len(result) > 1 else None
        retake = getattr(self, 'retake', None)
        owner = getattr(self.game, 'real_game', self.game)
        if retake is not None and (retake.preparing or (
                retake.launched
                and retake.utility.launch_tick == int(getattr(owner, 'battle_tick', 0)))):
            return result
        if (state.get('defender_setup_active')
                or getattr(char, 'plant_timer', 0) > 0
                or getattr(char, 'defuse_timer', 0) > 0
                or (char.name == 'Alfajer'
                    and getattr(self, 'engineer_recovery_target', None) is not None)
                or (isinstance(action, str) and action in {'PLANT', 'DEFUSE'})):
            return result
        neon = self.ultimates.revealed_neon(self, char, state)
        if neon is not None:
            return neon
        if retake is not None and retake.gathering:
            # Stationary MONITOR can still launch after 20 ticks. Other
            # abilities must not take players away from the gathering slots.
            if getattr(char, 'ultimate_name', '') == 'MONITOR':
                ultimate = self.ultimates.result(self, char, state)
                if ultimate is not None:
                    return ultimate
            return result
        ultimate = self.ultimates.result(self, char, state)
        if (getattr(self, 'defuse_names', ())
                and getattr(self, 'defuse_responder', None) == char.name):
            # Keep running to the defuser. Once seen, offensive ultimates can
            # stop the retake; a post-kill RAID can still escape into cover.
            # MONITOR is allowed immediately after its round timing threshold.
            if ultimate is not None:
                kind = ultimate[1]['ultimate']
                defuser_seen = any(c.name in self.defuse_names and c.team != char.team
                                   and c.is_alive and self._defuser_in_sight(char, c, state['grid'])
                                   for c in state.get('chars', ()))
                if kind in {'RAID', 'MONITOR'} or (kind in {'TUNNEL', 'NEON'} and defuser_seen):
                    return ultimate
            return result
        if ultimate is not None:
            return ultimate
        peek = self.reveal_peek.result(self, char, state)
        if peek is not None:
            return peek
        orb = self.orbs.result(self, char, state)
        if orb is not None:
            return orb
        if retake is not None and retake.launched and state['grid'][pos(char)] != 2:
            return result
        response = self.smoke_recon.result(self, char, state)
        if response is not None:
            return response
        tick = int(getattr(self.game, 'battle_tick', 0))
        if (isinstance(action, dict) and action.get('ability') == 'RECON'
                and not self.smoke_recon.normal_cast_ready(char, tick)):
            return self._result(char, pos(char))
        return result


class FnaticV3AttackerController(FnaticV3FacingMixin, FnaticV1AttackerController):
    def __init__(self, positions=None, engineer_map=None):
        self.positions = positions if positions is not None else TacticalPositions()
        self.engineer = FnaticEngineerRoute(engineer_map)
        super().__init__()

    def reset_round(self):
        super().reset_round()
        self.guard_anchor = None
        self.guard_targets = {}
        self.formation = FnaticFormation()
        self.engineer.reset_round()
        self.utility_pending = {}
        self.utility_last_use = {}
        self.defuse_names = ()
        self.defuse_responder = None
        self.force_a_attack = False
        self.engineer_recovery_target = None
        self.attack_region = None
        self.mid_entry_goal = None
        self.mid_entry_done = False

    def decide_move(self, char, state):
        self.positions.validate(state['grid'])
        self.engineer.validate(state['grid'])
        # Rosters are public before contact; use all members, including the
        # dead, rather than the current player's IQ-filtered enemy list.
        owner = getattr(self.game, 'real_game', self.game)
        roster = getattr(owner, 'defender_roster', None)
        if not roster:
            roster = [c.name for c in getattr(owner, 'chars', state.get('chars', ()))
                      if c.team == 'D']
        touyama = get_preset('Touyama Gaming').players
        self.force_a_attack = len(roster) == len(touyama) and set(roster) == set(touyama)
        return super().decide_move(char, state)

    def _plant_candidates(self, grid):
        registered = self.positions.plant_cells
        if self.force_a_attack:
            return tuple(p for p in registered if p[1] < grid.shape[1] // 2)
        return registered

    def _publish_target(self):
        if self.game is not None:
            # Our own tactical intent is shared, irrespective of IQ coordinate noise.
            owner = getattr(self.game, 'real_game', self.game)
            owner.target_plant_pos = self.target
            self.game.target_plant_pos = self.target

    def _rotate_target(self, holder, grid):
        """Select a reachable registered plant on the opposite map side."""
        if self.target is None or self.force_a_attack:
            return False
        left = self.target[1] < grid.shape[1] // 2
        reachable = distances(pos(holder), grid)
        candidates = [cell for cell in self.positions.plant_cells if cell in reachable
                      and (cell[1] < grid.shape[1] // 2) != left]
        if not candidates:
            return False
        self.target = random.choice(candidates)
        self.attack_region = region(self.target, grid)
        self.mid_entry_goal = None
        self.mid_entry_done = True
        self._publish_target()
        return True

    def _select_attack_target(self, candidates, lengths, grid):
        available = {region(p, grid) for p in candidates}
        mid = [p for p in MID_CONTROL_POSITIONS if p in lengths and region(p, grid) == 'MID']
        if not mid:
            mid = [p for p in lengths if region(p, grid) == 'MID' and grid[p] != 2]
        if mid and not self.force_a_attack:
            available.add('MID')
        selected = 'A' if self.force_a_attack else self.opponent_history.attack_region(sorted(available))
        pool = [p for p in candidates if region(p, grid) == selected] if selected in {'A', 'B'} else candidates
        self.target = random.choice(pool or candidates)
        self.attack_region = selected or region(self.target, grid)
        self.mid_entry_goal = None
        self.mid_entry_done = False
        if self.attack_region == 'MID':
            to_site = distances(self.target, grid)
            self.mid_entry_goal = min(mid, key=lambda p: (lengths[p] + to_site.get(p, float('inf')),
                                                        abs(p[0] - grid.shape[0] // 2), p))
        return self.target

    def _attack(self, char, state, grid, allies, blocked, plants, planted, visible, risks):
        owner = getattr(self.game, 'real_game', self.game)
        traps = getattr(owner, 'ramp_traps', state.get('ramp_traps', ()))
        engineer = next((getattr(c, 'real_character', c) for c in allies
                         if c.name == 'Alfajer' and c.ability_name == 'RAMP'), None)
        configured = engineer is not None and bool(self.engineer.candidates)
        self.formation.mid_exclusions = {'Alfajer'} if configured else set()
        active = configured and self.engineer.has_work(engineer, grid, traps)
        holder = next((getattr(c, 'real_character', c) for c in allies
                       if getattr(c, 'has_spike', False)), None)
        if self._urgent_engineer_recovery(state, grid, allies, holder, planted):
            active = False
        engineer_mid = (configured and not active and planted is None
                        and not engineer.has_spike
                        and holder is not None
                        and distances(pos(holder), grid).get(pos(engineer), float('inf')) > REGROUP_RADIUS)
        self.formation.detached_names = ({'Alfajer'} if (active or engineer_mid)
                                         and not engineer.has_spike else set())
        if not engineer_mid:
            self.engineer.mid_target = None
        if planted is not None:
            # Own planted spike coordinates and the public defuse tap are
            # shared even when IQ perception cannot currently see the enemy.
            anchor = getattr(owner, 'planted_pos', None)
            anchor = tuple(map(int, anchor if anchor is not None else planted))
            response = self._defuse_response(char, state, anchor, grid, allies, visible)
            if response is not None:
                return response
        else:
            self.defuse_names = ()
            self.defuse_responder = None
        if active and char.name == engineer.name:
            actual_allies = [getattr(c, 'real_character', c) for c in allies]
            occupied = {pos(c) for c in actual_allies if c.name != char.name}
            return self.engineer.result(self, char, grid, occupied, visible, traps)
        if engineer_mid and char.name == engineer.name:
            actual_allies = [getattr(c, 'real_character', c) for c in allies]
            if holder is not None:
                self.formation.assign_mid(actual_allies, holder)
            occupied = {pos(c) for c in actual_allies if c.name != char.name}
            occupied.update(pos(c) for c in visible)
            return self.engineer.mid_result(self, char, grid, occupied, visible)
        if planted is not None:
            guards = [c for c in allies if c.name not in self.formation.detached_names]
            result = self._postplant(char, anchor, grid, guards, blocked, visible)
            return self._attacker_utility(char, None, grid, visible, result, anchor, True)

        holder = next((c for c in allies if getattr(c, 'has_spike', False)), None)
        if holder is not None:
            registered = self._plant_candidates(grid)
            actual_holder = getattr(holder, 'real_character', holder)
            reachable = distances(pos(actual_holder), grid)
            if self.target not in registered or self.target not in reachable:
                candidates = [cell for cell in registered if cell in reachable]
                self.target = self._select_attack_target(candidates, reachable, grid) if candidates else None
            self._publish_target()
            if char.name == holder.name:
                if self.target is not None and pos(char) == self.target:
                    # Repeat PLANT until the game completes its required ticks.
                    return list(char.pos), 'PLANT'
                if self.target is None:
                    return self._result(char, pos(char))
            round_timer = getattr(owner, 'round_timer', state.get('round_timer', 100))
            if (self.attack_region == 'MID' and not self.mid_entry_done
                    and self.engineer_recovery_target is None):
                if pos(actual_holder) == self.mid_entry_goal:
                    self.mid_entry_done = True
                else:
                    return self.formation.result(self, char, holder, allies, grid, blocked, visible,
                                                 destination=self.mid_entry_goal, round_timer=round_timer,
                                                 transit=True)
            result = self.formation.result(self, char, holder, allies, grid, blocked, visible,
                                           round_timer=round_timer)
            return self._formation_utility(char, holder, grid, visible, result)
        dropped = state.get('spike_pos')
        if dropped is not None:
            # Recover with the main group; the separate Mid player stays Mid.
            anchor = tuple(map(int, dropped))
            eligible_mid = self.formation.mid_name
            if eligible_mid not in {c.name for c in allies}:
                eligible_mid = next((name for name in MID_PRIORITY
                                     if name not in self.formation.mid_exclusions
                                     and len(allies) > 1 and any(c.name == name for c in allies)), None)
            main = [c for c in allies if c.name != eligible_mid
                    and c.name not in self.formation.detached_names] or allies
            if self.retriever not in {c.name for c in main}:
                self.retriever = min(main, key=lambda c: (self._route(pos(c), [anchor], grid)[1], c.name)).name
            retriever = next(c for c in main if c.name == self.retriever)
            return self.formation.result(self, char, retriever, allies, grid, blocked,
                                         visible, destination=anchor)
        return super()._attack(char, state, grid, allies, blocked,
                               self._plant_candidates(grid), planted, visible, risks)

    def _urgent_engineer_recovery(self, state, grid, allies, holder, planted):
        if (planted is not None or len(allies) != 1 or allies[0].name != 'Alfajer'
                or getattr(allies[0], 'ability_name', '') != 'RAMP'):
            self.engineer_recovery_target = None
            return False
        engineer = getattr(allies[0], 'real_character', allies[0])
        registered = self._plant_candidates(grid)
        if holder is not None:
            # Continue the urgent recovery through planting, even though
            # automatic pickup has removed spike_pos from the next state.
            if self.engineer_recovery_target is None:
                return False
            lengths = distances(pos(engineer), grid)
        else:
            dropped = state.get('spike_pos')
            if dropped is None:
                self.engineer_recovery_target = None
                return False
            anchor = tuple(map(int, dropped))
            lengths = distances(anchor, grid)
            pickup = lengths.get(pos(engineer))
            candidates = [p for p in registered if p in lengths]
            if pickup is None or not candidates:
                self.engineer_recovery_target = None
                return False
            nearest = min(candidates, key=lambda p: (lengths[p], p))
            owner = getattr(self.game, 'real_game', self.game)
            remaining = getattr(owner, 'round_timer', state.get('round_timer', 100))
            # Pickup is automatic after movement. Already standing on a
            # dropped spike still needs one tick for that pickup to resolve.
            minimum = max(1, pickup) + lengths[nearest] + PLANT_REQUIRED_TICKS
            if self.engineer_recovery_target is None and remaining >= minimum + 5:
                return False
            self.engineer_recovery_target = nearest
        if self.engineer_recovery_target not in registered or self.engineer_recovery_target not in lengths:
            candidates = [p for p in registered if p in lengths]
            self.engineer_recovery_target = min(candidates, key=lambda p: (lengths[p], p)) if candidates else None
        if self.engineer_recovery_target is None:
            return False
        self.target = self.engineer_recovery_target
        self._publish_target()
        return True

    def _defuse_response(self, char, state, anchor, grid, allies, visible):
        info = state.get('defender_defuse_info') or {}
        names = tuple(sorted(name for name, (timer, _) in info.items() if timer > 0))
        if names != self.defuse_names:
            self.defuse_names = names
            self.defuse_responder = None
        if not names:
            return None
        # Distance uses shared real ally positions; enemy coordinates remain
        # limited to this player's own observation, not the tap notification.
        actual_allies = [getattr(c, 'real_character', c) for c in allies]
        lengths = distances(anchor, grid)
        eligible = {c.name: c for c in actual_allies if pos(c) in lengths and c.is_alive}
        if self.defuse_responder not in eligible:
            nearest = min(eligible.values(), key=lambda c: (lengths[pos(c)], c.name), default=None)
            self.defuse_responder = nearest.name if nearest is not None else None
        if char.name != self.defuse_responder:
            return None
        defuser = next((enemy for enemy in state.get('chars', ()) if enemy.name in names
                        and enemy.team != char.team and enemy.is_alive
                        and getattr(enemy, 'defuse_timer', 0) > 0
                        and self._defuser_in_sight(char, enemy, grid)), None)
        if defuser is not None:
            return self._result(char, pos(char), pos(defuser))
        occupied = {pos(c) for c in actual_allies if c.name != char.name}
        occupied.update(pos(c) for c in visible if c.is_alive)
        # A defuser farther along a narrow approach must not force a detour
        # before we can see them. Only this tick's next step must be empty.
        nxt, _, goal = self._route(pos(char), [anchor], grid)
        if nxt in occupied:
            nxt, _, goal = self._route(pos(char), [anchor], grid, occupied)
        if goal is None:
            # The spike itself can be occupied. Reach its adjacent defuse
            # area without trying to overlap another player.
            adjacent = [p for p in lengths if max(abs(p[0] - anchor[0]), abs(p[1] - anchor[1])) <= 1
                        and p not in occupied]
            nxt, _, _ = self._route(pos(char), adjacent, grid, occupied)
        return self._result(char, nxt, anchor)

    def _defuser_in_sight(self, char, enemy, grid):
        shot_los = getattr(self.game, 'check_shot_line_of_sight', None)
        if shot_los is not None:
            return shot_los(char, enemy)
        ignore_smoke = (getattr(char, 'sees_through_smoke', False)
                        or getattr(enemy, 'reveal_remaining', 0) > 0)
        return self._los(pos(char), pos(enemy), grid, smoke=not ignore_smoke)

    def _formation_utility(self, char, holder, grid, visible, result):
        return self._attacker_utility(char, holder, grid, visible, result, self.target, False)

    def _attacker_utility(self, char, holder, grid, visible, result, anchor, planted):
        # A cast occupies one movement tick. Roles may cast while advancing,
        # but planting must not be interrupted.
        if (anchor is None
                or getattr(char, 'plant_timer', 0) > 0
                or (len(result) > 1 and result[1] == 'PLANT')
                or char.name in self.formation.detached_names):
            return result
        if len(result) > 1 and isinstance(result[1], dict) and 'ability' in result[1]:
            return result
        kind = getattr(char, 'ability_name', '')
        if kind not in {'FLASH', 'RECON', 'SMOKE'}:
            return result
        charges = int(getattr(char, kind.lower() + '_charges', 0))
        tick = int(getattr(self.game, 'battle_tick', 0))
        pending = self.utility_pending.get(char.name)
        if pending is not None and charges < pending[0]:
            self.utility_last_use[char.name] = pending[1]
            del self.utility_pending[char.name]
        if charges <= 0:
            return result
        owner = getattr(self.game, 'real_game', self.game)
        enemies = sorted((c for c in visible if pos(c) != pos(char)
                          and self._utility_target_valid(pos(c), grid)),
                         key=lambda c: (max(abs(c.pos[0] - char.pos[0]),
                                            abs(c.pos[1] - char.pos[1])), c.name))
        site_lengths = distances(anchor, grid)
        near_site = site_lengths.get(pos(char), float('inf')) <= 10
        target = None
        if kind == 'FLASH':
            target = next((pos(c) for c in enemies
                           if max(abs(c.pos[0] - char.pos[0]), abs(c.pos[1] - char.pos[1])) <= 12
                           and self._los(pos(char), pos(c), grid, smoke=False)), None)
            if target is None and not planted and near_site and char.name != self.formation.mid_name:
                target = self._utility_site_target(char, anchor, grid)
        elif kind == 'RECON':
            # Preserve the second charge while the first projectile/reveal is
            # still useful, rather than throwing both on consecutive ticks.
            last = self.utility_last_use.get(char.name)
            in_flight = any(p.get('owner') == char.name
                            for p in getattr(owner, 'recon_projectiles', ()))
            if in_flight or (last is not None and tick - last < REVEAL_DURATION_TICKS + 3):
                return result
            target = next((pos(c) for c in enemies if getattr(c, 'reveal_remaining', 0) <= 0
                           and self._los(pos(char), pos(c), grid, smoke=False)), None)
            if target is None and near_site:
                target = self._utility_site_target(char, anchor, grid)
        else:
            # Smoke can be placed remotely by the Mid player to support entry.
            actual_holder = getattr(holder, 'real_character', holder)
            entering = planted or (actual_holder is not None
                                   and site_lengths.get(pos(actual_holder), float('inf')) <= 8)
            if entering or enemies:
                target = self._utility_smoke_target(char, anchor, grid, enemies, owner)
        if target is None:
            return result
        if kind in {'FLASH', 'RECON'}:
            path_builder = getattr(self.game, '_projectile_path', None)
            if path_builder is not None and len(path_builder(pos(char), target)) <= 1:
                return result
        ability = self._ability(char, target, kind)
        if ability is not None:
            self.utility_pending[char.name] = (charges, tick)
            return ability
        return result

    @staticmethod
    def _utility_target_valid(cell, grid):
        return (0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]
                and grid[cell] != 1)

    def _utility_site_target(self, char, anchor, grid):
        # At a corner the exact plant may be hidden; a visible cell in the
        # same site is a useful throw direction without aiming into a wall.
        lengths = distances(anchor, grid)
        candidates = [p for p, length in lengths.items() if length <= 3 and p != pos(char)
                      and self._los(pos(char), p, grid, smoke=False)]
        return min(candidates, key=lambda p: (grid[p] != 2, lengths[p],
                                              max(abs(p[0] - char.pos[0]), abs(p[1] - char.pos[1])), p),
                   default=None)

    def _utility_smoke_target(self, char, anchor, grid, enemies, owner):
        allies = [getattr(c, 'real_character', c) for c in getattr(owner, 'chars', ())
                  if c.team == char.team and c.is_alive]
        protected = {anchor, pos(char), *self.formation.entry_targets.values()}
        protected.update(pos(c) for c in allies)
        def safe(cell):
            return self._utility_target_valid(cell, grid) and all(
                max(abs(cell[0] - p[0]), abs(cell[1] - p[1])) > 1 for p in protected)
        for enemy in enemies:
            if safe(pos(enemy)):
                return pos(enemy)
        spawns = [tuple(map(int, p)) for p in zip(*np.where(grid == 4))]
        target = anchor
        for step in range(6):
            if not spawns:
                break
            target, _, goal = self._route(target, spawns, grid)
            if goal is None:
                break
            if step >= 2 and safe(target):
                return target
        return None

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
    """Mapped defense, synchronized retakes and enemy-smoke recon."""

    def __init__(self, position_map=None, retake_map=None, engineer_map=None):
        self.defender_positions = FnaticDefenderPositions(position_map)
        self.retake = FnaticRetake(retake_map)
        if engineer_map is None:
            from .map_data_defender_ability_fnatic import ENGINEER_LAMP_STR
            engineer_map = ENGINEER_LAMP_STR
        self.engineer = FnaticEngineerRoute(engineer_map)
        super().__init__()

    def reset_round(self):
        super().reset_round()
        self.defender_positions.reset_round()
        self.retake.reset_round()
        self.engineer.reset_round()

    def decide_move(self, char, state):
        grid = np.asarray(state['grid'])
        self.defender_positions.validate(grid)
        self.retake.validate(grid)
        self.engineer.validate(grid)
        self.opponent_history.observe(self, char, state)
        self.defender_positions.preferred_engineer_region = self.opponent_history.defender_site()
        dropped = (state.get('spike_pos') if not state.get('is_planted')
                   and not state.get('defender_setup_active') else None)
        self.defender_positions.focus_drop(dropped, grid)
        owner = getattr(self.game, 'real_game', self.game)
        traps = getattr(owner, 'ramp_traps', state.get('ramp_traps', ()))
        engineer = next((getattr(c, 'real_character', c) for c in state.get('chars', ())
                         if c.team == char.team and c.is_alive and c.name == 'Alfajer'), None)
        self.defender_positions.cover_traps(traps,
            [c for c in state.get('chars', ()) if c.team == char.team], grid)
        self.defender_positions.reserved = (
            set(self.engineer.available(engineer, grid, traps))
            if not state.get('is_planted') and engineer is not None
            and self.engineer.has_work(engineer, grid, traps) else set())
        if self.defender_positions.reserved and self.engineer.target is not None:
            setup = bool(state.get('defender_setup_active'))
            nxt, direct_length, _ = self._route(pos(engineer), [self.engineer.target], grid, setup=setup)
            occupied = {pos(getattr(c, 'real_character', c)) for c in state.get('chars', ())
                        if c.team == char.team and c.is_alive and c.name != engineer.name}
            if nxt in occupied:
                _, length, reachable = self._route(pos(engineer), [self.engineer.target], grid,
                                                   occupied, setup=setup)
                if reachable is None or length > direct_length + 2:
                    if nxt in self.defender_positions.targets.values():
                        self.defender_positions.reserved.add(nxt)
        if not state.get('is_planted'):
            self.retake.reset_round()
        main_changed = self.defender_positions.observe_main_attack(self, char, state)
        if self.defender_positions.main_region is not None and not state.get('is_planted'):
            # Reinforcing the confirmed attack takes priority over lamp chores.
            self.defender_positions.reserved.clear()
        if main_changed:
            allies = [c for c in state.get('chars', ()) if c.team == char.team and c.is_alive]
            self.defender_positions.goal(char, allies, grid)
        if state.get('defender_setup_active'):
            self.navigation.begin(char, state)
            allies = [c for c in state.get('chars', ()) if c.team == char.team and c.is_alive]
            result = self._engineer_position(char, state, grid, allies, [], setup=True)
            if result is not None:
                return self.navigation.finish(self, char, state, result)
            result = self._mapped_position(char, allies, grid, [], setup=True)
            if result is not None:
                return self.navigation.finish(self, char, state, result)
            self.navigation.context = None
        return super().decide_move(char, state)

    def _defend(self, char, state, grid, allies, blocked, plants, planted, visible, risks):
        if planted is not None:
            if getattr(char, 'defuse_timer', 0) > 0:
                return list(char.pos), 'DEFUSE'
            owner = getattr(self.game, 'real_game', self.game)
            remaining = getattr(owner, 'detonate_timer', state.get('detonate_timer'))
            result = self.retake.result(self, char, planted, grid, allies, visible,
                                        remaining_ticks=remaining, state=state)
            if result is not None:
                return result
            if self.retake.launched and max(abs(char.pos[i] - planted[i]) for i in (0, 1)) > 1:
                return self.retake.push(self, char, planted, grid, allies, visible)
        if planted is not None or visible:
            return super()._defend(char, state, grid, allies, blocked, plants, planted, visible, risks)
        if self.defender_positions.main_region is not None:
            result = self._mapped_position(char, allies, grid, visible)
            if result is not None:
                return result
        result = self._engineer_position(char, state, grid, allies, visible)
        if result is not None:
            return result
        if self.defender_positions.drop_region is not None:
            result = self._mapped_position(char, allies, grid, visible)
            if result is not None:
                return result
        support = self._cover_engagement(char, allies, risks, grid, blocked)
        if support is not None:
            return support
        result = self._mapped_position(char, allies, grid, visible)
        if result is not None:
            return result
        return super()._defend(char, state, grid, allies, blocked, plants, planted, visible, risks)

    def _engineer_position(self, char, state, grid, allies, visible, setup=False):
        if char.name != 'Alfajer' or char.ability_name != 'RAMP' or state.get('is_planted'):
            return None
        owner = getattr(self.game, 'real_game', self.game)
        traps = getattr(owner, 'ramp_traps', state.get('ramp_traps', ()))
        if not self.engineer.has_work(char, grid, traps):
            return None
        occupied = {pos(getattr(c, 'real_character', c)) for c in allies if c.name != char.name}
        occupied.update(pos(c) for c in visible if c.is_alive)
        return self.engineer.result(self, char, grid, occupied, visible, traps, setup=setup)

    def _mapped_position(self, char, allies, grid, visible, setup=False):
        goal = self.defender_positions.goal(char, allies, grid, setup)
        if goal is None:
            return None
        actual = [getattr(c, 'real_character', c) for c in allies]
        occupied = {pos(c) for c in actual if c.name != char.name}
        occupied.update(pos(c) for c in visible if c.is_alive)
        nxt, direct_length, _ = self._route(pos(char), [goal], grid, setup=setup)
        if nxt in occupied:
            direct_next = nxt
            nxt, length, reachable = self._route(pos(char), [goal], grid, occupied, setup=setup)
            if reachable is None or length > direct_length + 2:
                blocker = next((c for c in actual if pos(c) == direct_next
                                and self.defender_positions.targets.get(c.name) == pos(c)), None)
                if blocker is not None:
                    # Shared slots can be exchanged to clear a one-cell lane.
                    self.defender_positions.targets[blocker.name] = goal
                    goal = self.defender_positions.targets[char.name] = pos(blocker)
                    nxt = pos(char)
        return self._result(char, nxt, goal)
