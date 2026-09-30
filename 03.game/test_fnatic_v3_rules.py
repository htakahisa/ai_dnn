"""Fnatic v3 regressions for mapped planting and post-plant movement."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from fnatic_v3.controller import FnaticV3AttackerController, FnaticV3DefenderController
from fnatic_v3.positions import parse_grid
from game_core import PLANT_REQUIRED_TICKS
from iq_controller_adapter import IQAwareController
from map_data import NEW_MAZE_STR
from party_presets import get_preset
from test_ultimate_system import UltimateTestGame, make_character


def actor(name, position, **extra):
    return NS(name=name, team='A', pos=list(position), is_alive=True,
              facing='N', ability_name='NONE', has_spike=False, **extra)


class FnaticV3RulesTests(unittest.TestCase):
    def setUp(self):
        self.grid = parse_grid(NEW_MAZE_STR)
        self.ctrl = FnaticV3AttackerController(engineer_map='')

    def state(self, chars, **extra):
        return dict(grid=self.grid, chars=chars, round_timer=100, **extra)

    def test_carrier_passes_unregistered_plantable_cell(self):
        c = actor('carrier', (4, 39))
        c.has_spike = True
        result = self.ctrl.decide_move(c, self.state([c]))
        self.assertNotEqual(result[1], 'PLANT')
        self.assertNotEqual(result[0], c.pos)
        self.assertIn(self.ctrl.target, self.ctrl.positions.plant_cells)

    def test_deadline_keeps_registered_plant_requirement(self):
        c = actor('carrier', (4, 39))
        c.has_spike = True
        state = self.state([c])
        state['round_timer'] = 1
        self.assertNotEqual(self.ctrl.decide_move(c, state)[1], 'PLANT')

    def test_touyama_roster_selects_a_each_round_even_after_members_die(self):
        carrier = actor('carrier', (4, 39))
        carrier.has_spike = True
        enemies = [actor(name, (1, 17 + i)) for i, name in enumerate(
            reversed(get_preset('Touyama Gaming').players))]
        for enemy in enemies:
            enemy.team = 'D'
        # Prefer B whenever the selector offers it, proving that only A
        # candidates are passed to the random selection for this opponent.
        with patch('fnatic_v3.controller.random.choice', side_effect=lambda cells: max(cells)):
            for round_number in range(3):
                self.ctrl.reset_round()
                self.ctrl.target = next(p for p in self.ctrl.positions.plant_cells
                                        if p[1] > self.grid.shape[1] // 2)
                for enemy in enemies:
                    enemy.is_alive = round_number == 0
                self.ctrl.decide_move(carrier, self.state([carrier, *enemies]))
                self.assertTrue(self.ctrl.force_a_attack)
                self.assertLess(self.ctrl.target[1], self.grid.shape[1] // 2)

    def test_touyama_frontline_casualties_keep_attacking_a(self):
        carrier = actor('Leo', (23, 17))
        carrier.has_spike = True
        rear = actor('Boaster', (23, 18))
        mid = actor('Alfajer', (23, 19))
        owner = NS(defender_roster=list(get_preset('Touyama Gaming').players),
                   round_timer=100)
        self.ctrl.set_game(owner)
        target = next(p for p in self.ctrl.positions.plant_cells if p[1] < self.grid.shape[1] // 2)
        self.ctrl.target = target
        self.ctrl.formation.loss_context = (carrier.name, target)
        self.ctrl.formation.previous_frontline = {'Derke', 'Chronicle'}
        result = self.ctrl.decide_move(carrier, self.state([carrier, rear, mid]))
        self.assertTrue(self.ctrl.formation.exposed_after_losses)
        self.assertEqual(self.ctrl.target, target)
        self.assertNotEqual(result[0], carrier.pos)

    def test_other_rosters_use_random_sites_despite_touyama_team_label(self):
        carrier = actor('carrier', (23, 17))
        carrier.has_spike = True
        mixed = list(get_preset('Touyama Gaming').players)
        mixed[-1] = 'OtherPlayer'
        owner = NS(defender_roster=mixed, defender_team_name='Touyama Gaming', round_timer=100)
        self.ctrl.set_game(owner)
        with patch('fnatic_v3.controller.random.choice', side_effect=lambda cells: max(cells, key=lambda p: p[1])):
            self.ctrl.decide_move(carrier, self.state([carrier]))
        self.assertFalse(self.ctrl.force_a_attack)
        self.assertGreater(self.ctrl.target[1], self.grid.shape[1] // 2)

    def test_roster_is_rechecked_for_a_new_opponent(self):
        carrier = actor('carrier', (23, 17))
        carrier.has_spike = True
        owner = NS(defender_roster=list(get_preset('Touyama Gaming').players), round_timer=100)
        self.ctrl.set_game(owner)
        self.ctrl.decide_move(carrier, self.state([carrier]))
        self.assertTrue(self.ctrl.force_a_attack)
        self.ctrl.reset_round()
        owner.defender_roster = list(get_preset('Omoko Gaming').players)
        with patch('fnatic_v3.controller.random.choice', side_effect=lambda cells: max(cells, key=lambda p: p[1])):
            self.ctrl.decide_move(carrier, self.state([carrier]))
        self.assertFalse(self.ctrl.force_a_attack)
        self.assertGreater(self.ctrl.target[1], self.grid.shape[1] // 2)

    def test_iq_wrapper_plants_at_a_against_public_touyama_roster(self):
        carrier = make_character('Boaster', 'A', (4, 39))
        carrier.has_spike = True
        carrier.iq = carrier.effective_iq = 1
        enemies = [make_character(name, 'D', (1, 17 + i)) for i, name in enumerate(
            get_preset('Touyama Gaming').players)]
        game = self.game([carrier, *enemies])
        game.defender_roster = list(reversed(get_preset('Touyama Gaming').players))
        wrapper = IQAwareController(self.ctrl)
        wrapper.set_game(game)
        game.attacker_controller = wrapper
        for _ in range(120):
            game.move_character(carrier)
            game.battle_tick += 1
            if game.is_planted:
                break
        self.assertTrue(game.is_planted)
        self.assertLess(game.planted_pos[1], self.grid.shape[1] // 2)
        self.assertIn(game.planted_pos, self.ctrl.positions.plant_cells)

    def test_all_registered_positions_continue_plant_despite_contact(self):
        for cell in self.ctrl.positions.plant_cells:
            with self.subTest(cell=cell):
                self.ctrl.target = cell
                c = actor('carrier', cell)
                c.has_spike = True
                enemy = actor('enemy', (4, 39))
                enemy.team = 'D'
                for _ in range(PLANT_REQUIRED_TICKS):
                    self.assertEqual(self.ctrl.decide_move(c, self.state([c, enemy])),
                                     (list(cell), 'PLANT'))

    def test_teammate_vacates_carrier_plant_target(self):
        target = self.ctrl.positions.plants[7][0]
        self.ctrl.target = target
        carrier = actor('carrier', (4, 39))
        carrier.has_spike = True
        teammate = actor('teammate', target)
        result = self.ctrl.decide_move(teammate, self.state([carrier, teammate]))
        self.assertNotEqual(result[0], teammate.pos)
        self.assertNotEqual(result[0], carrier.pos)

    def test_guard_patterns_have_distinct_stable_assignments(self):
        for marker, plants in self.ctrl.positions.plants.items():
            with self.subTest(marker=marker):
                self.ctrl.reset_round()
                chars = [actor(str(i), p) for i, p in enumerate(
                    ((23, 17), (23, 18), (23, 19), (23, 20), (23, 21)))]
                state = self.state(chars, is_planted=True, planted_pos=plants[0])
                self.ctrl.decide_move(chars[0], state)
                targets = dict(self.ctrl.guard_targets)
                self.assertEqual(len(set(targets.values())), 5)
                registered = set(self.ctrl.positions.guards[marker])
                self.assertEqual(len(set(targets.values()) & registered),
                                 min(5, len(registered)))
                for c in reversed(chars):
                    self.ctrl.decide_move(c, state)
                self.assertEqual(self.ctrl.guard_targets, targets)
                c = chars[0]
                c.pos = list(targets[c.name])
                self.assertEqual(self.ctrl.decide_move(c, state)[0], c.pos)

    def test_dead_teammate_does_not_change_survivor_targets(self):
        chars = [actor(str(i), p) for i, p in enumerate(
            ((23, 17), (23, 18), (23, 19), (23, 20), (23, 21)))]
        state = self.state(chars, is_planted=True,
                           planted_pos=self.ctrl.positions.plants[7][0])
        self.ctrl.decide_move(chars[0], state)
        old = dict(self.ctrl.guard_targets)
        chars[0].is_alive = False
        self.ctrl.decide_move(chars[1], state)
        self.assertEqual(self.ctrl.guard_targets, {k: v for k, v in old.items() if k != '0'})

    def test_arrived_guard_exchanges_targets_to_clear_narrow_passage(self):
        follower = actor('follower', (14, 40))
        front = actor('front', (12, 40))
        side = actor('side', (7, 37))
        top = actor('top', (2, 37))
        anchor = self.ctrl.positions.plants[7][0]
        self.ctrl.guard_anchor = anchor
        self.ctrl.guard_targets = {'follower': (11, 40), 'front': (12, 40),
                                   'side': (7, 37), 'top': (2, 37)}
        self.ctrl.decide_move(follower, self.state([follower, front, side, top],
                              is_planted=True, planted_pos=anchor))
        self.assertEqual(self.ctrl.guard_targets,
                         {'follower': (12, 40), 'front': (11, 40),
                          'side': (7, 37), 'top': (2, 37)})

    def test_round_reset_discards_plant_and_guard_decisions(self):
        c = actor('carrier', (23, 17))
        c.has_spike = True
        self.ctrl.decide_move(c, self.state([c]))
        self.ctrl.decide_move(c, self.state([c], is_planted=True,
                                           planted_pos=self.ctrl.positions.plants[5][0]))
        self.ctrl.reset_round()
        self.assertIsNone(self.ctrl.target)
        self.assertIsNone(self.ctrl.guard_anchor)
        self.assertEqual(self.ctrl.guard_targets, {})

    def game(self, chars):
        game = UltimateTestGame(*self.grid.shape)
        game.grid = self.grid.copy()
        game.chars = chars
        game.attacker_controller = self.ctrl
        self.ctrl.set_game(game)
        return game

    def test_real_battle_moves_plants_then_moves_carrier_to_guard(self):
        c = make_character('Boaster', 'A', (4, 39))
        c.has_spike = True
        game = self.game([c])
        for _ in range(80):
            game.move_character(c)
            game.battle_tick += 1
            if game.is_planted:
                break
        self.assertTrue(game.is_planted)
        self.assertIn(game.planted_pos, self.ctrl.positions.plant_cells)
        self.assertFalse(c.has_spike)
        for _ in range(80):
            game.move_character(c)
            game.battle_tick += 1
            if tuple(c.pos) == self.ctrl.guard_targets.get(c.name):
                break
        self.assertEqual(tuple(c.pos), self.ctrl.guard_targets[c.name])
        marker = self.ctrl.positions.pattern_for(game.planted_pos, game.grid)
        self.assertIn(tuple(c.pos), self.ctrl.positions.guards[marker])

    def test_real_battle_all_five_reach_guards_including_shortage_pattern(self):
        for marker in self.ctrl.positions.plants:
            with self.subTest(marker=marker):
                self.ctrl.reset_round()
                chars = [make_character(name, 'A', (23, 17 + i)) for i, name in enumerate(
                    ('Boaster', 'Derke', 'Leo', 'Chronicle', 'Alfajer'))]
                game = self.game(chars)
                game.is_planted = True
                game.planted_pos = self.ctrl.positions.plants[marker][0]
                for _ in range(120):
                    for c in chars:
                        game.move_character(c)
                    game.battle_tick += 1
                    self.assertEqual(len({tuple(c.pos) for c in chars}), 5)
                    if all(tuple(c.pos) == self.ctrl.guard_targets.get(c.name) for c in chars):
                        break
                self.assertTrue(all(tuple(c.pos) == self.ctrl.guard_targets.get(c.name)
                                    for c in chars), [(c.name, c.pos,
                                    self.ctrl.guard_targets.get(c.name)) for c in chars])

    def test_iq_plant_position_noise_keeps_same_guard_pattern(self):
        c = make_character('Boaster', 'A', (23, 17))
        game = self.game([c])
        game.is_planted = True
        game.planted_pos = self.ctrl.positions.plants[6][0]
        self.ctrl.set_game(NS(real_game=game))
        for noisy in ((7, 4), (8, 3), (9, 3)):
            self.ctrl.decide_move(c, self.state([c], is_planted=True, planted_pos=noisy))
            self.assertEqual(self.ctrl.guard_anchor, game.planted_pos)
            self.assertIn(self.ctrl.guard_targets[c.name], self.ctrl.positions.guards[6])

    def test_game_iq_wrapper_reaches_registered_plant_and_guard(self):
        c = make_character('Boaster', 'A', (4, 39))
        c.has_spike = True
        game = self.game([c])
        wrapper = IQAwareController(self.ctrl)
        wrapper.set_game(game)
        game.attacker_controller = wrapper
        for _ in range(120):
            game.move_character(c)
            game.battle_tick += 1
            if game.is_planted and tuple(c.pos) == self.ctrl.guard_targets.get(c.name):
                break
        self.assertTrue(game.is_planted)
        self.assertIn(game.planted_pos, self.ctrl.positions.plant_cells)
        self.assertEqual(tuple(c.pos), self.ctrl.guard_targets[c.name])

    def test_invalid_plant_marker_rejected(self):
        self.grid[self.ctrl.positions.plants[5][0]] = 0
        with self.assertRaisesRegex(ValueError, 'plantable'):
            self.ctrl.decide_move(actor('one', (23, 17)), self.state([]))

    def test_new_version_can_be_built_and_selected(self):
        from run_game import _build_team_ai
        from roster_select import TEAM_AI_OPTIONS
        from run_competition_manager import CONTROLLER_OPTIONS
        ai = _build_team_ai('fnatic_v3')
        self.assertEqual(ai.name, 'Fnatic v3')
        self.assertIsInstance(ai.get_attacker_controller().inner, FnaticV3AttackerController)
        self.assertIsInstance(ai.get_defender_controller().inner, FnaticV3DefenderController)
        self.assertEqual(TEAM_AI_OPTIONS['Fnatic v3'], 'fnatic_v3')
        self.assertEqual(CONTROLLER_OPTIONS['Fnatic v3'], 'fnatic_v3')


if __name__ == '__main__':
    unittest.main()
