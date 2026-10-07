"""Fnatic front-player selection, main-group cohesion and Mid separation."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from fnatic_v3.controller import FnaticV3AttackerController
from fnatic_v3.formation import MID_CONTROL_POSITIONS, MAIN_GROUP_RADIUS, SITE_ENTRY_RADIUS, FRONT_SELECTION_RADIUS
from fnatic_v3.positions import distances, parse_grid
from map_data import NEW_MAZE_STR
from test_fnatic_v3_rules import actor
from test_ultimate_system import UltimateTestGame, make_character
from iq_controller_adapter import IQAwareController


class FnaticV3FormationTests(unittest.TestCase):
    def setUp(self):
        self.ctrl = FnaticV3AttackerController(engineer_map='')
        self.grid = np.zeros((9, 24), dtype=np.int32)
        self.ctrl.target = (4, 22)
        self.carrier = actor('carrier', (4, 4))
        self.carrier.has_spike = True

    def decide(self, char, allies, round_timer=100):
        alive = [c for c in allies if c.is_alive]
        blocked = {tuple(c.pos) for c in alive if c.name != char.name}
        return self.ctrl.formation.result(self.ctrl, char, self.carrier,
                                          alive, self.grid, blocked, [], round_timer=round_timer)

    def test_front_priority_uses_nearby_living_players_only(self):
        derke = actor('Derke', (4, 12))  # Eight cells away; no front assignment.
        chronicle = actor('Chronicle', (4, 6))
        leo = actor('Leo', (3, 6))
        allies = [self.carrier, derke, chronicle, leo]
        self.decide(self.carrier, allies)
        self.assertEqual(self.ctrl.formation.front_name, 'Chronicle')
        derke.pos = [4, 11]  # Exactly seven cells is eligible.
        self.decide(self.carrier, allies)
        self.assertEqual(self.ctrl.formation.front_name, 'Derke')
        derke.is_alive = False
        chronicle.is_alive = False
        self.decide(self.carrier, allies)
        self.assertEqual(self.ctrl.formation.front_name, 'Leo')

    def test_all_five_front_priorities_in_order(self):
        for expected, names in (
            ('Derke', ('Derke', 'Chronicle', 'Leo', 'Boaster', 'Alfajer')),
            ('Chronicle', ('Chronicle', 'Leo', 'Boaster', 'Alfajer')),
            ('Leo', ('Leo', 'Boaster', 'Alfajer')),
            ('Boaster', ('Boaster', 'Alfajer')),
            ('Alfajer', ('Alfajer',)),
        ):
            with self.subTest(expected=expected):
                self.ctrl.reset_round()
                self.ctrl.target = (4, 22)
                allies = [self.carrier] + [actor(name, (3, 4 + i)) for i, name in enumerate(names)]
                # A unit already serving in a separate group is excluded;
                # here suppress Mid so the final two priorities can be tested.
                self.ctrl.formation.mid_name = None
                with patch('fnatic_v3.formation.MID_PRIORITY', ()):
                    self.decide(self.carrier, allies)
                self.assertEqual(self.ctrl.formation.front_name, expected)

    def test_mid_uses_alfajer_then_boaster_and_never_carrier(self):
        alfajer = actor('Alfajer', (3, 4))
        boaster = actor('Boaster', (5, 4))
        allies = [self.carrier, alfajer, boaster]
        self.decide(self.carrier, allies)
        self.assertEqual(self.ctrl.formation.mid_name, 'Alfajer')
        alfajer.is_alive = False
        self.decide(self.carrier, allies)
        self.assertEqual(self.ctrl.formation.mid_name, 'Boaster')
        self.carrier.name = 'Boaster'
        self.decide(self.carrier, [self.carrier])
        self.assertIsNone(self.ctrl.formation.mid_name)

    def test_separate_mid_is_not_recalled_as_front_even_when_nearby(self):
        alfajer = actor('Alfajer', (4, 6))
        self.decide(self.carrier, [self.carrier, alfajer])
        self.assertEqual(self.ctrl.formation.mid_name, 'Alfajer')
        self.assertIsNone(self.ctrl.formation.front_name)

    def test_carrier_waits_until_nearby_front_gets_ahead(self):
        derke = actor('Derke', (4, 3))
        self.assertEqual(self.decide(self.carrier, [self.carrier, derke])[0], self.carrier.pos)
        for _ in range(10):
            derke.pos = self.decide(derke, [self.carrier, derke])[0]
            lengths = distances(self.ctrl.target, self.grid)
            if lengths[tuple(derke.pos)] <= lengths[tuple(self.carrier.pos)] - 2:
                break
        self.assertLessEqual(lengths[tuple(derke.pos)], lengths[tuple(self.carrier.pos)] - 2)
        self.assertEqual(self.decide(self.carrier, [self.carrier, derke])[0], [4, 5])

    def test_main_group_is_brought_near_carrier_instead_of_rushing_alone(self):
        derke = actor('Derke', (4, 6))
        chronicle = actor('Chronicle', (1, 12))
        allies = [self.carrier, derke, chronicle]
        self.assertEqual(self.decide(self.carrier, allies)[0], self.carrier.pos)
        before = distances(tuple(self.carrier.pos), self.grid)[tuple(chronicle.pos)]
        chronicle.pos = self.decide(chronicle, allies)[0]
        after = distances(tuple(self.carrier.pos), self.grid)[tuple(chronicle.pos)]
        self.assertLess(after, before)

    def test_regroup_wait_uses_twelve_step_boundary_on_approach_and_entry(self):
        for entering in (False, True):
            for walking, wait in ((12, True), (13, False)):
                with self.subTest(entering=entering, walking=walking):
                    self.ctrl.reset_round()
                    self.grid = np.zeros((9, 40), dtype=np.int32)
                    self.carrier.pos = [4, 16]
                    self.ctrl.target = (4, 22) if entering else (4, 35)
                    if entering:
                        self.grid[3:6, 21:24] = 2
                    derke = actor('Derke', (4, 22) if entering else (4, 18))
                    chronicle = actor('Chronicle', (1, 19 - walking))
                    self.assertEqual(distances(tuple(self.carrier.pos), self.grid)[tuple(chronicle.pos)], walking)
                    result = self.decide(self.carrier, [self.carrier, derke, chronicle])
                    self.assertEqual(result[0] == self.carrier.pos, wait)

    def test_wall_detour_over_twelve_steps_is_not_a_carrier_wait_target(self):
        self.grid = np.zeros((9, 28), dtype=np.int32)
        self.grid[:8, 10] = 1
        self.ctrl.target = (4, 25)
        chronicle = actor('Chronicle', (4, 11))
        self.assertEqual(distances(tuple(self.carrier.pos), self.grid)[tuple(chronicle.pos)], 15)
        result = self.decide(self.carrier, [self.carrier, chronicle])
        self.assertIn(chronicle.name, self.ctrl.formation.front_names)
        self.assertNotEqual(result[0], self.carrier.pos)

    def test_nearby_lurker_is_not_a_carrier_wait_target(self):
        derke = actor('Derke', (4, 6))
        lurker = actor('Alfajer', (1, 9))
        self.ctrl.formation.mid_exclusions = {'Alfajer'}
        self.ctrl.formation.detached_names = {'Alfajer'}
        result = self.decide(self.carrier, [self.carrier, derke, lurker])
        self.assertNotEqual(result[0], self.carrier.pos)
        self.assertNotIn(lurker.name, self.ctrl.formation.front_names)

    def test_old_passage_wait_for_remote_teammate_is_cancelled(self):
        derke = actor('Derke', (4, 6))
        remote = actor('Chronicle', (1, 17))
        self.ctrl.formation.passage_waits[self.carrier.name] = (
            remote.name, (1, 20), (1, 18), 100)
        result = self.decide(self.carrier, [self.carrier, derke, remote])
        self.assertNotIn(self.carrier.name, self.ctrl.formation.passage_waits)
        self.assertNotEqual(result[0], self.carrier.pos)

    def test_mid_stays_independent_in_both_rush_and_default(self):
        grid = parse_grid(NEW_MAZE_STR)
        for strategy in ('RUSH', 'DEFAULT'):
            with self.subTest(strategy=strategy):
                self.ctrl.reset_round()
                self.ctrl.strategy = strategy
                carrier = actor('Leo', (7, 3))
                carrier.has_spike = True
                alfajer = actor('Alfajer', (13, 23))
                state = dict(grid=grid, chars=[carrier, alfajer], round_timer=100)
                self.ctrl.decide_move(alfajer, state)
                self.assertEqual(self.ctrl.formation.mid_name, 'Alfajer')
                self.assertEqual(self.ctrl.formation.mid_target, (13, 23))
                self.assertEqual(self.ctrl.decide_move(alfajer, state)[0], alfajer.pos)
                self.assertIsNone(self.ctrl.formation.front_name)

    def test_mid_moves_to_registered_guard_after_plant(self):
        grid = parse_grid(NEW_MAZE_STR)
        carrier = actor('Leo', (7, 3))
        carrier.has_spike = True
        alfajer = actor('Alfajer', (13, 23))
        state = dict(grid=grid, chars=[carrier, alfajer], round_timer=100)
        self.ctrl.decide_move(alfajer, state)
        carrier.has_spike = False
        state.update(is_planted=True, planted_pos=(7, 3))
        self.ctrl.decide_move(alfajer, state)
        self.assertIn(self.ctrl.guard_targets['Alfajer'], self.ctrl.positions.guards[6])

    def test_dropped_spike_is_recovered_by_main_group_without_recalling_mid(self):
        grid = parse_grid(NEW_MAZE_STR)
        derke = actor('Derke', (10, 3))
        alfajer = actor('Alfajer', (13, 23))
        self.ctrl.formation.mid_name = 'Alfajer'
        self.ctrl.formation.mid_target = (13, 23)
        state = dict(grid=grid, chars=[derke, alfajer], round_timer=100, spike_pos=(9, 3))
        self.assertEqual(self.ctrl.decide_move(derke, state)[0], [9, 3])
        self.assertEqual(self.ctrl.retriever, 'Derke')
        self.assertEqual(self.ctrl.decide_move(alfajer, state)[0], alfajer.pos)
        self.assertEqual(self.ctrl.formation.mid_name, 'Alfajer')

    def test_main_group_keeps_recon_utility_when_holding_formation(self):
        grid = parse_grid(NEW_MAZE_STR)
        self.ctrl.target = (9, 3)
        carrier = actor('Leo', (12, 3))
        carrier.has_spike = True
        front = actor('Derke', (10, 3))
        second_front = actor('Chronicle', (11, 3))
        support = actor('Boaster', (13, 3), recon_charges=1)
        mid = actor('Alfajer', (13, 23))
        support.ability_name = 'RECON'
        state = dict(grid=grid, chars=[carrier, front, second_front, support, mid], round_timer=100)
        result = self.ctrl.decide_move(support, state)
        self.assertEqual(result[1]['ability'], 'RECON')
        self.assertEqual(result[0], support.pos)

    def test_chronicle_moves_ahead_of_leo_while_derke_is_alive(self):
        self.carrier.name = 'Leo'
        derke = actor('Derke', (4, 7))
        chronicle = actor('Chronicle', (4, 3))
        allies = [self.carrier, derke, chronicle]
        for _ in range(12):
            derke.pos = self.decide(derke, allies)[0]
            chronicle.pos = self.decide(chronicle, allies)[0]
            if distances(self.ctrl.target, self.grid)[tuple(chronicle.pos)] < 18:
                break
        self.assertEqual(self.ctrl.formation.front_names, ('Derke', 'Chronicle'))
        lengths = distances(self.ctrl.target, self.grid)
        self.assertLess(lengths[tuple(chronicle.pos)], lengths[tuple(self.carrier.pos)])

    def casualty_fixture(self, remaining):
        self.carrier.pos = [4, 8]
        self.ctrl.target = (4, 8 + remaining)
        self.ctrl.positions.plant_cells = (self.ctrl.target, (4, 2))
        derke = actor('Derke', (4, 11))
        chronicle = actor('Chronicle', (4, 10))
        boaster = actor('Boaster', (4, 6))
        mid = actor('Alfajer', (3, 6))
        allies = [self.carrier, derke, chronicle, boaster, mid]
        self.decide(self.carrier, allies)
        derke.is_alive = chronicle.is_alive = False
        return allies

    def test_all_front_players_dead_advance_when_close_or_late(self):
        for remaining, timer in ((7, 31), (8, 30), (10, 29), (7, 30)):
            with self.subTest(remaining=remaining, timer=timer):
                self.ctrl.reset_round()
                allies = self.casualty_fixture(remaining)
                target = self.ctrl.target
                self.assertEqual(self.decide(self.carrier, allies, timer)[0], [4, 9])
                self.assertEqual(self.ctrl.target, target)
                self.assertTrue(self.ctrl.formation.exposed_after_losses)

    @patch('fnatic_v3.formation.UNSCREENED_PLANT_RADIUS', 7)
    def test_all_front_players_dead_rotate_at_eight_steps_and_31_ticks(self):
        allies = self.casualty_fixture(8)
        old_target = self.ctrl.target
        self.ctrl.formation.entry_targets = {'Boaster': (4, 15)}
        self.ctrl.formation.passage_waits = {'carrier': ('Derke', old_target, (4, 9), 200)}
        result = self.decide(self.carrier, allies, 31)
        self.assertEqual(self.ctrl.target, (4, 2))
        self.assertEqual(result[0], [4, 7])
        self.assertEqual(self.ctrl.formation.entry_context[1], (4, 2))
        self.assertNotIn((4, 15), self.ctrl.formation.entry_targets.values())
        self.assertNotIn('Derke', self.ctrl.formation.previous_frontline)
        self.assertNotIn('carrier', self.ctrl.formation.passage_waits)
        for _ in range(4):
            self.decide(allies[-2], allies)
            self.assertEqual(self.ctrl.target, (4, 2))  # No repeated site flip.
        self.assertEqual(self.ctrl.formation.mid_name, 'Alfajer')

    def test_surviving_front_prevents_team_loss_retreat(self):
        allies = self.casualty_fixture(10)
        allies[2].is_alive = True
        self.decide(self.carrier, allies)
        self.assertFalse(self.ctrl.formation.exposed_after_losses)
        self.assertEqual(self.ctrl.target, (4, 18))

    def test_opening_with_front_players_behind_is_not_a_casualty(self):
        derke = actor('Derke', (4, 3))
        chronicle = actor('Chronicle', (3, 3))
        self.decide(self.carrier, [self.carrier, derke, chronicle])
        self.assertFalse(self.ctrl.formation.exposed_after_losses)
        self.assertEqual(self.ctrl.target, (4, 22))

    @patch('fnatic_v3.formation.UNSCREENED_PLANT_RADIUS', 7)
    def test_wall_detour_uses_shortest_path_for_seven_step_threshold(self):
        allies = self.casualty_fixture(7)
        target = self.ctrl.target
        self.grid[4, 9] = 1
        self.decide(self.carrier, allies)
        self.assertGreater(distances(target, self.grid)[tuple(self.carrier.pos)], 7)
        self.assertEqual(self.ctrl.target, (4, 2))

    @patch('fnatic_v3.formation.UNSCREENED_PLANT_RADIUS', 7)
    def test_lone_carrier_rotates_and_continues_without_waiting_for_dead_front(self):
        allies = self.casualty_fixture(10)
        for c in allies[3:]:
            c.is_alive = False
        for _ in range(6):
            self.carrier.pos = self.decide(self.carrier, allies)[0]
        self.assertEqual(self.carrier.pos, [4, 2])
        self.assertEqual(self.decide(self.carrier, allies)[0], self.carrier.pos)
        self.assertEqual(self.ctrl.target, (4, 2))

    def test_no_reachable_opposite_plant_keeps_current_entry(self):
        allies = self.casualty_fixture(8)
        old_target = self.ctrl.target
        self.grid[:, 5] = 1  # Disconnect the sole opposite-site candidate.
        result = self.decide(self.carrier, allies, 31)
        self.assertEqual(self.ctrl.target, old_target)
        self.assertEqual(result[0], [4, 9])

    def test_living_front_moving_behind_carrier_does_not_rotate(self):
        allies = self.casualty_fixture(8)
        allies[1].is_alive = True
        allies[1].pos = [3, 6]
        target = self.ctrl.target
        self.decide(self.carrier, allies, 31)
        self.assertEqual(self.ctrl.target, target)
        self.assertFalse(self.ctrl.formation.exposed_after_losses)

    @patch('fnatic_v3.formation.UNSCREENED_PLANT_RADIUS', 7)
    def test_shared_round_clock_controls_boundary_despite_iq_timer_noise(self):
        grid = parse_grid(NEW_MAZE_STR)
        for actual, perceived, rotate in ((30, 100, False), (31, 10, True)):
            with self.subTest(actual=actual, perceived=perceived):
                ctrl = FnaticV3AttackerController(engineer_map='')
                target = ctrl.positions.plants[5][0]
                ctrl.target = target
                carrier = actor('Leo', (15, 3))
                carrier.has_spike = True
                rear = actor('Boaster', (15, 6))
                mid = actor('Alfajer', (13, 23))
                ctrl.formation.loss_context = (carrier.name, target)
                ctrl.formation.previous_frontline = {'Derke', 'Chronicle'}
                owner = NS(round_timer=actual)
                ctrl.set_game(NS(real_game=owner, round_timer=perceived))
                ctrl.decide_move(carrier, dict(grid=grid, chars=[carrier, rear, mid],
                                               round_timer=perceived))
                self.assertEqual(ctrl.target != target, rotate)
                self.assertEqual(owner.target_plant_pos, ctrl.target)

    def test_front_finishes_passage_after_carrier_yields_during_rotation(self):
        grid = parse_grid(NEW_MAZE_STR)
        ctrl = FnaticV3AttackerController(engineer_map='')
        ctrl.target = (10, 42)
        carrier = actor('Leo', (16, 3))
        carrier.has_spike = True
        front = actor('Boaster', (16, 5))
        mid = actor('Alfajer', (13, 23))
        ctrl.formation.mid_name = 'Alfajer'
        ctrl.formation.passage_waits = {'Leo': ('Boaster', (13, 3), (15, 3), 4)}
        state = dict(grid=grid, chars=[carrier, front, mid], round_timer=31)
        game = NS(battle_tick=0, round_timer=31)
        ctrl.set_game(game)
        result = ctrl.decide_move(front, state)
        self.assertEqual(result[0], [15, 5])
        front.pos = result[0]
        remaining = distances(ctrl.target, grid)
        before = remaining[tuple(carrier.pos)]
        for _ in range(12):
            for char in (carrier, front, mid):
                char.pos = ctrl.decide_move(char, state)[0]
            game.battle_tick += 1
        self.assertLess(remaining[tuple(carrier.pos)], before)

    def test_round_reset_discards_frontline_loss_state(self):
        allies = self.casualty_fixture(10)
        self.decide(self.carrier, allies)
        self.ctrl.reset_round()
        self.assertEqual(self.ctrl.formation.previous_frontline, set())
        self.assertFalse(self.ctrl.formation.exposed_after_losses)
        self.assertEqual(self.ctrl.formation.entry_targets, {})
        self.assertEqual(self.ctrl.formation.passage_waits, {})

    def test_site_entries_clear_doorways_and_complete_plant(self):
        # Reproduced jams from the upper/lower approaches, including a rear
        # teammate trapped between the carrier and a one-cell site entrance.
        cases = (
            (5, ((11, 6), (12, 6), (9, 6), (12, 7))),
            (6, ((4, 6), (5, 6), (7, 6), (6, 6))),
            (7, ((11, 40), (10, 39), (8, 39), (12, 39))),
            (7, ((8, 42), (7, 42), (6, 42), (10, 42))),
            (8, ((4, 41), (7, 42), (3, 39), (6, 40))),
        )
        for marker, starts in cases:
            for use_iq in (False, True):
                with self.subTest(marker=marker, starts=starts, use_iq=use_iq):
                    ctrl = FnaticV3AttackerController(engineer_map='')
                    grid = parse_grid(NEW_MAZE_STR)
                    ctrl.target = ctrl.positions.plants[marker][0]
                    game = UltimateTestGame(*grid.shape)
                    game.grid = grid
                    names = ('Leo', 'Boaster', 'Derke', 'Chronicle', 'Alfajer')
                    chars = [make_character(name, 'A', p)
                             for name, p in zip(names, (*starts, (13, 23)))]
                    carrier = chars[0]
                    carrier.has_spike = True
                    game.chars = chars
                    controller = IQAwareController(ctrl) if use_iq else ctrl
                    controller.set_game(game)
                    game.attacker_controller = controller
                    for _ in range(80):
                        for c in game._move_order():
                            game.move_character(c)
                        game.battle_tick += 1
                        self.assertEqual(len({tuple(c.pos) for c in chars}), len(chars))
                        if game.is_planted:
                            break
                        self.assertEqual(chars[-1].pos, [13, 23])
                    self.assertTrue(game.is_planted, [(c.name, c.pos) for c in chars])
                    self.assertIn(tuple(carrier.pos), ctrl.positions.plant_cells)

    def test_rear_support_can_pass_spawn_queue_before_site_entry(self):
        ctrl = FnaticV3AttackerController(engineer_map='')
        grid = parse_grid(NEW_MAZE_STR)
        ctrl.target = ctrl.positions.plants[5][0]
        game = UltimateTestGame(*grid.shape)
        game.grid = grid
        names = ('Boaster', 'Derke', 'Chronicle', 'Alfajer', 'Leo')
        chars = [make_character(name, 'A', (23, 17 + i)) for i, name in enumerate(names)]
        chars[-1].has_spike = True
        game.chars = chars
        game.attacker_controller = ctrl
        ctrl.set_game(game)
        for _ in range(100):
            for c in game._move_order():
                game.move_character(c)
            game.battle_tick += 1
            if game.is_planted:
                break
        self.assertTrue(game.is_planted, [(c.name, c.pos) for c in chars])

    def test_carrier_catches_front_pair_at_bent_corridor(self):
        ctrl = FnaticV3AttackerController(engineer_map='')
        grid = parse_grid(NEW_MAZE_STR)
        ctrl.target = ctrl.positions.plants[8][0]
        game = UltimateTestGame(*grid.shape)
        game.grid = grid
        specs = (('Boaster', (22, 31)), ('Derke', (21, 32)),
                 ('Chronicle', (22, 33)), ('Alfajer', (13, 23)), ('Leo', (22, 37)))
        chars = [make_character(name, 'A', p) for name, p in specs]
        chars[1].has_spike = True
        game.chars = chars
        controller = IQAwareController(ctrl)
        controller.set_game(game)
        game.attacker_controller = controller
        for _ in range(100):
            for c in game._move_order():
                game.move_character(c)
            game.battle_tick += 1
            if game.is_planted:
                break
        self.assertTrue(game.is_planted, [(c.name, c.pos) for c in chars])

    @patch('fnatic_v3.formation.UNSCREENED_PLANT_RADIUS', 7)
    def test_real_frontline_losses_advance_or_rotate_then_plant(self):
        for marker in (5, 7, 8):
            for threshold, timer, use_iq in ((7, 31, False), (7, 31, True),
                                              (8, 30, False), (8, 30, True),
                                              (8, 31, False), (8, 31, True)):
                with self.subTest(marker=marker, threshold=threshold, timer=timer, use_iq=use_iq):
                    ctrl = FnaticV3AttackerController(engineer_map='')
                    grid = parse_grid(NEW_MAZE_STR)
                    ctrl.target = ctrl.positions.plants[marker][0]
                    game = UltimateTestGame(*grid.shape)
                    game.grid = grid
                    names = ('Leo', 'Boaster', 'Derke', 'Chronicle', 'Alfajer')
                    chars = [make_character(name, 'A', (23, 17 + i))
                             for i, name in enumerate(names)]
                    carrier = chars[0]
                    carrier.has_spike = True
                    game.chars = chars
                    controller = IQAwareController(ctrl) if use_iq else ctrl
                    game.attacker_controller = controller
                    controller.set_game(game)
                    lengths = distances(ctrl.target, grid)
                    original_target = ctrl.target
                    lost_at = rotated_target = None
                    for _ in range(240):
                        for c in chars:
                            if not c.is_alive:
                                continue
                            old_remaining = lengths[tuple(carrier.pos)]
                            game.move_character(c)
                            if lost_at is not None and (threshold <= 7 or timer <= 30):
                                self.assertLessEqual(lengths[tuple(carrier.pos)], old_remaining)
                            if ctrl.target != original_target:
                                if rotated_target is None:
                                    rotated_target = ctrl.target
                                self.assertEqual(ctrl.target, rotated_target)
                        game.battle_tick += 1
                        alive = [c for c in chars if c.is_alive]
                        self.assertEqual(len({tuple(c.pos) for c in alive}), len(alive))
                        if lost_at is None and lengths[tuple(carrier.pos)] == threshold:
                            ctrl.decide_move(carrier, dict(grid=grid, chars=chars, round_timer=100))
                            victims = [c for c in alive if c is not carrier
                                       and c.name != ctrl.formation.mid_name
                                       and lengths[tuple(c.pos)] < threshold]
                            # The casualty rule tracks everyone actually ahead;
                            # a second entry can still be passing a narrow lane.
                            self.assertTrue(victims)
                            self.assertLessEqual({c.name for c in victims}, {'Derke', 'Chronicle'})
                            for c in victims:
                                c.is_alive = False
                            lost_at = tuple(carrier.pos)
                            game.round_timer = timer
                        if game.is_planted:
                            break
                    self.assertIsNotNone(lost_at)
                    self.assertTrue(game.is_planted, [(c.name, c.pos) for c in chars if c.is_alive])
                    if threshold <= 7 or timer <= 30:
                        self.assertIsNone(rotated_target)
                        self.assertEqual(game.planted_pos, original_target)
                    else:
                        self.assertIsNotNone(rotated_target)
                        self.assertNotEqual(original_target[1] < grid.shape[1] // 2,
                                            rotated_target[1] < grid.shape[1] // 2)
                        self.assertEqual(game.planted_pos, rotated_target)
                    self.assertEqual(game.target_plant_pos, ctrl.target)
                    self.assertEqual(ctrl.formation.mid_name, 'Alfajer')

    def test_real_team_reaches_plant_while_mid_remains_separate(self):
        for strategy, use_iq, marker in (('RUSH', False, 5), ('DEFAULT', False, 5),
                                         ('RUSH', True, 5), ('RUSH', False, 7),
                                         ('RUSH', False, 8)):
            with self.subTest(strategy=strategy, use_iq=use_iq, marker=marker):
                self.ctrl.reset_round()
                self.ctrl.strategy = strategy
                self.ctrl.target = self.ctrl.positions.plants[marker][0]
                grid = parse_grid(NEW_MAZE_STR)
                game = UltimateTestGame(*grid.shape)
                game.grid = grid
                names = ('Leo', 'Boaster', 'Derke', 'Chronicle', 'Alfajer')
                chars = [make_character(name, 'A', (23, 17 + i)) for i, name in enumerate(names)]
                carrier = chars[0]
                carrier.has_spike = True
                game.chars = chars
                game.attacker_controller = self.ctrl
                self.ctrl.set_game(game)
                if use_iq:
                    wrapper = IQAwareController(self.ctrl)
                    wrapper.set_game(game)
                    game.attacker_controller = wrapper
                separated = False
                for _ in range(160):
                    for c in chars:
                        old_carrier = tuple(carrier.pos)
                        game.move_character(c)
                        if c is carrier and tuple(carrier.pos) != old_carrier:
                            remaining = distances(self.ctrl.target, grid)
                            if remaining[old_carrier] > SITE_ENTRY_RADIUS:
                                self.assertTrue(any(
                                    remaining[tuple(teammate.pos)] < remaining[tuple(carrier.pos)]
                                    for teammate in chars if teammate is not carrier
                                    and teammate.name != self.ctrl.formation.mid_name))
                    game.battle_tick += 1
                    self.assertEqual(len({tuple(c.pos) for c in chars}), 5)
                    if self.ctrl.formation.mid_target is not None:
                        separated |= tuple(chars[-1].pos) == self.ctrl.formation.mid_target
                    if game.is_planted:
                        break
                self.assertTrue(game.is_planted, [(c.name, c.pos) for c in chars])
                self.assertEqual(self.ctrl.formation.mid_name, 'Alfajer')
                self.assertTrue(separated)
                lengths = distances(tuple(carrier.pos), grid)
                self.assertTrue(all(lengths[tuple(c.pos)] <= FRONT_SELECTION_RADIUS + 1
                                    for c in chars[1:-1]))


if __name__ == '__main__':
    unittest.main()
