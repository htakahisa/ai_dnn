"""Shared weighted defensive slots through setup, live movement and retakes."""

from collections import Counter
import random
import unittest
from unittest.mock import patch

import numpy as np

from defender_setup_phase import DefenderSetupPhase
from fnatic_v3.controller import FnaticV3DefenderController
from fnatic_v3.defender_positions import FnaticDefenderPositions
from fnatic_v3.map_data_defender_fnatic import DEFENDER_POSITION_STR
from fnatic_v3.positions import distances, parse_grid, region
from iq_controller_adapter import IQAwareController
from map_data import NEW_MAZE_STR
from test_ultimate_system import UltimateTestGame, make_character


def placement_map(grid, candidates):
    marked = np.where(grid == 1, 1, 0)
    for marker, cells in candidates.items():
        for cell in cells:
            marked[cell] = marker
    return '\n'.join(''.join(map(str, row)) for row in marked)


class FnaticDefenderPositionTests(unittest.TestCase):
    def fixture(self, grid=None, candidates=None, names=('Alfajer', 'Boaster', 'Chronicle', 'Derke', 'Leo')):
        if grid is None:
            grid = np.zeros((9, 24), dtype=np.int32)
            grid[1, 1] = 2
        if candidates is None:
            candidates = {marker: ((2, 2 + (marker - 3) * 3),) for marker in range(3, 10)}
        game = UltimateTestGame(*grid.shape)
        game.grid = grid
        game.chars = [make_character(name, 'D', (5, 1 + i)) for i, name in enumerate(names)]
        ctrl = FnaticV3DefenderController(placement_map(grid, candidates), retake_map='', engineer_map='')
        ctrl.set_game(game)
        game.defender_controller = ctrl
        return game, ctrl

    @staticmethod
    def state(game, setup=False):
        return dict(grid=game.grid, chars=game.chars, round_timer=100,
                    is_planted=game.is_planted, planted_pos=game.planted_pos,
                    spike_pos=game.spike_pos,
                    defender_setup_active=setup)

    def test_drop_redeploys_all_survivors_only_to_its_region_and_holds_positions(self):
        candidates = {3: tuple((2, c) for c in (2, 4, 6, 9, 11, 13, 17, 19, 21))}
        for col, label in ((3, 'A'), (12, 'MID'), (20, 'B')):
            with self.subTest(region=label):
                game, ctrl = self.fixture(candidates=candidates, names=('Alfajer', 'Derke', 'Leo'))
                ctrl.decide_move(game.chars[0], self.state(game))
                game.spike_pos = (4, col)
                ctrl.decide_move(game.chars[-1], self.state(game))
                targets = dict(ctrl.defender_positions.targets)
                self.assertEqual(len(set(targets.values())), 3)
                self.assertTrue(all(region(p, game.grid) == label for p in targets.values()))
                for _ in range(35):
                    for char in game.chars:
                        game.move_character(char)
                    game.battle_tick += 1
                    if all(tuple(c.pos) == targets[c.name] for c in game.chars):
                        break
                self.assertEqual(ctrl.defender_positions.targets, targets)
                self.assertTrue(all(tuple(c.pos) == targets[c.name] for c in game.chars))

    def test_drop_chooses_random_candidate_instead_of_nearest(self):
        game, ctrl = self.fixture(candidates={3: ((2, 2), (2, 7), (2, 20))}, names=('Derke',))
        game.spike_pos = (4, 3)
        char = game.chars[0]
        with patch('fnatic_v3.defender_positions.random.choice', side_effect=lambda cells: max(cells)):
            result = ctrl.decide_move(char, self.state(game))
        goal = ctrl.defender_positions.targets[char.name]
        self.assertEqual(goal, (2, 7))
        lengths = distances(goal, game.grid)
        self.assertLess(lengths[tuple(result[0])], lengths[tuple(char.pos)])

    def test_pickup_restores_normal_slots_and_a_new_drop_can_change_region(self):
        game, ctrl = self.fixture(candidates={3: ((2, 2), (2, 12), (2, 20))}, names=('Derke',))
        char = game.chars[0]
        ctrl.defender_positions.targets = {char.name: (2, 20)}
        game.spike_pos = (4, 3)
        ctrl.decide_move(char, self.state(game))
        self.assertEqual(ctrl.defender_positions.targets[char.name], (2, 2))
        game.spike_pos = None
        ctrl.decide_move(char, self.state(game))
        self.assertIsNone(ctrl.defender_positions.drop_region)
        self.assertEqual(ctrl.defender_positions.targets[char.name], (2, 20))
        game.spike_pos = (4, 12)
        ctrl.decide_move(char, self.state(game))
        self.assertEqual(ctrl.defender_positions.targets[char.name], (2, 12))
        ctrl.reset_round()
        self.assertIsNone(ctrl.defender_positions.drop_region)
        self.assertIsNone(ctrl.defender_positions.saved_targets)

    def test_drop_orders_preserve_combat_priority_and_end_when_planted(self):
        game, ctrl = self.fixture(candidates={3: ((2, 2), (2, 20))}, names=('Derke',))
        char = game.chars[0]
        game.spike_pos = (4, 20)
        enemy = make_character('Enemy', 'A', (5, 3))
        game.chars.append(enemy)
        self.assertEqual(ctrl.decide_move(char, self.state(game))[0], char.pos)
        game.chars.remove(enemy)
        ctrl.decide_move(char, self.state(game))
        self.assertEqual(ctrl.defender_positions.targets[char.name], (2, 20))
        game.is_planted = True
        game.planted_pos = (5, 2)
        self.assertEqual(ctrl.decide_move(char, self.state(game))[1], 'DEFUSE')
        self.assertIsNone(ctrl.defender_positions.drop_region)

    def test_iq_noise_does_not_reroll_shared_drop_orders_across_regions(self):
        game, ctrl = self.fixture(candidates={3: ((2, 7), (2, 9), (2, 12))}, names=('Derke',))
        state = self.state(game)
        state['spike_pos'] = (4, 7)
        ctrl.decide_move(game.chars[0], state)
        targets = dict(ctrl.defender_positions.targets)
        state['spike_pos'] = (4, 9)
        ctrl.decide_move(game.chars[0], state)
        self.assertEqual(ctrl.defender_positions.targets, targets)
        self.assertEqual(ctrl.defender_positions.drop_region, 'A')

    def test_weighted_groups_are_independent_of_the_number_of_cells(self):
        game, ctrl = self.fixture(candidates={3: ((2, 2),), 9: tuple((2, c) for c in range(3, 23))},
                                  names=('Derke',))
        char = game.chars[0]
        counts = Counter()
        rng = random.Random(17)
        with patch('fnatic_v3.defender_positions.random.choices', rng.choices), \
                patch('fnatic_v3.defender_positions.random.choice', rng.choice):
            for _ in range(800):
                ctrl.reset_round()
                goal = ctrl.defender_positions.goal(char, game.chars, game.grid)
                counts[3 if goal == (2, 2) else 9] += 1
        # Group 3 has weight 7 even though group 9 contains twenty cells.
        self.assertGreater(counts[3], 650)
        self.assertLess(counts[3], 750)
        self.assertGreater(counts[9], 50)

    def coverage_fixture(self, columns=(11,), names=None):
        candidates = {3: tuple((2, c) for c in (1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21))}
        kwargs = {'candidates': candidates}
        if names is not None:
            kwargs['names'] = names
        game, ctrl = self.fixture(**kwargs)
        game.ramp_traps = [{'pos': (3, c), 'owner': 'Alfajer', 'team': 'D'} for c in columns]
        return game, ctrl

    def test_lamps_reduce_regional_staffing_and_reassign_old_slots(self):
        for col, covered in ((3, 'A'), (11, 'MID'), (20, 'B')):
            with self.subTest(region=covered):
                game, ctrl = self.coverage_fixture((col,))
                old_cells = [p for p in ctrl.defender_positions.candidates[3]
                             if region(p, game.grid) == covered]
                ctrl.defender_positions.targets = dict(zip((c.name for c in game.chars), old_cells))
                ctrl.decide_move(game.chars[-1], self.state(game))
                targets = dict(ctrl.defender_positions.targets)
                counts = Counter(region(p, game.grid) for p in targets.values())
                self.assertEqual(counts[covered], 1)
                self.assertTrue(all(counts[label] == 2 for label in ('A', 'MID', 'B') if label != covered))
                self.assertEqual(region(targets['Alfajer'], game.grid), covered)
                self.assertEqual(len(set(targets.values())), 5)
                for char in reversed(game.chars):
                    ctrl.decide_move(char, self.state(game))
                self.assertEqual(ctrl.defender_positions.targets, targets)

    def test_multiple_trapped_regions_favor_the_remaining_untrapped_route(self):
        game, ctrl = self.coverage_fixture((3, 11))
        ctrl.decide_move(game.chars[0], self.state(game))
        counts = Counter(region(p, game.grid) for p in ctrl.defender_positions.targets.values())
        self.assertEqual(counts, {'A': 1, 'MID': 1, 'B': 3})

    def test_all_trapped_regions_and_fewer_survivors_still_get_distinct_slots(self):
        for columns in ((11,), (3, 11, 20)):
            for alive_count in range(1, 6):
                with self.subTest(columns=columns, alive_count=alive_count):
                    game, ctrl = self.coverage_fixture(columns)
                    for char in game.chars[alive_count:]:
                        char.is_alive = False
                    ctrl.decide_move(game.chars[0], self.state(game))
                    targets = ctrl.defender_positions.targets
                    self.assertEqual(len(targets), alive_count)
                    self.assertEqual(len(set(targets.values())), alive_count)
                    counts = Counter(region(p, game.grid) for p in targets.values())
                    if len(columns) == 1:
                        self.assertEqual(counts['MID'], 1)
                    else:
                        self.assertLessEqual(max(counts.values()), 2)

    def test_enemy_other_owner_and_dead_owner_lamps_do_not_change_staffing(self):
        game, ctrl = self.coverage_fixture()
        game.ramp_traps = [dict(pos=(3, 11), owner='Alfajer', team='A'),
                           dict(pos=(3, 3), owner='Other', team='D')]
        ctrl.decide_move(game.chars[-1], self.state(game))
        self.assertEqual(ctrl.defender_positions.trap_regions, set())
        self.assertIsNone(ctrl.defender_positions.region_limits)
        game.ramp_traps.append(dict(pos=(3, 11), owner='Alfajer', team='D'))
        game.chars[0].is_alive = False
        ctrl.decide_move(game.chars[-1], self.state(game))
        self.assertEqual(ctrl.defender_positions.trap_regions, set())
        self.assertNotIn('Alfajer', ctrl.defender_positions.targets)

    def test_dropped_spike_region_takes_priority_over_trap_distribution(self):
        game, ctrl = self.coverage_fixture()
        ctrl.decide_move(game.chars[0], self.state(game))
        game.spike_pos = (3, 11)
        ctrl.decide_move(game.chars[-1], self.state(game))
        self.assertEqual(Counter(region(p, game.grid) for p in ctrl.defender_positions.targets.values()),
                         {'MID': 5})
        game.spike_pos = None
        ctrl.decide_move(game.chars[-1], self.state(game))
        self.assertEqual(Counter(region(p, game.grid) for p in ctrl.defender_positions.targets.values()),
                         {'A': 2, 'MID': 1, 'B': 2})

    def test_lamp_loss_and_owner_death_update_remaining_team_assignments(self):
        game, ctrl = self.coverage_fixture()
        ctrl.decide_move(game.chars[-1], self.state(game))
        game.chars[1].is_alive = False
        ctrl.decide_move(game.chars[-1], self.state(game))
        self.assertEqual(len(ctrl.defender_positions.targets), 4)
        self.assertEqual(sum(region(p, game.grid) == 'MID'
                             for p in ctrl.defender_positions.targets.values()), 1)
        game.ramp_traps.clear()
        ctrl.decide_move(game.chars[-1], self.state(game))
        self.assertIsNone(ctrl.defender_positions.region_limits)
        self.assertEqual(len(ctrl.defender_positions.targets), 4)

    def test_all_players_share_the_map_and_receive_distinct_stable_slots(self):
        game, ctrl = self.fixture()
        ctrl.decide_move(game.chars[-1], self.state(game))
        targets = dict(ctrl.defender_positions.targets)
        self.assertEqual(set(targets), {c.name for c in game.chars})
        self.assertEqual(len(set(targets.values())), 5)
        for _ in range(10):
            for char in game.chars:
                ctrl.decide_move(char, self.state(game))
            self.assertEqual(ctrl.defender_positions.targets, targets)

    def test_decision_order_does_not_change_the_team_assignment(self):
        assignments = []
        for first in (0, 4):
            game, ctrl = self.fixture()
            rng = random.Random(29)
            with patch('fnatic_v3.defender_positions.random.choices', rng.choices), \
                    patch('fnatic_v3.defender_positions.random.choice', rng.choice), \
                    patch('fnatic_v3.defender_positions.random.shuffle', rng.shuffle):
                ctrl.decide_move(game.chars[first], self.state(game))
            assignments.append(ctrl.defender_positions.targets)
        self.assertEqual(*assignments)

    def test_setup_respects_reachable_area_and_live_can_use_forbidden_slot(self):
        game, ctrl = self.fixture(candidates={3: ((2, 2),), 9: ((2, 20),)}, names=('Derke',))
        allowed = lambda r, c: c < 10
        with patch('fnatic_v3.defender_positions.is_setup_position_allowed', allowed):
            goal = ctrl.defender_positions.goal(game.chars[0], game.chars, game.grid, setup=True)
        self.assertEqual(goal, (2, 2))
        self.assertEqual(ctrl.defender_positions.goal(game.chars[0], game.chars, game.grid), goal)
        # A map containing only a forbidden setup candidate becomes usable
        # after the live round starts; setup itself uses the old fallback.
        game, ctrl = self.fixture(candidates={3: ((2, 20),)}, names=('Derke',))
        with patch('fnatic_v3.defender_positions.is_setup_position_allowed', allowed):
            self.assertIsNone(ctrl.defender_positions.goal(game.chars[0], game.chars, game.grid, setup=True))
        self.assertEqual(ctrl.defender_positions.goal(game.chars[0], game.chars, game.grid), (2, 20))

    def test_unreachable_candidates_are_not_selected(self):
        grid = np.zeros((9, 24), dtype=np.int32)
        grid[:, 10] = 1
        grid[1, 1] = 2
        game, ctrl = self.fixture(grid, {3: ((2, 20),), 9: ((2, 2),)}, names=('Derke',))
        self.assertEqual(ctrl.defender_positions.goal(game.chars[0], game.chars, grid), (2, 2))

    def test_mapped_defenders_reach_and_hold_their_slots_in_real_runtime(self):
        for use_iq in (False, True):
            game, ctrl = self.fixture()
            if use_iq:
                adapter = IQAwareController(ctrl)
                adapter.set_game(game)
                game.defender_controller = adapter
            for _ in range(60):
                for char in game.chars:
                    game.move_character(char)
                game.battle_tick += 1
                self.assertEqual(len({tuple(c.pos) for c in game.chars}), 5)
                if all(tuple(c.pos) == ctrl.defender_positions.targets.get(c.name) for c in game.chars):
                    break
            self.assertTrue(all(tuple(c.pos) == ctrl.defender_positions.targets[c.name] for c in game.chars))
            positions = [tuple(c.pos) for c in game.chars]
            for _ in range(3):
                for char in game.chars:
                    game.move_character(char)
            self.assertEqual([tuple(c.pos) for c in game.chars], positions)

    def test_setup_and_live_round_keep_same_slots_on_actual_game_map(self):
        grid = parse_grid(NEW_MAZE_STR)
        candidates = {marker: ((2, 18 + marker - 3),) for marker in range(3, 8)}
        game, ctrl = self.fixture(grid, candidates)
        for index, char in enumerate(game.chars):
            char.pos = [1, 18 + index]
        game.defender_setup_phase = DefenderSetupPhase()
        game.defender_setup_phase.start()
        for _ in range(20):
            for char in game.chars:
                game._move_character_during_defender_setup(char)
        targets = dict(ctrl.defender_positions.targets)
        self.assertTrue(all(tuple(c.pos) == targets[c.name] for c in game.chars))
        wrapper = IQAwareController(ctrl)
        wrapper.set_game(game)
        game.defender_controller = wrapper
        for _ in range(5):
            for char in game.chars:
                game.move_character(char)
        self.assertEqual(ctrl.defender_positions.targets, targets)
        self.assertTrue(all(tuple(c.pos) == targets[c.name] for c in game.chars))

    def test_shared_slots_exchange_to_clear_one_cell_corridor(self):
        grid = np.ones((7, 12), dtype=np.int32)
        grid[4, 1:11] = 0
        grid[4, 10] = 2
        game, ctrl = self.fixture(grid, {3: ((4, 4), (4, 8))}, names=('Derke', 'Leo'))
        rear, front = game.chars
        rear.pos = [4, 3]
        front.pos = [4, 4]
        front.recon_charges = 0
        ctrl.defender_positions.targets = {rear.name: (4, 8), front.name: (4, 4)}
        for _ in range(8):
            for char in game.chars:
                game.move_character(char)
        self.assertEqual(tuple(rear.pos), (4, 4))
        self.assertEqual(tuple(front.pos), (4, 8))
        self.assertEqual(set(ctrl.defender_positions.targets.values()), {(4, 4), (4, 8)})

    def test_enemy_contact_and_postplant_defuse_take_priority(self):
        game, ctrl = self.fixture(candidates={3: ((2, 20),)}, names=('Derke',))
        char = game.chars[0]
        enemy = make_character('Leo', 'A', (5, 3))
        game.chars.append(enemy)
        self.assertEqual(ctrl.decide_move(char, self.state(game))[0], char.pos)
        game.chars.remove(enemy)
        game.is_planted = True
        game.planted_pos = (5, 2)
        self.assertEqual(ctrl.decide_move(char, self.state(game))[1], 'DEFUSE')

    def test_enemy_smoke_recon_overrides_mapped_movement(self):
        game, ctrl = self.fixture(candidates={3: ((2, 20),)}, names=('Leo',))
        game.smokes.append(dict(cells={(5, 8)}, center=(5, 8), team='A',
                                owner='Boaster', remaining_ticks=10))
        game.move_character(game.chars[0])
        self.assertEqual(game.chars[0].recon_charges, 1)
        self.assertEqual(len(game.recon_projectiles), 1)

    def test_dead_players_release_slots_and_reset_clears_assignments(self):
        game, ctrl = self.fixture()
        ctrl.decide_move(game.chars[0], self.state(game))
        game.chars[0].is_alive = False
        ctrl.decide_move(game.chars[1], self.state(game))
        self.assertNotIn(game.chars[0].name, ctrl.defender_positions.targets)
        ctrl.reset_round()
        self.assertEqual(ctrl.defender_positions.targets, {})

    def test_invalid_map_dimensions_and_walls_are_rejected(self):
        for error in ('dimensions', 'walls'):
            game, ctrl = self.fixture()
            if error == 'dimensions':
                game.grid = np.zeros((8, 24), dtype=np.int32)
            else:
                game.grid[3, 3] = 1
            with self.assertRaisesRegex(ValueError, error):
                ctrl.decide_move(game.chars[0], self.state(game))

    def test_empty_map_and_candidate_shortage_keep_old_fallback(self):
        game, _ = self.fixture()
        ctrl = FnaticV3DefenderController('', retake_map='', engineer_map='')
        ctrl.set_game(game)
        self.assertIsNotNone(ctrl.decide_move(game.chars[0], self.state(game)))
        game, ctrl = self.fixture(candidates={3: ((2, 2),)})
        for char in game.chars:
            self.assertIsNotNone(ctrl.decide_move(char, self.state(game)))
        self.assertEqual(len(ctrl.defender_positions.targets), 1)

    def test_default_map_has_walls_matching_game(self):
        template = parse_grid(DEFENDER_POSITION_STR)
        grid = parse_grid(NEW_MAZE_STR)
        np.testing.assert_array_equal(template == 1, grid == 1)
        self.assertTrue(set(np.unique(template)) <= set(range(10)))


if __name__ == '__main__':
    unittest.main()
