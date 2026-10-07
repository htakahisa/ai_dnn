"""Observed tendencies, role/map transitions and resulting team movement."""

import contextlib
import io
import random
import unittest
from unittest.mock import patch

import numpy as np

from fnatic_v3.controller import FnaticV3AttackerController, FnaticV3DefenderController
from fnatic_v3.opponent_history import FnaticOpponentHistory
from fnatic_v3.positions import TacticalPositions, parse_grid, region
from game_core import WINNING_ROUNDS
from iq_controller_adapter import IQAwareController
from iq_perception import PerceivedCharacter
from map_data import NEW_MAZE_STR
from party_presets import get_preset
from test_fnatic_v3_defender_positions import placement_map
from test_fnatic_v3_engineer import lamp_map
from test_fnatic_v3_utility import advance_effects
from test_ultimate_system import UltimateTestGame, make_character


def attack_row(*labels):
    return dict(map=1, round=1,
                contacts={str(i): dict(pos=(2, i), region=label) for i, label in enumerate(labels)})


class FnaticOpponentHistoryTests(unittest.TestCase):
    def fixture(self, side='A', names=('Leo',), memory=None, grid=None):
        game = UltimateTestGame(14, 24)
        if grid is not None:
            game.grid = grid
        game.grid[2, 2] = game.grid[2, 21] = 2
        game.attacker_team_name = 'Fnatic' if side == 'A' else 'Opponent'
        game.defender_team_name = 'Fnatic' if side == 'D' else 'Opponent'
        game.series_context = dict(maps_played=0, fnatic_memory=memory if memory is not None else {})
        game.chars = [make_character(name, side, (10, 3 + i)) for i, name in enumerate(names)]
        for char in game.chars:
            char.ability_name = char.ultimate_name = 'NONE'
        marked = np.where(game.grid == 1, 1, 0)
        marked[2, 2], marked[2, 21] = 5, 6
        text = '\n'.join(''.join(map(str, row)) for row in marked)
        slots = {3: ((3, 2), (3, 4), (3, 11), (3, 13), (3, 19), (3, 21))}
        ctrl = (FnaticV3AttackerController(TacticalPositions(text, text), engineer_map='')
                if side == 'A' else FnaticV3DefenderController(
                    placement_map(game.grid, slots), retake_map='', engineer_map=''))
        ctrl.set_game(game)
        setattr(game, 'attacker_controller' if side == 'A' else 'defender_controller', ctrl)
        return game, ctrl

    @staticmethod
    def state(game, **extra):
        return dict(grid=game.grid, chars=game.chars, round_timer=game.round_timer,
                    is_planted=game.is_planted, planted_pos=game.planted_pos,
                    spike_pos=game.spike_pos, **extra)

    @staticmethod
    def add_enemy(game, name, cell):
        char = make_character(name, 'D' if game.chars[0].team == 'A' else 'A', cell)
        game.chars.append(char)
        return char

    def test_first_contact_per_enemy_is_shared_and_not_overwritten_after_rotation(self):
        game, ctrl = self.fixture(names=('Leo', 'Derke'))
        enemy = self.add_enemy(game, 'Enemy1', (4, 2))
        second = self.add_enemy(game, 'Enemy2', (4, 12))
        ctrl.decide_move(game.chars[0], self.state(game))
        enemy.pos = [4, 21]
        ctrl.decide_move(game.chars[1], self.state(game))
        ctrl.record_opponent_round_end()
        ctrl.record_opponent_round_end()
        ctrl.reset_round()
        rows = ctrl.opponent_history.data['attack_rounds']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['contacts'][enemy.name], dict(pos=(4, 2), region='A'))
        self.assertEqual(rows[0]['contacts'][second.name]['region'], 'MID')

    def test_hidden_real_coordinates_are_not_recorded_but_revealed_observations_are(self):
        grid = np.zeros((14, 24), dtype=np.int32)
        grid[:, 8] = 1
        game, ctrl = self.fixture(grid=grid)
        enemy = self.add_enemy(game, 'Enemy', (4, 21))
        ctrl.opponent_history.observe(ctrl, game.chars[0], self.state(game))
        self.assertEqual(ctrl.opponent_history.current['contacts'], {})
        enemy.reveal_remaining = 5
        # IQ reports a different coordinate; the series history must keep it.
        observed = PerceivedCharacter(enemy, pos=[5, 19])
        state = self.state(game)
        state['chars'] = [game.chars[0], observed]
        ctrl.opponent_history.observe(ctrl, game.chars[0], state)
        ctrl.record_opponent_round_end()
        self.assertEqual(ctrl.opponent_history.data['attack_rounds'][0]['contacts'][enemy.name]['pos'], (5, 19))

    def test_setup_and_new_postplant_contacts_do_not_count_as_initial_positions(self):
        game, ctrl = self.fixture()
        self.add_enemy(game, 'Enemy', (4, 2))
        ctrl.opponent_history.observe(ctrl, game.chars[0], self.state(game, defender_setup_active=True))
        game.is_planted, game.planted_pos = True, (2, 2)
        ctrl.opponent_history.observe(ctrl, game.chars[0], self.state(game))
        ctrl.record_opponent_round_end()
        self.assertEqual(ctrl.opponent_history.data['attack_rounds'][0]['contacts'], {})
        self.assertIsNone(ctrl.opponent_history.attack_region(['A', 'MID', 'B']))

    def test_round_reset_archives_and_keeps_history_but_allows_new_first_contact(self):
        game, ctrl = self.fixture()
        enemy = self.add_enemy(game, 'Enemy', (4, 2))
        ctrl.opponent_history.observe(ctrl, game.chars[0], self.state(game))
        ctrl.reset_round()
        enemy.pos = [4, 21]
        game.current_round += 1
        ctrl.opponent_history.observe(ctrl, game.chars[0], self.state(game))
        ctrl.record_opponent_round_end()
        self.assertEqual([r['contacts'][enemy.name]['region'] for r in ctrl.opponent_history.data['attack_rounds']], ['A', 'B'])

    def test_roles_and_fresh_map_controllers_share_team_history_only_within_series(self):
        memory = {}
        game, attacker = self.fixture(memory=memory)
        self.add_enemy(game, 'Enemy', (4, 2))
        attacker.opponent_history.observe(attacker, game.chars[0], self.state(game))
        attacker.record_opponent_round_end()
        game2, defender = self.fixture('D', ('Alfajer',), memory)
        game2.series_context['maps_played'] = 1
        self.assertIs(defender.opponent_history.data, attacker.opponent_history.data)
        game2.is_planted, game2.planted_pos = True, (2, 21)
        defender.opponent_history.observe(defender, game2.chars[0], self.state(game2))
        defender.record_opponent_round_end()
        self.assertEqual(defender.opponent_history.data['defence_rounds'][0], dict(map=2, round=1, site='B'))
        _, fresh = self.fixture()
        self.assertEqual(fresh.opponent_history.data, dict(attack_rounds=[], defence_rounds=[]))

    def test_halftime_binding_preserves_both_roles_for_the_same_team(self):
        game, attacker = self.fixture()
        self.add_enemy(game, 'Enemy', (4, 2))
        attacker.opponent_history.observe(attacker, game.chars[0], self.state(game))
        attacker.record_opponent_round_end()
        game.attacker_team_name, game.defender_team_name = 'Opponent', 'Fnatic'
        game.chars = [make_character('Alfajer', 'D', (10, 3))]
        defender = FnaticV3DefenderController('', '', engineer_map='')
        defender.set_game(game)
        self.assertIs(defender.opponent_history.data, attacker.opponent_history.data)
        self.assertEqual(len(defender.opponent_history.data['attack_rounds']), 1)

    def test_thin_region_uses_completed_round_average_and_randomizes_similar_counts(self):
        history = FnaticOpponentHistory()
        history.data['attack_rounds'] = [attack_row('A', 'A', 'MID', 'MID', 'B')]
        self.assertEqual(history.attack_region(['A', 'MID', 'B']), 'B')
        history.data['attack_rounds'] = [attack_row('A', 'MID', 'B')] * 4 + [attack_row('A', 'MID')]
        with patch('fnatic_v3.opponent_history.random.choice', side_effect=lambda cells: cells[0]) as choice:
            history.attack_region(['A', 'MID', 'B'])
        self.assertEqual(set(choice.call_args.args[0]), {'A', 'MID', 'B'})
        history.data['attack_rounds'] = [attack_row('A', 'A', 'MID', 'MID', 'B')]
        self.assertIn(history.attack_region(['A', 'MID']), {'A', 'MID'})

    def test_attacker_selects_thin_site_and_keeps_target_for_the_round(self):
        for label, history in (('A', attack_row('A', 'MID', 'MID', 'B', 'B')),
                               ('B', attack_row('A', 'A', 'MID', 'MID', 'B'))):
            game, ctrl = self.fixture()
            game.chars[0].has_spike = True
            ctrl.opponent_history.data['attack_rounds'].append(history)
            for _ in range(3):
                ctrl.decide_move(game.chars[0], self.state(game))
                self.assertEqual(region(ctrl.target, game.grid), label)
                self.assertEqual(ctrl.attack_region, label)

    def test_no_history_preserves_random_site_choice_and_touyama_still_forces_a(self):
        game, ctrl = self.fixture()
        game.chars[0].has_spike = True
        with patch('fnatic_v3.controller.random.choice', side_effect=lambda cells: max(cells)):
            ctrl.decide_move(game.chars[0], self.state(game))
        self.assertEqual(ctrl.target, (2, 21))
        ctrl.reset_round()
        ctrl.opponent_history.data['attack_rounds'] = [attack_row('A', 'A', 'MID', 'MID', 'B')]
        game.defender_roster = list(get_preset('Touyama Gaming').players)
        ctrl.decide_move(game.chars[0], self.state(game))
        self.assertEqual(ctrl.target, (2, 2))
        self.assertEqual(ctrl.attack_region, 'A')

    def test_unreachable_thin_site_falls_back_to_a_reachable_entry(self):
        grid = np.zeros((14, 24), dtype=np.int32)
        grid[:, 8] = 1
        game, ctrl = self.fixture(grid=grid)
        game.chars[0].has_spike = True
        ctrl.opponent_history.data['attack_rounds'] = [attack_row('A', 'A', 'MID', 'MID', 'B')]
        ctrl.decide_move(game.chars[0], self.state(game))
        self.assertEqual(ctrl.target, (2, 2))
        self.assertEqual(ctrl.attack_region, 'A')

    def test_thin_mid_main_team_reaches_waypoint_then_registered_plant(self):
        game, ctrl = self.fixture(names=('Leo', 'Derke', 'Chronicle', 'Boaster', 'Alfajer'))
        game.chars[0].has_spike = True
        ctrl.opponent_history.data['attack_rounds'] = [attack_row('A', 'A', 'MID', 'B', 'B')]
        random.seed(3)
        visited = False
        for _ in range(110):
            for char in game.chars:
                game.move_character(char)
                visited |= tuple(game.chars[0].pos) == ctrl.mid_entry_goal
            game.battle_tick += 1
            if game.is_planted:
                break
        self.assertEqual(ctrl.attack_region, 'MID')
        self.assertTrue(visited)
        self.assertTrue(ctrl.mid_entry_done)
        self.assertTrue(game.is_planted)
        self.assertIn(game.planted_pos, ctrl.positions.plant_cells)
        self.assertEqual(game.planted_pos, ctrl.target)

    def test_defender_records_final_planted_site_even_after_an_a_fake(self):
        game, ctrl = self.fixture('D', ('Alfajer',))
        self.add_enemy(game, 'Enemy', (4, 2))
        ctrl.opponent_history.observe(ctrl, game.chars[0], self.state(game))
        # Plant completes after this defender's last decision in the round.
        game.is_planted, game.planted_pos = True, (2, 21)
        ctrl.record_opponent_round_end()
        self.assertEqual(ctrl.opponent_history.data['defence_rounds'][0]['site'], 'B')

    def test_real_map_mid_route_with_lamps_and_iq_finishes_before_time_limit(self):
        for use_iq in (False, True):
            with self.subTest(use_iq=use_iq):
                random.seed(2)
                grid = parse_grid(NEW_MAZE_STR)
                game = UltimateTestGame(*grid.shape)
                game.grid = grid
                game.chars = [make_character(name, 'A', (23, 17 + i))
                              for i, name in enumerate(('Leo', 'Derke', 'Chronicle', 'Boaster', 'Alfajer'))]
                game.chars[0].has_spike = True
                ctrl = FnaticV3AttackerController(engineer_map=lamp_map(grid, ((22, 21), (21, 22))))
                adapter = IQAwareController(ctrl) if use_iq else ctrl
                adapter.set_game(game)
                game.attacker_controller = adapter
                ctrl.opponent_history.data['attack_rounds'] = [attack_row('A', 'A', 'MID', 'B', 'B')]
                for tick in range(100):
                    game.round_timer = 100 - tick
                    for char in game._move_order():
                        game.move_character(char)
                    advance_effects(game)
                    self.assertEqual(len({tuple(c.pos) for c in game.chars}), 5)
                    if game.is_planted:
                        break
                self.assertEqual(ctrl.attack_region, 'MID')
                self.assertTrue(ctrl.mid_entry_done)
                self.assertTrue(game.is_planted)
                self.assertIn(game.planted_pos, ctrl.positions.plant_cells)
                self.assertEqual(game.chars[-1].ramp_charges, 0)

    def test_defender_bias_finishes_lamps_before_heading_to_frequent_site(self):
        game, _ = self.fixture('D', ('Alfajer',))
        char = game.chars[0]
        char.ability_name, char.ramp_charges = 'RAMP', 2
        ctrl = FnaticV3DefenderController(
            placement_map(game.grid, {3: ((3, 2), (3, 21))}), retake_map='',
            engineer_map=lamp_map(game.grid, ((10, 3), (10, 4))))
        ctrl.set_game(game)
        game.defender_controller = ctrl
        ctrl.opponent_history.data['defence_rounds'] = [dict(site='B')] * 2
        for _ in range(35):
            game.move_character(char)
            game.battle_tick += 1
            if char.ramp_charges > 0:
                self.assertGreaterEqual(char.pos[0], 9)
            if char.ramp_charges == 0 and char.pos == [3, 21]:
                break
        self.assertEqual(char.ramp_charges, 0)
        self.assertEqual(char.pos, [3, 21])

    def test_unreachable_frequent_defender_site_uses_reachable_mapped_fallback(self):
        grid = np.zeros((14, 24), dtype=np.int32)
        grid[:, 8] = 1
        game, ctrl = self.fixture('D', ('Alfajer',), grid=grid)
        ctrl.opponent_history.data['defence_rounds'] = [dict(site='B')]
        ctrl.decide_move(game.chars[0], self.state(game))
        self.assertEqual(region(ctrl.defender_positions.targets['Alfajer'], grid), 'A')

    def test_without_plant_latest_observations_identify_final_site_and_ties_stay_unknown(self):
        game, ctrl = self.fixture('D', ('Alfajer',))
        first = self.add_enemy(game, 'Enemy1', (4, 2))
        self.add_enemy(game, 'Enemy2', (4, 21))
        ctrl.opponent_history.observe(ctrl, game.chars[0], self.state(game))
        first.pos = [4, 21]
        ctrl.opponent_history.observe(ctrl, game.chars[0], self.state(game))
        ctrl.reset_round()
        first.pos = [4, 2]
        game.current_round += 1
        ctrl.opponent_history.observe(ctrl, game.chars[0], self.state(game))
        ctrl.record_opponent_round_end()
        self.assertEqual([r['site'] for r in ctrl.opponent_history.data['defence_rounds']], ['B', None])

    def test_biased_final_sites_assign_alfajer_a_unique_mapped_slot(self):
        for label in ('A', 'B'):
            game, ctrl = self.fixture('D', ('Alfajer', 'Boaster', 'Chronicle', 'Derke', 'Leo'))
            ctrl.opponent_history.data['defence_rounds'] = [dict(site=label)] * 3 + [dict(site='B' if label == 'A' else 'A')]
            ctrl.decide_move(game.chars[-1], self.state(game))
            targets = dict(ctrl.defender_positions.targets)
            self.assertEqual(region(targets['Alfajer'], game.grid), label)
            self.assertEqual(len(set(targets.values())), 5)
            ctrl.decide_move(game.chars[0], self.state(game))
            self.assertEqual(ctrl.defender_positions.targets, targets)

    def test_bias_ties_leave_normal_assignment_and_dropped_spike_takes_priority(self):
        game, ctrl = self.fixture('D', ('Alfajer',))
        ctrl.opponent_history.data['defence_rounds'] = [dict(site='A'), dict(site='B'), dict(site=None)]
        ctrl.decide_move(game.chars[0], self.state(game))
        self.assertIsNone(ctrl.defender_positions.preferred_engineer_region)
        ctrl.opponent_history.data['defence_rounds'].append(dict(site='B'))
        game.spike_pos = (4, 2)
        ctrl.decide_move(game.chars[0], self.state(game))
        self.assertEqual(region(ctrl.defender_positions.targets['Alfajer'], game.grid), 'A')
        game.spike_pos = None
        ctrl.decide_move(game.chars[0], self.state(game))
        self.assertEqual(region(ctrl.defender_positions.targets['Alfajer'], game.grid), 'B')

    def test_trap_staffing_still_works_when_frequent_site_differs_from_lamps(self):
        game, ctrl = self.fixture('D', ('Alfajer', 'Boaster', 'Chronicle', 'Derke', 'Leo'))
        ctrl.opponent_history.data['defence_rounds'] = [dict(site='B')] * 2
        game.ramp_traps = [dict(pos=(4, 12), team='D', owner='Alfajer')]
        ctrl.decide_move(game.chars[0], self.state(game))
        self.assertEqual(region(ctrl.defender_positions.targets['Alfajer'], game.grid), 'B')
        self.assertLessEqual(sum(region(p, game.grid) == 'MID' for p in ctrl.defender_positions.targets.values()), 1)

    def test_engine_finish_hook_records_once_before_next_round_and_in_final_round(self):
        for final in (False, True):
            game, ctrl = self.fixture('D', ('Alfajer',))
            ctrl.opponent_history.observe(ctrl, game.chars[0], self.state(game))
            game.is_planted, game.planted_pos = True, (2, 21)
            game.attacker_wins = WINNING_ROUNDS if final else 1
            game.defender_wins = 0
            game.overtime = game.is_defused = False
            game.match_over = False
            archived_at_reset = []
            game.init_round = lambda: archived_at_reset.append(len(ctrl.opponent_history.data['defence_rounds']))
            with contextlib.redirect_stdout(io.StringIO()):
                game.check_match_winner()
            self.assertEqual(len(ctrl.opponent_history.data['defence_rounds']), 1)
            self.assertEqual(ctrl.opponent_history.defender_site(), 'B')
            self.assertEqual(game.match_over, final)
            self.assertEqual(archived_at_reset, [] if final else [1])


if __name__ == '__main__':
    unittest.main()
