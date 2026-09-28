"""Enemy-smoke recon through real movement, casts and both team controllers."""

import unittest
from types import SimpleNamespace

import numpy as np

from fnatic_v3.controller import FnaticV3AttackerController, FnaticV3DefenderController
from fnatic_v3.positions import TacticalPositions
from iq_controller_adapter import IQAwareController
from game_core import COMBO_BANNER_HEIGHT
from rendering_ui import RenderingUIMixin
from test_ultimate_system import UltimateTestGame, make_character


class FnaticSmokeReconTests(unittest.TestCase):
    def fixture(self, side='A', steps=7, open_grid=False, use_iq=False):
        bottom = steps + 2
        grid = np.ones((bottom + 2, 11), dtype=np.int32)
        grid[1:bottom + 1, 1] = 0
        grid[bottom, 1:10] = 0
        if open_grid:
            grid[1:-1, 1:-1] = 0
        grid[1, 1] = 2
        tactical = grid.copy()
        tactical[1, 1] = 5
        text = '\n'.join(''.join(map(str, row)) for row in tactical)
        game = UltimateTestGame(*grid.shape)
        game.grid = grid
        char = make_character('Leo', side, (2, 1))
        char.iq = char.effective_iq = 200
        enemy = make_character('Boaster', 'D' if side == 'A' else 'A', (1, 1))
        game.chars = [char, enemy]
        ctrl = (FnaticV3AttackerController(TacticalPositions(text, text), engineer_map='')
                if side == 'A' else FnaticV3DefenderController(position_map='', retake_map='', engineer_map=''))
        adapter = IQAwareController(ctrl) if use_iq else ctrl
        adapter.set_game(game)
        if side == 'A':
            game.attacker_controller = adapter
        else:
            game.defender_controller = adapter
        self.assertTrue(game.execute_ai_ability(enemy, {'ability': 'SMOKE', 'target': (bottom, 8)}))
        return game, ctrl, char, enemy

    @staticmethod
    def state(game):
        return dict(grid=game.grid, chars=game.chars, round_timer=100,
                    is_planted=game.is_planted, planted_pos=game.planted_pos)

    def test_both_sides_walk_seven_cells_then_cast_in_real_runtime(self):
        for side in ('A', 'D'):
            for use_iq in (False, True):
                with self.subTest(side=side, use_iq=use_iq):
                    game, ctrl, char, _ = self.fixture(side, use_iq=use_iq)
                    for step in range(7):
                        game.move_character(char)
                        self.assertEqual(char.pos, [3 + step, 1])
                        self.assertEqual(char.recon_charges, 2)
                        game.battle_tick += 1
                    game.move_character(char)
                    self.assertEqual(char.pos, [9, 1])
                    self.assertEqual(char.recon_charges, 1)
                    self.assertEqual(len(game.recon_projectiles), 1)
                    game.battle_tick += 1
                    for _ in range(3):
                        game.move_character(char)
                        game.battle_tick += 1
                    self.assertEqual(char.recon_charges, 1)

    def test_eight_walking_cells_is_outside_limit(self):
        for side in ('A', 'D'):
            game, ctrl, char, _ = self.fixture(side, steps=8)
            self.assertIsNone(ctrl.smoke_recon.result(ctrl, char, self.state(game)))

    def test_current_wall_los_casts_in_place_ignoring_bodies_and_smoke(self):
        for side in ('A', 'D'):
            game, ctrl, char, enemy = self.fixture(side)
            char.pos = [9, 1]
            blocker = make_character('Derke', side, (9, 4))
            game.chars.append(blocker)
            game.smokes.append(dict(cells={(9, 2), (9, 3)}, remaining_ticks=10,
                                    owner=blocker.name, team=side, center=(9, 3)))
            result = ctrl.decide_move(char, self.state(game))
            self.assertEqual(result[0], char.pos)
            self.assertEqual(result[1], {'ability': 'RECON', 'target': [9, 8]})
            self.assertTrue(game.execute_ai_ability(char, result[1]))

    def test_wall_blocks_despite_small_geometric_distance(self):
        game, ctrl, char, _ = self.fixture(steps=8)
        # The smoke is less than seven cells away diagonally, but the nearest
        # firing cell requires eight walking steps around the wall.
        char.pos = [2, 1]
        game.smokes[0]['cells'] = {(8, 7)}
        game.grid[8, 7] = 0  # Isolated smoke pocket with no reachable firing position.
        game.smokes[0]['center'] = (8, 7)
        self.assertIsNone(ctrl.smoke_recon.result(ctrl, char, self.state(game)))

    def test_nearest_firing_cell_stops_movement_immediately(self):
        game, ctrl, char, _ = self.fixture(steps=2)
        for _ in range(2):
            game.move_character(char)
        self.assertEqual(char.pos, [4, 1])
        game.move_character(char)
        self.assertEqual(char.pos, [4, 1])
        self.assertEqual(char.recon_charges, 1)

    def test_friendly_unknown_and_expired_smokes_do_not_trigger(self):
        for kind in ('friendly', 'unknown', 'expired'):
            game, ctrl, char, _ = self.fixture(open_grid=True)
            smoke = game.smokes[0]
            if kind == 'friendly':
                smoke['team'] = char.team
            elif kind == 'unknown':
                smoke.pop('team')
                smoke['owner'] = 'unknown'
            else:
                smoke['remaining_ticks'] = 0
            self.assertIsNone(ctrl.smoke_recon.result(ctrl, char, self.state(game)))

    def test_dead_and_mirrored_smoke_owners_use_recorded_team(self):
        game, ctrl, char, enemy = self.fixture(open_grid=True)
        enemy.is_alive = False
        game.chars.append(make_character('Boaster', char.team, (1, 2)))
        result = ctrl.smoke_recon.result(ctrl, char, self.state(game))
        self.assertEqual(result[1]['ability'], 'RECON')

    def test_legacy_smoke_owner_fallback_includes_dead_characters(self):
        game, ctrl, char, enemy = self.fixture(open_grid=True)
        game.smokes[0].pop('team')
        game.smokes[0].pop('center')
        enemy.is_alive = False
        result = ctrl.smoke_recon.result(ctrl, char, self.state(game))
        self.assertEqual(result[1]['ability'], 'RECON')

    def test_no_recon_or_no_charge_does_not_react(self):
        for attribute, value in (('ability_name', 'FLASH'), ('recon_charges', 0), ('is_alive', False)):
            game, ctrl, char, _ = self.fixture(open_grid=True)
            setattr(char, attribute, value)
            self.assertIsNone(ctrl.smoke_recon.result(ctrl, char, self.state(game)))

    def test_failed_cast_retries_and_new_smoke_can_use_second_charge(self):
        game, ctrl, char, _ = self.fixture(open_grid=True)
        state = self.state(game)
        result = ctrl.smoke_recon.result(ctrl, char, state)
        # Merely requesting a cast must not mark the cloud as completed.
        self.assertEqual(ctrl.smoke_recon.result(ctrl, char, state), result)
        self.assertTrue(game.execute_ai_ability(char, result[1]))
        game.battle_tick += 1
        self.assertIsNone(ctrl.smoke_recon.result(ctrl, char, state))
        game.smokes.append(dict(game.smokes[0]))
        self.assertIsNone(ctrl.smoke_recon.result(ctrl, char, state))
        game.recon_projectiles.clear()
        game.battle_tick += 18
        second = ctrl.smoke_recon.result(ctrl, char, state)
        self.assertEqual(second[1]['ability'], 'RECON')
        self.assertTrue(game.execute_ai_ability(char, second[1]))
        self.assertEqual(char.recon_charges, 0)

    def test_each_recon_player_responds_and_reset_clears_completed_clouds(self):
        game, ctrl, char, _ = self.fixture(open_grid=True)
        other = make_character('Shao', char.team, (3, 1))
        other.ability_name = 'RECON'
        other.recon_charges = 2
        game.chars.append(other)
        for recon in (char, other):
            action = ctrl.smoke_recon.result(ctrl, recon, self.state(game))
            self.assertTrue(game.execute_ai_ability(recon, action[1]))
            self.assertIsNone(ctrl.smoke_recon.result(ctrl, recon, self.state(game)))
        ctrl.reset_round()
        game.recon_projectiles.clear()
        self.assertEqual(ctrl.smoke_recon.result(ctrl, char, self.state(game))[1]['ability'], 'RECON')

    def test_plant_and_defuse_remain_uninterrupted(self):
        for side in ('A', 'D'):
            game, ctrl, char, _ = self.fixture(side, open_grid=True)
            if side == 'A':
                char.pos = [1, 1]
                char.has_spike = True
                # The enemy stood on the plant marker only for the fixture.
                game.chars[1].pos = [2, 3]
                expected = 'PLANT'
            else:
                game.is_planted = True
                game.planted_pos = (1, 1)
                char.defuse_timer = 1  # An already-started channel takes priority.
                expected = 'DEFUSE'
            self.assertEqual(ctrl.decide_move(char, self.state(game))[1], expected)
            self.assertEqual(char.recon_charges, 2)

    def test_expiring_smoke_cancels_approach(self):
        game, ctrl, char, _ = self.fixture()
        self.assertEqual(ctrl.smoke_recon.result(ctrl, char, self.state(game))[0], [3, 1])
        game.smokes.clear()
        self.assertIsNone(ctrl.smoke_recon.result(ctrl, char, self.state(game)))

    def test_movement_does_not_overlap_a_player(self):
        game, ctrl, char, _ = self.fixture()
        game.chars.append(make_character('Derke', char.team, (3, 1)))
        action = ctrl.smoke_recon.result(ctrl, char, self.state(game))
        self.assertEqual(action[0], char.pos)

    def test_ai_smoke_records_team_and_center(self):
        game, _, _, enemy = self.fixture()
        self.assertEqual(game.smokes[0]['team'], enemy.team)
        self.assertEqual(game.smokes[0]['center'], (9, 8))

    def test_manual_smoke_records_team_and_center(self):
        game, _, _, enemy = self.fixture(open_grid=True)
        game.smokes.clear()
        enemy.smoke_charges = 1
        game.ability_mode = ('SMOKE', enemy.team, enemy.name)
        game.ultimate_mode = None
        game.map_offset_x = 0
        game.cell_size = 1
        game.map_pixel_width = game.width
        game._handle_team_panel_click = lambda x, y: False
        game._selected_user_character = lambda: None
        game._plant_button_bounds = lambda: None
        game._orb_button_bounds = lambda: None
        game.draw = lambda: None
        event = SimpleNamespace(x=8, y=9 + COMBO_BANNER_HEIGHT)
        RenderingUIMixin.on_canvas_click(game, event)
        self.assertEqual(game.smokes[0]['team'], enemy.team)
        self.assertEqual(game.smokes[0]['center'], (9, 8))

    def test_defender_setup_does_not_start_smoke_mission(self):
        game, ctrl, char, _ = self.fixture('D', open_grid=True)
        state = self.state(game)
        state['defender_setup_active'] = True
        action = ctrl.decide_move(char, state)
        self.assertNotEqual(action[1].get('ability'), 'RECON')

    def test_closest_of_multiple_smokes_is_selected(self):
        game, ctrl, char, _ = self.fixture()
        # The first cloud needs seven steps; the second needs one.
        game.smokes.append(dict(cells={(4, 1)}, center=(4, 1), team='D',
                                owner='Boaster', remaining_ticks=10))
        action = ctrl.smoke_recon.result(ctrl, char, self.state(game))
        self.assertEqual(action[0], char.pos)
        self.assertEqual(action[1]['target'], [4, 1])


if __name__ == '__main__':
    unittest.main()
