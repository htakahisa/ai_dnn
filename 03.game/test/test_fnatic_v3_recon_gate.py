"""One confirmed recon for each continuously satisfied trigger."""

import unittest

from fnatic_v3.controller import FnaticV3AttackerController, FnaticV3DefenderController
from fnatic_v3.positions import TacticalPositions
from iq_controller_adapter import IQAwareController
from test_fnatic_v3_defender_positions import placement_map
from test_fnatic_v3_utility import advance_effects
from test_ultimate_system import UltimateTestGame, make_character


class FnaticReconGateTests(unittest.TestCase):
    def fixture(self, side='A', use_iq=False):
        game = UltimateTestGame(10, 24)
        game.attacker_team_name, game.defender_team_name = 'Attackers', 'Defenders'
        game.grid[1, 1] = game.grid[1, 22] = 2
        char = make_character('Leo', side, (5, 4))
        char.ultimate_name = 'NONE'
        char.iq = char.effective_iq = 200
        enemy = make_character('Enemy', 'D' if side == 'A' else 'A', (5, 7))
        game.chars = [char, enemy]
        if side == 'A':
            text = placement_map(game.grid, {5: ((1, 1),), 6: ((1, 22),)})
            ctrl = FnaticV3AttackerController(TacticalPositions(text, text), engineer_map='')
            game.is_planted, game.planted_pos = True, (1, 1)
        else:
            ctrl = FnaticV3DefenderController('', '', engineer_map='')
        adapter = IQAwareController(ctrl) if use_iq else ctrl
        adapter.set_game(game)
        setattr(game, 'attacker_controller' if side == 'A' else 'defender_controller', adapter)
        return game, ctrl, char, enemy

    @staticmethod
    def state(game):
        return dict(grid=game.grid, chars=game.chars, is_planted=game.is_planted,
                    planted_pos=game.planted_pos, round_timer=100)

    def test_continuous_contact_does_not_consume_second_charge_even_after_30_ticks(self):
        for side in ('A', 'D'):
            for use_iq in (False, True):
                with self.subTest(side=side, use_iq=use_iq):
                    game, ctrl, char, enemy = self.fixture(side, use_iq)
                    game.move_character(char)
                    self.assertEqual(char.recon_charges, 1)
                    for _ in range(30):
                        advance_effects(game)
                        enemy.pos[0] = 5 + game.battle_tick % 2
                        game.move_character(char)
                    self.assertEqual(char.recon_charges, 1)

    def test_lost_then_renewed_contact_can_trigger_remaining_charge(self):
        for side in ('A', 'D'):
            game, ctrl, char, enemy = self.fixture(side)
            game.move_character(char)
            for _ in range(22):
                advance_effects(game)
            game.chars.remove(enemy)
            ctrl.decide_move(char, self.state(game))
            game.chars.append(enemy)
            game.move_character(char)
            self.assertEqual(char.recon_charges, 0)

    def test_failed_request_does_not_consume_trigger_and_success_is_confirmed(self):
        game, ctrl, char, _ = self.fixture('D')
        first = ctrl.decide_move(char, self.state(game))
        retry = ctrl.decide_move(char, self.state(game))
        self.assertEqual(first[1]['ability'], 'RECON')
        self.assertEqual(retry[1]['ability'], 'RECON')
        self.assertEqual(ctrl.recon_gate.used[char.name], set())
        self.assertTrue(game.execute_ai_ability(char, retry[1]))
        ctrl.decide_move(char, self.state(game))
        self.assertIn(('enemy', 'A', 'Enemy'), ctrl.recon_gate.used[char.name])

    def test_smoke_and_normal_recon_share_cooldown_but_each_new_cloud_can_trigger(self):
        game, ctrl, char, _ = self.fixture('D')
        game.move_character(char)
        cloud = dict(cells={(7, 6)}, center=(7, 6), remaining_ticks=100,
                     team='A', owner='Enemy')
        game.smokes.append(cloud)
        self.assertIsNone(ctrl.smoke_recon.result(ctrl, char, self.state(game)))
        for _ in range(18):
            advance_effects(game)
        action = ctrl.smoke_recon.result(ctrl, char, self.state(game))
        self.assertEqual(action[1]['ability'], 'RECON')
        self.assertTrue(game.execute_ai_ability(char, action[1]))
        self.assertEqual(char.recon_charges, 0)

    def test_same_entry_does_not_cast_again_and_keeps_advancing(self):
        game, ctrl, char, enemy = self.fixture()
        game.chars.remove(enemy)
        game.is_planted, game.planted_pos = False, None
        char.has_spike = True
        ctrl.target = (1, 1)
        action = ctrl.decide_move(char, self.state(game))
        self.assertEqual(action[1]['ability'], 'RECON')
        self.assertTrue(game.execute_ai_ability(char, action[1]))
        for _ in range(22):
            advance_effects(game)
            ctrl.decide_move(char, self.state(game))
        movement = ctrl.decide_move(char, self.state(game))
        self.assertNotIn('ability', movement[1])
        self.assertNotEqual(movement[0], char.pos)
        self.assertEqual(char.recon_charges, 1)

    def test_round_reset_reenables_same_condition(self):
        game, ctrl, char, _ = self.fixture('D')
        game.move_character(char)
        ctrl.decide_move(char, self.state(game))
        ctrl.reset_round()
        game.recon_projectiles.clear()
        char.recon_charges = 2
        game.move_character(char)
        self.assertEqual(char.recon_charges, 1)


if __name__ == '__main__':
    unittest.main()
