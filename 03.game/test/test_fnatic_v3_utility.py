"""Fnatic attacker utility through real casts and complete team movement."""

import unittest

from fnatic_v3.controller import FnaticV3AttackerController
from fnatic_v3.positions import parse_grid
from game_core import PLANT_REQUIRED_TICKS, REVEAL_DURATION_TICKS
from iq_controller_adapter import IQAwareController
from map_data import NEW_MAZE_STR
from test_fnatic_v3_engineer import lamp_map
from test_ultimate_system import UltimateTestGame, make_character


def advance_effects(game):
    game.battle_tick += 1
    for char in game.chars:
        char.blind_remaining = max(0, char.blind_remaining - 1)
        char.reveal_remaining = max(0, char.reveal_remaining - 1)
    game._advance_flash_projectiles()
    game._advance_recon_projectiles()
    for smoke in game.smokes:
        smoke['remaining_ticks'] -= 1
    game.smokes = [s for s in game.smokes if s['remaining_ticks'] > 0]


class FnaticUtilityTests(unittest.TestCase):
    def fixture(self, players, engineer_map=''):
        grid = parse_grid(NEW_MAZE_STR)
        game = UltimateTestGame(*grid.shape)
        game.grid = grid
        game.chars = [make_character(name, 'A', cell) for name, cell in players]
        ctrl = FnaticV3AttackerController(engineer_map=engineer_map)
        ctrl.set_game(game)
        game.attacker_controller = ctrl
        ctrl.target = (8, 4)
        return game, ctrl

    def state(self, game):
        return dict(grid=game.grid, chars=game.chars, round_timer=100,
                    is_planted=game.is_planted, planted_pos=game.planted_pos)

    def test_moving_entry_uses_flash_and_resumes_movement(self):
        game, ctrl = self.fixture((('Leo', (15, 3)), ('Derke', (12, 3)),
                                   ('Chronicle', (13, 3)), ('Alfajer', (13, 23))))
        holder, _, char, _ = game.chars
        holder.has_spike = True
        blocked = {tuple(c.pos) for c in game.chars if c is not char}
        movement = ctrl.formation.result(ctrl, char, holder, game.chars, game.grid, blocked, [])
        self.assertNotEqual(movement[0], char.pos)
        game.move_character(char)
        self.assertEqual(char.flash_charges, 0)
        self.assertEqual(len(game.flash_projectiles), 1)
        self.assertEqual(char.pos, [13, 3])
        advance_effects(game)
        game.move_character(char)
        self.assertNotEqual(char.pos, [13, 3])

    def test_carrier_recon_throws_toward_visible_site_around_corner(self):
        game, ctrl = self.fixture((('Leo', (13, 3)), ('Derke', (11, 3)),
                                   ('Chronicle', (12, 3)), ('Alfajer', (13, 23))))
        char = game.chars[0]
        char.has_spike = True
        self.assertFalse(ctrl._los(tuple(char.pos), ctrl.target, game.grid, smoke=False))
        action = ctrl.decide_move(char, self.state(game))
        self.assertEqual(action[1]['ability'], 'RECON')
        self.assertNotEqual(tuple(action[1]['target']), ctrl.target)
        self.assertTrue(ctrl._los(tuple(char.pos), tuple(action[1]['target']), game.grid, smoke=False))
        self.assertTrue(game.execute_ai_ability(char, action[1]))
        self.assertEqual(char.recon_charges, 1)

    def test_mid_smoke_supports_entry_without_covering_spike_or_allies(self):
        grid = parse_grid(NEW_MAZE_STR)
        game, ctrl = self.fixture((('Leo', (12, 3)), ('Derke', (10, 3)),
                                   ('Chronicle', (11, 3)), ('Boaster', (13, 23)),
                                   ('Alfajer', (22, 23))), lamp_map(grid, ((22, 23),)))
        game.chars[0].has_spike = True
        char = game.chars[3]
        game.move_character(char)
        self.assertEqual(ctrl.formation.mid_name, 'Boaster')
        self.assertEqual(char.pos, [13, 23])
        self.assertEqual(char.smoke_charges, 0)
        self.assertEqual(len(game.smokes), 1)
        smoke = game.smokes[0]['cells']
        protected = {ctrl.target, *ctrl.formation.entry_targets.values()}
        protected.update(tuple(c.pos) for c in game.chars)
        self.assertFalse(smoke & protected)

    def test_postplant_remaining_abilities_have_real_effects(self):
        for name, ability in (('Chronicle', 'FLASH'), ('Leo', 'RECON'), ('Boaster', 'SMOKE')):
            with self.subTest(ability=ability):
                game, ctrl = self.fixture(((name, (11, 3)),))
                char = game.chars[0]
                enemy = make_character('Demon1', 'D', (13, 3))
                game.chars.append(enemy)
                game.is_planted = True
                game.planted_pos = (9, 3)
                ctrl.guard_anchor = game.planted_pos
                ctrl.guard_targets = {name: (12, 3)}
                before = getattr(char, ability.lower() + '_charges')
                game.move_character(char)
                self.assertEqual(getattr(char, ability.lower() + '_charges'), before - 1)
                for _ in range(3):
                    advance_effects(game)
                if ability == 'FLASH':
                    self.assertGreater(enemy.blind_remaining, 0)
                elif ability == 'RECON':
                    self.assertGreater(enemy.reveal_remaining, 0)
                else:
                    self.assertIn(tuple(enemy.pos), game._smoke_cells())

    def test_recon_second_charge_requires_a_new_condition_after_cooldown(self):
        game, ctrl = self.fixture((('Leo', (11, 3)),))
        char = game.chars[0]
        enemy = make_character('Demon1', 'D', (13, 3))
        game.chars.append(enemy)
        game.is_planted = True
        game.planted_pos = (9, 3)
        game.move_character(char)
        self.assertEqual(char.recon_charges, 1)
        game.battle_tick += REVEAL_DURATION_TICKS + 3
        action = ctrl.decide_move(char, self.state(game))
        self.assertNotIn('ability', action[1])  # Own projectile still flying.
        for _ in range(3):
            game._advance_recon_projectiles()
        self.assertGreater(enemy.reveal_remaining, 0)
        # Establish a fresh round to exercise the normal cooldown separately.
        ctrl.reset_round()
        char.recon_charges = 2
        enemy.reveal_remaining = 0
        action = ctrl.decide_move(char, self.state(game))
        self.assertTrue(game.execute_ai_ability(char, action[1]))
        for _ in range(REVEAL_DURATION_TICKS + 2):
            advance_effects(game)
            action = ctrl.decide_move(char, self.state(game))
            self.assertNotIn('ability', action[1])
        advance_effects(game)
        action = ctrl.decide_move(char, self.state(game))
        self.assertNotIn('ability', action[1])  # The same contact/site still holds.
        self.assertEqual(char.recon_charges, 1)
        game.chars.remove(enemy)
        ctrl.decide_move(char, self.state(game))
        game.chars.append(enemy)
        enemy.reveal_remaining = 0
        action = ctrl.decide_move(char, self.state(game))
        self.assertEqual(action[1]['ability'], 'RECON')
        self.assertTrue(game.execute_ai_ability(char, action[1]))
        self.assertEqual(char.recon_charges, 0)
        ctrl.reset_round()
        self.assertEqual(ctrl.utility_pending, {})
        self.assertEqual(ctrl.utility_last_use, {})

    def test_plant_takes_priority_over_utility(self):
        game, ctrl = self.fixture((('Leo', (9, 3)), ('Derke', (10, 3))))
        ctrl.target = (9, 3)
        char = game.chars[0]
        char.has_spike = True
        for _ in range(PLANT_REQUIRED_TICKS):
            game.move_character(char)
            advance_effects(game)
        self.assertTrue(game.is_planted)
        self.assertEqual(char.recon_charges, 2)

    def test_invalid_or_immediately_blocked_throw_preserves_movement(self):
        game, ctrl = self.fixture((('Leo', (15, 4)),))
        char = game.chars[0]
        enemy = make_character('Demon1', 'D', (14, 4))  # A wall/noisy observation.
        movement = ([15, 3], {'facing': 'W'})
        result = ctrl._attacker_utility(char, char, game.grid, [enemy], movement, (7, 40), False)
        self.assertEqual(result, movement)
        # Guard against a throw direction rejected by the actual cast engine.
        ctrl.target = (9, 3)
        game._projectile_path = lambda start, end: [start]
        char.pos = [13, 3]
        self.assertEqual(ctrl._formation_utility(char, char, game.grid, [], movement), movement)
        self.assertEqual(char.recon_charges, 2)

    def test_whole_team_uses_all_active_roles_then_plants_and_holds_guards(self):
        for use_iq in (False, True):
            with self.subTest(use_iq=use_iq):
                grid = parse_grid(NEW_MAZE_STR)
                names = ('Leo', 'Derke', 'Chronicle', 'Boaster', 'Alfajer')
                game, ctrl = self.fixture(tuple((name, (23, 17 + i)) for i, name in enumerate(names)),
                                          lamp_map(grid, ((22, 21), (21, 22))))
                ctrl.target = None
                game.chars[0].has_spike = True
                if use_iq:
                    game.attacker_controller = IQAwareController(ctrl)
                    game.attacker_controller.set_game(game)
                for _ in range(150):
                    for char in game._move_order():
                        game.move_character(char)
                    advance_effects(game)
                    self.assertEqual(len({tuple(c.pos) for c in game.chars}), 5)
                leo, _, chronicle, boaster, alfajer = game.chars
                self.assertLess(leo.recon_charges, 2)
                self.assertEqual(chronicle.flash_charges, 0)
                self.assertEqual(boaster.smoke_charges, 0)
                self.assertEqual(alfajer.ramp_charges, 0)
                self.assertTrue(game.is_planted)
                self.assertEqual(ctrl.formation.mid_name, 'Boaster')
                self.assertTrue(all(ctrl.guard_targets.get(c.name) == tuple(c.pos) for c in game.chars))


if __name__ == '__main__':
    unittest.main()
