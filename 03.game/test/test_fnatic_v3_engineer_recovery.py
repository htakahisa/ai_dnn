"""Lone engineer stops placing lamps when dropped-spike planting is urgent."""

from types import SimpleNamespace
import unittest

import numpy as np

from fnatic_v3.controller import FnaticV3AttackerController
from fnatic_v3.positions import TacticalPositions
from party_presets import get_preset
from test_fnatic_v3_engineer import lamp_map
from test_ultimate_system import UltimateTestGame, make_character


class FnaticEngineerRecoveryTests(unittest.TestCase):
    def fixture(self, remaining=14, detour=False):
        game = UltimateTestGame(7, 18)
        if detour:
            game.grid[:6, 10] = 1
        game.grid[4, 3] = game.grid[4, 15] = 2
        marked = np.where(game.grid == 1, 1, 0)
        marked[4, 3], marked[4, 15] = 5, 6
        text = '\n'.join(''.join(map(str, row)) for row in marked)
        ctrl = FnaticV3AttackerController(TacticalPositions(text, text),
                                          engineer_map=lamp_map(game.grid, ((4, 9), (5, 9))))
        engineer = make_character('Alfajer', 'A', (4, 9))
        game.chars = [engineer]
        game.spike_pos = (4, 11)
        game.round_timer = remaining
        ctrl.engineer.target = (4, 9)
        ctrl.set_game(game)
        game.attacker_controller = ctrl
        return game, ctrl, engineer

    @staticmethod
    def state(game, remaining=None):
        return dict(grid=game.grid, chars=game.chars, spike_pos=game.spike_pos,
                    is_planted=game.is_planted, planted_pos=game.planted_pos,
                    round_timer=game.round_timer if remaining is None else remaining)

    def test_exact_deadline_boundary_keeps_lamps_until_below_minimum_plus_five(self):
        # Two moves to pickup + four to B + four plant ticks = ten ticks.
        for remaining, urgent in ((14, True), (15, False), (16, False)):
            with self.subTest(remaining=remaining):
                game, ctrl, char = self.fixture(remaining)
                result = ctrl.decide_move(char, self.state(game))
                if urgent:
                    self.assertEqual(result[0], [4, 10])
                    self.assertEqual(ctrl.retriever, 'Alfajer')
                    self.assertNotIn('Alfajer', ctrl.formation.detached_names)
                else:
                    self.assertEqual(result, ([4, 9], {'ability': 'RAMP'}))

    def test_wall_detour_counts_as_ticks_in_the_deadline(self):
        # The wall makes pickup six moves, giving a nineteen-tick threshold.
        game, ctrl, char = self.fixture(17, detour=True)
        self.assertEqual(ctrl.decide_move(char, self.state(game))[0], [5, 9])
        self.assertEqual(ctrl.engineer_recovery_target, (4, 15))

    def test_unregistered_plant_cell_is_not_counted_as_an_available_shortcut(self):
        game, ctrl, char = self.fixture(12)
        game.grid[4, 12] = 2
        self.assertEqual(ctrl.decide_move(char, self.state(game))[0], [4, 10])
        self.assertEqual(ctrl.target, (4, 15))

    def test_real_round_clock_overrides_perception_timer(self):
        for actual, perceived, urgent in ((14, 100, True), (15, 1, False)):
            with self.subTest(actual=actual):
                game, ctrl, char = self.fixture(actual)
                ctrl.set_game(SimpleNamespace(real_game=game, round_timer=perceived))
                result = ctrl.decide_move(char, self.state(game, perceived))
                self.assertEqual(result[0], [4, 10] if urgent else [4, 9])

    def test_live_teammate_prevents_lone_survivor_exception_but_dead_ones_do_not(self):
        for alive in (True, False):
            with self.subTest(alive=alive):
                game, ctrl, char = self.fixture(1)
                mate = make_character('Derke', 'A', (2, 2))
                mate.is_alive = alive
                game.chars.append(mate)
                result = ctrl.decide_move(char, self.state(game))
                self.assertEqual(result[0], [4, 9] if alive else [4, 10])

    def test_no_drop_and_existing_carrier_keep_normal_lamp_priority(self):
        game, ctrl, char = self.fixture(1)
        game.spike_pos = None
        for has_spike in (False, True):
            char.has_spike = has_spike
            self.assertEqual(ctrl.decide_move(char, self.state(game)),
                             ([4, 9], {'ability': 'RAMP'}))

    def test_already_on_dropped_spike_waits_for_pickup_without_casting(self):
        # One tick for pickup + six moves to B + four plant ticks + margin5.
        game, ctrl, char = self.fixture(15)
        game.spike_pos = tuple(char.pos)
        result = ctrl.decide_move(char, self.state(game))
        self.assertEqual(result[0], char.pos)
        self.assertNotEqual(result[1], {'ability': 'RAMP'})

    def test_recovery_then_nearest_registered_plant_preserves_lamp_charges(self):
        game, ctrl, char = self.fixture(14)
        ctrl.target = (4, 3)  # Earlier random selection was the farther site.
        game.available_orbs = {(4, 9)}
        char.ultimate_name = 'MONITOR'
        char.ultimate_points = char.ultimate_cost
        game.battle_tick = 25
        for _ in range(14):
            game.move_character(char)
            if game.spike_pos is not None and tuple(char.pos) == game.spike_pos:
                # Apply the engine's automatic pickup between movement ticks.
                char.has_spike, game.spike_pos = True, None
            game.battle_tick += 1
            if game.is_planted:
                break
            game.round_timer -= 1
        self.assertTrue(game.is_planted)
        self.assertEqual(game.planted_pos, (4, 15))
        self.assertEqual(char.ramp_charges, 2)
        self.assertEqual(getattr(game, 'ramp_traps', []), [])
        self.assertIn((4, 9), game.available_orbs)
        self.assertEqual(char.ultimate_points, char.ultimate_cost)
        self.assertGreater(game.round_timer, 0)
        ctrl.decide_move(char, self.state(game))
        self.assertIsNone(ctrl.engineer_recovery_target)

    def test_touyama_roster_keeps_a_only_during_urgent_recovery(self):
        game, ctrl, char = self.fixture(18)
        game.defender_roster = get_preset('Touyama Gaming').players
        result = ctrl.decide_move(char, self.state(game))
        self.assertEqual(result[0], [4, 10])
        self.assertEqual(ctrl.target, (4, 3))

    def test_recovery_stays_active_after_pickup_and_round_reset_clears_it(self):
        game, ctrl, char = self.fixture()
        ctrl.decide_move(char, self.state(game))
        char.pos, char.has_spike, game.spike_pos = [4, 11], True, None
        game.round_timer = 100  # Recovery must not flip back to lamps.
        result = ctrl.decide_move(char, self.state(game))
        self.assertEqual(result[0], [4, 12])
        ctrl.reset_round()
        self.assertIsNone(ctrl.engineer_recovery_target)
        self.assertEqual(ctrl.decide_move(char, self.state(game))[0], [4, 10])


if __name__ == '__main__':
    unittest.main()
