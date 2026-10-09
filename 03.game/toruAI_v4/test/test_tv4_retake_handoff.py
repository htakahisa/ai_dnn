"""A dead defuser cannot reserve the legal action for the entire team."""
from dataclasses import replace
import unittest
from frc_v1.actions import build_masks, KINDS


class RetakeHandoffTests(unittest.TestCase):
    def test_engine_death_keeps_timer_but_successor_has_legal_defuse(self):
        from test.test_ultimate_system import UltimateTestGame, make_character
        from frc_v1.perception import FrcPerceptionBuilder
        game = UltimateTestGame()
        shooter = make_character('Demon1', 'A', (4, 8))
        defenders = [make_character(n, 'D', p) for n, p in zip(
            ('Xdll', 'Chronicle', 'Leo', 'Derke', 'Boaster'), ((4, 4), (4, 5), (2, 2), (2, 3), (2, 4)))]
        game.chars = [shooter, *defenders]
        game.is_planted, game.planted_pos = True, (4, 4)
        fallen = defenders[0]
        fallen.defuse_timer, game.active_defuser_name = 3, fallen.name
        game._kill_character(shooter, fallen)
        self.assertFalse(fallen.is_alive)
        self.assertEqual(fallen.defuse_timer, 3)
        self.assertIsNone(game.active_defuser_name)
        snapshot = FrcPerceptionBuilder('D').build(game)
        successor = next(a for a in snapshot.allies if a.name == defenders[1].name)
        self.assertTrue(build_masks(snapshot).kind[successor.slot, KINDS.index('DEFUSE')])

    def test_dead_progress_releases_defuse_but_living_progress_reserves_it(self):
        from toruAI_v4.test.test_tv4_retake import RetakeTests
        from toruAI_v4.tv4_scenario import Scenario
        fixture = RetakeTests()
        fixture.scenario = Scenario()
        snapshot = fixture.snapshot()
        fallen = replace(snapshot.allies[0], alive=False, defuse_progress=3, position=(7, 3))
        successor = replace(snapshot.allies[1], alive=True, position=(8, 4), defuse_progress=0)
        state = replace(snapshot, allies=(fallen, successor, *snapshot.allies[2:]))
        index = KINDS.index('DEFUSE')
        self.assertTrue(build_masks(state).kind[successor.slot, index])
        living = replace(state, allies=(replace(fallen, alive=True), successor, *snapshot.allies[2:]))
        self.assertFalse(build_masks(living).kind[successor.slot, index])
        self.assertTrue(build_masks(living).kind[fallen.slot, index])


if __name__ == '__main__':
    unittest.main()
