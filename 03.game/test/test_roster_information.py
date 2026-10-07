"""Public lineups reach controllers during setup and live IQ perception."""

import unittest
from types import SimpleNamespace

from game_core import get_character_combat_stats
from iq_controller_adapter import IQAwareController
from iq_perception import UNKNOWN_ENEMY_POS
from roster_utils import roster_information
from team_ai import PrivateInfoController
from test_enemy_position_visibility import RecordingController
from test_ultimate_system import UltimateTestGame, make_character


class RosterInformationTests(unittest.TestCase):
    def fixture(self):
        game = UltimateTestGame()
        ally = make_character("Absol", "A", (1, 1))
        enemy = make_character("Leo", "D", (7, 8))
        duplicate = make_character("Absol", "D", (6, 8))
        duplicate.name = "Absol_2"
        duplicate.is_alive = False
        game.chars = [ally, enemy, duplicate]
        return game, ally, enemy

    def test_lineup_includes_hidden_dead_and_duplicate_players(self):
        game, ally, enemy = self.fixture()
        state = roster_information(game.chars, ally.team)
        self.assertEqual([p["name"] for p in state["enemy_roster"]], ["Leo", "Absol_2"])
        self.assertEqual(state["enemy_roster"][1]["base_name"], "Absol")
        entry = state["enemy_roster"][0]
        self.assertEqual(entry["role"], enemy.role)
        self.assertEqual(entry["ability_name"], enemy.ability_name)
        self.assertEqual(entry["ultimate_name"], enemy.ultimate_name)
        self.assertEqual(entry["stats"], get_character_combat_stats("Leo"))
        self.assertEqual(set(entry), {"name", "base_name", "team", "role", "ability_name",
                                      "ultimate_name", "ultimate_cost", "stats"})
        enemy.pos = [4, 4]
        enemy.hp = 1
        enemy.has_spike = True
        enemy.accuracy = 99
        self.assertEqual(roster_information(game.chars, ally.team), state)
        state["enemy_roster"][0]["stats"]["accuracy"] = -1
        self.assertNotEqual(roster_information(game.chars, ally.team), state)

    def test_live_input_survives_iq_and_private_information_wrappers(self):
        for wrapper in (IQAwareController, PrivateInfoController):
            for team in ("A", "D"):
                with self.subTest(wrapper=wrapper.__name__, team=team):
                    game, ally, enemy = self.fixture()
                    actor = ally if team == "A" else enemy
                    inner = RecordingController()
                    controller = wrapper(inner)
                    controller.set_game(game)
                    if team == "A":
                        game.attacker_controller = controller
                    else:
                        game.defender_controller = controller
                    game.move_character(actor)
                    state = inner.seen[1]
                    self.assertEqual(state["ally_roster"][0]["name"], actor.name)
                    self.assertEqual([p["name"] for p in state["enemy_roster"]],
                                     ["Leo", "Absol_2"] if team == "A" else ["Absol"])
                    hidden = next(c for c in state["chars"] if c.team != team)
                    self.assertEqual(tuple(hidden.pos), UNKNOWN_ENEMY_POS)

    def test_setup_receives_full_lineup_and_side_swap_reverses_it(self):
        game, ally, enemy = self.fixture()
        game.defender_setup_phase = SimpleNamespace(
            ticks_remaining=5, can_move_to=lambda team, row, col: True)
        inner = RecordingController()
        game.defender_controller = inner
        game._move_character_during_defender_setup(enemy)
        self.assertEqual([p["name"] for p in inner.seen[1]["enemy_roster"]], ["Absol"])
        for char in game.chars:
            char.team = "D" if char.team == "A" else "A"
        state = roster_information(game.chars, "D")
        self.assertEqual([p["name"] for p in state["enemy_roster"]], ["Leo", "Absol_2"])


if __name__ == "__main__":
    unittest.main()
