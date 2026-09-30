import unittest

from combo_awakening import ComboAwakeningMixin
from game_core import Character
from map_data import NEW_MAZE_STR
from run_game import VisualFPSBattle, _build_team_ai


class RoundAbilityInventoryTests(unittest.TestCase):
    def test_recon_starts_with_two_charges_on_both_sides_every_round(self):
        game = VisualFPSBattle(
            NEW_MAZE_STR,
            _build_team_ai("default"),
            _build_team_ai("default"),
            headless=True,
            attacker_roster=["Alfajer", "Boaster", "Chronicle", "Derke", "Leo"],
            defender_roster=["Demon1", "jawgemo", "Ethan", "Boostio", "C0M"],
        )
        for round_number in (1, 2):
            with self.subTest(round=round_number):
                seekers = [char for char in game.chars if char.ability_name == "RECON"]
                self.assertEqual({char.team for char in seekers}, {"A", "D"})
                for char in seekers:
                    self.assertEqual(char.recon_charges, 2)
                    char.recon_charges = 0
            if round_number == 1:
                game.current_round = 2
                game.init_round()

    def test_round_start_preserves_defaults_and_explicit_ability_bonus(self):
        game = ComboAwakeningMixin()
        game.chars = [
            Character(name, "A", [1, 1], "white", "red")
            for name in ("Leo", "Boaster", "Chronicle", "Alfajer")
        ]
        game._apply_round_start_effects()
        self.assertEqual(game.chars[0].recon_charges, 2)
        self.assertEqual(game.chars[1].smoke_charges, 1)
        self.assertEqual(game.chars[2].flash_charges, 1)
        self.assertEqual(game.chars[3].ramp_charges, 2)

        for char in game.chars[:3]:
            char.abilities_per_round = 3
        game._apply_round_start_effects()
        self.assertEqual(game.chars[0].recon_charges, 3)
        self.assertEqual(game.chars[1].smoke_charges, 3)
        self.assertEqual(game.chars[2].flash_charges, 3)


if __name__ == "__main__":
    unittest.main()
