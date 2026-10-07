import unittest
from types import SimpleNamespace
from unittest.mock import patch

import combo_awakening


class TestGame(combo_awakening.ComboAwakeningMixin):
    def __init__(self, chars):
        self.chars = chars
        self.active_player_combos = []


class PlayerComboRenameTest(unittest.TestCase):
    def test_renamed_player_can_trigger_following_combo(self):
        player_a = self._player("a")
        player_c = self._player("c")
        player_x = self._player("x")

        game = TestGame([player_a, player_c, player_x])

        combos = [
            {
                "name": "rename a to b",
                "players": ("a", "x"),
                "bonuses": {},
                "renames": {"a": "b"},
            },
            {
                "name": "b and c",
                "players": ("b", "c"),
                "bonuses": {},
                "renames": {},
            },
        ]

        with patch.object(combo_awakening, "PLAYER_COMBOS", combos):
            game._apply_player_combos()

        self.assertEqual(player_a.display_name, "b")
        self.assertEqual(
            [combo["name"] for combo in game.active_player_combos],
            ["rename a to b", "b and c"],
        )

    @staticmethod
    def _player(name):
        return SimpleNamespace(
            name=name,
            display_name=name,
            team="A",
            active_combos=[],
            accuracy=0.8,
            hs_rate=0.3,
            dodge_rate=0.2,
            reaction=100.0,
            iq=100.0,
            effective_iq=100.0,
            mental=5.0,
            condition_bonus=0.0,
            form_variance=0.0,
            move_steps_per_tick=1,
        )


if __name__ == "__main__":
    unittest.main()