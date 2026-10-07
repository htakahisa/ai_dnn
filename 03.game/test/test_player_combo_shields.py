"""Shield combo operations affect combat, respect toggles, and reset each round."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import combo_awakening
import game_core
from effect_text_generator import generate_combo_effect_text
from game_core import _apply_combo_bonus, absorb_shield_damage
from map_data import NEW_MAZE_STR
from run_game import VisualFPSBattle, _build_team_ai
from test_ultimate_system import UltimateTestGame, make_character


class ComboGame(combo_awakening.ComboAwakeningMixin):
    def __init__(self, chars):
        self.chars = chars


class PlayerComboShieldTests(unittest.TestCase):
    def player(self, name="a", team="A", **kwargs):
        values = dict(name=name, display_name=name, team=team, active_combos=[],
                      max_shield_hp=50, shield_hp=50, shield_piercer=True,
                      shield_crash=20, accuracy=.5, hs_rate=.2, iq=100,
                      effective_iq=100)
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_shield_capacity_and_crash_add_subtract_and_set(self):
        for stat, initial in (("shield_hp", 50), ("shield_crash", 20)):
            for bonus, expected in ((25, initial + 25), (-10, initial - 10),
                                    (-100, 0), ({"set": 12.5}, 12.5),
                                    ({"set": -20}, 0), ({"add": 5}, initial + 5)):
                with self.subTest(stat=stat, bonus=bonus):
                    char = self.player()
                    self.assertTrue(_apply_combo_bonus(char, stat, bonus))
                    self.assertEqual(getattr(char, stat), expected)
                    if stat == "shield_hp":
                        self.assertEqual(char.max_shield_hp, expected)

    def test_capacity_adjustment_preserves_already_absorbed_damage(self):
        char = self.player(shield_hp=10)
        self.assertTrue(_apply_combo_bonus(char, "shield_hp", 20))
        self.assertEqual((char.max_shield_hp, char.shield_hp), (70, 30))
        self.assertTrue(_apply_combo_bonus(char, "shield_hp", {"set": 60}))
        self.assertEqual((char.max_shield_hp, char.shield_hp), (60, 20))
        self.assertTrue(_apply_combo_bonus(char, "shield_hp", -50))
        self.assertEqual((char.max_shield_hp, char.shield_hp), (10, 0))

    def test_piercer_enables_and_disables_existing_trait(self):
        char = self.player()
        for value, expected in ((False, False), (True, True), ({"set": False}, False)):
            self.assertTrue(_apply_combo_bonus(char, "shield_piercer", value))
            self.assertIs(char.shield_piercer, expected)

    def test_invalid_operations_do_not_change_any_shield_fields(self):
        cases = [(stat, value) for stat in ("shield_hp", "shield_crash")
                 for value in (True, None, float("nan"), float("inf"),
                               {"set": float("nan")}, {"multiply": 2},
                               {"set": 50, "add": 10}, {}, [])]
        cases += [("shield_piercer", value) for value in
                  (0, 1, "True", None, {"set": 1}, {"add": True})]
        for stat, value in cases:
            with self.subTest(stat=stat, value=value):
                char = self.player()
                before = vars(char).copy()
                self.assertFalse(_apply_combo_bonus(char, stat, value))
                self.assertEqual(vars(char), before)

    def test_aliases_match_execution_and_description(self):
        for key, field, value, expected in (
            ("シールド", "shield_hp", 10, 60),
            ("シールドHP", "shield_hp", {"set": 15}, 15),
            ("シールドピアサー", "shield_piercer", False, False),
            ("シールドクラッシュ", "shield_crash", -5, 15),
            ("shield_crush", "shield_crash", 5, 25),
        ):
            with self.subTest(key=key):
                char = self.player()
                self.assertTrue(_apply_combo_bonus(char, key, value))
                self.assertEqual(getattr(char, field), expected)
                text = generate_combo_effect_text({"bonuses": {key: value}})
                self.assertIn("シールド", text)

    def test_common_and_individual_bonuses_stack_in_definition_order_on_both_sides(self):
        chars = [self.player(name, team) for team in ("A", "D") for name in ("a", "b", "c")]
        combos = [
            dict(name="first", players=("a", "b"),
                 bonuses=dict(shield_hp=20, shield_crash=5, shield_piercer=False),
                 player_bonuses={"a": dict(shield_hp={"set": 100}, shield_piercer=True)}),
            dict(name="second", players=("a", "b"),
                 bonuses=dict(shield_hp=-10, shield_crash={"set": 7})),
            dict(name="missing partner", players=("a", "missing"), bonuses=dict(shield_hp=999)),
        ]
        game = ComboGame(chars)
        with patch.object(combo_awakening, "PLAYER_COMBOS", combos):
            game._apply_player_combos()
        self.assertEqual(len(game.active_player_combos), 4)
        for char in chars:
            expected = {"a": (90, True, 7), "b": (60, False, 7), "c": (50, True, 20)}[char.name]
            self.assertEqual((char.shield_hp, char.shield_piercer, char.shield_crash), expected)

    def test_combo_shields_and_attack_traits_change_actual_gunfire(self):
        for piercer, expected in ((True, (40, 60)), (False, (0, 100))):
            with self.subTest(piercer=piercer):
                table = game_core._character_stats.CHARACTER_TABLE
                with patch.dict(table, {name: replace(table[name], shield_hp=0,
                                                     shield_piercer=False, shield_crash=0)
                                        for name in ("Chronicle", "Demon1")}):
                    shooter = make_character("Chronicle", "A", (4, 4))
                    target = make_character("Demon1", "D", (4, 6))
                combos = [dict(name="attack", players=("Chronicle",),
                               bonuses=dict(shield_piercer=piercer, shield_crash={"set": 10})),
                          dict(name="defense", players=("Demon1",), bonuses=dict(shield_hp=50))]
                with patch.object(combo_awakening, "PLAYER_COMBOS", combos):
                    ComboGame([shooter, target])._apply_player_combos()
                shooter.facing = target.facing = "E"
                shooter.hs_rate = 0
                game = UltimateTestGame()
                game.chars = [shooter, target]
                with patch("battle_logic.random.random", return_value=0):
                    game._resolve_all_shots()
                self.assertEqual((target.shield_hp, target.hp), expected)

    def test_each_round_resets_bonuses_and_disabled_shields_stay_disabled(self):
        attackers = ["Alfajer", "Boaster", "Chronicle", "Derke", "Leo"]
        defenders = ["Demon1", "jawgemo", "Ethan", "Boostio", "C0M"]
        combos = [dict(name="shields", players=tuple(roster[:2]),
                       bonuses=dict(shield_hp=20, shield_crash=5, shield_piercer=True))
                  for roster in (attackers, defenders)]
        table = game_core._character_stats.CHARACTER_TABLE
        definitions = {name: replace(table[name], shield_hp=50, shield_crash=10, shield_piercer=False)
                       for name in attackers + defenders}
        for enabled in (True, False):
            with self.subTest(enabled=enabled), patch.dict(table, definitions), \
                    patch.object(combo_awakening, "PLAYER_COMBOS", combos), redirect_stdout(StringIO()):
                game = VisualFPSBattle(NEW_MAZE_STR, _build_team_ai("default"), _build_team_ai("default"),
                                       headless=True, attacker_roster=attackers, defender_roster=defenders,
                                       shield_abilities_enabled=enabled)
                for round_number in (1, 2):
                    for char in game.chars:
                        affected = char.name in attackers[:2] + defenders[:2]
                        expected = (70 if affected else 50, affected, 15 if affected else 10) if enabled else (0, False, 0)
                        self.assertEqual((char.shield_hp, char.shield_piercer, char.shield_crash), expected)
                        self.assertEqual(char.max_shield_hp, expected[0])
                        char.hp -= absorb_shield_damage(char, 200)
                    if round_number == 1:
                        game.current_round = 2
                        game.init_round()

    def test_shield_toggle_does_not_disable_existing_combo_effects(self):
        char = self.player(shield_abilities_enabled=False, max_shield_hp=0,
                           shield_hp=0, shield_crash=0, shield_piercer=False)
        combos = [dict(name="mixed", players=("a",),
                       bonuses=dict(accuracy=.1, iq=10, shield_hp=100,
                                    shield_piercer=True, shield_crash=20))]
        with patch.object(combo_awakening, "PLAYER_COMBOS", combos):
            ComboGame([char])._apply_player_combos()
        self.assertAlmostEqual(char.accuracy, .6)
        self.assertEqual((char.iq, char.effective_iq), (110, 110))
        self.assertEqual((char.shield_hp, char.max_shield_hp, char.shield_piercer, char.shield_crash),
                         (0, 0, False, 0))

    def test_generated_descriptions_include_sets_and_disabling_piercer(self):
        combo = dict(bonuses=dict(shield_hp=50, shield_crash=-10, shield_piercer=False),
                     player_bonuses={"a": dict(shield_hp={"set": 100}, shield_crash={"set": 20})})
        text = generate_combo_effect_text(combo)
        for expected in ("シールド +50HP", "シールドクラッシュ -10HP", "シールドピアサー 無効化",
                         "シールドを100HPに設定", "シールドクラッシュを20HPに設定"):
            self.assertIn(expected, text)
        description = ComboGame([])._describe_bonuses(dict(shield_hp={"set": 100}, shield_piercer=False))
        self.assertIn("シールドを100HPに設定", description)
        self.assertIn("シールドピアサー 無効化", description)
        self.assertEqual(generate_combo_effect_text(dict(combo, effect_text="手動説明")), "手動説明")

    def test_expiring_awakening_retains_combo_capacity_without_refunding_damage(self):
        char = self.player()
        _apply_combo_bonus(char, "shield_hp", 20)
        snapshot = combo_awakening._snapshot_character_awakening_state(char)
        _apply_combo_bonus(char, "shield_hp", 30)
        _apply_combo_bonus(char, "shield_crash", 10)
        _apply_combo_bonus(char, "shield_piercer", False)
        absorb_shield_damage(char, 40)
        combo_awakening._restore_character_awakening_state(char, snapshot)
        self.assertEqual((char.max_shield_hp, char.shield_hp, char.shield_crash, char.shield_piercer),
                         (70, 30, 20, True))


if __name__ == "__main__":
    unittest.main()
