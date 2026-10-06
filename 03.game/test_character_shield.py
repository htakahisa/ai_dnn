"""Round shields absorb damage before HP across gunfire and ongoing effects."""

from contextlib import redirect_stdout
from dataclasses import asdict, replace
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import game_core
from character_stats import CharacterStats
from game_core import absorb_shield_damage
from map_data import NEW_MAZE_STR
from realtime_season import SeasonSaveError, player_from_save, validate_player
from run_game import VisualFPSBattle, _build_team_ai
from test_ultimate_system import UltimateTestGame, make_character


class CharacterShieldTests(unittest.TestCase):
    def shield_definition(self, name, hp):
        table = game_core._character_stats.CHARACTER_TABLE
        return patch.dict(table, {name: replace(table[name], shield_hp=hp)})

    def shooting_game(self, shield_hp, *, target_name="Demon1", headshot=False,
                      piercer=False, crash=0):
        with self.shield_definition(target_name, shield_hp):
            target = make_character(target_name, "D", (4, 6))
        table = game_core._character_stats.CHARACTER_TABLE
        with patch.dict(table, {"Chronicle": replace(table["Chronicle"],
                                                     shield_piercer=piercer, shield_crash=crash)}):
            shooter = make_character("Chronicle", "A", (4, 4))
        shooter.facing = target.facing = "E"
        shooter.hs_rate = 1 if headshot else 0
        game = UltimateTestGame()
        game.chars = [shooter, target]
        return game, shooter, target

    def fire(self, game):
        with patch("battle_logic.random.random", return_value=0):
            game._resolve_all_shots()

    def test_stat_order_and_defaults(self):
        player = CharacterStats("Example", .3, .2, 100, .7, 100, "フラッシュ", 50,
                                5, 9, 30, True, 20, 100_000, 7, 2)
        self.assertEqual((player.mental, player.shield_hp, player.shield_piercer,
                          player.shield_crash, player.monthly_salary,
                          player.loyalty, player.debut_chapter), (9, 30, True, 20, 100_000, 7, 2))
        self.assertEqual(game_core.get_character_combat_stats("unknown")["shield_hp"], 0)
        defaults = game_core.get_character_combat_stats("unknown")
        self.assertEqual((defaults["shield_piercer"], defaults["shield_crash"]), (False, 0))

    def test_piercer_bypasses_shield_on_body_and_headshots(self):
        for headshot in (False, True):
            with self.subTest(headshot=headshot):
                game, shooter, target = self.shooting_game(200, headshot=headshot, piercer=True)
                self.assertTrue(shooter.shield_piercer)
                with patch.object(game, "_kill_character") as kill:
                    self.fire(game)
                    self.assertEqual(target.shield_hp, 200)
                    self.assertEqual(target.hp, 0 if headshot else 60)
                    self.assertEqual(kill.call_count, 1 if headshot else 0)

    def test_crash_only_adds_shield_damage_and_applies_each_hit(self):
        for shield, expected_shield, expected_hp in ((0, 0, 60), (30, 0, 90),
                                                    (50, 0, 100), (90, 30, 100)):
            with self.subTest(shield=shield):
                game, shooter, target = self.shooting_game(shield, crash=20)
                self.assertEqual(shooter.shield_crash, 20)
                self.fire(game)
                self.assertEqual((target.shield_hp, target.hp), (expected_shield, expected_hp))
                self.fire(game)
                self.assertEqual((target.shield_hp, target.hp),
                                 (0, expected_hp - max(0, 40 - expected_shield)))

    def test_piercer_and_crash_combine(self):
        game, _, target = self.shooting_game(50, piercer=True, crash=20)
        self.fire(game)
        self.assertEqual((target.shield_hp, target.hp), (30, 60))

    def test_misses_do_not_trigger_crash(self):
        game, _, target = self.shooting_game(50, crash=20)
        with patch("battle_logic.random.random", return_value=1):
            game._resolve_all_shots()
        self.assertEqual((target.shield_hp, target.hp), (50, 100))

    def test_attack_traits_do_not_change_absorption_without_attacker(self):
        target = SimpleNamespace(shield_hp=50, shield_piercer=True, shield_crash=20)
        self.assertEqual(absorb_shield_damage(target, 40), 0)
        self.assertEqual(target.shield_hp, 10)

    def test_body_shots_absorb_break_and_pass_excess_to_hp(self):
        for shield, expected_shield, expected_hp in ((0, 0, 60), (30, 0, 90),
                                                    (40, 0, 100), (70, 30, 100)):
            with self.subTest(shield=shield):
                game, _, target = self.shooting_game(shield)
                self.fire(game)
                self.assertEqual((target.shield_hp, target.hp), (expected_shield, expected_hp))
                self.assertTrue(target.is_alive)
                self.assertEqual(game.last_shot["damage"], 40)
                self.fire(game)
                self.assertEqual(target.hp, expected_hp - max(0, 40 - expected_shield))

    def test_headshot_can_be_fully_absorbed_or_kill_with_excess(self):
        game, shooter, target = self.shooting_game(200, headshot=True)
        self.fire(game)
        self.assertEqual((target.shield_hp, target.hp), (40, 100))
        with patch.object(game, "_kill_character") as kill:
            self.fire(game)
            self.assertEqual((target.shield_hp, target.hp), (0, 0))
            kill.assert_called_once_with(shooter, target)

    def test_iron_will_uses_only_lethal_damage_to_body(self):
        game, _, target = self.shooting_game(170, target_name="ひつじさん", headshot=True)
        self.fire(game)
        self.assertEqual((target.shield_hp, target.hp, target.iron_will_charges), (10, 100, 1))
        self.fire(game)
        self.assertEqual((target.shield_hp, target.hp, target.iron_will_charges), (0, 1, 0))

    def test_neon_uses_shield_then_hp(self):
        game, owner, target = self.shooting_game(15)
        game.neon_bursts = [{"phase": "active", "remaining_ticks": 10,
                             "owner": owner.name, "team": owner.team,
                             "cells": {tuple(target.pos)}}]
        game._advance_engineer_effects()
        self.assertEqual((target.shield_hp, target.hp), (5, 100))
        game._advance_engineer_effects()
        self.assertEqual((target.shield_hp, target.hp), (0, 95))

    def test_neon_uses_owner_traits_even_after_owner_dies(self):
        for piercer, crash, expected_shield, expected_hp in (
            (True, 0, 50, 80), (False, 20, 0, 100), (True, 20, 10, 80)
        ):
            with self.subTest(piercer=piercer, crash=crash):
                game, owner, target = self.shooting_game(50, piercer=piercer, crash=crash)
                owner.is_alive = False
                game.neon_bursts = [{"phase": "active", "remaining_ticks": 10,
                                     "owner": owner.name, "team": owner.team,
                                     "cells": {tuple(target.pos)}}]
                game._advance_engineer_effects()
                game._advance_engineer_effects()
                self.assertEqual((target.shield_hp, target.hp), (expected_shield, expected_hp))

    def test_life_contract_reduces_body_hp_cap_only_after_shield(self):
        game, owner, target = self.shooting_game(15)
        target.life_contract_remaining = 3
        target.life_contract_owner = owner.name
        for expected in ((5, 100, 100, 0), (0, 95, 95, 5), (0, 85, 85, 15)):
            game._advance_destruction_areas()
            self.assertEqual((target.shield_hp, target.hp, target.max_hp,
                              target.contract_max_hp_lost), expected)

    def test_life_contract_uses_owner_traits_each_damage_tick(self):
        for piercer, crash, expected_shield, expected_hp in (
            (True, 0, 50, 80), (False, 20, 0, 100), (True, 20, 10, 80)
        ):
            with self.subTest(piercer=piercer, crash=crash):
                game, owner, target = self.shooting_game(50, piercer=piercer, crash=crash)
                target.life_contract_remaining = 3
                target.life_contract_owner = owner.name
                game._advance_destruction_areas()
                game._advance_destruction_areas()
                self.assertEqual((target.shield_hp, target.hp, target.max_hp),
                                 (expected_shield, expected_hp, expected_hp))

    def test_every_round_restores_shields_on_both_sides_and_records_replay(self):
        with self.shield_definition("Leo", 50), self.shield_definition("Ethan", 50), redirect_stdout(StringIO()):
            game = VisualFPSBattle(
                NEW_MAZE_STR, _build_team_ai("default"), _build_team_ai("default"),
                headless=True,
                attacker_roster=["Alfajer", "Boaster", "Chronicle", "Derke", "Leo"],
                defender_roster=["Demon1", "jawgemo", "Ethan", "Boostio", "C0M"],
            )
            for round_number in (1, 2):
                shielded = [c for c in game.chars if c.name in {"Leo", "Ethan"}]
                self.assertEqual({c.team for c in shielded}, {"A", "D"})
                for char in shielded:
                    self.assertEqual((char.shield_hp, char.max_shield_hp), (50, 50))
                    char.hp -= absorb_shield_damage(char, 60)
                    self.assertEqual((char.shield_hp, char.hp), (0, 90))
                if round_number == 1:
                    game._record_replay_frame()
                    row = next(c for c in game.replay_frames[-1]["chars"] if c["name"] == "Leo")
                    self.assertEqual((row["shield_hp"], row["max_shield_hp"]), (0, 50))
                    game.current_round = 2
                    game.init_round()

    def test_saved_shield_values_and_legacy_defaults(self):
        player = CharacterStats("Example", .3, .2, 100, .7, 100, "フラッシュ", 50,
                                shield_hp=25)
        saved = asdict(player)
        self.assertEqual(player_from_save(saved).shield_hp, 25)
        saved.pop("shield_hp")
        self.assertEqual(player_from_save(saved).shield_hp, 0)
        with patch("realtime_season.get_by_name", return_value=player):
            self.assertEqual(player_from_save(saved).shield_hp, 25)
        validate_player(player_from_save(asdict(player)))

    def test_legacy_targets_and_zero_damage(self):
        target = SimpleNamespace(hp=100)
        self.assertEqual(absorb_shield_damage(target, 40), 40)
        target.shield_hp = 30
        self.assertEqual(absorb_shield_damage(target, 0), 0)
        self.assertEqual(target.shield_hp, 30)

    def test_zero_damage_does_not_pierce_or_crash(self):
        target = SimpleNamespace(shield_hp=30)
        attacker = SimpleNamespace(shield_piercer=True, shield_crash=20)
        self.assertEqual(absorb_shield_damage(target, 0, attacker), 0)
        self.assertEqual(target.shield_hp, 30)

    def test_save_preserves_traits_and_legacy_rows_inherit_catalog_defaults(self):
        player = CharacterStats("Example", .3, .2, 100, .7, 100, "フラッシュ", 50,
                                shield_piercer=True, shield_crash=20)
        saved = asdict(player)
        loaded = player_from_save(saved)
        validate_player(loaded)
        self.assertEqual((loaded.shield_piercer, loaded.shield_crash), (True, 20))
        saved.pop("shield_piercer")
        saved.pop("shield_crash")
        loaded = player_from_save(saved)
        self.assertEqual((loaded.shield_piercer, loaded.shield_crash), (False, 0))
        with patch("realtime_season.get_by_name", return_value=player):
            loaded = player_from_save(saved)
            self.assertEqual((loaded.shield_piercer, loaded.shield_crash), (True, 20))
            saved.update(shield_piercer=False, shield_crash=5)
            loaded = player_from_save(saved)
            self.assertEqual((loaded.shield_piercer, loaded.shield_crash), (False, 5))

    def test_save_rejects_invalid_attack_trait_values(self):
        player = CharacterStats("Example", .3, .2, 100, .7, 100, "フラッシュ", 50)
        for value in (1, 0, "True", "False", None):
            with self.subTest(piercer=value), self.assertRaises(SeasonSaveError):
                validate_player(replace(player, shield_piercer=value))
        for value in (-1, float("nan"), float("inf"), True):
            with self.subTest(crash=value), self.assertRaises(SeasonSaveError):
                validate_player(replace(player, shield_crash=value))


if __name__ == "__main__":
    unittest.main()
