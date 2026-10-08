"""HP-only attack traits, exact durations, movement, and persistent HP caps."""

from contextlib import redirect_stdout
from dataclasses import asdict, fields, replace
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import game_core
from abilities_los import MonitorDrone
from character_stats import CharacterStats
from combo_awakening import (
    ComboAwakeningMixin,
    _restore_character_awakening_state,
    _snapshot_character_awakening_state,
)
from game_core import apply_on_hp_damage
from player_details_ui import shield_stats_text
from realtime_season import SeasonSaveError, player_from_save, validate_player
from rendering_ui import RenderingUIMixin
from test_ultimate_system import FixedController, UltimateTestGame, make_character


class StepController:
    def decide_move(self, char, state):
        return [char.pos[0], char.pos[1] + 1], {"facing": "N"}


class OnHitTraitsTests(unittest.TestCase):
    def shooting_game(self, *, shield=0, curse=3, fate=2, piercer=False,
                      target_name="Demon1", headshot=False):
        table = game_core._character_stats.CHARACTER_TABLE
        with patch.dict(table, {
            "Chronicle": replace(table["Chronicle"], erosion_curse=curse,
                                 fate_loom=fate, shield_piercer=piercer),
            target_name: replace(table[target_name], shield_hp=shield),
        }):
            shooter = make_character("Chronicle", "A", (4, 4))
            target = make_character(target_name, "D", (4, 6))
        shooter.facing = target.facing = "E"
        shooter.hs_rate = 1 if headshot else 0
        game = UltimateTestGame()
        game.chars = [shooter, target]
        return game, shooter, target

    def fire(self, game):
        with patch("battle_logic.random.random", return_value=0):
            game._resolve_all_shots()

    def test_stats_follow_shield_crash_without_changing_positional_salary(self):
        names = [field.name for field in fields(CharacterStats)]
        start = names.index("shield_crash")
        self.assertEqual(names[start:start + 4],
                         ["shield_crash", "erosion_curse", "fate_loom", "monthly_salary"])
        player = CharacterStats("Example", .3, .2, 100, .7, 100, "フラッシュ", 50,
                                5, 9, 30, True, 20, 100_000, 7, 2,
                                erosion_curse=3, fate_loom=2)
        self.assertEqual((player.monthly_salary, player.loyalty, player.debut_chapter),
                         (100_000, 7, 2))
        self.assertEqual((player.erosion_curse, player.fate_loom), (3, 2))
        unknown = game_core.get_character_combat_stats("unknown")
        self.assertEqual((unknown["erosion_curse"], unknown["fate_loom"]), (0, 0))

    def test_shields_partial_damage_and_piercing(self):
        for shield, piercer, expected_hp in ((0, False, 60), (30, False, 90),
                                            (40, False, 100), (70, False, 100),
                                            (70, True, 60)):
            with self.subTest(shield=shield, piercer=piercer):
                game, _, target = self.shooting_game(shield=shield, piercer=piercer)
                self.fire(game)
                self.assertEqual(target.hp, expected_hp)
                self.assertEqual(target.max_hp, expected_hp)
                applied = expected_hp < 100
                self.assertEqual((target.ability_seal_remaining, target.fate_loom_remaining),
                                 (3, 2) if applied else (0, 0))
                self.assertEqual(target.fate_max_hp_lost, 100 - expected_hp)

    def test_misses_crash_only_and_disabled_traits_do_not_apply(self):
        game, shooter, target = self.shooting_game(shield=50)
        shooter.shield_crash = 50
        self.fire(game)
        self.assertEqual((target.hp, target.shield_hp, target.max_hp,
                          target.ability_seal_remaining, target.fate_loom_remaining),
                         (100, 0, 100, 0, 0))
        with patch("battle_logic.random.random", return_value=1):
            game._resolve_all_shots()
        self.assertEqual((target.hp, target.ability_seal_remaining), (100, 0))
        shooter.erosion_curse = shooter.fate_loom = 0
        self.fire(game)
        self.assertEqual((target.hp, target.max_hp, target.ability_seal_remaining,
                          target.fate_loom_remaining), (60, 100, 0, 0))

    def test_traits_are_independent_and_repeated_damage_accumulates_hp_cap(self):
        for curse, fate in ((3, 0), (0, 2), (3, 2)):
            with self.subTest(curse=curse, fate=fate):
                game, _, target = self.shooting_game(curse=curse, fate=fate)
                self.fire(game)
                self.fire(game)
                self.assertEqual((target.hp, target.max_hp), (20, 20 if fate else 100))
                self.assertEqual((target.ability_seal_remaining, target.fate_loom_remaining),
                                 (curse, fate))

    def test_iron_will_uses_actual_hp_lost_and_survives(self):
        game, _, target = self.shooting_game(target_name="ひつじさん", headshot=True)
        self.fire(game)
        self.assertEqual((target.hp, target.max_hp, target.fate_max_hp_lost), (1, 1, 99))
        self.assertTrue(target.is_alive)
        self.assertEqual(target.iron_will_charges, 0)
        self.fire(game)
        self.assertFalse(target.is_alive)
        self.assertEqual((target.hp, target.max_hp), (0, 0))

    def test_no_trait_effect_on_allies_drones_or_unowned_damage(self):
        game, shooter, target = self.shooting_game()
        for attacker in (None, target):
            target.hp = 90
            apply_on_hp_damage(target, attacker, 100, game.battle_tick)
            self.assertEqual((target.max_hp, target.ability_seal_remaining), (100, 0))
        drone = MonitorDrone(target, 1)
        drone.hp -= 40
        apply_on_hp_damage(drone, shooter, drone.hp + 40, game.battle_tick)
        self.assertEqual(drone.max_hp, game_core.MONITOR_DRONE_HP)
        self.assertFalse(hasattr(drone, "ability_seal_remaining"))

    def test_neon_and_contract_apply_only_after_shields_with_dead_owner(self):
        for effect in ("neon", "contract"):
            with self.subTest(effect=effect):
                game, owner, target = self.shooting_game(shield=15)
                owner.is_alive = False
                if effect == "neon":
                    game.neon_bursts = [{"phase": "active", "remaining_ticks": 10,
                                         "owner": owner.name, "team": owner.team,
                                         "cells": {tuple(target.pos)}}]
                    advance = game._advance_engineer_effects
                else:
                    target.life_contract_remaining = 3
                    target.life_contract_owner = owner.name
                    advance = game._advance_destruction_areas
                advance()
                self.assertEqual((target.hp, target.max_hp, target.ability_seal_remaining),
                                 (100, 100, 0))
                advance()
                self.assertEqual((target.ability_seal_remaining, target.fate_loom_remaining), (3, 2))
                self.assertEqual(target.max_hp, 95 if effect == "neon" else 90)
                self.assertEqual(target.fate_max_hp_lost, 5)

    def test_exact_full_tick_durations_and_refresh_without_shortening(self):
        game, shooter, target = self.shooting_game()
        self.fire(game)
        game._advance_on_hit_effects()  # Damage tick does not consume the effect.
        self.assertEqual((target.ability_seal_remaining, target.fate_loom_remaining), (3, 2))
        for expected in ((2, 1), (1, 0), (0, 0)):
            game.battle_tick += 1
            game._advance_on_hit_effects()
            self.assertEqual((target.ability_seal_remaining, target.fate_loom_remaining), expected)
        target.ability_seal_remaining = 8
        target.fate_loom_remaining = 5
        target.hp -= 1
        apply_on_hp_damage(target, shooter, target.hp + 1, game.battle_tick)
        self.assertEqual((target.ability_seal_remaining, target.fate_loom_remaining), (8, 5))

    def test_root_blocks_multiple_step_moves_but_allows_turning_and_shooting(self):
        game, shooter, target = self.shooting_game(curse=0)
        target.fate_loom_remaining = 2
        target.move_steps_per_tick = 3
        game.defender_controller = StepController()
        start = list(target.pos)
        for _ in range(2):
            game.move_character(target)
            self.assertEqual(target.pos, start)
            self.assertEqual(target.facing, "N")
            game.battle_tick += 1
            game._advance_on_hit_effects()
        game.move_character(target)
        self.assertNotEqual(target.pos, start)
        shooter.fate_loom_remaining = 2
        self.fire(game)
        self.assertTrue(game.last_shots)

    def test_seal_blocks_casts_without_spending_and_releases_after_n_ticks(self):
        game, caster, _ = self.shooting_game()
        caster.ultimate_points = caster.ultimate_cost
        caster.ability_seal_remaining = 2
        initial_charges = caster.flash_charges
        flash = {"ability": "FLASH", "target": (4, 9)}
        for _ in range(2):
            self.assertFalse(game.execute_ai_ability(caster, flash))
            self.assertFalse(game.execute_ai_ultimate(caster, {"ultimate": "TUNNEL"}))
            self.assertEqual(caster.flash_charges, initial_charges)
            self.assertEqual(caster.ultimate_points, caster.ultimate_cost)
            game.battle_tick += 1
            game._advance_on_hit_effects()
        self.assertTrue(game.execute_ai_ability(caster, flash))
        self.assertTrue(game.execute_ai_ultimate(caster, {"ultimate": "TUNNEL"}))

    def test_root_allows_stationary_ability_plant_and_defuse(self):
        game, caster, target = self.shooting_game(curse=0)
        caster.fate_loom_remaining = 2
        self.assertTrue(game.execute_ai_ability(caster, {"ability": "FLASH", "target": (4, 9)}))
        game.grid[tuple(caster.pos)] = 2
        caster.has_spike = True
        game.attacker_controller = FixedController("PLANT")
        game.move_character(caster)
        self.assertEqual(caster.plant_timer, 1)
        target.fate_loom_remaining = 2
        game.is_planted = True
        game.planted_pos = tuple(target.pos)
        game.defender_controller = FixedController("DEFUSE")
        game.move_character(target)
        self.assertEqual(target.defuse_timer, 1)

    def test_root_blocks_dash_portal_and_awakening_teleport(self):
        game = UltimateTestGame()
        for name, payload in (("something", {"ultimate": "RAID"}),
                              ("Demon1", {"ultimate": "ESCAPE", "target": (7, 10)})):
            caster = make_character(name, "A", (2, 2), 99)
            caster.fate_loom_remaining = 1
            self.assertFalse(game.execute_ai_ultimate(caster, payload))
            self.assertEqual(caster.ultimate_points, caster.ultimate_cost)
        game.chars = [caster]
        game.escape_portals = [{"owner": caster.name, "pos": (7, 10), "remaining_ticks": 0}]
        game._advance_escape_portals()
        self.assertEqual(caster.pos, [2, 2])
        caster.fate_loom_remaining = 0
        game._advance_escape_portals()
        self.assertEqual(caster.pos, [7, 10])
        caster.fate_loom_remaining = 1
        caster.active_awakenings = {"Leap": {}}
        event = {"name": "Leap", "leap_on_kill": True}
        with patch("combo_awakening.AWAKENING_EVENTS", [event]):
            ComboAwakeningMixin._maybe_trigger_leap_awakening(game, caster, SimpleNamespace(pos=[2, 5]))
        self.assertEqual(caster.pos, [7, 10])

    def test_healing_and_awakening_expiration_preserve_reduced_max_hp(self):
        game, _, target = self.shooting_game()
        snapshot = _snapshot_character_awakening_state(target)
        target.max_hp = target.hp = 150
        self.fire(game)
        self.assertEqual((target.hp, target.max_hp), (110, 110))
        _restore_character_awakening_state(target, snapshot)
        self.assertEqual((target.hp, target.max_hp, target.fate_max_hp_lost), (60, 60, 40))
        healer = make_character("Leo", "D", (4, 7))
        healer.ability_name = "DANCE"
        healer.dance_charges = 2
        game.chars.append(healer)
        target.hp = 50
        self.assertTrue(game.execute_ai_ability(healer, {"ability": "DANCE", "target_name": target.name}))
        self.assertEqual((target.hp, target.max_hp), (60, 60))
        self.assertFalse(game.execute_ai_ability(healer, {"ability": "DANCE", "target_name": target.name}))
        fresh = make_character(target.name, target.team, target.pos)
        self.assertEqual((fresh.hp, fresh.max_hp, fresh.ability_seal_remaining,
                          fresh.fate_loom_remaining, fresh.fate_max_hp_lost), (100, 100, 0, 0, 0))

    def test_save_round_trip_legacy_defaults_validation_and_display(self):
        player = CharacterStats("Example", .3, .2, 100, .7, 100, "フラッシュ", 50,
                                erosion_curse=3, fate_loom=2)
        saved = asdict(player)
        loaded = player_from_save(saved)
        validate_player(loaded)
        self.assertEqual((loaded.erosion_curse, loaded.fate_loom), (3, 2))
        saved.pop("erosion_curse")
        saved.pop("fate_loom")
        loaded = player_from_save(saved)
        self.assertEqual((loaded.erosion_curse, loaded.fate_loom), (0, 0))
        with patch("realtime_season.get_by_name", return_value=player):
            loaded = player_from_save(saved)
            self.assertEqual((loaded.erosion_curse, loaded.fate_loom), (3, 2))
        for stat in ("erosion_curse", "fate_loom"):
            for value in (-1, float("nan"), float("inf"), True):
                with self.subTest(stat=stat, value=value), self.assertRaises(SeasonSaveError):
                    validate_player(replace(player, **{stat: value}))
        self.assertIn("摩耗の呪い: 3tick\n運命の織機: 2tick", shield_stats_text(player))

    def test_mouse_casts_cannot_bypass_seal_even_if_already_armed(self):
        game, caster, _ = self.shooting_game()
        caster.ability_seal_remaining = 2
        game.ability_mode = ("FLASH", caster.team, caster.name)
        game.ultimate_mode = None
        game.map_offset_x = 0
        game.map_pixel_width = 240
        game.cell_size = 20
        game._selected_user_character = lambda: caster
        game._handle_team_panel_click = lambda x, y: False
        game._plant_button_bounds = lambda: None
        game._orb_button_bounds = lambda: None
        game._ultimate_button_bounds = lambda: None
        game.draw = lambda: None
        event = SimpleNamespace(x=180, y=80 + game_core.COMBO_BANNER_HEIGHT)
        charges = caster.flash_charges
        RenderingUIMixin.on_canvas_click(game, event)
        self.assertEqual(caster.flash_charges, charges)
        self.assertEqual(game.flash_projectiles, [])
        caster.ultimate_points = caster.ultimate_cost
        self.assertFalse(RenderingUIMixin._selected_ultimate_ready(game, caster))

    def test_live_battle_ticks_replay_and_next_round_reset(self):
        from map_data import NEW_MAZE_STR
        from run_game import VisualFPSBattle, _build_team_ai

        table = game_core._character_stats.CHARACTER_TABLE
        with patch.dict(table, {"Chronicle": replace(table["Chronicle"],
                                                     erosion_curse=3, fate_loom=2)}), redirect_stdout(StringIO()):
            game = VisualFPSBattle(
                NEW_MAZE_STR, _build_team_ai("default"), _build_team_ai("default"),
                headless=True,
                attacker_roster=["Alfajer", "Boaster", "Chronicle", "Derke", "Leo"],
                defender_roster=["Demon1", "jawgemo", "Ethan", "Boostio", "C0M"],
            )
            shooter = next(char for char in game.chars if char.name == "Chronicle")
            target = next(char for char in game.chars if char.name == "Demon1")
            game.chars = [shooter, target]
            # Open corridor in the actual map, outside starting positions.
            shooter.pos, target.pos = [22, 17], [22, 19]
            shooter.facing = target.facing = "E"
            shooter.hs_rate = 0
            with patch("battle_logic.random.random", return_value=0):
                game.process_battle()
            self.assertEqual((target.hp, target.max_hp, target.ability_seal_remaining,
                              target.fate_loom_remaining), (60, 60, 3, 2))
            game._record_replay_frame()
            frame = next(char for char in game.replay_frames[-1]["chars"] if char["name"] == target.name)
            self.assertEqual((frame["max_hp"], frame["ability_seal"], frame["fate_loom_root"]), (60, 3, 2))
            game.defender_controller = StepController()
            start = list(target.pos)
            with patch.object(game, "_resolve_all_shots"):
                for _ in range(2):
                    game.move_character(target)
                    self.assertEqual(target.pos, start)
                    game.process_battle()
                self.assertEqual((target.ability_seal_remaining, target.fate_loom_remaining), (1, 0))
                game.process_battle()
                self.assertEqual(target.ability_seal_remaining, 0)
            game.current_round += 1
            game.init_round()
            target = next(char for char in game.chars if char.name == "Demon1")
            shooter = next(char for char in game.chars if char.name == "Chronicle")
            self.assertEqual((target.hp, target.max_hp, target.ability_seal_remaining,
                              target.fate_loom_remaining, target.fate_max_hp_lost), (100, 100, 0, 0, 0))
            self.assertEqual((shooter.erosion_curse, shooter.fate_loom), (3, 2))


if __name__ == "__main__":
    unittest.main()
