"""Competition shield settings stay local to a game and reach every render path."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import queue
from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

import game_core
import run_competition_manager as manager
from game_core import Character, absorb_shield_damage
from map_data import NEW_MAZE_STR
from run_game import VisualFPSBattle, _build_team_ai
from tactical_simulator import TacticalSimulator, create_sample_retake_scenario


class CompetitionShieldTests(unittest.TestCase):
    def setUp(self):
        table = game_core._character_stats.CHARACTER_TABLE
        settings = patch.dict(table, {
            name: replace(table[name], shield_hp=50, shield_piercer=True, shield_crash=20)
            for name in ("Leo", "Ethan")
        })
        settings.start()
        self.addCleanup(settings.stop)
        self.attackers = ["Alfajer", "Boaster", "Chronicle", "Derke", "Leo"]
        self.defenders = ["Demon1", "jawgemo", "Ethan", "Boostio", "C0M"]

    def assert_traits(self, chars, enabled):
        for char in chars:
            if char.name in {"Leo", "Ethan"}:
                self.assertEqual((char.shield_hp, char.max_shield_hp,
                                  char.shield_piercer, char.shield_crash),
                                 (50, 50, True, 20) if enabled else (0, 0, False, 0))

    def test_off_persists_across_rounds_and_does_not_change_other_games(self):
        with redirect_stdout(StringIO()):
            for enabled in (False, True, False):
                game = VisualFPSBattle(
                    NEW_MAZE_STR, _build_team_ai("default"), _build_team_ai("default"),
                    headless=True, attacker_roster=self.attackers, defender_roster=self.defenders,
                    shield_abilities_enabled=enabled,
                )
                for round_number in (1, 2):
                    self.assert_traits(game.chars, enabled)
                    if not enabled:
                        actor = next(c for c in game.chars if c.name == "Leo")
                        target = SimpleNamespace(shield_hp=100)
                        self.assertEqual(absorb_shield_damage(target, 40, actor), 0)
                        self.assertEqual(target.shield_hp, 60)
                    game.current_round = round_number + 1
                    game.init_round()
                ordinary = Character("Leo", "A", (1, 1), "white", "red")
                self.assert_traits([ordinary], True)

    def test_play_map_passes_setting_into_real_game(self):
        teams = [SimpleNamespace(name=name, players=roster, spike_holder=roster[0], igl=roster[0])
                 for name, roster in (("One", self.attackers), ("Two", self.defenders))]
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                with patch.object(VisualFPSBattle, "run", autospec=True) as run:
                    manager.play_map(*teams, 1, 42, False, "default", "default",
                                     shield_abilities_enabled=enabled)
                self.assert_traits(run.call_args.args[0].chars, enabled)

    def test_headless_controller_uses_competition_setting(self):
        app = SimpleNamespace(live_render_enabled=False, current_shield_abilities_enabled=False)
        render = manager.CompetitionApp._RenderController(app)
        with patch.object(manager, "play_map", return_value="result") as play:
            result = render.play_map(None, None, 1, 42, "default", "default", 0, 0, 1)
        self.assertEqual(result, "result")
        self.assertIs(play.call_args.kwargs["shield_abilities_enabled"], False)

    def test_render_queue_captures_setting_for_rendered_and_user_matches(self):
        for live_render, controller in ((True, "default"), (False, "user")):
            for enabled in (False, True):
                with self.subTest(render=live_render, controller=controller, enabled=enabled):
                    app = manager.CompetitionApp.__new__(manager.CompetitionApp)
                    app.live_render_enabled = live_render
                    app.current_shield_abilities_enabled = enabled
                    app.tick_time_ms = 100
                    requests = []

                    def process(request):
                        requests.append(request)
                        # The queued request keeps the captured value.
                        app.current_shield_abilities_enabled = not enabled
                        app.render_requests = queue.Queue()
                        app.render_requests.put(request)
                        app._process_render_requests()

                    proxy = SimpleNamespace(live_render_enabled=live_render,
                                            current_shield_abilities_enabled=enabled,
                                            render_requests=SimpleNamespace(put=process))
                    render = manager.CompetitionApp._RenderController(proxy)
                    with patch.object(manager, "play_map", return_value="result") as play:
                        result = render.play_map(None, None, 1, 42, controller, "default", 0, 0, 1)
                    self.assertEqual(result, "result")
                    self.assertIs(requests[0]["shield_abilities_enabled"], enabled)
                    self.assertIs(play.call_args.kwargs["shield_abilities_enabled"], enabled)
                    self.assertTrue(play.call_args.args[4])

    def test_tactical_simulator_uses_setting_for_scenario_characters(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), redirect_stdout(StringIO()):
                scenario = create_sample_retake_scenario()
                scenario.attackers = [{"name": "Leo", "pos": (7, 3)}]
                scenario.defenders = [{"name": "Ethan", "pos": (9, 3)}]
                simulator = TacticalSimulator(scenario, "default", "default",
                                              shield_abilities_enabled=enabled)
                self.assert_traits(simulator.chars, enabled)


class CompetitionShieldUITests(unittest.TestCase):
    def test_checkbox_and_run_lock(self):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        app = manager.CompetitionApp.__new__(manager.CompetitionApp)
        app.root = root
        for name, value in (("maps_to_win_var", "2"), ("lower_final_maps_to_win_var", "3"),
                            ("grand_final_maps_to_win_var", "3"), ("seed_mode_var", "random"),
                            ("seed_var", "42"), ("tick_time_var", "100")):
            setattr(app, name, tk.StringVar(root, value=value))
        app.render_var = tk.BooleanVar(root, value=False)
        app.rating_enabled_var = tk.BooleanVar(root, value=True)
        app.shield_abilities_var = tk.BooleanVar(root, value=True)
        app._build_common_settings()
        self.assertTrue(app.shield_abilities_var.get())
        app.shield_abilities_check.invoke()
        self.assertFalse(app.shield_abilities_var.get())
        for name in ("power_refresh_button", "team1_box", "team2_box", "team1_controller_box",
                     "team2_controller_box", "swiss_slots", "tournament_seed_editor",
                     "qualifier_count_spin", "qualifier_import_button", "league_slots"):
            setattr(app, name, Mock())
        app.set_enabled(False)
        self.assertEqual(str(app.shield_abilities_check.cget("state")), "disabled")
        app.set_enabled(True)
        self.assertEqual(str(app.shield_abilities_check.cget("state")), "normal")
        app.shield_abilities_check.invoke()
        self.assertTrue(app.shield_abilities_var.get())


if __name__ == "__main__":
    unittest.main()
