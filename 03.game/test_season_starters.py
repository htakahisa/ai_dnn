from dataclasses import replace
import json
from pathlib import Path
from random import Random
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

import realtime_season_config
import realtime_season_teams
from realtime_season import INITIAL_MONEY, SeasonSaveError, SeasonStore, new_season
from run_realtime_season import RealtimeSeasonApp


CANDIDATES = ("Leo", "Boaster", "Derke", "Chronicle", "Alfajer", "Meiy", "まーやまくん")
CHOSEN = ("Leo", "Derke", "Alfajer", "Meiy", "まーやまくん")


def destroy_window(window):
    try:
        window.destroy()
    except tk.TclError:
        pass


class StarterSelectionTest(unittest.TestCase):
    def setUp(self):
        for module, key, value in ((realtime_season_config, "INITIAL_OWNED_PLAYERS", list(CANDIDATES)),
                                   (realtime_season_teams, "SEASON_TEAMS", [])):
            context = patch.object(module, key, value)
            context.start()
            self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "save.json"
        self.store = SeasonStore(self.path)

    def test_large_pool_draws_seven_unique_candidates_and_varies_between_new_games(self):
        pool = (*CANDIDATES, "Aspas", "valyn", "trent", "leaf", "tex")
        draws = []
        with patch.object(realtime_season_config, "INITIAL_OWNED_PLAYERS", pool):
            for seed in range(5):
                with patch("realtime_season.Random", return_value=Random(seed)):
                    state = new_season()
                names = tuple(p.name for p in state.starter_candidates)
                self.assertEqual(len(names), 7)
                self.assertEqual(len(set(names)), 7)
                self.assertTrue(set(names).issubset(pool))
                draws.append(names)
                omitted = next(name for name in pool if name not in names)
                with self.assertRaises(SeasonSaveError):
                    state.with_initial_selection((*names[:4], omitted))
        self.assertGreater(len(set(draws)), 1)

    def test_random_candidates_and_partial_choice_survive_restart_without_redrawing(self):
        pool = (*CANDIDATES, "Aspas", "valyn", "trent")
        with patch.object(realtime_season_config, "INITIAL_OWNED_PLAYERS", pool):
            state = self.store.load_or_create()
        chosen = tuple(p.name for p in state.starter_candidates[:5])
        state = state.with_starter_selection(chosen[:2])
        self.store.save(state)
        with patch("realtime_season.Random", side_effect=AssertionError("redrawn on reload")):
            loaded = self.store.load_or_create()
        self.assertEqual(loaded, state)
        self.assertEqual(tuple(p.name for p in loaded.starter_candidates), tuple(p.name for p in state.starter_candidates))
        started = loaded.with_initial_selection(chosen)
        self.assertEqual(tuple(p.name for p in started.owned_players), chosen)

    def test_five_or_six_configured_candidates_are_all_available(self):
        for size in (5, 6):
            with self.subTest(size=size), patch.object(realtime_season_config, "INITIAL_OWNED_PLAYERS", CANDIDATES[:size]):
                self.assertEqual(tuple(p.name for p in new_season().starter_candidates), CANDIDATES[:size])

    def test_bad_entry_outside_draw_is_still_rejected(self):
        for names in ((*CANDIDATES, "missing"), (*CANDIDATES, "Leo")):
            with self.subTest(names=names), patch.object(realtime_season_config, "INITIAL_OWNED_PLAYERS", names), patch("realtime_season.Random") as rng:
                with self.assertRaises(SeasonSaveError):
                    new_season()
                rng.assert_not_called()

    def test_first_launch_owns_nobody_then_acquires_only_the_five_selected(self):
        state = self.store.load_or_create()
        self.assertTrue(state.starter_selection_pending)
        self.assertEqual(state.owned_players, ())
        self.assertEqual(state.contracts, ())
        self.assertEqual(tuple(p.name for p in state.starter_candidates), CANDIDATES)
        chosen = state.with_starter_selection(CHOSEN).with_initial_selection()
        self.assertFalse(chosen.starter_selection_pending)
        self.assertEqual(tuple(p.name for p in chosen.owned_players), CHOSEN)
        self.assertEqual(chosen.money, INITIAL_MONEY)
        self.assertEqual(len(chosen.contracts), 5)
        self.assertTrue(all(c.kind == "short" and c.duration_months == 6 and c.start_month == 0
                            and c.monthly_salary == chosen.player(c.player_name).monthly_salary for c in chosen.contracts))
        self.assertTrue({"Boaster", "Chronicle"}.issubset(p.name for p in chosen.lft_players))
        self.store.save(chosen)
        self.assertEqual(self.store.load_or_create(), chosen)
        with self.assertRaises(SeasonSaveError):
            chosen.with_initial_selection(CANDIDATES[:5])

    def test_initial_short_contracts_expire_after_six_game_months_and_can_be_renewed(self):
        state = self.store.load_or_create().with_initial_selection(CHOSEN)
        five = state.advance_months(5, pay_salaries=False)
        self.assertTrue(all(five.can_play(name) for name in CHOSEN))
        six = five.advance_months(pay_salaries=False)
        self.assertTrue(all(not six.can_play(name) for name in CHOSEN))
        self.assertEqual(tuple(p.name for p in six.owned_players), CHOSEN)
        self.store.save(six)
        six = self.store.load_or_create()
        renewed = six.with_renewed_contract("Leo", "year1")
        self.assertTrue(renewed.can_play("Leo"))
        self.assertEqual((renewed.contract("Leo").start_month, renewed.contract("Leo").duration_months), (6, 12))

    def test_incomplete_excess_duplicate_or_unknown_selection_cannot_acquire_players(self):
        state = self.store.load_or_create()
        for names in ((), CHOSEN[:4], CANDIDATES[:6], ("Leo",) * 5, (*CHOSEN[:4], "Aspas")):
            with self.subTest(names=names), self.assertRaises(SeasonSaveError):
                state.with_initial_selection(names)
            self.assertEqual(state.owned_players, ())
        for names in (CANDIDATES[:6], ("Leo", "Leo"), ("Aspas",)):
            with self.subTest(names=names), self.assertRaises(SeasonSaveError):
                state.with_starter_selection(names)

    def test_pending_selection_preserves_snapshots_and_draft_after_config_changes(self):
        state = self.store.load_or_create().with_starter_selection(CHOSEN[:3])
        state = replace(state, starter_candidates=(replace(state.starter_candidates[0], iq=180), *state.starter_candidates[1:]))
        self.store.save(state)
        with patch.object(realtime_season_config, "INITIAL_OWNED_PLAYERS", ["missing"]):
            loaded = self.store.load_or_create()
        self.assertEqual(loaded, state)
        self.assertTrue(loaded.starter_selection_pending)
        self.assertEqual(loaded.with_initial_selection(CHOSEN).player("Leo").iq, 180)

    def test_gameplay_actions_cannot_grant_players_or_advance_time_before_selection(self):
        state = self.store.load_or_create()
        original = self.path.read_bytes()
        for action in (lambda: state.with_added_players(("Leo",)),
                       lambda: state.with_scouted_player("Leo", "year1"),
                       lambda: state.advance_days(1), lambda: state.advance_months(1)):
            with self.assertRaises(SeasonSaveError):
                action()
        self.assertEqual(self.path.read_bytes(), original)

    def test_too_few_candidates_are_rejected_and_initial_selection_overrides_rival_reserves(self):
        for names in ([], ["Leo"], list(CANDIDATES[:4])):
            with patch.object(realtime_season_config, "INITIAL_OWNED_PLAYERS", names), self.assertRaises(SeasonSaveError):
                self.store.load_or_create()
            self.assertFalse(self.path.exists())
        rival = [{"name": "Rival", "players": ["Aspas", "valyn", "trent", "leaf", "tex", "Sato"]}]
        with patch.object(realtime_season_teams, "SEASON_TEAMS", rival), patch.object(realtime_season_config, "INITIAL_OWNED_PLAYERS", [*CANDIDATES[:6], "Sato"]):
            state = self.store.load_or_create().with_initial_selection([*CHOSEN[:4], "Sato"])
            self.assertIsNotNone(state.player("Sato"))
            self.assertIsNone(state.opponent_owner("Sato"))

    def test_legacy_version_five_inventory_and_roster_are_preserved(self):
        state = new_season(CANDIDATES).with_roster(CANDIDATES[:5]).with_confirmed_team()
        self.store.save(state)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["version"] = 5
        for player in data["owned_players"]:
            player["loyalty"] *= 10
        for key in ("starter_candidates", "starter_selection", "starter_selection_pending"):
            data.pop(key)
        self.path.write_text(json.dumps(data), encoding="utf-8")
        original = self.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertFalse(loaded.starter_selection_pending)
        self.assertEqual(loaded.owned_players, state.owned_players)
        self.assertEqual(loaded.teams, state.teams)
        self.assertEqual(loaded.contracts, state.contracts)
        self.assertEqual(self.path.read_bytes(), original)

    def test_invalid_saved_selection_is_preserved(self):
        self.store.load_or_create()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["starter_selection"] = ["Leo"] * 5
        self.path.write_text(json.dumps(data), encoding="utf-8")
        invalid = self.path.read_bytes()
        with self.assertRaises(SeasonSaveError):
            self.store.load_or_create()
        self.assertEqual(self.path.read_bytes(), invalid)

    def test_failed_save_keeps_previous_draft(self):
        state = self.store.load_or_create().with_starter_selection(CHOSEN[:2])
        self.store.save(state)
        before = self.path.read_bytes()
        with patch("realtime_season.os.replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                self.store.save(state.with_initial_selection(CHOSEN))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertTrue(self.store.load_or_create().starter_selection_pending)


class StarterScreenTest(StarterSelectionTest):
    def setUp(self):
        super().setUp()
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(destroy_window, self.root)
        self.app = RealtimeSeasonApp(self.root, self.store, self.store.load_or_create())

    def choose(self, name):
        self.app.starter_players.selection_set(name)
        self.app.preview_starter()
        self.app.toggle_starter()

    def test_select_replace_confirm_and_return_to_home(self):
        app = self.app
        self.assertEqual(app.current_screen, "starter")
        app.show_editor()
        self.assertEqual(app.current_screen, "starter")
        self.assertEqual(str(app.starter_confirm_button["state"]), "disabled")
        for name in CHOSEN:
            self.choose(name)
        self.assertEqual(str(app.starter_confirm_button["state"]), "normal")
        app.starter_players.selection_set("Boaster")
        app.preview_starter()
        self.assertEqual(str(app.starter_toggle_button["state"]), "disabled")
        app.toggle_starter()
        self.assertEqual(len(app.state.starter_selection), 5)
        self.choose("Leo")
        self.assertEqual(str(app.starter_confirm_button["state"]), "disabled")
        self.choose("Boaster")
        app.confirm_starters()
        self.assertEqual(app.current_screen, "home")
        self.assertEqual({p.name for p in app.state.owned_players}, {*CHOSEN[1:], "Boaster"})
        self.assertEqual(self.store.load_or_create(), app.state)
        app.show_screen("starter")
        self.assertEqual(app.current_screen, "home")
        # New games acquire more players through scouting, not the legacy free-input panel.
        app.name_input.insert("1.0", "Leo")
        app.register_players()
        self.assertIsNone(app.state.player("Leo"))

    def test_close_and_restart_resumes_selection_then_completed_save_opens_home(self):
        for name in CHOSEN[:2]:
            self.choose(name)
        self.app.close()
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(destroy_window, self.root)
        self.app = RealtimeSeasonApp(self.root, self.store, self.store.load_or_create())
        self.assertEqual(self.app.current_screen, "starter")
        self.assertEqual(self.app.state.starter_selection, CHOSEN[:2])
        for name in CHOSEN[2:]:
            self.choose(name)
        self.app.confirm_starters()
        other = tk.Toplevel(self.root)
        other.withdraw()
        self.addCleanup(destroy_window, other)
        restarted = RealtimeSeasonApp(other, self.store, self.store.load_or_create())
        self.assertEqual(restarted.current_screen, "home")
        self.assertEqual(tuple(p.name for p in restarted.state.owned_players), CHOSEN)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(StarterSelectionTest))
    for name in ("test_select_replace_confirm_and_return_to_home", "test_close_and_restart_resumes_selection_then_completed_save_opens_home"):
        suite.addTest(StarterScreenTest(name))
    return suite


if __name__ == "__main__":
    unittest.main()
