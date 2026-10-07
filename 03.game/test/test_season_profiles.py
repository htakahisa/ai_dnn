from dataclasses import replace
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

import realtime_season_config
import realtime_season_teams
from realtime_season import SeasonSaveError, SeasonStore, new_season
from run_realtime_season import RealtimeSeasonApp
from season.season_profiles import SeasonProfile, SeasonProfiles


class SeasonProfilesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.profiles = SeasonProfiles(self.directory / "teams")
        for module, key in ((realtime_season_config, "INITIAL_OWNED_PLAYERS"), (realtime_season_teams, "SEASON_TEAMS")):
            value = ["Leo", "Boaster", "Derke", "Chronicle", "Alfajer"] if key == "INITIAL_OWNED_PLAYERS" else []
            config = patch.object(module, key, value)
            config.start()
            self.addCleanup(config.stop)

    def test_independent_save_history_and_logs(self):
        first, state = self.profiles.create("チーム一")
        second, other = self.profiles.create("チーム二")
        state = state.with_initial_selection(tuple(p.name for p in state.starter_candidates))
        first.save(replace(state, money=state.money + 1234))
        (first.path.parent / "scrims").mkdir()
        (first.path.parent / "scrims" / "result.json").write_text("{}")
        self.assertEqual(second.load_or_create(), other)
        self.assertNotEqual(first.history_path, second.history_path)
        self.assertTrue(first.history_path.is_file())
        profile = next(p for p in self.profiles.list() if p.name == "チーム一")
        self.profiles.delete(profile)
        self.assertFalse(first.path.parent.exists())
        self.assertEqual(second.load_or_create(), other)

    def test_names_cannot_escape_or_collide(self):
        stores = [self.profiles.create(name)[0] for name in ("../A", "..\\A", "CON", "日本語")]
        self.assertEqual(len({store.path for store in stores}), 4)
        for store in stores:
            self.assertEqual(store.path.parent.parent, self.profiles.directory)
        with self.assertRaises(SeasonSaveError):
            self.profiles.create("con")
        with self.assertRaises(SeasonSaveError):
            self.profiles.create("   ")

    def test_legacy_load_is_preserved_and_delete_is_guarded(self):
        legacy = SeasonStore(self.directory / "save.json")
        legacy.save(new_season(()))
        original = legacy.path.read_bytes()
        profile = self.profiles.list()[0]
        self.assertTrue(profile.legacy)
        self.profiles.load(profile)
        self.assertEqual(legacy.path.read_bytes(), original)
        with self.assertRaises(SeasonSaveError):
            self.profiles.delete(profile)
        with self.assertRaises(SeasonSaveError):
            self.profiles.delete(SeasonProfile("outside", self.directory / "save.json"))

    def test_bad_or_missing_save_is_not_replaced(self):
        store, _ = self.profiles.create("broken")
        store.path.write_text("invalid", encoding="utf-8")
        profile = self.profiles.list()[0]
        with self.assertRaises(SeasonSaveError):
            self.profiles.load(profile)
        self.assertEqual(store.path.read_text(), "invalid")
        store.path.unlink()
        with self.assertRaises(SeasonSaveError):
            self.profiles.load(profile)
        self.assertFalse(store.path.exists())

    def test_rename_keeps_save_directory_and_updates_list(self):
        store, state = self.profiles.create("before")
        store.save(state.with_team_name("after"))
        profile = self.profiles.list()[0]
        self.assertEqual(profile.name, "after")
        self.assertEqual(profile.path, store.path)
        loaded_store, loaded = self.profiles.load(profile)
        self.assertEqual(loaded.team_name, "after")
        self.assertEqual(loaded_store.path, store.path)

    def test_switch_clears_old_widgets_and_scheduled_callbacks(self):
        app = RealtimeSeasonApp.__new__(RealtimeSeasonApp)
        app.root = Mock()
        widget = Mock()
        app.root.winfo_children.return_value = [widget]
        app.profiles = self.profiles
        app.store, app.state = self.profiles.create("first")
        selected = self.profiles.create("second")
        app.current_screen = "home"
        app.scrim_job = app.competition_job = None
        app._scrim_after_id = "scrim-callback"
        app._competition_after_id = "competition-callback"
        with patch("run_realtime_season.choose_season_profile", return_value=selected), patch.object(RealtimeSeasonApp, "__init__", return_value=None) as rebuild:
            app.manage_profiles()
        widget.destroy.assert_called_once()
        self.assertEqual(app.root.after_cancel.call_count, 2)
        rebuild.assert_called_once_with(app.root, *selected, self.profiles)

    def test_switch_rebuilds_screens_and_blocks_during_match(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.addCleanup(root.destroy)
        root.withdraw()
        first, state = self.profiles.create("first")
        second, other = self.profiles.create("second")
        app = RealtimeSeasonApp(root, first, state, self.profiles)
        old_home = app.home_host
        with patch("run_realtime_season.choose_season_profile", return_value=(second, other)) as choose:
            app.scrim_job = object()
            app.manage_profiles()
            choose.assert_not_called()
            app.scrim_job = None
            app.manage_profiles()
        self.assertEqual(app.store.path, second.path)
        self.assertEqual(app.state.team_name, "second")
        self.assertFalse(old_home.winfo_exists())
        self.assertEqual(app.club_name.get(), "second")


if __name__ == "__main__":
    unittest.main()
