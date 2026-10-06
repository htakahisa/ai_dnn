"""Chapter selection, gated recruitment, shared career and legacy migration."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

import character_stats
import realtime_season_config as config
import realtime_season_competitions as calendar
import realtime_season_leagues as leagues
import realtime_season_teams as teams
import run_realtime_season as app_module
from realtime_season import SeasonSaveError, SeasonStore, new_season
from season_leagues import chapter_save_path, configured_leagues
from season_league_ui import SeasonChapterSelection
from season_transfers import with_randomized_clubs


OWN = ("Leo", "Boaster", "Derke", "Chronicle", "Alfajer")
EARLY = ("Aspas", "valyn", "trent", "leaf", "tex")
LATE = ("Boostio", "Ethan", "jawgemo", "C0M", "Demon1")


class LeagueTests(unittest.TestCase):
    def setUp(self):
        catalog = {name: replace(p, monthly_salary=100_000, debut_chapter=1)
                   for name, p in character_stats.CHARACTER_TABLE.items()}
        for name in LATE:
            catalog[name] = replace(catalog[name], debut_chapter=2)
        catalog["Demon1"] = replace(catalog["Demon1"], debut_chapter=3)
        self.definitions = [dict(name="Early", players=[*EARLY, "Boostio"], debut_chapter=1),
                            dict(name="Middle", players=list(LATE), debut_chapter=2),
                            dict(name="Late", players=[], debut_chapter=3)]
        for context in (patch.dict(character_stats.CHARACTER_TABLE, catalog),
                        patch.object(config, "INITIAL_OWNED_PLAYERS", (*OWN, *LATE)),
                        patch.object(teams, "SEASON_TEAMS", self.definitions),
                        patch.object(calendar, "TOURNAMENTS", []),
                        patch.object(calendar, "START_DATE", "2026-01-01"),
                        patch.object(leagues, "LEAGUE_NAMES", {1: "地域リーグ", 2: "国内リーグ", 3: "世界リーグ"})):
            context.start()
            self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name) / "save.json"

    def assert_allowed(self, state):
        self.assertTrue(all(c.debut_chapter == state.chapter for c in state.opponent_teams))
        players = (*state.scout_players, *state.starter_candidates, *state.owned_players,
                   *(p for c in state.opponent_teams for p in c.players))
        self.assertTrue(all(p.debut_chapter <= state.chapter for p in players))

    def test_each_chapter_includes_only_available_teams_players_and_initial_candidates(self):
        for chapter, names in ((1, ["Early"]), (2, ["Middle"]), (3, ["Late"])):
            with self.subTest(chapter=chapter):
                state = new_season(chapter=1) if chapter == 1 else new_season(OWN, chapter=chapter)
                self.assertEqual([c.name for c in state.opponent_teams], names)
                self.assert_allowed(state)
                self.assertEqual(len(state.starter_candidates), 5 if chapter == 1 else 0)
                self.assertEqual(state.league_name, leagues.LEAGUE_NAMES[chapter])
        early = new_season(chapter=1)
        self.assertEqual(early.opponent_teams[0].members, EARLY)
        self.assertNotIn("Boostio", [p.name for p in early.lft_players])
        middle = new_season(OWN, chapter=2)
        self.assertEqual(len(middle.opponent_teams[0].players), 4)
        self.assertEqual(middle.player_affiliation("Aspas"), "LFT")
        self.assertNotIn("Demon1", [p.name for p in middle.scout_players])

    def test_removed_future_starter_clears_role_and_allows_a_short_roster(self):
        self.definitions[0].update(players=["Boostio", *EARLY[:4]], igl="Boostio", carrier="Boostio")
        state = new_season(OWN, chapter=1)
        self.assertEqual(state.opponent_teams[0].members, EARLY[:4])
        self.assertIsNone(state.opponent_teams[0].igl)
        self.assertIsNone(state.opponent_teams[0].carrier)
        state.validate()

    def test_initial_mixing_monthly_fills_and_regular_returns_cannot_introduce_future_players(self):
        state = new_season(chapter=1).with_initial_selection(OWN)
        self.assert_allowed(state)
        for _ in range(3):
            state = with_randomized_clubs(state).advance_months()
            self.assert_allowed(state)
        # An import filters the configured bench and the chapter 2/3 teams too.
        state = SeasonStore(self.base).import_season_teams(state)
        self.assert_allowed(state)
        self.assertEqual(len(state.opponent_teams), 1)

    def test_monthly_replacement_pool_is_gated_even_when_future_player_is_stronger(self):
        self.definitions[0]["players"] = list(EARLY[:4])
        state = new_season(OWN, chapter=1)
        from season_monthly_events import choose_recruit
        pools = []
        def record(club, candidates, rating):
            pools.append(candidates)
            return choose_recruit(club, candidates, rating)
        with patch("season_monthly_events.choose_recruit", side_effect=record):
            state = state.advance_months()
        self.assertTrue(pools)
        self.assertTrue(all(p.debut_chapter == 1 for pool in pools for p in pool))
        self.assertEqual(len(state.opponent_teams[0].players), 5)

    def test_direct_scout_or_player_add_cannot_bypass_the_chapter(self):
        state = replace(new_season(OWN, chapter=1), money=100_000_000)
        for action in (lambda: state.with_scouted_player("Boostio", "year1"),
                       lambda: state.with_added_players(["Boostio"]),
                       lambda: new_season(["Boostio"], chapter=1)):
            with self.assertRaises(SeasonSaveError):
                action()
        unlocked = replace(new_season(OWN, chapter=2), money=100_000_000)
        self.assertEqual(unlocked.with_scouted_player("Boostio", "year1").player("Boostio").debut_chapter, 2)

    def test_chapter_two_tournament_excludes_future_and_incomplete_teams_and_can_resume(self):
        self.definitions[0]["debut_chapter"] = 2
        cup = dict(id="cup", name="Cup", start_date="2026-01-02", team_count=8,
                   format="single_elimination", prizes={}, normal_maps_to_win=1, grand_final_maps_to_win=1)
        with patch.object(calendar, "TOURNAMENTS", [cup]):
            state = new_season(OWN, chapter=2).with_roster(OWN).with_confirmed_team()
        state = state.with_tournament_entry("cup", state.teams[0].id).advance_days()
        run = state.tournament("cup")
        self.assertEqual({t.name for t in run.entrants}, {state.team_name, "Early"})
        store = SeasonStore(chapter_save_path(self.base, 2))
        store.save(state)
        self.assertEqual(store.load_or_create(chapter=2), state)
        from season_competitions import next_match, SeriesScore
        match, _ = next_match(state.tournament_definition("cup"), run)
        state = state.with_tournament_result("cup", SeriesScore(match.id, match.left, match.right,
                    int(match.left == state.club_id) * match.maps_to_win,
                    int(match.right == state.club_id) * match.maps_to_win))
        self.assertTrue(state.tournament("cup").completed)
        self.assertEqual(state.chapter, 2)
        self.assert_allowed(state)

    def test_import_uses_only_the_selected_chapter_even_when_a_team_changes_chapter(self):
        store = SeasonStore(chapter_save_path(self.base, 2))
        state = new_season(OWN, chapter=2)
        original = state.opponent_teams[0]
        self.definitions[1]["debut_chapter"] = 3
        self.definitions[0]["debut_chapter"] = 2
        state = store.import_season_teams(state)
        self.assertEqual([c.name for c in state.opponent_teams], ["Early"])
        self.assertNotIn(original.id, [c.id for c in state.opponent_teams])
        self.assert_allowed(state)
        self.assertEqual(store.load_or_create(chapter=2), state)

    def test_previous_active_tournament_keeps_its_registered_entrants_after_migration(self):
        self.definitions[0]["debut_chapter"] = 2
        cup = dict(id="cup", name="Cup", start_date="2026-01-02", team_count=2,
                   format="single_elimination", prizes={}, normal_maps_to_win=1)
        with patch.object(calendar, "TOURNAMENTS", [cup]):
            state = new_season(OWN, chapter=2).with_roster(OWN).with_confirmed_team()
        state = state.with_tournament_entry("cup", state.teams[0].id).advance_days()
        store = SeasonStore(chapter_save_path(self.base, 2))
        store.save(state)
        data = json.loads(store.path.read_text(encoding="utf-8"))
        data["version"] = 29
        next(c for c in data["opponent_teams"] if c["name"] == "Early")["debut_chapter"] = 1
        store.path.write_text(json.dumps(data), encoding="utf-8")
        loaded = store.load_or_create(chapter=2)
        self.assertEqual([c.name for c in loaded.opponent_teams], ["Middle"])
        self.assertEqual(loaded.tournaments, state.tournaments)
        from season_competitions import next_match, SeriesScore
        match, _ = next_match(loaded.tournament_definition("cup"), loaded.tournament("cup"))
        loaded = loaded.with_tournament_result("cup", SeriesScore(match.id, match.left, match.right,
                    int(match.left == loaded.club_id) * match.maps_to_win,
                    int(match.right == loaded.club_id) * match.maps_to_win))
        self.assertTrue(loaded.tournament("cup").completed)
        self.assert_allowed(loaded)
        store.save(loaded)
        self.assertEqual(store.load_or_create(chapter=2), loaded)

    def test_previous_chapter_save_removes_earlier_teams_without_resetting_progress(self):
        store = SeasonStore(chapter_save_path(self.base, 2))
        state = new_season(OWN, chapter=2).advance_days(3)
        earlier = new_season(OWN, chapter=1)
        club = earlier.opponent_teams[0]
        club = replace(club, players=tuple(replace(p, iq=p.iq + 10) for p in club.players))
        store.save(state)
        data = json.loads(store.path.read_text(encoding="utf-8"))
        data["version"] = 29
        data["opponent_teams"].append(asdict(club))
        data["ratings"].append(asdict(next(r for r in earlier.ratings if r.team_id == club.id)))
        data["transfer_offers"].append(dict(id="old-offer", team_id=club.id, player_name="Leo",
            created_month=0, fee=5_000_000, status="pending", contract_kind="year1",
            contract_months=12, monthly_salary=100_000))
        store.path.write_text(json.dumps(data), encoding="utf-8")
        before = store.path.read_bytes()
        loaded = store.load_or_create(chapter=2)
        self.assertEqual([c.name for c in loaded.opponent_teams], ["Middle"])
        self.assertEqual((loaded.money, loaded.date, loaded.owned_players, loaded.contracts, loaded.history),
                         (state.money, state.date, state.owned_players, state.contracts, state.history))
        self.assertEqual(loaded.transfer_offers[0].status, "cancelled")
        self.assertEqual(loaded.player_affiliation("Aspas"), "LFT")
        self.assertEqual(next(p for p in loaded.scout_players if p.name == "Aspas").iq, club.players[0].iq)
        self.assertEqual(loaded.team_loyalty("Aspas", club.id), club.contracts[0].team_loyalty)
        self.assert_allowed(loaded)
        self.assertEqual(store.path.read_bytes(), before)
        store.save(loaded)
        self.assertEqual(store.load_or_create(chapter=2), loaded)

    def test_assets_contracts_and_history_follow_the_career_between_chapters(self):
        store = SeasonStore(self.base)
        state = store.load_or_create().with_initial_selection(OWN).advance_days(3)
        state = replace(state, unlocked_chapters=(1, 2, 3))
        initial = state
        for chapter in (2, 3, 1):
            state = state.with_chapter(chapter)
            self.assertEqual((state.money, state.date, state.owned_players, state.contracts,
                              state.roster, state.contract_bans, state.starter_candidates,
                              state.starter_selection_pending),
                             (initial.money, initial.date, initial.owned_players, initial.contracts,
                              initial.roster, initial.contract_bans, initial.starter_candidates, False))
            for name, relationships in initial.team_loyalties.items():
                for identifier, loyalty in relationships.items():
                    self.assertEqual(state.team_loyalties[name][identifier], loyalty)
            for name, counts in initial.contract_signings.items():
                for identifier, count in counts.items():
                    self.assertEqual(state.contract_signings[name][identifier], count)
            store.save(state)
            self.assertEqual(store.load_or_create(), state)
            self.assertTrue(store.history_path.exists())
        self.assertEqual(len(state.owned_players), 5)
        self.assertGreater(len(state.history), len(initial.history))
        self.assertFalse(chapter_save_path(self.base, 2).exists())
        for chapter in (2, 3):
            with self.assertRaises(SeasonSaveError):
                new_season(chapter=chapter)

    def test_old_save_defaults_to_chapter_one_without_rewriting_or_resetting(self):
        store = SeasonStore(self.base)
        state = new_season(OWN).advance_days(3)
        store.save(state)
        data = json.loads(self.base.read_text(encoding="utf-8"))
        data["version"] = 28
        data.pop("chapter")
        for p in data["owned_players"]:
            p.pop("debut_chapter")
        for club in data["opponent_teams"]:
            club.pop("debut_chapter")
            for p in club["players"]:
                p.pop("debut_chapter")
        self.base.write_text(json.dumps(data), encoding="utf-8")
        before = self.base.read_bytes()
        self.assertEqual(store.load_or_create(chapter=1), state)
        self.assertEqual(self.base.read_bytes(), before)

    def test_invalid_chapters_and_debut_fields_are_rejected(self):
        for chapter in (0, 4, True, "2"):
            with self.assertRaises(SeasonSaveError):
                new_season(OWN, chapter=chapter)
        for debut in (0, -1, True, 1.5, "2"):
            self.definitions[0]["debut_chapter"] = debut
            with self.assertRaises(SeasonSaveError):
                new_season(OWN)
        self.definitions[0]["debut_chapter"] = 1
        with patch.dict(character_stats.CHARACTER_TABLE, {"Leo": replace(character_stats.get_by_name("Leo"), debut_chapter=0)}):
            with self.assertRaises(SeasonSaveError):
                new_season(OWN)
        with patch.object(leagues, "LEAGUE_NAMES", {1: "地域リーグ"}):
            with self.assertRaises(ValueError):
                configured_leagues()

    def test_additional_configured_chapter_is_selectable_and_keeps_team_exclusivity(self):
        self.definitions[0]["debut_chapter"] = 4
        with patch.object(leagues, "LEAGUE_NAMES", {**leagues.LEAGUE_NAMES, 4: "モンスターリーグ"}):
            state = new_season(OWN, chapter=4)
            self.assertEqual(state.league_name, "モンスターリーグ")
            self.assertEqual([c.name for c in state.opponent_teams], ["Early"])
            self.assert_allowed(state)
            self.assertEqual(chapter_save_path(self.base, 4).name, "save_chapter4.json")
            root = tk.Tk()
            root.withdraw()
            self.addCleanup(root.destroy)
            selected = []
            selector = SeasonChapterSelection(root, selected.append, state=state)
            selector.buttons[4].invoke()
            self.assertEqual(selected, [4])

    def test_selector_and_main_hide_locked_chapters_and_create_only_one_save(self):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        selected = []
        selector = SeasonChapterSelection(root, selected.append)
        self.assertEqual(set(selector.buttons), {1})
        selector.buttons[1].invoke()
        self.assertEqual(selected, [1])
        selector.host.destroy()
        def choose():
            self.assertTrue(self.base.exists())
            frame = next(w for w in root.winfo_children() if isinstance(w, app_module.ttk.Frame))
            buttons = [w for w in frame.winfo_children() if isinstance(w, app_module.ttk.Button)]
            self.assertEqual(len(buttons), 1)
            buttons[0].invoke()
        with patch.object(app_module.tk, "Tk", return_value=root), patch.object(root, "mainloop", side_effect=choose), \
                patch.object(app_module, "RealtimeSeasonApp") as app:
            self.assertEqual(app_module.main(["--save-file", str(self.base)]), 0)
        _, store, state = app.call_args.args
        self.assertEqual(state.chapter, 1)
        self.assertEqual(store.path, self.base)
        self.assertTrue(state.starter_selection_pending)
        self.assertFalse(chapter_save_path(self.base, 2).exists())

    def test_league_name_and_debut_metadata_are_visible_in_game(self):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        state = new_season(OWN, chapter=2)
        app = app_module.RealtimeSeasonApp(root, SeasonStore(self.base), state)
        self.assertIn("第2章 国内リーグ", root.title())
        self.assertIn("第2章　国内リーグ", app.home_summary.get())
        app.show_screen("scout")
        app.scout_filter.set("全選手")
        app.scout_players.selection_set("Boostio")
        app.refresh_offer("scout")
        self.assertIn("登場章: 第2章", app.scout_details.get())

    def test_home_can_return_switch_chapters_and_resume_saved_progress(self):
        state = replace(new_season(OWN).advance_days(), unlocked_chapters=(1, 2))
        SeasonStore(self.base).save(state)
        states = {1: state}
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        apps = []
        app_class = app_module.RealtimeSeasonApp
        def build_app(*args, **kwargs):
            app = app_class(*args, **kwargs)
            apps.append(app)
            return app
        def select(chapter):
            host = next(w for w in root.winfo_children() if isinstance(w, app_module.ttk.Frame))
            button = next(w for w in host.winfo_children() if isinstance(w, app_module.ttk.Button)
                          and w.cget("text").startswith(f"第{chapter}章"))
            button.invoke()
            return apps[-1]
        def navigate():
            first = select(1)
            self.assertEqual(first.current_screen, "home")
            initial_widget_count = len(root.winfo_children())
            # Return saves the current state even if it has not yet been written.
            first.state = first.state.advance_days(2)
            states[1] = first.state
            events = []
            first._competition_after_id = root.after(0, lambda: events.append("old chapter"))
            previous_close = root.protocol("WM_DELETE_WINDOW")
            popup = tk.Toplevel(root)
            first.chapter_selection_button.invoke()
            self.assertFalse(popup.winfo_exists())
            self.assertIn("章選択", root.title())
            self.assertEqual(root.minsize(), (640, 400))
            self.assertFalse(root.tk.call("info", "commands", previous_close))
            self.assertEqual(first.search.trace_info(), [])
            self.assertEqual(first.offer_kind["scout"].trace_info(), [])
            root.update()
            self.assertEqual(events, [])
            self.assertEqual(SeasonStore(self.base).load_or_create(chapter=1), states[1])

            second = select(2)
            self.assertEqual((second.state.money, second.state.date, second.state.owned_players, second.state.contracts),
                             (states[1].money, states[1].date, states[1].owned_players, states[1].contracts))
            second.state = replace(second.state, money=second.state.money - 123)
            self.assertEqual(len(root.winfo_children()), initial_widget_count)
            second.chapter_selection_button.invoke()
            resumed = select(1)
            self.assertEqual(resumed.state.chapter, 1)
            self.assertEqual(resumed.state.money, states[1].money - 123)
            self.assertEqual(resumed.state.date, states[1].date)
            self.assertEqual(len(root.winfo_children()), initial_widget_count)
            resumed.chapter_selection_button.invoke()
            self.assertIn("章選択", root.title())
            self.assertEqual(root.protocol("WM_DELETE_WINDOW"), "")
        with patch.object(app_module.tk, "Tk", return_value=root), \
                patch.object(root, "deiconify"), patch.object(root, "mainloop", side_effect=navigate), \
                patch.object(app_module, "RealtimeSeasonApp", side_effect=build_app):
            self.assertEqual(app_module.main(["--save-file", str(self.base)]), 0)

    def test_return_save_failure_keeps_the_current_chapter_open(self):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        callback = Mock()
        store = SeasonStore(self.base)
        state = new_season(OWN)
        app = app_module.RealtimeSeasonApp(root, store, state, on_chapter_selection=callback)
        previous_close = root.protocol("WM_DELETE_WINDOW")
        with patch.object(store, "save", side_effect=OSError("disk full")), \
                patch.object(app_module.messagebox, "showerror") as error:
            self.assertFalse(app.return_to_chapter_selection())
        callback.assert_not_called()
        error.assert_called_once()
        self.assertTrue(app.home_host.winfo_exists())
        self.assertTrue(app.search.trace_info())
        self.assertEqual(root.protocol("WM_DELETE_WINDOW"), previous_close)
        self.assertEqual(app.state, state)
        self.assertIn("保存に失敗", app.status.get())

    def test_reopening_a_chapter_does_not_repeat_startup_configuration_imports(self):
        SeasonStore(self.base).save(new_season(OWN, chapter=2))
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        def navigate():
            for _ in range(2):
                host = next(w for w in root.winfo_children() if isinstance(w, app_module.ttk.Frame))
                button = next(w for w in host.winfo_children() if isinstance(w, app_module.ttk.Button)
                              and w.cget("text").startswith("第2章"))
                button.invoke()
                app.call_args.kwargs["on_chapter_selection"]()
        team_import = SeasonStore.import_season_teams
        competition_import = SeasonStore.import_competitions
        with patch.object(app_module.tk, "Tk", return_value=root), patch.object(root, "deiconify"), \
                patch.object(root, "mainloop", side_effect=navigate), \
                patch.object(app_module, "RealtimeSeasonApp") as app, \
                patch.object(SeasonStore, "import_season_teams", autospec=True, side_effect=team_import) as import_teams, \
                patch.object(SeasonStore, "import_competitions", autospec=True, side_effect=competition_import) as import_events:
            self.assertEqual(app_module.main(["--save-file", str(self.base), "--import-season-teams", "--import-competitions"]), 0)
        self.assertEqual(app.call_count, 2)
        import_teams.assert_called_once()
        import_events.assert_called_once()

    def test_running_match_blocks_return_without_saving_or_cancelling_the_match(self):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        callback = Mock()
        store = SeasonStore(self.base)
        app = app_module.RealtimeSeasonApp(root, store, new_season(OWN), on_chapter_selection=callback)
        for attribute in ("scrim_job", "competition_job"):
            job = Mock()
            setattr(app, attribute, job)
            app.refresh_home()
            self.assertEqual(str(app.chapter_selection_button.cget("state")), "disabled")
            with patch.object(store, "save") as save:
                self.assertFalse(app.return_to_chapter_selection())
            save.assert_not_called()
            callback.assert_not_called()
            job.cancel.assert_not_called()
            self.assertTrue(app.home_host.winfo_exists())
            setattr(app, attribute, None)


if __name__ == "__main__":
    unittest.main()
