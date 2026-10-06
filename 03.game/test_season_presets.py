"""Club identity across lineup presets, old saves, and initial setup."""

from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from realtime_season import SeasonSaveError, new_season
from season_competitions import next_match
from season_scrim import build_scrim_request
from season_series import build_series_request
from season_world_levels import world_level_for_rating
from test_season_competitions import (OWN, CompetitionScreenTest, SeasonCompetitionTest,
                                      finish, record_next)


class PresetIdentityTest(SeasonCompetitionTest):
    def test_unspecified_ai_reaches_scrims_tournaments_and_new_drafts(self):
        state = self.state()
        self.assertEqual(state.preset_ai, "toru_ai_v3.1")
        self.assertTrue(all(c.ai == "toru_ai_v3.1" for c in state.opponent_teams))
        request = build_scrim_request(state, state.selected_team_id, state.opponent_teams[0].id)
        self.assertEqual(request["own"]["ai"], "toru_ai_v3.1")
        self.assertEqual(request["opponent"]["ai"], "toru_ai_v3.1")
        registered = state.with_tournament_entry("cup", state.selected_team_id).advance_days(33)
        request = build_series_request(registered, "cup")
        self.assertEqual(request["own"]["ai"], "toru_ai_v3.1")
        self.assertEqual(request["opponent"]["ai"], "toru_ai_v3.1")
        self.assertEqual(state.with_new_team().preset_ai, "toru_ai_v3.1")
        self.assertEqual(state.with_preset_settings().selected_team.ai, "toru_ai_v3.1")

    def test_missing_ai_in_saved_lineups_rivals_and_tournament_snapshots_uses_toru(self):
        state = self.entered()
        self.store.save(state)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data.pop("preset_ai")
        for team in (*data["teams"], *data["opponent_teams"], *data["tournaments"][0]["entrants"]):
            team.pop("ai")
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, state)
        self.assertEqual(loaded.preset_ai, "toru_ai_v3.1")
        self.assertTrue(all(t.ai == "toru_ai_v3.1" for t in loaded.tournament("cup").entrants))

    def test_settings_saved_per_preset_and_used_by_scrims_and_tournaments(self):
        state = self.state().with_preset_settings(igl="Boaster", carrier="Derke", ai="fnatic_v3")
        first = state.selected_team
        state = state.with_new_team().with_preset_name("Second").with_roster(OWN)
        state = state.with_preset_settings(igl="Leo", carrier="Alfajer", ai="frc_v1_baseline").with_confirmed_team()
        second = state.teams[1]
        self.store.save(state)
        state = self.store.load_or_create()
        for team, expected in ((first, ("Boaster", "Derke", "fnatic_v3")),
                               (second, ("Leo", "Alfajer", "frc_v1_baseline"))):
            with self.subTest(team=team.name):
                editing = state.with_editing_team(team.id)
                self.assertEqual((editing.preset_igl, editing.preset_carrier, editing.preset_ai), expected)
                request = build_scrim_request(editing, team.id, state.opponent_teams[0].id)
                self.assertEqual(tuple(request["own"][key] for key in ("igl", "spike_holder", "ai")), expected)
                tournament = editing.with_tournament_entry("cup", team.id).advance_days(33)
                request = build_series_request(tournament, "cup")
                self.assertEqual(tuple(request["own"][key] for key in ("igl", "spike_holder", "ai")), expected)
                self.assertEqual(next(t.ai for t in tournament.tournament("cup").entrants
                                      if t.id == tournament.club_id), expected[2])
        overridden = build_scrim_request(state, first.id, state.opponent_teams[0].id,
            own_ai="default", own_igl="Chronicle", own_spike="Leo")
        self.assertEqual((overridden["own"]["ai"], overridden["own"]["igl"], overridden["own"]["spike_holder"]),
                         ("default", "Chronicle", "Leo"))
        self.assertEqual(state.team(first.id), first)

    def test_partial_draft_survives_reload_and_removed_roles_revert_to_auto(self):
        state = self.state().with_preset_settings(igl="Boaster", carrier="Derke", ai="fnatic_v3")
        saved = state.selected_team
        state = state.with_roster(("Leo", "Chronicle", "Alfajer"))
        self.assertEqual((state.preset_igl, state.preset_carrier), (None, None))
        state = state.with_preset_settings(igl="Chronicle", carrier="Leo", ai="frc_v1_baseline")
        self.assertEqual(state.selected_team, saved)
        self.store.save(state)
        state = self.store.load_or_create()
        self.assertEqual((state.preset_igl, state.preset_carrier, state.preset_ai),
                         ("Chronicle", "Leo", "frc_v1_baseline"))
        state = state.with_roster(OWN).with_confirmed_team()
        self.assertEqual((state.selected_team.igl, state.selected_team.carrier, state.selected_team.ai),
                         ("Chronicle", "Leo", "frc_v1_baseline"))
        state = state.with_new_team()
        self.assertEqual((state.preset_igl, state.preset_carrier, state.preset_ai), (None, None, "toru_ai_v3.1"))

    def test_departure_clears_roles_and_registered_preset_without_losing_draft_ai(self):
        state = self.state().with_preset_settings(igl="Leo", carrier="Leo", ai="fnatic_v3")
        state = state._with_departed_player("Leo")
        state.validate()
        self.assertIsNone(state.preset_igl)
        self.assertIsNone(state.preset_carrier)
        self.assertEqual(state.preset_ai, "fnatic_v3")
        self.assertFalse(state.teams)
        self.store.save(state)
        self.assertEqual(self.store.load_or_create(), state)

    def test_invalid_settings_cannot_replace_save(self):
        state = self.state()
        self.store.save(state)
        original = self.path.read_bytes()
        for kwargs in ({"igl": "Aspas"}, {"carrier": "missing"}, {"igl": []}, {"carrier": ""},
                       {"ai": "missing"}, {"ai": None}, {"ai": []}):
            with self.subTest(kwargs=kwargs), self.assertRaises(SeasonSaveError):
                state.with_preset_settings(**kwargs)
            self.assertEqual(self.path.read_bytes(), original)
        invalid = replace(state.teams[0], carrier="Aspas")
        with self.assertRaises(SeasonSaveError):
            self.store.save(replace(state, teams=(invalid,)))
        self.assertEqual(self.path.read_bytes(), original)

    def test_version_twelve_save_loads_default_settings_without_resetting_progress(self):
        state = record_next(self.entered())
        self.store.save(state)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["version"] = 12
        data.pop("developed_players")
        data.pop("world_level_lock")
        for key in ("preset_igl", "preset_carrier", "preset_ai"):
            data.pop(key)
        for team in data["teams"]:
            for key in ("igl", "carrier", "ai"):
                team.pop(key)
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        before = self.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, replace(state, developed_players=(), world_level_lock=world_level_for_rating(state.rating(state.club_id))))
        self.assertEqual(self.path.read_bytes(), before)
        self.store.save(loaded)
        self.assertEqual(self.store.load_or_create(), loaded)

    def test_registered_tournament_keeps_settings_snapshot_when_preset_changes(self):
        state = self.state().with_preset_settings(igl="Boaster", carrier="Derke", ai="fnatic_v3")
        state = state.with_tournament_entry("cup", state.selected_team_id).advance_days(33)
        request = build_series_request(state, "cup")
        changed = state.with_preset_settings(igl="Leo", carrier="Alfajer", ai="default")
        self.assertEqual(build_series_request(changed, "cup"), request)
        self.assertEqual(changed.selected_team.ai, "default")

    def test_shared_rosters_keep_one_club_rating_and_sponsor(self):
        state = self.state().with_team_name("Real Club").with_preset_name("Main").with_confirmed_team()
        first = state.selected_team
        state = state.with_rated_result("first win", first.id, state.opponent_teams[0].id, 1, 0)
        rating, income = state.rating(state.club_id), state.monthly_sponsor_income
        state = state.with_new_team().with_preset_name("Alternative").with_roster(tuple(reversed(OWN))).with_confirmed_team()
        state = state.with_selected_team(state.teams[1].id)
        self.assertEqual(state.team_name, "Real Club")
        self.assertEqual(state.rating(state.selected_team_id), rating)
        self.assertEqual(state.monthly_sponsor_income, income)
        self.assertEqual(len(state.rating_ranking), len(state.opponent_teams) + 1)
        self.assertEqual({state.player_affiliation(name) for name in OWN}, {"Real Club"})
        for preset in state.teams:
            request = build_scrim_request(state, preset.id, state.opponent_teams[0].id, render=False)
            self.assertEqual(request["own"]["name"], "Real Club")
            self.assertEqual(tuple(p["name"] for p in request["own"]["players"]), preset.roster)
        self.store.save(state)
        self.assertEqual(self.store.load_or_create(), state)

    def test_preset_and_club_rename_are_independent(self):
        state = self.state().with_team_name("Club A")
        state = state.with_rated_result("win", state.club_id, state.opponent_teams[0].id, 1, 0)
        state = state.with_preset_name("Lineup A").with_confirmed_team()
        self.assertEqual(state.team_name, "Club A")
        preset = state.selected_team
        renamed = state.with_team_name("Club B")
        self.assertEqual(renamed.selected_team, preset)
        self.assertEqual(renamed.rating(renamed.club_id), state.rating(state.club_id))
        self.assertEqual(next(r.team_name for r in renamed.rating_ranking if r.team_id == renamed.club_id), "Club B")
        for name in ("", " " * 4, "x" * 41, renamed.opponent_teams[0].name):
            with self.subTest(name=name), self.assertRaises(SeasonSaveError):
                renamed.with_team_name(name)

    def test_tournament_uses_actual_club_and_allows_applying_changed_preset(self):
        state = self.state().with_team_name("Real Club")
        first = state.selected_team
        state = state.with_new_team().with_preset_name("Second").with_roster(tuple(reversed(OWN))).with_confirmed_team()
        second = state.teams[1]
        state = state.with_tournament_entry("cup", first.id).advance_days(33)
        run = state.tournament("cup")
        self.assertEqual(run.own_team_id, state.club_id)
        self.assertEqual(run.preset_id, first.id)
        self.assertEqual(next(t.name for t in run.entrants if t.id == state.club_id), "Real Club")
        self.assertEqual(build_series_request(state, "cup")["own"]["name"], "Real Club")
        state = state.with_editing_team(second.id).with_roster(OWN).with_confirmed_team()
        state = state.with_editing_team(first.id).with_roster(tuple(reversed(OWN))).with_confirmed_team()
        self.assertEqual(tuple(p.name for p in state.tournament_team("cup").players), OWN)
        state = state.with_tournament_roster("cup", state.team(first.id).roster, preset_id=first.id)
        self.assertEqual(tuple(p.name for p in state.tournament_team("cup").players), tuple(reversed(OWN)))
        finished = finish(state)
        self.assertEqual(finished.tournament("cup").ranking[0], finished.club_id)
        finished.with_editing_team(first.id).with_roster(tuple(reversed(OWN))).with_confirmed_team()

    def legacy_save(self, state):
        self.store.save(state)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["version"] = 10
        data.pop("club_id")
        data.pop("preset_name")
        data["team_name"] = "Old editing draft"
        own_id = state.teams[0].id
        data["ratings"] = [r for r in data["ratings"] if r["team_id"] != state.club_id]
        data["ratings"].extend({"team_id": t.id, "team_name": t.name,
                                "value": state.rating(state.club_id) if t.id == own_id else 1400}
                               for t in state.teams)
        for run in data["tournaments"]:
            run.pop("preset_id")
            if run["own_team_id"] is None:
                continue
            run["own_team_id"] = own_id
            for team in run["entrants"]:
                if team["id"] == state.club_id:
                    team.update(id=own_id, name=state.teams[0].name)
            for score in run["results"]:
                for key in ("left_id", "right_id"):
                    if score[key] == state.club_id:
                        score[key] = own_id
            run["ranking"] = [own_id if key == state.club_id else key for key in run["ranking"]]
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def test_old_active_tournament_migrates_without_resetting_progress(self):
        state = self.entered().with_preset_name("Main").with_confirmed_team()
        state = record_next(state)
        state = state.with_new_team().with_preset_name("Second").with_roster(OWN).with_confirmed_team()
        state = state.with_selected_team(state.teams[1].id)
        self.legacy_save(state)
        before = self.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(loaded.team_name, "Main")
        self.assertEqual(loaded.preset_name, "Old editing draft")
        self.assertEqual(loaded.teams, state.teams)
        self.assertEqual(loaded.money, state.money)
        self.assertEqual(loaded.contracts, state.contracts)
        self.assertEqual(loaded.rating(loaded.club_id), state.rating(state.club_id))
        self.assertEqual(loaded.tournament("cup").results, state.tournament("cup").results)
        with self.assertRaisesRegex(SeasonSaveError, "1日1試合"):
            build_series_request(loaded, "cup")
        completed = finish(loaded)
        self.assertEqual(completed.tournament("cup").prize_paid, 5000000)
        self.store.save(completed)
        self.assertEqual(self.store.load_or_create(), completed)

    def test_old_completed_tournament_preserves_prize_and_ranking(self):
        state = finish(self.entered().with_preset_name("Main").with_confirmed_team())
        self.legacy_save(state)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.money, state.money)
        self.assertEqual(loaded.tournament("cup").ranking, state.tournament("cup").ranking)
        self.assertEqual(loaded.tournament("cup").completed_date, "2026-02-08")
        self.assertEqual(loaded.tournament("cup").prize_paid, state.tournament("cup").prize_paid)


class PresetScreenTest(CompetitionScreenTest):
    def set_settings(self, igl, carrier, ai):
        app = self.app
        app.preset_igl_choice.set(igl)
        app.preset_carrier_choice.set(carrier)
        app.preset_ai_choice.set(ai)
        app.preset_settings_menus["ai"].event_generate("<<ComboboxSelected>>")

    def test_editor_settings_autosave_switch_presets_and_reach_preparation(self):
        app = self.app
        app.show_editor()
        self.assertEqual(tuple(app.preset_settings_menus["igl"]["values"]), ("自動", *OWN))
        self.set_settings("Boaster", "Derke", "Fnatic v3")
        first = self.store.load_or_create().teams[0]
        self.assertEqual((first.igl, first.carrier, first.ai), ("Boaster", "Derke", "fnatic_v3"))
        app.new_team()
        app.preset_name.set("Second")
        self.assertEqual(app.preset_ai_choice.get(), "Toru AI v3.1")
        app.commit(app.state.with_roster(OWN), "roster")
        self.set_settings("Leo", "Alfajer", "FRC v1（基礎ルール）")
        app.confirm()
        second = app.state.teams[1]
        app.edit_team_choice.set(first.name)
        app.edit_saved_team()
        self.assertEqual((app.preset_igl_choice.get(), app.preset_carrier_choice.get(), app.preset_ai_choice.get()),
                         ("Boaster", "Derke", "Fnatic v3"))
        app.show_preparation()
        for team, expected in ((second, ("Leo", "Alfajer", "FRC v1（基礎ルール）")),
                               (first, ("Boaster", "Derke", "Fnatic v3"))):
            app.prep_team_choice.set(team.name)
            app.preview_preparation()
            self.assertEqual((app.own_igl.get(), app.own_spike.get(), app.own_ai_choice.get()), expected)
        app.own_igl.set("Chronicle")
        app.own_ai_choice.set("ロジック")
        app.preview_preparation()
        self.assertEqual((app.own_igl.get(), app.own_ai_choice.get()), ("Chronicle", "ロジック"))
        self.assertEqual(app.state.team(first.id), first)
        self.root.update_idletasks()
        self.assertLessEqual(app.editor_host.winfo_reqheight(), 800)

    def test_editor_role_removal_reverts_to_auto_and_save_failure_restores_choices(self):
        app = self.app
        app.show_editor()
        self.set_settings("Boaster", "Derke", "Fnatic v3")
        before = app.state
        with patch.object(self.store, "save", side_effect=OSError("test failure")), patch("run_realtime_season.messagebox.showerror"):
            self.set_settings("Leo", "Alfajer", "ロジック")
        self.assertEqual(app.state, before)
        self.assertEqual((app.preset_igl_choice.get(), app.preset_carrier_choice.get(), app.preset_ai_choice.get()),
                         ("Boaster", "Derke", "Fnatic v3"))
        app.roster.selection_set("1")
        app.remove_player()
        self.assertEqual(app.preset_igl_choice.get(), "自動")
        self.assertNotIn("Boaster", app.preset_settings_menus["igl"]["values"])

    def test_tournament_entry_uses_selected_preset_even_when_scrim_choices_differ(self):
        app = self.app
        app.show_editor()
        self.set_settings("Boaster", "Derke", "Fnatic v3")
        app.show_screen("competitions")
        app.competition_list.selection_set("cup")
        app.own_igl.set("Leo")
        app.own_spike.set("Alfajer")
        app.own_ai_choice.set("ロジック")
        app.enter_competition()
        run = app.state.tournament("cup")
        own = next(t for t in run.entrants if t.id == app.state.club_id)
        self.assertEqual((own.igl, own.carrier, own.ai), ("Boaster", "Derke", "fnatic_v3"))

    def test_initial_name_and_preset_name_remain_separate_on_screens(self):
        app = self.app
        app.commit(new_season(), "New season")
        app.show_screen("starter")
        app.starter_team_name.set("Season Club")
        app.save_starter_team_name()
        self.assertEqual(self.store.load_or_create().team_name, "Season Club")
        for name in OWN:
            app.starter_players.selection_set(name)
            app.toggle_starter()
        app.confirm_starters()
        self.assertEqual(app.state.team_name, "Season Club")
        self.assertEqual(app.state.monthly_sponsor_income, 7500000)
        app.show_editor()
        app.preset_name.set("Main lineup")
        app.commit(app.state.with_roster(OWN), "roster")
        app.confirm()
        app.new_team()
        app.preset_name.set("Second lineup")
        app.commit(app.state.with_roster(tuple(reversed(OWN))), "roster")
        app.confirm()
        app.show_home()
        self.assertEqual(len(app.home_teams.get_children()), len(app.state.opponent_teams) + 1)
        self.assertEqual(app.home_teams.set(app.state.club_id, "name"), "Season Club")
        app.show_screen("ratings")
        self.assertEqual(len(app.ratings_table.get_children()), len(app.state.opponent_teams) + 1)
        self.assertIn("Season Club", app.sponsor_summary.get())
        app.club_name.set("Renamed Club")
        app.rename_club()
        self.assertEqual(app.state.team_name, "Renamed Club")
        self.assertEqual([t.name for t in app.state.teams], ["Main lineup", "Second lineup"])
        self.assertEqual(self.store.load_or_create(), app.state)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for case in (PresetIdentityTest, PresetScreenTest):
        for name in case.__dict__:
            if name.startswith("test_"):
                suite.addTest(case(name))
    return suite


if __name__ == "__main__":
    unittest.main()
