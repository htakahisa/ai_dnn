"""Annual calendar dates and independent registration, prizes, and histories."""

from dataclasses import replace
from datetime import date
import json
import tkinter as tk
import unittest
from unittest.mock import patch

import test_season_competitions as fixtures
from realtime_season import SeasonSaveError, new_season
from run_realtime_season import RealtimeSeasonApp
from season_competitions import CompetitionError, configured_calendar, definition_from_dict, extend_annual_calendar, next_match
from season_history import snapshot


def annual(**changes):
    return fixtures.definition(**{"start_date": "01-03", "visible_from": "01-01", "format": "single_elimination", "team_count": 2,
                                  "prizes": {1: 5_000_000, 2: 2_000_000}, **changes})


class AnnualCompetitionTests(unittest.TestCase):
    setUp = fixtures.SeasonCompetitionTest.setUp
    state = fixtures.SeasonCompetitionTest.state

    def configure(self, rows, *, start="2026-01-01"):
        for key, value in (("START_DATE", start), ("TOURNAMENTS", rows)):
            context = patch.object(fixtures.config, key, value)
            context.start()
            self.addCleanup(context.stop)

    def test_same_month_day_and_year_specific_dates_coexist(self):
        self.configure([annual(), fixtures.definition(id="once", start_date="2026-02-01")])
        _, _, events = configured_calendar()
        by_id = {event.id: event for event in events}
        self.assertEqual(set(by_id), {"cup_2026", "cup_2027", "once"})
        self.assertEqual((by_id["cup_2026"].start_date, by_id["cup_2027"].start_date), ("2026-01-03", "2027-01-03"))
        self.assertEqual(by_id["cup_2027"].visible_from, "2027-01-01")
        self.assertEqual(by_id["cup_2027"].display_name, "Cup 2027")
        self.assertEqual(by_id["once"].display_name, "Cup")
        expanded = extend_annual_calendar(events, date(2027, 1, 1), date(2026, 1, 1))
        self.assertIn("cup_2028", {event.id for event in expanded})
        self.assertEqual(sum(event.id == "once" for event in expanded), 1)

    def test_notice_previous_year_and_default_notice(self):
        row = annual(visible_from="12-01")
        self.configure([row])
        _, _, events = configured_calendar()
        self.assertEqual(events[0].visible_from, "2025-12-01")
        self.assertEqual(events[1].visible_from, "2026-12-01")
        row.pop("visible_from")
        with patch.object(fixtures.config, "TOURNAMENTS", [row]):
            _, _, events = configured_calendar()
        self.assertEqual(events[0].visible_from, "2026-01-01")
        self.assertEqual(events[1].visible_from, "2027-01-01")

    def test_leap_day_occurs_only_in_leap_years_and_notice_clamps(self):
        self.configure([annual(start_date="02-29")])
        _, _, events = configured_calendar()
        self.assertEqual([(e.id, e.start_date) for e in events], [("cup_2028", "2028-02-29")])
        expanded = extend_annual_calendar(events, date(2031, 1, 1), date(2026, 1, 1))
        self.assertIn("cup_2032", {e.id for e in expanded})
        notice = definition_from_dict(annual(start_date="03-03", visible_from="02-29"), year=2026)
        self.assertEqual(notice.visible_from, "2026-02-28")
        self.assertEqual(notice.for_year(2028).visible_from, "2028-02-29")

    def test_invalid_dates_ids_and_dependencies_preserve_saved_file(self):
        state = self.state()
        self.store.save(state)
        before = self.path.read_bytes()
        rows = ([annual(start_date="2-03")], [annual(start_date="02-30")],
                [annual(visible_from="04-31")], [annual(id="bad/id")],
                [annual(), annual(start_date="02-29")],
                [annual(id="cup", appearance_conditions={"completed_tournaments": ["cup"]})],
                [annual(id="cup", appearance_conditions={"completed_tournaments": ["second"]}),
                 annual(id="second", appearance_conditions={"completed_tournaments": ["cup"]})],
                [annual(id="cup", start_date="02-29", appearance_conditions={"completed_tournaments": ["second"]}),
                 annual(id="second", appearance_conditions={"completed_tournaments": ["cup"]})])
        for configured in rows:
            with self.subTest(configured=configured), patch.object(fixtures.config, "TOURNAMENTS", configured):
                with self.assertRaises(SeasonSaveError):
                    self.store.import_competitions(state)
            self.assertEqual(self.path.read_bytes(), before)

    def test_new_season_and_import_do_not_replay_earlier_annual_occurrences(self):
        self.configure([annual()], start="2026-07-01")
        state = self.state()
        self.assertEqual([event.id for event in state.tournament_definitions], ["cup_2027"])
        self.assertEqual(state.advance_days().tournaments, ())
        with patch.object(fixtures.config, "START_DATE", "2026-01-01"):
            imported = self.store.import_competitions(state)
        self.assertEqual([event.id for event in imported.tournament_definitions], ["cup_2027"])

    def test_npc_events_repeat_each_year_without_overwriting_results(self):
        self.configure([annual(allow_player_entry=False)])
        state = self.state().advance_days(4)
        first = state.tournament("cup_2026")
        self.assertTrue(first.completed)
        self.assertEqual(first.completed_date, "2026-01-03")
        self.assertIsNone(first.own_team_id)
        # Real daily progression includes monthly finance and contract changes.
        next_year = state.advance_days(365)
        self.assertEqual(next_year.tournament("cup_2026"), first)
        second = next_year.tournament("cup_2027")
        self.assertTrue(second.completed)
        self.assertEqual(second.completed_date, "2027-01-03")
        self.assertNotEqual(first.seed, second.seed)
        self.assertEqual(len(next_year.rated_results), 2)
        self.assertIn("cup_2028", {event.id for event in next_year.tournament_definitions})
        self.store.save(next_year)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, next_year)
        self.assertEqual(loaded.advance_days(2).rated_results, next_year.rated_results)
        self.assertEqual([item["大会名"] for item in snapshot(loaded)["大会順位"]], ["Cup 2026", "Cup 2027"])

    def test_annual_batch_progress_matches_daily_progress(self):
        self.configure([annual(allow_player_entry=False)], start="2026-12-30")
        state = self.state()
        bulk = state.advance_days(8)
        daily = state
        for _ in range(8):
            daily = daily.advance_days()
        self.assertEqual(bulk, daily)
        self.assertTrue(bulk.tournament("cup_2027").completed)
        self.assertIsNone(bulk.tournament("cup_2026"))

    def test_december_bracket_continues_in_january(self):
        self.configure([fixtures.definition(start_date="12-31", visible_from="12-01", allow_player_entry=False)], start="2026-12-30")
        state = self.state().advance_days(2)
        run = state.tournament("cup_2026")
        self.assertEqual(state.game_date, "2027-01-01")
        self.assertFalse(run.completed)
        self.assertEqual(len(run.results), 2)
        self.assertEqual([r.tournament_id for r in state.active_tournaments], ["cup_2026"])
        finished = state.advance_days(5)
        self.assertTrue(finished.tournament("cup_2026").completed)
        # Three real rival clubs fill the four-team setting: four series.
        self.assertEqual(finished.tournament("cup_2026").completed_date, "2027-01-03")
        self.assertIsNone(finished.tournament("cup_2027"))
        self.assertIsNone(finished.world_level_lock)
        self.store.save(finished)
        self.assertEqual(self.store.load_or_create(), finished)

    def test_player_can_enter_again_prizes_history_and_yearly_conditions(self):
        qualifier = annual(id="qualifier")
        final = annual(id="final", start_date="02-03", appearance_conditions={
            "completed_tournaments": ["qualifier"], "best_rank": {"qualifier": 1}})
        self.configure([qualifier, final])
        state = self.state()
        state = state.with_tournament_entry("qualifier_2026", state.selected_team_id)
        state = state.advance_days(2)
        completed = fixtures.finish(state, "qualifier_2026")
        first = completed.tournament("qualifier_2026")
        self.assertEqual(first.prize_paid, 5_000_000)
        self.assertTrue(completed.tournament_definition("final_2026").appears(completed))
        # Give the fixture a renewed lineup without simulating unrelated actions.
        year_two = replace(completed, game_date="2027-01-01", game_month=12, monthly_events_through=12,
                           contracts=tuple(replace(c, signed_on="2027-01-01", start_month=12) for c in completed.contracts))
        self.assertFalse(year_two.tournament_definition("final_2027").appears(year_two))
        entered = year_two.with_tournament_entry("qualifier_2027", year_two.selected_team_id)
        self.assertEqual(entered.tournament("qualifier_2026"), first)
        finished = fixtures.finish(entered.advance_days(2), "qualifier_2027")
        self.assertEqual(finished.tournament("qualifier_2027").prize_paid, 5_000_000)
        self.assertTrue(finished.tournament_definition("final_2027").appears(finished))
        awards = [item for entry in finished.history for item in entry["収入"] if item["内訳"] == "大会賞金"]
        self.assertEqual(len(awards), 2)
        self.store.save(finished)
        self.assertEqual(self.store.load_or_create(), finished)

    def test_import_preserves_old_snapshot_and_updates_next_annual_rules(self):
        self.configure([fixtures.definition(id="cup_2026", start_date="2026-01-03")])
        state = self.state()
        state = state.with_tournament_entry("cup_2026", state.selected_team_id).advance_days(2)
        state = fixtures.finish(state, "cup_2026")
        original = state.tournament_definition("cup_2026")
        run = state.tournament("cup_2026")
        self.store.save(state)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["version"] = 27
        for event in data["tournament_definitions"]:
            event.pop("annual_id")
            event.pop("annual_notice")
        self.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(loaded, state)
        updated = annual(name="Updated Cup", prizes={1: 7_000_000, 2: 3_000_000})
        final = annual(id="final", start_date="02-15", appearance_conditions={"completed_tournaments": ["cup"]})
        with patch.object(fixtures.config, "TOURNAMENTS", [updated, final]):
            imported = self.store.import_competitions(loaded)
        self.assertEqual(imported.tournament_definition("cup_2026"), original)
        self.assertEqual(imported.tournament("cup_2026"), run)
        self.assertEqual(imported.tournament_definition("cup_2027").prizes, {1: 7_000_000, 2: 3_000_000})
        self.assertTrue(imported.tournament_definition("final_2026").appears(imported))
        self.assertEqual(imported.money, loaded.money)
        self.assertEqual(imported.rated_results, loaded.rated_results)
        self.assertEqual(self.store.load_or_create(), imported)
        with patch.object(fixtures.config, "TOURNAMENTS", [annual(start_date="02-03"), final]):
            changed = self.store.import_competitions(imported)
        self.assertEqual(changed.tournament_definition("cup_2026"), original)
        self.assertEqual(changed.tournament_definition("cup_2027").start_date, "2027-02-03")
        following = extend_annual_calendar(changed.tournament_definitions, date(2027, 1, 1), date(2026, 1, 1))
        self.assertEqual(next(event for event in following if event.id == "cup_2028").start_date, "2028-02-03")

    def test_tournament_screen_identifies_years_and_deadline(self):
        self.configure([annual()])
        state = self.state()
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        app = RealtimeSeasonApp(root, self.store, state)
        app.show_screen("competitions")
        self.assertEqual(app.competition_list.item("cup_2026", "values")[0], "Cup 2026")
        self.assertFalse(app.competition_list.exists("cup_2027"))
        app.competition_list.selection_set("cup_2026")
        app.preview_competition()
        self.assertIn("2026-01-03", app.competition_info.get())
        self.assertEqual(str(app.competition_enter_button["state"]), "normal")
        next_year = replace(app.state, game_date="2027-01-01", game_month=12, monthly_events_through=12)
        app.commit(next_year, "yearly calendar")
        self.assertEqual(app.competition_list.item("cup_2027", "values")[0], "Cup 2027")
        self.assertEqual(app.state.tournament_definition("cup_2027").start_date, "2027-01-03")

    def test_changed_yearly_conditions_do_not_form_cycles_with_old_snapshots(self):
        self.configure([annual(id="qualifier"), annual(id="final", start_date="01-05",
                        appearance_conditions={"completed_tournaments": ["qualifier"]})])
        state = self.state()
        state = state.with_tournament_entry("qualifier_2026", state.selected_team_id).advance_days(2)
        state = fixtures.finish(state, "qualifier_2026")
        state = state.with_tournament_entry("final_2026", state.selected_team_id).advance_days(2)
        state = fixtures.finish(state, "final_2026")
        # The next season reverses the qualifying order; the first year's
        # completed rules remain valid and do not create a cross-year cycle.
        changed_rules = [annual(id="qualifier", start_date="02-03", appearance_conditions={"completed_tournaments": ["final"]}),
                         annual(id="final", start_date="01-05")]
        with patch.object(fixtures.config, "TOURNAMENTS", changed_rules):
            changed = self.store.import_competitions(state)
        self.assertEqual(changed.tournament_definition("qualifier_2026"), state.tournament_definition("qualifier_2026"))
        self.assertEqual(changed.tournament_definition("final_2026"), state.tournament_definition("final_2026"))
        self.assertEqual(changed.tournament_definition("qualifier_2027").appearance_conditions,
                         {"completed_tournaments": ["final"]})
        self.assertEqual(changed.tournament_definition("final_2027").appearance_conditions, {})


if __name__ == "__main__":
    unittest.main()
