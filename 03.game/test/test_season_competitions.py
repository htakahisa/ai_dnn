from dataclasses import replace
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import realtime_season_competitions as config
import realtime_season_config
import realtime_season_teams
import realtime_season_rival_economy
import character_stats
from realtime_season import SeasonSaveError, SeasonStore, new_season
from run_realtime_season import RealtimeSeasonApp
from season.season_competitions import (CompetitionError, SeriesScore, TournamentProgress, bracket,
    configured_calendar, definition_from_dict, next_match, phase_for)
from season.season_scrim import ScrimJob
from season.season_series import build_series_request, play_series
from season.season_world_levels import world_level_for_rating


OWN = ("Leo", "Boaster", "Derke", "Chronicle", "Alfajer")
RIVALS = [("Aspas", "valyn", "trent", "leaf", "tex", "Sato"),
          ("Boostio", "Ethan", "jawgemo", "C0M", "Demon1"),
          ("F0rsakeN", "Jinggg", "d4v41", "something", "PatMen")]


def definition(**changes):
    return {"id": "cup", "name": "Cup", "start_date": "2026-02-03",
            "visible_from": "2026-01-01", "team_count": 4, "prizes": {1: 5000000, 2: 2000000, 3: 1000000, 4: 500000},
            "normal_maps_to_win": 1, "lower_final_maps_to_win": 2, "grand_final_maps_to_win": 3, **changes}


def finish(state, event_id="cup", own_wins=True):
    while not state.tournament(event_id).completed:
        run = state.tournament(event_id)
        if run.last_match_date == state.game_date:
            state = state.advance_days(1)
        match, _ = next_match(state.tournament_definition(event_id), run)
        winner = run.own_team_id if own_wins and run.own_team_id in (match.left, match.right) else match.right
        state = state.with_tournament_result(event_id, SeriesScore(match.id, match.left, match.right,
                    match.maps_to_win if winner == match.left else 0, match.maps_to_win if winner == match.right else 0))
    return state


def record_next(state, event_id="cup", *, left_wins=True):
    match, _ = next_match(state.tournament_definition(event_id), state.tournament(event_id))
    return state.with_tournament_result(event_id, SeriesScore(match.id, match.left, match.right,
        match.maps_to_win if left_wins else 0, 0 if left_wins else match.maps_to_win))


class SeasonCompetitionTest(unittest.TestCase):
    def setUp(self):
        fixture = {name: replace(p, monthly_salary=100_000, loyalty=5, debut_chapter=1)
                   for name, p in character_stats.CHARACTER_TABLE.items()}
        context = patch.dict(character_stats.CHARACTER_TABLE, fixture)
        context.start()
        self.addCleanup(context.stop)
        changes = [(config, "START_DATE", "2026-01-01"), (config, "IN_SEASON_PERIODS", [("03-01", "11-30")]),
                   (realtime_season_rival_economy, "NON_REGULAR_OFFER_CHANCE", 0),
                   (config, "TOURNAMENTS", [definition()]), (realtime_season_config, "INITIAL_OWNED_PLAYERS", list(OWN)),
                   (realtime_season_teams, "SEASON_TEAMS", [{"name": f"Rival{i}", "players": players} for i, players in enumerate(RIVALS)])]
        for module, key, value in changes:
            context = patch.object(module, key, value)
            context.start()
            self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "save.json"
        self.store = SeasonStore(self.path)

    def state(self):
        state = new_season().with_initial_selection(OWN).with_roster(OWN).with_confirmed_team()
        return replace(state.with_selected_team(state.teams[0].id), money=10_000_000)

    def entered(self):
        state = self.state()
        state = state.with_tournament_entry("cup", state.selected_team_id)
        return state.advance_days(33)

    def test_real_calendar_phase_boundary_and_single_month_payroll(self):
        state = self.state()
        self.assertEqual(state.date, date(2026, 1, 1))
        self.assertEqual(state.phase, "off_season")
        state = state.advance_days(30)
        self.assertEqual(state.money, 10000000)
        state = state.advance_days()
        self.assertEqual(state.date, date(2026, 2, 1))
        self.assertEqual(state.money, 17000000)
        state = state.with_declined_tournament("cup").advance_days(27)
        expected_money = state.money + state.monthly_sponsor_income - state.monthly_payroll
        state = state.advance_days()
        self.assertEqual(state.date, date(2026, 3, 1))
        self.assertEqual(state.phase, "in_season")
        self.assertEqual(state.money, expected_money)
        self.assertEqual(phase_for(date(2026, 1, 2), [("11-01", "02-28")]), "in_season")
        self.assertEqual(phase_for(date(2026, 3, 1), [("11-01", "02-28")]), "off_season")

    def test_unregistered_tournament_does_not_stop_calendar(self):
        state = self.state().advance_days(100)
        self.assertEqual(state.date, date(2026, 4, 11))
        self.assertEqual(state.advance_days().date, date(2026, 4, 12))
        self.assertIsNone(state.tournament("cup").own_team_id)
        self.assertTrue(state.tournament("cup").completed)

    def test_mandatory_entry_and_decline_rejection(self):
        with patch.object(config, "TOURNAMENTS", [definition(participation_optional=False)]):
            state = self.state().advance_days(100)
        self.assertEqual(state.date, date(2026, 4, 11))
        self.assertIsNone(state.tournament("cup").own_team_id)
        self.assertIsNone(state.tournament("cup").preset_id)
        with self.assertRaises(SeasonSaveError):
            state.with_declined_tournament("cup")

    def test_registration_checks_preset_and_rival_count(self):
        state = self.state()
        with self.assertRaises(SeasonSaveError):
            state.with_tournament_entry("cup", "missing-preset")
        reduced = replace(state, opponent_teams=state.opponent_teams[:2]).with_tournament_entry("cup", state.selected_team_id)
        self.assertEqual(len(reduced.tournament("cup").entrants), 3)
        free_player = state.lft_players[0].name
        state = state.with_added_players((free_player,)).with_tournament_entry("cup", state.selected_team_id)
        with self.assertRaises(SeasonSaveError):
            state.with_opponent_teams(())
        changed = state.with_roster((*OWN[:4], free_player)).with_confirmed_team()
        changed.validate()
        self.assertEqual(tuple(p.name for p in changed.tournament_team("cup").players), OWN)
        with self.assertRaises(ValueError):
            build_series_request(state, "cup")

    def test_series_stages_prizes_and_restart_cannot_pay_twice(self):
        state = self.entered()
        balance = state.money
        state = finish(state)
        run = state.tournament("cup")
        self.assertEqual(run.ranking[0], state.club_id)
        self.assertEqual(run.prize_paid, 5000000)
        self.assertEqual(state.money, balance + 5000000)
        self.assertEqual(len(run.results), 6)
        self.assertEqual(next(r for r in run.results if r.match_id == "LOWER_FINAL").right_wins, 2)
        self.assertEqual(max(run.results[-1].left_wins, run.results[-1].right_wins), 3)
        self.store.save(state)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, state)
        with self.assertRaises(SeasonSaveError):
            loaded.with_tournament_result("cup", run.results[-1])
        self.assertEqual(self.store.load_or_create().money, state.money)

    def test_one_series_per_day_survives_restart_and_bulk_date_advance(self):
        state = record_next(self.entered(), left_wins=False)
        self.store.save(state)
        state = self.store.load_or_create()
        with self.assertRaisesRegex(SeasonSaveError, "1日1試合"):
            build_series_request(state, "cup")
        with self.assertRaisesRegex(SeasonSaveError, "1日1試合"):
            record_next(state)
        following = state.advance_days(100)
        self.assertEqual(following.date, date(2026, 2, 4))
        self.assertEqual(following.advance_days(), following)
        with self.assertRaisesRegex(SeasonSaveError, "レート"):
            build_series_request(following, "cup", render=False)
        following = following.with_tournament_rating_result("cup")
        self.assertEqual(following.game_date, "2026-02-05")
        self.assertEqual(following.tournament("cup").last_match_date, "2026-02-04")
        # NPC results advance the date, so our next series is immediately ready.
        following = following.with_tournament_forfeit("cup")
        self.assertEqual(following.tournament("cup").last_match_date, "2026-02-08")
        self.assertEqual(following.game_date, "2026-02-09")

    def test_final_completes_after_planned_end_and_releases_calendar(self):
        state = finish(replace(self.entered(), game_date="2026-02-10"))
        run = state.tournament("cup")
        self.assertEqual(state.date, date(2026, 2, 15))
        self.assertEqual(run.completed_date, "2026-02-15")
        self.assertEqual(run.last_match_date, run.completed_date)
        self.assertFalse(state.pending_tournaments)
        self.assertEqual(state.advance_days().date, date(2026, 2, 16))

    def test_manual_end_date_is_ignored_and_final_completes_on_calculated_date(self):
        with patch.object(config, "TOURNAMENTS", [definition(end_date="2026-02-20")]):
            state = finish(self.entered())
        self.assertTrue(state.tournament("cup").completed)
        self.assertEqual(state.tournament("cup").completed_date, "2026-02-08")
        self.assertEqual(state.tournament_definition("cup").end_date, "2026-02-08")
        self.assertEqual(state.tournament("cup").prize_paid, 5000000)

    def test_daily_tournament_advance_runs_monthly_finance_and_events_once(self):
        with patch.object(config, "TOURNAMENTS", [definition(start_date="2026-01-31", end_date="2026-01-31")]):
            state = self.state()
            state = state.with_tournament_entry("cup", state.selected_team_id).advance_days(30)
        state = record_next(state)
        balance, income = state.money, state.monthly_sponsor_income
        payroll = sum(c.monthly_salary for c in state.contracts if state.contract_active(c))
        following = state.advance_days(100)
        self.assertEqual(following.date, date(2026, 2, 1))
        self.assertEqual(following.money, balance + income - payroll)
        self.assertEqual(following.game_month, 1)
        self.assertEqual(following.contract("Leo").team_loyalty, state.contract("Leo").team_loyalty - .5)
        self.assertEqual(len([e for e in following.monthly_events if e.kind == "month_completed"]), 1)
        self.store.save(following)
        reloaded = self.store.load_or_create()
        self.assertEqual(reloaded.advance_days(100), following)
        reloaded.with_tournament_rating_result("cup")

    def test_planned_end_still_closes_new_registration(self):
        state = self.state().advance_days(40, stop_for_tournaments=False)
        self.assertFalse(state.pending_tournaments)
        with self.assertRaises(SeasonSaveError):
            state.with_tournament_entry("cup", state.selected_team_id)

    def test_version_nine_migrates_match_dates_without_rewriting(self):
        for state in (record_next(self.entered()), finish(self.entered())):
            with self.subTest(completed=state.tournament("cup").completed):
                self.store.save(state)
                data = json.loads(self.path.read_text(encoding="utf-8"))
                data["version"] = 9
                data.pop("developed_players")
                data.pop("world_level_lock")
                for run in data["tournaments"]:
                    run.pop("last_match_date")
                    run.pop("completed_date")
                self.path.write_text(json.dumps(data), encoding="utf-8")
                before = self.path.read_bytes()
                locked = world_level_for_rating(state.rating(state.club_id)) if state.active_tournaments else None
                self.assertEqual(self.store.load_or_create(), replace(state, developed_players=(), world_level_lock=locked))
                self.assertEqual(self.path.read_bytes(), before)

    def test_invalid_match_dates_cannot_replace_saved_results(self):
        state = record_next(self.entered())
        self.store.save(state)
        before = self.path.read_bytes()
        run = state.tournament("cup")
        for bad in (replace(run, last_match_date=None), replace(run, last_match_date="2026-02-04"),
                    replace(run, last_match_date="2026-02-02"), replace(run, completed_date=run.last_match_date)):
            with self.subTest(run=bad), self.assertRaises(SeasonSaveError):
                self.store.save(replace(state, tournaments=(bad,)))
            self.assertEqual(self.path.read_bytes(), before)

    def test_opponent_only_tournament_cannot_pay_player_prize(self):
        with patch.object(config, "TOURNAMENTS", [definition(format="single_elimination", team_count=3,
                prizes={1: 5000000, 2: 2000000, 3: 1000000}, allow_player_entry=False)]):
            state = self.state().advance_days(33)
        self.assertIsNone(state.tournament("cup").own_team_id)
        with self.assertRaises(SeasonSaveError):
            build_series_request(state, "cup", render=True)
        balance = state.money
        state = state.advance_days(2)
        self.assertEqual(state.money, balance)
        self.assertEqual(state.tournament("cup").prize_paid, 0)

    def test_short_departure_before_event_can_continue_through_forfeit(self):
        state = self.state()
        leo = state.contract("Leo")
        state = replace(state, contracts=tuple(replace(c, kind="short", duration_months=6, team_loyalty=0)
                                               if c == leo else c for c in state.contracts))
        state = state.with_tournament_entry("cup", state.selected_team_id).advance_days(33)
        self.assertIsNone(state.player("Leo"))
        self.assertIsNone(state.selected_team)
        request = build_series_request(state, "cup")
        own = request["own"] if request["left_id"] == state.club_id else request["opponent"]
        self.assertIn("友達", [p["name"] for p in own["players"]])
        state = state.with_tournament_forfeit("cup")
        while not state.tournament("cup").completed:
            state = state.advance_days(1)
            run = state.tournament("cup")
            match, _ = next_match(state.tournament_definition("cup"), run)
            if run.own_team_id in (match.left, match.right):
                state = state.with_tournament_forfeit("cup")
            else:
                state = state.with_tournament_rating_result("cup")
        state.validate()

    def test_bad_or_out_of_order_results_preserve_save(self):
        state = self.entered()
        self.store.save(state)
        before = self.path.read_bytes()
        match, _ = next_match(state.tournament_definition("cup"), state.tournament("cup"))
        for score in (SeriesScore("other", match.left, match.right, 1, 0),
                      SeriesScore(match.id, match.left, match.right, 0, 0),
                      SeriesScore(match.id, match.left, match.right, True, 0)):
            with self.subTest(score=score), self.assertRaises(ValueError):
                state.with_tournament_result("cup", score)
            self.assertEqual(self.path.read_bytes(), before)

    def test_appearance_conditions_unlock_by_record_and_phase(self):
        next_event = definition(id="next", name="Next", start_date="2026-03-03", end_date="2026-03-07",
            appearance_conditions={"completed_tournaments": ["cup"], "best_rank": {"cup": 2}, "min_money": 1000000,
                                   "min_owned_players": 5, "phase": "in_season"})
        with patch.object(config, "TOURNAMENTS", [definition(), next_event]):
            state = self.entered()
        self.assertEqual([e.id for e in state.visible_tournaments], ["cup"])
        state = finish(state)
        state = state.advance_days((date(2026, 3, 1) - state.date).days)
        self.assertEqual([e.id for e in state.visible_tournaments], ["cup", "next"])
        self.assertNotIn("next", [e.id for e in replace(state, money=0).visible_tournaments])

    def test_save_resume_mid_bracket_and_keep_definition_snapshot_on_import(self):
        state = self.entered()
        match, _ = next_match(state.tournament_definition("cup"), state.tournament("cup"))
        state = state.with_tournament_result("cup", SeriesScore(match.id, match.left, match.right, 1, 0))
        self.store.save(state)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, state)
        with patch.object(config, "TOURNAMENTS", [definition(prizes={1: 999}, grand_final_maps_to_win=1),
            definition(id="later", name="Later", start_date="2026-04-01", end_date="2026-04-07")]):
            loaded = self.store.import_competitions(loaded)
        self.assertEqual(loaded.tournament_definition("cup").grand_final_maps_to_win, 3)
        self.assertEqual(loaded.tournament_definition("cup").prizes[1], 5000000)
        self.assertIsNotNone(loaded.tournament_definition("later"))
        self.assertEqual(loaded.date, state.date)
        self.assertEqual(loaded.money, state.money)

    def test_old_version_four_keeps_contracts_money_and_elapsed_months(self):
        state = self.state().advance_months(2)
        self.store.save(state)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["version"] = 4
        for key in ("start_date", "game_date", "in_season_periods", "tournament_definitions", "tournaments"):
            data.pop(key)
        self.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.date, date(2026, 3, 1))
        self.assertEqual(loaded.money, state.money)
        self.assertEqual(loaded.contracts, state.contracts)
        self.assertEqual(loaded.teams, state.teams)
        self.assertEqual(self.path.read_bytes(), before)

    def test_version_six_converts_tournament_snapshots_without_changing_progress(self):
        state = self.entered()
        self.store.save(state)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["version"] = 6
        data.pop("developed_players")
        data.pop("world_level_lock")
        data.pop("monthly_events")
        data.pop("monthly_events_through")
        snapshots = [*data["owned_players"], *data["starter_candidates"],
                     *(p for club in data["opponent_teams"] for p in club["players"]),
                     *(p for run in data["tournaments"] for team in run["entrants"] for p in team["players"])]
        for player in snapshots:
            player["loyalty"] *= 10
        self.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, replace(state, monthly_events=(), developed_players=()))
        self.assertEqual(self.path.read_bytes(), before)
        self.store.save(loaded)
        self.assertEqual(self.store.load_or_create(), loaded)

    def test_invalid_configuration_import_does_not_overwrite_save(self):
        state = self.state()
        self.store.save(state)
        before = self.path.read_bytes()
        bad = [definition(format="invalid"), definition(start_date="2026-02-30"), definition(team_count=3),
               definition(normal_maps_to_win=True), definition(prizes={5: 100}), definition(prizes={1: -1}),
               definition(opponent_teams="Rival0"), definition(appearance_conditions={"unknown": 1}),
               definition(appearance_conditions={"completed_tournaments": ["cup"]}), definition(id="../outside")]
        for row in bad:
            with self.subTest(row=row), patch.object(config, "TOURNAMENTS", [row]), self.assertRaises(ValueError):
                self.store.import_competitions(state)
            self.assertEqual(self.path.read_bytes(), before)

    def test_missing_notice_date_defaults_to_configured_game_start(self):
        row = definition()
        row.pop("visible_from")
        with patch.object(config, "START_DATE", "2025-01-01"), patch.object(config, "TOURNAMENTS", [row]):
            start, _, events = configured_calendar()
        self.assertEqual(start, "2025-01-01")
        self.assertEqual(events[0].visible_from, "2025-01-01")

    def test_series_request_contains_saved_abilities_and_rival_settings(self):
        state = self.entered()
        request = build_series_request(state, "cup", render=False, tick_time_ms=15)
        self.assertFalse(request["render"])
        self.assertEqual(request["maps_to_win"], 1)
        self.assertEqual(request["own"]["players"][0]["name"], "Leo")
        self.assertEqual(request["tick_time_ms"], 15)
        opponent = next(t for t in state.tournament("cup").entrants if t.id == request["right_id"])
        self.assertEqual(request["opponent"]["igl"], opponent.igl)
        self.assertEqual(tuple(p["name"] for p in request["opponent"]["players"]),
                         tuple(p.name for p in opponent.players))

    def test_series_plays_until_target_and_alternates_sides(self):
        request = build_series_request(self.entered(), "cup", render=False)
        request["maps_to_win"] = 2
        winners = [request["own"]["name"], request["opponent"]["name"], request["own"]["name"]]
        calls = []
        def play(single):
            calls.append(single)
            return {"status": "completed", "winner": winners[len(calls) - 1]}
        with patch("season.season_scrim_worker.play_scrim", side_effect=play):
            result = play_series(request)
        self.assertEqual((result["left_wins"], result["right_wins"]), (2, 1))
        self.assertEqual([r["initial_side"] for r in calls], ["A", "D", "A"])
        self.assertEqual(len({r["seed"] for r in calls}), 3)
        with patch("season.season_scrim_worker.play_scrim", return_value={"status": "cancelled"}):
            self.assertEqual(play_series(request)["status"], "cancelled")

    def test_series_preserves_frc_for_different_members_without_changing_progress(self):
        state = self.entered()
        run = state.tournament("cup")
        for ai in ("frc_v1", "frc_v1_baseline"):
            changed_run = replace(run, entrants=tuple(replace(team, ai=ai) for team in run.entrants))
            changed = replace(state, tournaments=(changed_run,))
            request = build_series_request(changed, "cup", render=False)
            with self.subTest(ai=ai):
                self.assertEqual(request["own"]["ai"], ai)
                self.assertEqual(request["opponent"]["ai"], ai)
                self.assertTrue(all(team.ai == ai for team in changed.tournament("cup").entrants))
                self.assertEqual(changed.tournament("cup").results, run.results)

    def test_real_background_series_worker(self):
        request = build_series_request(self.entered(), "cup", render=False, tick_time_ms=1)
        job = ScrimJob(request, self.path.parent / "series")
        try:
            end = time.monotonic() + 120
            result = None
            while result is None and time.monotonic() < end:
                result = job.poll()
                if result is None:
                    time.sleep(.05)
            self.assertIsNotNone(result)
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(max(result["left_wins"], result["right_wins"]), 1)
            self.assertEqual(len(result["maps"]), 1)
            self.assertEqual(result["match_id"], request["match_id"])
        finally:
            job.cancel()
            job.process.wait(timeout=10)


class BracketTest(unittest.TestCase):
    def test_all_team_counts_byes_loss_paths_and_final_targets(self):
        for mode, minimum in (("double_elimination", 4), ("single_elimination", 2)):
            for count in range(minimum, 33):
                for choose_right in (False, True):
                    with self.subTest(mode=mode, count=count, choose_right=choose_right):
                        event = definition_from_dict(definition(format=mode, team_count=count, prizes={1: 5000000}))
                        teams = tuple(SimpleNamespace(id=str(i)) for i in range(count))
                        generator = bracket(event, teams)
                        losses = {t.id: 0 for t in teams}
                        matches = []
                        try:
                            match = next(generator)
                            while True:
                                matches.append(match)
                                self.assertNotEqual(match.left, match.right)
                                loser = match.left if choose_right else match.right
                                winner = match.right if choose_right else match.left
                                losses[loser] += 1
                                match = generator.send(winner)
                        except StopIteration as finished:
                            ranking = finished.value
                        self.assertEqual(len(matches), 2 * count - 2 if mode == "double_elimination" else count - 1)
                        self.assertEqual(event.match_count, len(matches))
                        self.assertEqual(event.end_date, (date(2026, 2, 3) + timedelta(days=len(matches) - 1)).isoformat())
                        self.assertEqual(set(ranking), set(losses))
                        self.assertEqual(len(ranking), count)
                        self.assertEqual(matches[-1].stage, "grand_final")
                        self.assertEqual(matches[-1].maps_to_win, 3)
                        if mode == "double_elimination":
                            self.assertEqual(sum(m.stage == "lower_final" for m in matches), 1)
                            self.assertTrue(all(losses[t] == 2 for t in ranking[2:]))


class CompetitionScreenTest(SeasonCompetitionTest):
    # Custom load_tests below prevents the model tests from being collected twice.
    def setUp(self):
        super().setUp()
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.app = RealtimeSeasonApp(self.root, self.store, self.state())

    def test_home_calendar_icon_entry_real_poll_and_prize_display(self):
        app = self.app
        self.assertIn("2026/01/01", app.calendar_label.get())
        self.assertEqual(app.phase_label.get(), "オフシーズン")
        self.assertTrue(app.phase_icon.find_all())
        app.show_screen("competitions")
        app.competition_list.selection_set("cup")
        app.preview_competition()
        app.enter_competition()
        app.advance_to_competition()
        self.assertEqual(app.state.date, date(2026, 2, 3))
        while not app.state.tournament("cup").completed:
            if app.state.tournament("cup").last_match_date == app.state.game_date:
                self.assertEqual(str(app.competition_play_button["state"]), "disabled")
                app.advance_calendar(1)
            with patch("season.season_competition_ui.ScrimJob") as factory:
                app.start_competition_series()
            if not factory.called:
                self.assertTrue(app.state.tournament("cup").results[-1].decided_by_rating)
                continue
            request = factory.call_args.args[0]
            own = app.state.tournament("cup").own_team_id
            left_wins = request["maps_to_win"] if own == request["left_id"] or own not in (request["left_id"], request["right_id"]) else 0
            factory.return_value.poll.return_value = {"status": "completed", "match_id": request["match_id"],
                "left_id": request["left_id"], "right_id": request["right_id"], "left_wins": left_wins,
                "right_wins": request["maps_to_win"] if left_wins == 0 else 0}
            self.root.after_cancel(app._competition_after_id)
            app.poll_competition_series()
        self.assertIn("5,000,000", app.competition_status.get())
        self.assertEqual(self.store.load_or_create(), app.state)
        self.assertEqual(app.competition_job, None)
        self.assertEqual(str(app.competition_play_button["state"]), "disabled")

    def test_running_competition_blocks_date_and_scrim_and_handles_cancel(self):
        app = self.app
        app.competition_job = Mock()
        before = app.state
        app.advance_calendar(1)
        app.advance_game_month()
        app.start_scrim()
        self.assertEqual(app.state, before)
        self.assertIsNone(app.scrim_job)
        app.cancel_competition_series()
        app.competition_job.cancel.assert_called_once()
        app.competition_job = None

    def test_auto_execution_advances_date_between_series_and_stops_at_final(self):
        app = self.app
        app.commit(self.entered(), "entered")
        app.show_screen("competitions")
        app.competition_list.selection_set("cup")
        app.competition_auto.set(True)
        dates = []

        def make_job(request, output):
            dates.append(app.state.game_date)
            own = app.state.tournament("cup").own_team_id
            left = own == request["left_id"] or own not in (request["left_id"], request["right_id"])
            return Mock(poll=Mock(return_value={"status": "completed", "match_id": request["match_id"],
                "left_id": request["left_id"], "right_id": request["right_id"],
                "left_wins": request["maps_to_win"] if left else 0,
                "right_wins": 0 if left else request["maps_to_win"]}))

        with patch("season.season_competition_ui.ScrimJob", side_effect=make_job):
            app.start_competition_series()
            while not app.state.tournament("cup").completed:
                self.root.after_cancel(app._competition_after_id)
                if app.competition_job is not None:
                    app.poll_competition_series()
                else:
                    app._start_next_competition_series("cup")
        self.assertEqual(dates, [f"2026-02-{day:02}" for day in (3, 6, 8)])
        self.assertEqual(len(app.state.tournament("cup").results), 6)
        self.assertEqual(app.state.game_date, "2026-02-08")
        self.assertTrue(app.state.tournament("cup").completed)
        self.assertEqual(self.store.load_or_create(), app.state)
        self.assertIn("2026-02-08", app.competition_info.get())

    def test_cancelled_series_does_not_consume_daily_slot(self):
        app = self.app
        app.commit(self.entered(), "entered")
        app.show_screen("competitions")
        app.competition_list.selection_set("cup")
        before = app.state
        with patch("season.season_competition_ui.ScrimJob") as factory:
            app.start_competition_series()
        factory.return_value.poll.return_value = {"status": "cancelled"}
        self.root.after_cancel(app._competition_after_id)
        app.poll_competition_series()
        self.assertEqual(app.state, before)
        self.assertEqual(str(app.competition_play_button["state"]), "normal")
        build_series_request(app.state, "cup")

    def test_auto_next_day_save_failure_keeps_completed_series_and_current_date(self):
        app = self.app
        app.commit(self.entered(), "entered")
        app.show_screen("competitions")
        app.competition_list.selection_set("cup")
        app.competition_auto.set(True)
        with patch("season.season_competition_ui.ScrimJob") as factory:
            app.start_competition_series()
        request = factory.call_args.args[0]
        factory.return_value.poll.return_value = {"status": "completed", "match_id": request["match_id"],
            "left_id": request["left_id"], "right_id": request["right_id"], "left_wins": request["maps_to_win"], "right_wins": 0}
        self.root.after_cancel(app._competition_after_id)
        save = self.store.save
        calls = []

        def fail_next_day(candidate):
            calls.append(candidate.game_date)
            if len(calls) == 2:
                raise OSError("test save failure")
            save(candidate)

        with patch.object(self.store, "save", side_effect=fail_next_day), patch("run_realtime_season.messagebox.showerror"):
            app.poll_competition_series()
        self.assertEqual(calls, ["2026-02-03", "2026-02-04"])
        self.assertEqual(len(app.state.tournament("cup").results), 1)
        self.assertEqual(app.state.game_date, "2026-02-03")
        self.assertIsNone(app.competition_job)
        self.assertEqual(self.store.load_or_create(), app.state)
        with self.assertRaisesRegex(SeasonSaveError, "1日1試合"):
            build_series_request(app.state, "cup")
        app.advance_calendar(1)
        app.state.with_tournament_rating_result("cup")

    def test_phase_badge_updates_and_screen_layout_fits_default_window(self):
        app = self.app
        state = app.state.with_declined_tournament("cup").advance_days(59)
        app.commit(state, "date")
        self.assertEqual(app.phase_label.get(), "インシーズン")
        self.assertIn("2026/03/01", app.calendar_label.get())
        self.root.update_idletasks()
        # Home and tournament controls remain visible at the default height.
        self.assertLessEqual(app.home_host.winfo_reqheight(), 800)
        self.assertLessEqual(app.competition_host.winfo_reqheight(), 800)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(SeasonCompetitionTest))
    suite.addTests(loader.loadTestsFromTestCase(BracketTest))
    for name in ("test_home_calendar_icon_entry_real_poll_and_prize_display", "test_running_competition_blocks_date_and_scrim_and_handles_cancel",
                 "test_phase_badge_updates_and_screen_layout_fits_default_window",
                 "test_auto_execution_advances_date_between_series_and_stops_at_final", "test_cancelled_series_does_not_consume_daily_slot",
                 "test_auto_next_day_save_failure_keeps_completed_series_and_current_date"):
        suite.addTest(CompetitionScreenTest(name))
    return suite


if __name__ == "__main__":
    unittest.main()
