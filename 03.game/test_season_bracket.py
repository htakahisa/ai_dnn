"""Bracket presentation and daily completion after player elimination."""

from dataclasses import replace
from datetime import timedelta
import json
from random import Random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from realtime_season import SeasonSaveError
from season_competitions import (SeriesScore, TournamentProgress, bracket_view,
                                 definition_from_dict, next_match, player_eliminated)
from season_ratings import expected_score, series_ratings
from test_season_competitions import (CompetitionScreenTest, SeasonCompetitionTest,
                                      config, definition, record_next)


def before_elimination(state):
    state = record_next(state, left_wins=False)
    return record_next(state.advance_days()).advance_days()


def losing_score(state):
    run = state.tournament("cup")
    match, _ = next_match(state.tournament_definition("cup"), run)
    assert run.own_team_id in (match.left, match.right)
    return SeriesScore(match.id, match.left, match.right,
                       0 if match.left == run.own_team_id else match.maps_to_win,
                       0 if match.right == run.own_team_id else match.maps_to_win)


def legacy_eliminated(state):
    """Construct a valid old save, before automatic completion existed."""
    score = losing_score(state)
    run = replace(state.tournament("cup"), results=(*state.tournament("cup").results, score),
                  last_match_date=state.game_date)
    state = replace(state, tournaments=(run,))
    return state.with_rated_result(f"tournament:cup:{score.match_id}", score.left_id,
                                  score.right_id, score.left_wins, score.right_wins)


class RatingCompletionTest(SeasonCompetitionTest):
    def test_upper_loss_continues_second_loss_finishes_daily_and_pays_once(self):
        first_loss = record_next(self.entered(), left_wins=False)
        self.assertFalse(first_loss.tournament("cup").completed)
        self.assertFalse(player_eliminated(first_loss.tournament_definition("cup"), first_loss.tournament("cup")))
        self.assertFalse(any(s.decided_by_rating for s in first_loss.tournament("cup").results))
        with self.assertRaisesRegex(SeasonSaveError, "敗退"):
            first_loss.with_tournament_rating_finish("cup")
        state = record_next(first_loss.advance_days()).advance_days()
        self.assertEqual(next_match(state.tournament_definition("cup"), state.tournament("cup"))[0].stage, "lower")
        legacy = legacy_eliminated(state)
        snapshot = {team.id: legacy.rating(team.id) for team in legacy.tournament("cup").entrants}
        finished = state.with_tournament_result("cup", losing_score(state))
        run = finished.tournament("cup")
        self.assertTrue(run.completed)
        self.assertEqual(run.ranking[-1], state.club_id)
        self.assertEqual(run.prize_paid, 500000)
        self.assertEqual(finished.money, state.money + 500000)
        self.assertEqual((finished.game_date, run.completed_date, run.last_match_date),
                         ("2026-02-09", "2026-02-08", "2026-02-08"))
        self.assertEqual(len(run.results), 6)
        actual = run.results[:3]
        generated = run.results[3:]
        self.assertTrue(all(not s.decided_by_rating for s in actual))
        self.assertTrue(all(s.decided_by_rating for s in generated))
        # Each draw uses ratings updated by the preceding result.
        expected_ratings = dict(snapshot)
        for index, score in enumerate(generated, len(actual)):
            winner = score.left_id if score.left_wins > score.right_wins else score.right_id
            probability = expected_score(expected_ratings[score.left_id], expected_ratings[score.right_id])
            draw = Random((run.seed + index * 1000) % (2**31)).random()
            self.assertEqual(winner, score.left_id if draw < probability else score.right_id)
            self.assertNotIn(state.club_id, (score.left_id, score.right_id))
            left, right = series_ratings(expected_ratings[score.left_id], expected_ratings[score.right_id],
                                         score.left_wins, score.right_wins)
            expected_ratings.update({score.left_id: left, score.right_id: right})
        self.assertEqual({t: finished.rating(t) for t in snapshot}, expected_ratings)
        self.assertEqual(snapshot[generated[0].left_id], snapshot[generated[0].right_id])
        self.assertEqual(len([r for r in finished.rated_results if r.startswith("tournament:cup:")]), 6)
        self.assertEqual(finished.contract("Leo").team_loyalty, legacy.contract("Leo").team_loyalty)
        self.store.save(finished)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, finished)
        self.assertIs(loaded.with_tournament_rating_finish("cup"), loaded)
        self.assertEqual(loaded.advance_days().date, finished.date + timedelta(days=1))

    def test_single_elimination_first_loss_finishes_and_allows_editing(self):
        with patch.object(config, "TOURNAMENTS", [definition(format="single_elimination")]):
            state = self.entered()
        finished = state.with_tournament_forfeit("cup")
        run = finished.tournament("cup")
        self.assertTrue(run.completed)
        self.assertEqual(run.completed_date, "2026-02-05")
        self.assertEqual(finished.game_date, "2026-02-06")
        self.assertEqual(len(run.results), 3)
        self.assertEqual([s.decided_by_rating for s in run.results], [False, True, True])
        self.assertEqual(run.ranking[-1], state.club_id)
        self.assertEqual(finished.money, state.money + run.prize_paid)
        finished.with_roster(tuple(reversed(state.roster))).with_confirmed_team()

    def test_version_eleven_eliminated_save_is_read_only_until_explicit_completion(self):
        legacy = legacy_eliminated(before_elimination(self.entered()))
        self.store.save(legacy)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["version"] = 11
        for score in data["tournaments"][0]["results"]:
            score.pop("decided_by_rating")
        self.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, legacy)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(loaded.tournament("cup").completed)
        finished = loaded.with_tournament_rating_finish("cup")
        self.store.save(finished)
        self.assertEqual(self.store.load_or_create(), finished)
        self.assertEqual(finished.money, loaded.money + finished.tournament("cup").prize_paid)
        self.assertEqual(finished.date, loaded.date + timedelta(days=4))

    def test_lower_final_elimination_keeps_played_cards_and_only_awards_final(self):
        state = self.entered()
        for left_wins in (True, True, True, False):
            if state.tournament("cup").last_match_date == state.game_date:
                state = state.advance_days()
            state = record_next(state, left_wins=left_wins)
        self.assertFalse(state.tournament("cup").completed)
        state = state.advance_days()
        match, _ = next_match(state.tournament_definition("cup"), state.tournament("cup"))
        self.assertEqual(match.stage, "lower_final")
        self.assertEqual(match.maps_to_win, 2)
        finished = state.with_tournament_forfeit("cup")
        run = finished.tournament("cup")
        self.assertEqual(run.results[:4], state.tournament("cup").results)
        self.assertEqual([s.decided_by_rating for s in run.results], [False] * 5 + [True])
        self.assertEqual(run.ranking[2], state.club_id)
        self.assertEqual(run.prize_paid, 1000000)
        self.assertEqual(max(run.results[-1].left_wins, run.results[-1].right_wins), 3)
        self.assertEqual(run.completed_date, (state.date + timedelta(days=1)).isoformat())
        self.assertEqual(finished.date, state.date + timedelta(days=2))

    def test_skipped_matches_cross_month_pay_finances_and_monthly_events_once(self):
        with patch.object(config, "TOURNAMENTS", [definition(start_date="2026-01-28")]):
            state = self.state()
            state = state.with_tournament_entry("cup", state.selected_team_id)
            state = before_elimination(state.advance_days(27))
        legacy = legacy_eliminated(state)
        income = legacy.monthly_sponsor_income
        payroll = legacy.monthly_payroll
        finished = state.with_tournament_forfeit("cup")
        run = finished.tournament("cup")
        self.assertEqual(state.game_date, "2026-01-30")
        self.assertEqual((finished.game_date, run.completed_date), ("2026-02-03", "2026-02-02"))
        self.assertEqual(finished.game_month, 1)
        self.assertEqual(finished.money, state.money + income - payroll + run.prize_paid)
        self.assertEqual(finished.contract("Leo").team_loyalty, legacy.contract("Leo").team_loyalty - .5)
        self.assertEqual(len([e for e in finished.monthly_events if e.kind == "month_completed"]), 1)
        self.store.save(finished)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, finished)
        self.assertEqual(loaded.with_tournament_rating_finish("cup"), loaded)

    def test_invalid_rating_markers_reject_without_overwriting_save(self):
        state = record_next(self.entered())
        self.store.save(state)
        before = self.path.read_bytes()
        for value in (True, 1, "false"):
            with self.subTest(value=value):
                run = state.tournament("cup")
                run = replace(run, results=(replace(run.results[0], decided_by_rating=value),))
                with self.assertRaises(ValueError):
                    self.store.save(replace(state, tournaments=(run,)))
                self.assertEqual(self.path.read_bytes(), before)


class BracketProjectionTest(unittest.TestCase):
    def test_all_sizes_and_byes_use_actual_results_and_forward_feeders(self):
        for mode, minimum in (("single_elimination", 2), ("double_elimination", 4)):
            for count in range(minimum, 33):
                with self.subTest(mode=mode, count=count):
                    event = definition_from_dict(definition(format=mode, team_count=count, prizes={1: 1}))
                    teams = tuple(SimpleNamespace(id=str(i)) for i in range(count))
                    run = TournamentProgress("cup", "0", teams)
                    expected_count = count - 1 if mode == "single_elimination" else count * 2 - 2
                    while True:
                        cards = bracket_view(event, run)
                        self.assertEqual(len(cards), expected_count)
                        self.assertEqual(sum(c.pending for c in cards), 0 if run.completed else 1)
                        seen = {}
                        for card in cards:
                            for slot in (card.left, card.right):
                                if slot.source_match is None:
                                    self.assertIsNotNone(slot.team_id)
                                    continue
                                source = seen[slot.source_match]
                                if source.score is None:
                                    self.assertIsNone(slot.team_id)
                                else:
                                    s = source.score
                                    winner, loser = (s.left_id, s.right_id) if s.left_wins > s.right_wins else (s.right_id, s.left_id)
                                    self.assertEqual(slot.team_id, winner if slot.source_outcome == "winner" else loser)
                            if card.score:
                                self.assertEqual((card.left.team_id, card.right.team_id), (card.score.left_id, card.score.right_id))
                            seen[card.match.id] = card
                        if run.completed:
                            break
                        match, _ = next_match(event, run)
                        # Alternate winners: future projection must not predict these IDs.
                        left = len(run.results) % 2 == 0
                        score = SeriesScore(match.id, match.left, match.right,
                                            match.maps_to_win if left else 0, 0 if left else match.maps_to_win)
                        run = replace(run, results=(*run.results, score))
                        pending, ranking = next_match(event, run)
                        run = replace(run, completed=pending is None, ranking=ranking)


class BracketScreenTest(CompetitionScreenTest):
    def select_entered(self, state):
        self.app.commit(state, "entered")
        self.app.show_screen("competitions")
        self.app.competition_list.selection_set("cup")
        self.app.preview_competition()

    def test_bracket_cards_and_completed_results_fit_window(self):
        self.select_entered(self.entered())
        app = self.app
        self.assertEqual(app.competition_tabs.tab(app.competition_tabs.select(), "text"), "トーナメント表")
        self.assertEqual(len(app.competition_bracket.cards), 6)
        canvas = app.competition_bracket.canvas
        self.assertTrue(canvas.find_withtag("connector"))
        self.assertTrue(canvas.find_withtag("unresolved"))
        self.assertTrue(canvas.find_withtag(f"team:{app.state.club_id}"))
        for state in (app.state, before_elimination(app.state).with_tournament_forfeit("cup")):
            app.commit(state, "update")
            app.preview_competition()
            self.root.update_idletasks()
            self.assertLessEqual(app.competition_host.winfo_reqheight(), 800)
        texts = [canvas.itemcget(i, "text") for i in canvas.find_all() if canvas.type(i) == "text"]
        self.assertEqual(sum("レート判定" in t for t in texts), 4)  # Legend and three awarded cards.
        self.assertFalse(canvas.find_withtag("unresolved"))
        self.assertEqual(sum(canvas.itemcget(i, "fill") == "#14532d" for i in canvas.find_all()
                             if canvas.type(i) == "rectangle"), 6)
        self.assertIn("敗退", app.competition_info.get())

    def test_elimination_stops_worker_and_advances_skipped_match_days(self):
        state = before_elimination(self.entered())
        self.select_entered(state)
        app = self.app
        app.competition_auto.set(True)
        score = losing_score(state)
        with patch("season_competition_ui.ScrimJob") as factory:
            app.start_competition_series()
            factory.return_value.poll.return_value = {"status": "completed", "match_id": score.match_id,
                "left_id": score.left_id, "right_id": score.right_id,
                "left_wins": score.left_wins, "right_wins": score.right_wins}
            self.root.after_cancel(app._competition_after_id)
            app.poll_competition_series()
            factory.assert_called_once()
        self.assertIsNone(app.competition_job)
        self.assertEqual(app.state.game_date, "2026-02-09")
        self.assertTrue(app.state.tournament("cup").completed)
        self.assertEqual(self.store.load_or_create(), app.state)
        self.assertIn("500,000", app.competition_status.get())
        rows = [app.competition_matches.item(i, "values") for i in app.competition_matches.get_children()]
        self.assertEqual(sum("レート判定" in row[2] for row in rows), 3)

    def test_old_eliminated_save_finishes_without_launching_worker(self):
        state = legacy_eliminated(before_elimination(self.entered()))
        self.select_entered(state)
        self.assertEqual(str(self.app.competition_play_button["state"]), "normal")
        self.assertEqual(self.app.competition_play_button["text"], "レート判定で大会を終了")
        with patch("season_competition_ui.ScrimJob") as factory:
            self.app.start_competition_series()
            factory.assert_not_called()
        self.assertTrue(self.app.state.tournament("cup").completed)
        self.assertEqual(self.app.state.game_date, "2026-02-09")
        self.assertEqual(self.store.load_or_create(), self.app.state)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for cls in (RatingCompletionTest, BracketProjectionTest, BracketScreenTest):
        for name in cls.__dict__:
            if name.startswith("test_"):
                suite.addTest(cls(name))
    return suite


if __name__ == "__main__":
    unittest.main()
