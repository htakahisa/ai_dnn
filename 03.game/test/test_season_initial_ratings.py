"""Configured starting ratings, ranking, imports and save compatibility."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import realtime_season_competitions as calendar
import realtime_season_config as config
import realtime_season_teams as teams
import realtime_season_world_levels as levels
from realtime_season import SeasonSaveError, SeasonStore, new_season
from season.season_ratings import series_ratings


OWN = ("Leo", "Boaster", "Derke", "Chronicle", "Alfajer")
RIVALS = (("Aspas", "valyn", "trent", "leaf", "tex"),
          ("Boostio", "Ethan", "jawgemo", "C0M", "Demon1"))


class InitialRatingTests(unittest.TestCase):
    def setUp(self):
        self.clubs = [dict(name=f"Rival{i}", players=list(names), initial_rating=rating)
                      for i, (names, rating) in enumerate(zip(RIVALS, (2100.5, 900)))]
        for context in (
            patch.object(teams, "SEASON_TEAMS", self.clubs),
            patch.object(config, "INITIAL_TEAM_RATING", 1700),
            patch.object(config, "INITIAL_OWNED_PLAYERS", OWN),
            patch.object(calendar, "START_DATE", "2026-01-01"),
            patch.object(calendar, "TOURNAMENTS", []),
            patch.object(levels, "WORLD_LEVELS", [
                {"レベル": 1, "必要レート": 0, "敵倍率": 1, "スポンサー資金": 100},
                {"レベル": 2, "必要レート": 1600, "敵倍率": 1.5, "スポンサー資金": 200},
                {"レベル": 3, "必要レート": 2000, "敵倍率": 2, "スポンサー資金": 300},
            ]),
        ):
            context.start()
            self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.store = SeasonStore(Path(directory.name) / "save.json")

    def state(self):
        return new_season(OWN).with_roster(OWN).with_confirmed_team()

    def test_new_game_uses_configured_ratings_in_ranking_and_world_level(self):
        state = self.state()
        self.assertEqual([r.value for r in state.rating_ranking], [2100.5, 1700, 900])
        self.assertEqual(state.world_rank, (2, 3))
        self.assertEqual(state.world_level, 2)
        self.assertEqual(state.rating(state.teams[0].id), 1700)
        pending = new_season()
        self.assertEqual(pending.ratings, state.ratings)
        chosen = pending.with_initial_selection(OWN)
        self.assertEqual(chosen.ratings, pending.ratings)
        self.assertEqual(chosen.world_rank, (2, 3))

    def test_omitted_rating_defaults_to_1500_and_zero_is_allowed(self):
        self.clubs[0].pop("initial_rating")
        self.clubs[1]["initial_rating"] = 0
        with patch.object(config, "INITIAL_TEAM_RATING", 0):
            state = self.state()
        self.assertEqual(state.rating(state.club_id), 0)
        self.assertEqual([state.rating(c.id) for c in state.opponent_teams], [1500, 0])

    def test_first_match_updates_from_configured_ratings(self):
        state = self.state()
        rival = state.opponent_teams[0]
        expected = series_ratings(1700, 2100.5, 2, 1)
        won = state.with_rated_result("first", state.club_id, rival.id, 2, 1)
        self.assertEqual((won.rating(state.club_id), won.rating(rival.id)), expected)
        self.assertEqual(won.opponent_teams[0].initial_rating, 2100.5)
        self.assertEqual(won.with_registered_ratings().ratings, won.ratings)

    def test_saved_progress_survives_config_changes_and_import(self):
        state = self.state()
        won = state.with_rated_result("first", state.club_id, state.opponent_teams[0].id, 1, 0)
        self.store.save(won)
        self.clubs[0]["initial_rating"] = 3000
        self.clubs[1]["initial_rating"] = 4000
        with patch.object(config, "INITIAL_TEAM_RATING", 5000):
            loaded = self.store.load_or_create()
            self.assertEqual(loaded, won)
            imported = self.store.import_season_teams(loaded)
            self.assertEqual(imported.ratings, won.ratings)
            self.assertEqual(imported.opponent_teams[0].initial_rating, 3000)
            fresh = self.state()
            self.assertEqual(fresh.rating(fresh.club_id), 5000)
            self.assertEqual(fresh.rating(fresh.opponent_teams[0].id), 3000)
        self.assertEqual(self.store.load_or_create(), imported)

    def test_import_seeds_only_new_teams(self):
        with patch.object(teams, "SEASON_TEAMS", self.clubs[:1]):
            state = self.state()
        state = state.with_rated_result("first", state.club_id, state.opponent_teams[0].id, 0, 1)
        imported = self.store.import_season_teams(state)
        self.assertEqual(imported.rating(imported.opponent_teams[1].id), 900)
        for record in state.ratings:
            self.assertEqual(imported.rating(record.team_id), record.value)

    def test_legacy_save_retains_actual_ratings_without_rewriting_file(self):
        state = self.state()
        state = state.with_rated_result("first", state.club_id, state.opponent_teams[0].id, 0, 1)
        self.store.save(state)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 19
        for club in data["opponent_teams"]:
            club.pop("initial_rating")
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.store.path.read_bytes()
        self.clubs[0]["initial_rating"] = 9000
        with patch.object(config, "INITIAL_TEAM_RATING", 8000):
            loaded = self.store.load_or_create()
        self.assertEqual(loaded.ratings, state.ratings)
        self.assertEqual(loaded.world_rank, state.world_rank)
        self.assertEqual(self.store.path.read_bytes(), before)
        self.store.save(loaded)
        self.assertEqual(self.store.load_or_create(), loaded)

    def test_invalid_configuration_leaves_existing_save_untouched(self):
        state = self.state()
        self.store.save(state)
        before = self.store.path.read_bytes()
        for value in (-1, True, "1800", None, float("inf"), float("nan")):
            with self.subTest(value=value):
                self.clubs[0]["initial_rating"] = value
                with self.assertRaisesRegex(SeasonSaveError, "初期レート"):
                    self.store.import_season_teams(state)
                self.assertEqual(self.store.path.read_bytes(), before)
                self.clubs[0]["initial_rating"] = 2100.5
                with patch.object(config, "INITIAL_TEAM_RATING", value):
                    with self.assertRaisesRegex(SeasonSaveError, "初期レート"):
                        self.state()
                    self.assertEqual(self.store.load_or_create(), state)
                self.assertEqual(self.store.path.read_bytes(), before)

    def test_invalid_saved_initial_rating_is_rejected(self):
        state = self.state()
        self.store.save(state)
        original = json.loads(self.store.path.read_text(encoding="utf-8"))
        for value in (-1, True, "1800", None, float("inf"), float("nan")):
            with self.subTest(value=value):
                data = json.loads(json.dumps(original))
                data["opponent_teams"][0]["initial_rating"] = value
                self.store.path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(SeasonSaveError, "初期レート"):
                    self.store.load_or_create()
        original["opponent_teams"][0].pop("initial_rating")
        self.store.path.write_text(json.dumps(original), encoding="utf-8")
        with self.assertRaisesRegex(SeasonSaveError, "初期レート"):
            self.store.load_or_create()

    def test_unregistered_club_uses_initial_rating_even_with_tournament_entrant(self):
        state = self.state()
        club = state.opponent_teams[0]
        # A tournament snapshot must not create a default rating ahead of the club.
        from types import SimpleNamespace
        state = replace(state, ratings=(), tournaments=(SimpleNamespace(entrants=(club,)),))
        self.assertEqual(state.rating(club.id), 2100.5)
        self.assertEqual(state.with_registered_ratings().rating(club.id), 2100.5)


if __name__ == "__main__":
    unittest.main()
