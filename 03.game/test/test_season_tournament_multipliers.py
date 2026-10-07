"""Tournament difficulty applies to opponents and survives saved event snapshots."""

from dataclasses import asdict, replace
import json
import tkinter as tk
import unittest
from unittest.mock import patch

import character_stats
import realtime_season_pair_familiarity as pair_settings
import test_season_world_levels as fixtures
from realtime_season import SeasonSaveError
from run_realtime_season import RealtimeSeasonApp
from season.season_competitions import CompetitionError, definition_from_dict, next_match
from season.season_scrim import build_scrim_request
from season.season_series import build_series_request


class TournamentMultiplierTests(unittest.TestCase):
    def setUp(self):
        context = patch.dict(character_stats.CHARACTER_TABLE, {
            name: replace(player, debut_chapter=1)
            for name, player in character_stats.CHARACTER_TABLE.items()})
        context.start()
        self.addCleanup(context.stop)
        fixtures.WorldLevelTests.setUp(self)

    def state(self, multiplier=1.0, rank=2):
        state = fixtures.WorldLevelTests.state(self, rank)
        return replace(state, tournament_definitions=tuple(
            replace(event, enemy_multiplier=multiplier, team_count=2)
            for event in state.tournament_definitions))

    def at_start(self, state):
        state = state.with_tournament_entry("cup", state.selected_team_id)
        state = replace(state, game_date="2026-03-01", game_month=2, monthly_events_through=2)
        return state._with_tournament_world_level()

    def sides(self, state, request):
        if request["left_id"] == state.club_id:
            return request["own"], request["opponent"]
        return request["opponent"], request["own"]

    def test_multiplies_world_and_pair_effects_and_keeps_owned_abilities(self):
        for multiplier in (0.5, 1.0, 1.4):
            with self.subTest(multiplier=multiplier):
                state = self.at_start(self.state(multiplier))
                original = asdict(state)
                request = build_series_request(state, "cup", render=False)
                own, enemy = self.sides(state, request)
                self.assertEqual(own["players"], [asdict(p) for p in state.match_players(
                    state.tournament_team("cup").players)])
                rival = next(t for t in state.tournament("cup").entrants if t.id != state.club_id)
                pair_multiplier = state.pair_metrics(tuple(p.name for p in rival.players))[1]
                for player, effective in zip(rival.players, enemy["players"]):
                    combined = state.world_level_settings.enemy_multiplier * pair_multiplier * multiplier
                    self.assertAlmostEqual(effective["iq"], player.iq * combined)
                    self.assertAlmostEqual(effective["hit_pct"], player.hit_pct * combined)
                    self.assertAlmostEqual(effective["hs_pct"], min(1, player.hs_pct * combined))
                self.assertEqual(asdict(state), original)

    def test_multiplier_targets_enemy_when_own_team_is_on_the_right(self):
        state = self.at_start(self.state(1.4))
        run = state.tournament("cup")
        if next_match(state.tournament_definition("cup"), run)[0].left == state.club_id:
            run = replace(run, entrants=tuple(reversed(run.entrants)))
            state = replace(state, tournaments=(run,))
        self.assertEqual(next_match(state.tournament_definition("cup"), run)[0].right, state.club_id)
        request = build_series_request(state, "cup", render=False)
        own, enemy = self.sides(state, request)
        self.assertEqual(own["players"], [asdict(p) for p in state.match_players(state.tournament_team("cup").players)])
        rival = next(t for t in run.entrants if t.id != state.club_id)
        expected = state.match_players(rival.players, enemy=True, tournament_multiplier=1.4)
        self.assertEqual(enemy["players"], [asdict(p) for p in expected])

    @patch.object(pair_settings, "pair_familiarity_enabled", False)
    def test_caps_apply_after_combining_multipliers(self):
        state = self.at_start(self.state(0.5))
        run = state.tournament("cup")
        state = replace(state, tournaments=(replace(run, entrants=tuple(
            replace(team, players=tuple(replace(p, hs_pct=0.8, dodge_pct=0.9,
                                               mental=9, hit_pct=2) for p in team.players))
            if team.id != state.club_id else team for team in run.entrants)),))
        _, enemy = self.sides(state, build_series_request(state, "cup", render=False))
        self.assertAlmostEqual(enemy["players"][0]["hs_pct"], 0.6)
        self.assertAlmostEqual(enemy["players"][0]["dodge_pct"], 0.675)
        self.assertAlmostEqual(enemy["players"][0]["mental"], 6.75)
        self.assertAlmostEqual(enemy["players"][0]["hit_pct"], 1.5)

    def test_scrim_does_not_use_tournament_multiplier(self):
        state = self.state(1.7)
        neutral = replace(state, tournament_definitions=tuple(
            replace(event, enemy_multiplier=1) for event in state.tournament_definitions))
        request = build_scrim_request(state, state.selected_team_id, state.opponent_teams[0].id, seed=12)
        expected = build_scrim_request(neutral, neutral.selected_team_id, neutral.opponent_teams[0].id, seed=12)
        self.assertEqual(request, expected)

    def test_invalid_multiplier_import_preserves_saved_file(self):
        state = self.state()
        self.store.save(state)
        original = self.store.path.read_bytes()
        for value in (0, -1, True, None, "1.2", float("inf"), float("nan")):
            with self.subTest(value=value), patch.object(fixtures.calendar, "TOURNAMENTS", [
                    {**fixtures.CUP, "enemy_multiplier": value}]):
                with self.assertRaises(SeasonSaveError):
                    self.store.import_competitions(state)
                self.assertEqual(self.store.path.read_bytes(), original)
                with self.assertRaises(CompetitionError):
                    definition_from_dict({**fixtures.CUP, "enemy_multiplier": value})

    def test_registered_multiplier_survives_import_restart_and_chapter_archive(self):
        state = self.at_start(self.state(1.35))
        self.store.save(state)
        with patch.object(fixtures.calendar, "TOURNAMENTS", [{**fixtures.CUP, "enemy_multiplier": 2}]):
            imported = self.store.import_competitions(self.store.load_or_create())
        self.assertEqual(imported.tournament_definition("cup").enemy_multiplier, 1.35)
        loaded = self.store.load_or_create()
        self.assertEqual(build_series_request(loaded, "cup", render=False),
                         build_series_request(state, "cup", render=False))
        from season.season_chapter_progression import ChapterEnvironment, environment_from_save
        archive = ChapterEnvironment.capture(state)
        restored = environment_from_save(json.loads(json.dumps(asdict(archive))))
        self.assertEqual(restored.tournament_definitions[0].enemy_multiplier, 1.35)

    def test_legacy_definitions_default_to_one_and_invalid_saved_value_is_rejected(self):
        state = self.state(1.2)
        self.store.save(state)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["tournament_definitions"][0].pop("enemy_multiplier")
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.store.load_or_create().tournament_definition("cup").enemy_multiplier, 1)
        data["tournament_definitions"][0]["enemy_multiplier"] = 0
        original = json.dumps(data)
        self.store.path.write_text(original, encoding="utf-8")
        with self.assertRaises(SeasonSaveError):
            self.store.load_or_create()
        self.assertEqual(self.store.path.read_text(encoding="utf-8"), original)

    def test_unregistered_import_and_next_year_keep_configured_multiplier(self):
        state = self.state()
        with patch.object(fixtures.calendar, "TOURNAMENTS", [
                {**fixtures.CUP, "start_date": "03-01", "enemy_multiplier": 1.3}]):
            imported = self.store.import_competitions(state)
        self.assertEqual(imported.tournament_definition("cup_2026").enemy_multiplier, 1.3)
        self.assertEqual(imported.tournament_definition("cup_2027").enemy_multiplier, 1.3)
        self.assertEqual(imported.tournament_definition("cup_2027").for_year(2028).enemy_multiplier, 1.3)

    def test_competition_details_show_multiplier(self):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        app = RealtimeSeasonApp(root, self.store, self.state(1.4))
        app.show_screen("competitions")
        app.competition_list.selection_set("cup")
        app.preview_competition()
        self.assertIn("大会の敵ステータス倍率: 1.4倍", app.competition_info.get())


if __name__ == "__main__":
    unittest.main()
