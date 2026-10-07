import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import realtime_season_teams
from realtime_season import SeasonSaveError, new_season
from season.season_scrim import ScrimJob, build_scrim_request, validate_scrim_request


OWN = ("Leo", "Boaster", "Derke", "Chronicle", "Alfajer")
RIVAL = ("Aspas", "valyn", "trent", "leaf", "tex", "Sato")
FRC = ("Furina", "Lisa", "Lohen", "Jean", "Arlecchino")


def scrim_state():
    with patch.object(realtime_season_teams, "SEASON_TEAMS", [{"name": "Rival", "players": list(RIVAL)}]):
        return new_season(OWN).with_roster(OWN).with_team_name("Own").with_confirmed_team()


class SeasonScrimTests(unittest.TestCase):
    def request(self, **kwargs):
        state = scrim_state()
        return build_scrim_request(state, state.teams[0].id, state.opponent_teams[0].id, seed=17, **kwargs)

    @patch("realtime_season_pair_familiarity.pair_familiarity_enabled", False)
    def test_request_uses_saved_abilities_and_only_five_rival_players(self):
        state = scrim_state()
        state = replace(state, owned_players=(replace(state.owned_players[0], iq=180), *state.owned_players[1:]))
        request = build_scrim_request(state, state.teams[0].id, state.opponent_teams[0].id, render=False, initial_side="D", tick_time_ms=15, seed=4)
        self.assertEqual(request["own"]["players"][0]["iq"], 180)
        self.assertEqual(request["own"]["igl"], "Leo")
        self.assertEqual(tuple(row["name"] for row in request["opponent"]["players"]), RIVAL[:5])
        self.assertEqual(request["initial_side"], "D")
        self.assertEqual(request["tick_time_ms"], 15)

    def test_request_defaults_to_club_settings_and_allows_per_match_overrides(self):
        with patch.object(realtime_season_teams, "SEASON_TEAMS", [{"name": "Rival", "players": list(RIVAL), "igl": "Aspas", "carrier": "trent", "ai": "fnatic_v3"}]):
            state = new_season(OWN).with_roster(OWN).with_confirmed_team()
        request = build_scrim_request(state, state.teams[0].id, state.opponent_teams[0].id)
        self.assertEqual(request["opponent"]["igl"], "Aspas")
        self.assertEqual(request["opponent"]["spike_holder"], "trent")
        self.assertEqual(request["opponent"]["ai"], "fnatic_v3")
        overridden = build_scrim_request(state, state.teams[0].id, state.opponent_teams[0].id, opponent_igl="leaf", opponent_spike="valyn", opponent_ai="default")
        self.assertEqual(overridden["opponent"]["igl"], "leaf")
        self.assertEqual(overridden["opponent"]["spike_holder"], "valyn")
        self.assertEqual(overridden["opponent"]["ai"], "default")
        self.assertEqual(state.opponent_teams[0].ai, "fnatic_v3")

    def test_rejects_invalid_roles_duplicate_members_and_headless_user_input(self):
        for kwargs in ({"own_igl": "Sato"}, {"opponent_spike": "Sato"}, {"render": False, "own_ai": "user"}, {"tick_time_ms": 0}, {"own_ai": "missing"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.request(**kwargs)
        request = self.request()
        request["opponent"]["players"][0] = copy.deepcopy(request["own"]["players"][0])
        with self.assertRaisesRegex(SeasonSaveError, "重複"):
            validate_scrim_request(request)

    def test_frc_keeps_selected_ai_with_replacement_and_bench_players(self):
        from character_stats import get_by_name
        for ai in ("frc_v1", "frc_v1_baseline"):
            with self.subTest(ai=ai), patch.object(realtime_season_teams, "SEASON_TEAMS",
                    [{"name": "Furina Party", "players": list(reversed(FRC)), "ai": ai}]):
                state = new_season(OWN).with_roster(OWN).with_confirmed_team()
                own, rival = state.teams[0], state.opponent_teams[0]
                request = build_scrim_request(state, own.id, rival.id, own_ai=ai)
                self.assertEqual(request["own"]["ai"], ai)
                self.assertEqual(request["opponent"]["ai"], ai)
                # A replacement or a member on the bench does not change the AI.
                mixed = replace(rival, players=(get_by_name("Tartaglia"), *rival.players[1:], rival.players[0]))
                changed = replace(state, opponent_teams=(mixed,))
                request = build_scrim_request(changed, own.id, rival.id)
                self.assertEqual(request["opponent"]["ai"], ai)
                self.assertEqual(changed.opponent_teams[0].ai, ai)
                self.assertEqual(build_scrim_request(state, own.id, rival.id)["opponent"]["ai"], ai)

    def test_preparation_keeps_frc_for_different_members_and_allows_start(self):
        import tkinter as tk
        from realtime_season import SeasonStore
        from run_realtime_season import RealtimeSeasonApp
        state = scrim_state()
        state = state.with_selected_team(state.teams[0].id)
        state = replace(state, opponent_teams=(replace(state.opponent_teams[0], ai="frc_v1"),))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = tk.Tk()
            root.withdraw()
            try:
                app = RealtimeSeasonApp(root, SeasonStore(Path(directory) / "save.json"), state)
                app.opponent_choice.set("Rival")
                app.preview_preparation()
                self.assertEqual(app.opponent_ai_choice.get(), "FRC v1（学習モデル）")
                self.assertNotIn("ロジックAI", app.prep_hint.get())
                self.assertEqual(str(app.scrim_start_button["state"]), "normal")
                app.opponent_ai_choice.set("ロジック")
                app.update_scrim_start_state()
                self.assertNotIn("FRC専用の5人", app.prep_hint.get())
                self.assertEqual(app.state.opponent_teams[0].ai, "frc_v1")
            finally:
                root.destroy()

    def test_worker_applies_saved_abilities_to_actual_game_characters(self):
        import character_stats
        import game_core
        from run_game import VisualFPSBattle
        from season.season_scrim_worker import play_scrim
        request = self.request(render=False)
        request["own"]["players"][0]["iq"] = 180
        observed = []

        def finish_match(game):
            observed.append(next(char.base_iq for char in game.chars if char.name == "Leo"))
            game.attacker_wins, game.defender_wins = 13, 0
            game.match_over = True

        with patch.dict(character_stats.CHARACTER_TABLE), patch.object(game_core, "_character_stats", game_core._character_stats), patch.object(VisualFPSBattle, "run", finish_match):
            result = play_scrim(request)
        self.assertEqual(observed, [180])
        self.assertEqual(result["own_score"], 13)
        self.assertEqual(result["winner"], "Own")

    def test_headless_worker_runs_a_complete_real_map_and_saves_result(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            # Both FRC modes must run their own controllers on non-FRC members.
            request = self.request(render=False, initial_side="D")
            request["own"]["ai"] = "frc_v1"
            request["opponent"]["ai"] = "frc_v1_baseline"
            job = ScrimJob(request, directory)
            try:
                deadline = time.monotonic() + 120
                result = None
                while result is None and time.monotonic() < deadline:
                    result = job.poll()
                    if result is None:
                        time.sleep(0.05)
                self.assertIsNotNone(result, f"Scrim timed out. Log: {job.log_path}")
                self.assertEqual(result["status"], "completed", result)
                own, rival = result["own_score"], result["opponent_score"]
                self.assertGreaterEqual(max(own, rival), 13)
                if min(own, rival) >= 12:
                    self.assertGreaterEqual(abs(own - rival), 2)
                self.assertEqual(result["winner"], "Own" if own > rival else "Rival")
                self.assertTrue(job.result_path.is_file())
                self.assertEqual(len(result["player_stats"]), 10)
            finally:
                job.cancel()
                job.process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
