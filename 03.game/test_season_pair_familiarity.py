"""Co-membership, season-only multipliers, migration, news and touch UI."""

from dataclasses import asdict, replace
from itertools import combinations
import json
import math
from pathlib import Path
from random import Random
import tempfile
import unittest
from unittest.mock import patch

import character_stats
import realtime_season_competitions as calendar
import realtime_season_config as config
import realtime_season_pair_familiarity as settings
import realtime_season_rival_economy as economy
import realtime_season_teams as teams
import realtime_season_world_levels as world
from realtime_season import SeasonSaveError, SeasonStore, new_season
from season_pair_familiarity import familiarity, initial_pair_days, pair_key, stage_index, team_metrics
from season_rival_economy import make_offer


OWN = ("Leo", "Boaster", "Derke", "Chronicle", "Alfajer")
RIVAL = ("Aspas", "valyn", "trent", "leaf", "tex", "Sato")


class PairFamiliarityTests(unittest.TestCase):
    def setUp(self):
        fixture = {name: replace(p, monthly_salary=100_000, loyalty=10)
                   for name, p in character_stats.CHARACTER_TABLE.items()}
        for context in (
            patch.dict(character_stats.CHARACTER_TABLE, fixture),
            patch.object(config, "INITIAL_OWNED_PLAYERS", OWN),
            patch.object(teams, "SEASON_TEAMS", [dict(name="Rival", players=list(RIVAL), initial_money=100_000_000)]),
            patch.object(calendar, "START_DATE", "2026-01-01"),
            patch.object(calendar, "TOURNAMENTS", []),
            patch.object(economy, "NON_REGULAR_OFFER_CHANCE", 0),
            patch.object(settings, "pair_familiarity_enabled", True),
            patch.object(world, "WORLD_LEVELS", [
                {"レベル": 1, "上位%": 100, "敵倍率": 1.5, "スポンサー資金": 10_000_000}]),
        ):
            context.start()
            self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.store = SeasonStore(Path(directory.name) / "save.json")

    def state(self, *, bench=False):
        state = new_season((*OWN, "Meiy") if bench else OWN)
        state = state.with_roster(OWN).with_confirmed_team()
        return replace(state.with_selected_team(state.teams[0].id), money=100_000_000)

    def test_daily_counts_include_bench_and_both_clubs_but_not_lft(self):
        state = self.state(bench=True)
        advanced = state.advance_days(3)
        self.assertEqual(advanced.pair_days[pair_key("Leo", "Meiy")], 3)
        self.assertEqual(advanced.pair_days[pair_key("Aspas", "Sato")],
                         state.pair_days[pair_key("Aspas", "Sato")] + 3)
        self.assertNotIn(pair_key("Leo", "Aspas"), advanced.pair_days)
        self.assertNotIn(pair_key("Leo", "Less"), advanced.pair_days)
        self.assertNotIn(pair_key("Leo", "Meiy"), state.pair_days)

    def test_cumulative_days_survive_move_to_another_club(self):
        state = self.state().advance_days(3)
        for name in ("Leo", "Boaster"):
            buyer = state.opponent_teams[0]
            state = make_offer(state, buyer, state.player(name), lambda *args: None)
            state = state.with_transfer_response(state.transfer_offer(name).id, True)
        state = state.advance_days(5)
        self.assertEqual(state.pair_days[pair_key("Leo", "Boaster")], 8)

    def test_lft_pause_and_reunion_preserve_days(self):
        state = self.state(bench=True).advance_days(3)
        state = state.without_player("Meiy").advance_days(5)
        self.assertEqual(state.pair_days[pair_key("Leo", "Meiy")], 3)
        # The signing action itself now supplies the first reunion day.
        reunited = state.with_scouted_player("Meiy", "year1").advance_days(4)
        self.assertEqual(reunited.pair_days[pair_key("Leo", "Meiy")], 8)

    def test_batched_days_match_individual_days_across_month_boundary(self):
        state = self.state(bench=True)
        together = state.advance_days(34)
        individual = state
        for _ in range(34):
            individual = individual.advance_days()
        self.assertEqual(together, individual)
        self.assertEqual(together.history, individual.history)
        self.assertEqual(together.pair_days[pair_key("Leo", "Meiy")], 34)

    def test_month_end_uses_final_roster_after_departures(self):
        state = self.state(bench=True)
        state = replace(state, contracts=tuple(
            replace(c, kind="short", duration_months=6, team_loyalty=0) if c.player_name == "Meiy" else c
            for c in state.contracts))
        state = state.advance_days(31)
        self.assertIsNone(state.player("Meiy"))
        self.assertEqual(state.pair_days[pair_key("Leo", "Meiy")], 30)

    def test_randomized_ai_seeds_only_retained_regular_members(self):
        state = new_season()
        self.assertEqual(state.pair_days, {})
        with patch("season_transfers.Random", return_value=Random(12)):
            state = state.with_initial_selection(OWN)
        club = state.opponent_teams[0]
        retained = set(club.members).intersection(club.regular_members)
        self.assertTrue(set(club.members) - set(club.regular_members))
        self.assertEqual(set(state.pair_days), {pair_key(a, b) for a, b in combinations(retained, 2)})
        self.assertTrue(all(160 <= days <= 720 for days in state.pair_days.values()))
        seeded = initial_pair_days(replace(state, pair_days={}))
        self.assertEqual(seeded.pair_days, state.pair_days)
        for a, b in combinations(OWN, 2):
            self.assertNotIn(pair_key(a, b), state.pair_days)

    def test_monthly_return_starts_new_pairs_at_zero_without_backfill(self):
        with patch("season_transfers.Random", return_value=Random(12)):
            state = new_season().with_initial_selection(OWN)
        club = state.opponent_teams[0]
        missing = next(n for n in club.regular_members if n not in club.members)
        retained = next(n for n in club.members if n in club.regular_members)
        state = state.advance_days(30)
        self.assertNotIn(pair_key(missing, retained), state.pair_days)
        returned = state.advance_days()
        self.assertIn(missing, returned.opponent_teams[0].members)
        self.assertEqual(returned.pair_days[pair_key(missing, retained)], 1)

    def test_initialization_does_not_consume_global_game_randomness(self):
        import random
        state = self.state()
        before = random.getstate()
        with patch("realtime_season.Random", side_effect=AssertionError("game RNG consumed")):
            initial_pair_days(replace(state, pair_days={}))
        self.assertEqual(random.getstate(), before)

    def test_save_roundtrip_and_old_save_migration_are_not_destructive(self):
        state = self.state().advance_days(3)
        self.store.save(state)
        self.assertEqual(self.store.load_or_create(), state)
        self.assertEqual(self.store.load_or_create().pair_days, state.pair_days)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 20
        data.pop("pair_days")
        data.pop("pair_familiarity_seed")
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.store.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertNotIn(pair_key("Leo", "Boaster"), loaded.pair_days)
        self.assertTrue(loaded.pair_days)
        self.assertEqual(loaded.money, state.money)
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_bad_pairs_fail_without_overwriting_save(self):
        state = self.state()
        self.store.save(state)
        before = self.store.path.read_bytes()
        for days in (0, -1, True, 1.5):
            with self.assertRaises(SeasonSaveError):
                self.store.save(replace(state, pair_days={pair_key("Leo", "Boaster"): days}))
            self.assertEqual(self.store.path.read_bytes(), before)

    def test_familiarity_curve_and_stage_boundaries(self):
        self.assertEqual(familiarity(0), 0)
        self.assertAlmostEqual(familiarity(180), 0.6321205588)
        self.assertAlmostEqual(familiarity(414), 0.90, places=3)
        for index, (_, boundary) in enumerate(settings.STAGES[1:], 1):
            self.assertEqual(stage_index(boundary - 1e-10), index - 1)
            self.assertEqual(stage_index(boundary), index)
            self.assertEqual(stage_index(boundary + 1e-10), index)

    def test_team_metrics_endpoints_and_replacement(self):
        self.assertEqual(team_metrics({}, OWN), (0, 0.8))
        days = {pair_key(a, b): 1_000_000 for a, b in combinations(OWN, 2)}
        c, multiplier = team_metrics(days, OWN)
        self.assertEqual(c, 1)
        self.assertAlmostEqual(multiplier, 1.1)
        replaced = (*OWN[:4], "Meiy")
        self.assertAlmostEqual(team_metrics(days, replaced)[0], 0.6)
        self.assertEqual(len(days), 10)

    def test_scrim_applies_combined_multiplier_once_to_both_sides(self):
        from season_scrim import build_scrim_request
        from season_world_levels import scale_enemy_player
        state = self.state()
        rival = state.opponent_teams[0]
        request = build_scrim_request(state, state.selected_team_id, rival.id, render=False, seed=123)
        self.assertEqual(request["own"]["players"], [asdict(scale_enemy_player(state.player(n), 0.8)) for n in OWN])
        multiplier = state.pair_metrics(rival.roster)[1] * 1.5
        self.assertEqual(request["opponent"]["players"], [asdict(scale_enemy_player(p, multiplier)) for p in rival.players[:5]])
        self.assertEqual(state.player("Leo"), character_stats.get_by_name("Leo"))

    def test_disabled_requests_are_identical_to_legacy_match_snapshots(self):
        from season_scrim import build_scrim_request
        state = self.state().advance_days(3)
        rival = state.opponent_teams[0]
        with patch.object(settings, "pair_familiarity_enabled", False):
            request = build_scrim_request(state, state.selected_team_id, rival.id, render=False, seed=123)
        self.assertEqual(request["own"]["players"], [asdict(state.player(n)) for n in OWN])
        self.assertEqual(request["opponent"]["players"], [asdict(state.enemy_player(p)) for p in rival.players[:5]])
        self.assertEqual(request["seed"], 123)
        own = state.selected_team
        own_players = [state.player(n) for n in own.roster]
        opponent_players = [state.enemy_player(p) for p in rival.players[:5]]
        legacy = {
            "own": {"name": state.team_name, "players": [asdict(p) for p in own_players], "ai": own.ai,
                    "igl": own.igl or max(own_players, key=lambda p: p.iq).name,
                    "spike_holder": own.carrier or own_players[0].name},
            "opponent": {"name": rival.name, "players": [asdict(p) for p in opponent_players], "ai": rival.ai,
                         "igl": rival.effective_igl, "spike_holder": rival.effective_carrier},
            "render": False, "initial_side": "A", "tick_time_ms": 100, "seed": 123,
        }
        self.assertEqual(json.dumps(request, sort_keys=True), json.dumps(legacy, sort_keys=True))

    def test_series_uses_registered_five_and_history_records_all_clubs(self):
        from season_series import build_series_request
        cup = dict(id="cup", name="Cup", start_date="2026-01-02", team_count=2,
                   format="single_elimination", prizes={}, normal_maps_to_win=1)
        with patch.object(calendar, "TOURNAMENTS", [cup]):
            state = self.state()
            state = state.with_tournament_entry("cup", state.selected_team_id)
        self.assertEqual(state.history[-1]["種別"], "大会参加登録")
        self.assertEqual(len(state.history[-1]["ペア練度"]), 2)
        state = state.advance_days()
        request = build_series_request(state, "cup", render=False)
        run = state.tournament("cup")
        for key, team_id in (("own", request["left_id"]), ("opponent", request["right_id"])):
            entrant = next(t for t in run.entrants if t.id == team_id)
            players = [state.player(p.name) for p in entrant.players] if team_id == state.club_id else entrant.players
            self.assertEqual(request[key]["players"], [asdict(p) for p in state.match_players(players, enemy=team_id != state.club_id)])
        with patch.object(settings, "pair_familiarity_enabled", False):
            disabled = build_series_request(state, "cup", render=False)
        self.assertEqual(disabled["seed"], request["seed"])
        for key, team_id in (("own", disabled["left_id"]), ("opponent", disabled["right_id"])):
            entrant = next(t for t in run.entrants if t.id == team_id)
            expected = [asdict(state.player(p.name) if team_id == state.club_id else state.enemy_player(p)) for p in entrant.players]
            self.assertEqual(disabled[key]["players"], expected)
        self.store.save(state)
        self.assertEqual(self.store.load_or_create(), state)

    def test_monthly_history_contains_final_daily_metrics(self):
        state = self.state().advance_days(31)
        entry = next(e for e in state.history if e["種別"] == "月次決算")
        own = next(r for r in entry["ペア練度"] if r["チームID"] == state.club_id)
        self.assertAlmostEqual(own["C"], familiarity(31))
        self.assertAlmostEqual(own["M"], state.pair_metrics(OWN)[1])
        self.store.save(state)
        export = json.loads(self.store.history_path.read_text(encoding="utf-8"))
        self.assertTrue(any("ペア練度" in e for e in export["履歴"]))

    def test_combo_conditions_and_effects_do_not_depend_on_familiarity_switch(self):
        from test_player_combo_rename import TestGame, PlayerComboRenameTest
        import combo_awakening
        combos = [{"name": "Independent combo", "players": ("Leo", "Boaster"),
                   "bonuses": {"accuracy": 0.05, "iq": 20}, "renames": {}}]
        outcomes = []
        for enabled in (True, False):
            game = TestGame([PlayerComboRenameTest._player(n) for n in OWN])
            with patch.object(settings, "pair_familiarity_enabled", enabled), patch.object(combo_awakening, "PLAYER_COMBOS", combos):
                game._apply_player_combos()
            outcomes.append((game.active_player_combos, [(p.accuracy, p.iq, p.active_combos) for p in game.chars]))
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertAlmostEqual(outcomes[0][1][0][0], 0.85)

    def ui(self, state):
        import tkinter as tk
        from run_realtime_season import RealtimeSeasonApp
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        app = RealtimeSeasonApp(root, self.store, state)
        return app

    def test_team_matrix_and_partner_ui_support_clicks_and_refresh(self):
        state = self.state(bench=True).advance_days(3)
        app = self.ui(state)
        own = app.show_team_pairs(state.club_id)
        rival = app.show_team_pairs(state.opponent_teams[0].id)
        self.assertEqual(len(own.pair_cells), 10)
        self.assertEqual(len(rival.pair_cells), 10)
        own.pair_cells[pair_key("Leo", "Boaster")].invoke()
        self.assertIn("通算3日", own.pair_detail.get())
        self.assertIn("練度", own.pair_detail.get())
        partners = app.show_player_pairs("Meiy")
        self.assertEqual(len(partners.partner_tree.get_children()), 5)
        self.assertTrue(all("●" in partners.partner_tree.item(i, "values")[3] for i in partners.partner_tree.get_children()))
        app.state = state.advance_days()
        app.refresh_pair_windows()
        own.pair_cells[pair_key("Leo", "Boaster")].invoke()
        self.assertIn("通算4日", own.pair_detail.get())

    def test_ranking_ui_limits_filters_and_reuses_cached_sort(self):
        state = self.state().advance_days()
        extras = [p.name for p in state.lft_players[:30]]
        state = replace(state, pair_days={**state.pair_days, **{pair_key("Leo", n): 100 + i for i, n in enumerate(extras)}})
        cached = state.pair_ranking
        app = self.ui(state)
        window = app.show_pair_ranking()
        self.assertEqual(len(window.ranking_tree.get_children()), 20)
        window.same_only.set(True)
        window.refresh_ranking()
        self.assertLessEqual(len(window.ranking_tree.get_children()), 20)
        self.assertTrue(all("同じチーム" in window.ranking_tree.item(i, "values")[2] for i in window.ranking_tree.get_children()))
        self.assertIs(state.pair_ranking, cached)

    def test_upgrade_news_daily_cap_player_priority_and_no_repeats(self):
        state = self.state()
        days = {pair_key(a, b): 91 for a, b in combinations(OWN, 2)}
        days[pair_key("Aspas", "Sato")] = 414
        state = replace(state, pair_days=days)
        advanced = state.advance_days()
        self.assertEqual(len(advanced.pair_news), 3)
        self.assertTrue(all(e.player_priority and e.stage == "相棒" for e in advanced.pair_news))
        self.assertEqual(advanced._record_history("同日再表示").pair_news, advanced.pair_news)
        self.assertEqual(advanced.advance_days().pair_news, advanced.pair_news)
        with patch.object(settings, "MAX_NEWS_PER_DAY", 20):
            complete = state.advance_days()
        self.assertEqual(len(complete.pair_news), 11)
        self.assertTrue(any(e.stage == "名コンビ" and not e.player_priority for e in complete.pair_news))

    def test_rival_intermediate_stage_is_silent_but_top_stage_is_announced(self):
        state = self.state()
        for old_days, count in ((91, 0), (188, 0), (414, 1)):
            with self.subTest(days=old_days):
                state = replace(state, pair_days={pair_key("Aspas", "Sato"): old_days})
                self.assertEqual(len(state.advance_days().pair_news), count)

    def test_separation_news_for_transfer_and_release_and_disabled_silence(self):
        state = self.state(bench=True)
        state = replace(state, pair_days={pair_key("Leo", "Meiy"): 100, pair_key("Aspas", "Sato"): 200})
        released = state.without_player("Meiy")
        self.assertEqual(len(released.pair_news), 1)
        self.assertEqual(released.pair_news[0].kind, "pair_separation")
        self.assertTrue(released.pair_news[0].player_priority)
        transferred = state.with_scouted_player("Sato", "year1")
        self.assertTrue(any(e.pair == pair_key("Aspas", "Sato") for e in transferred.pair_news))
        with patch.object(settings, "pair_familiarity_enabled", False):
            self.assertFalse(state.without_player("Meiy").pair_news)
            self.assertFalse(replace(state, pair_days={pair_key("Leo", "Meiy"): 91}).advance_days().pair_news)
        self.store.save(released)
        self.assertEqual(self.store.load_or_create().pair_news, released.pair_news)

    def test_monthly_separation_news_uses_final_day_and_news_ui_renders_it(self):
        state = self.state(bench=True)
        state = replace(state, pair_days={pair_key("Leo", "Meiy"): 100}, contracts=tuple(
            replace(c, kind="short", duration_months=6, team_loyalty=0) if c.player_name == "Meiy" else c
            for c in state.contracts))
        advanced = state.advance_days(31)
        event = next(e for e in advanced.pair_news if e.kind == "pair_separation")
        self.assertEqual(event.date, "2026-02-01")
        self.assertEqual(event.days, 130)
        app = self.ui(advanced)
        app.show_screen("monthly")
        self.assertTrue(app.monthly_table.exists(event.id))
        app.monthly_table.selection_set(event.id)
        app.preview_monthly_event()
        self.assertIn("別々", app.monthly_detail.get())

    def test_pair_contract_button_calls_existing_scout_twice_and_preserves_days(self):
        state = replace(self.state(), pair_days={pair_key("Aspas", "valyn"): 200})
        app = self.ui(state)
        window = app.show_player_pairs("Aspas")
        window.partner_tree.selection_set("valyn")
        app.root.update()
        self.assertIn("2,400,000", window.pair_contract_button.cget("text"))
        window.pair_contract_button.invoke()
        self.assertIsNotNone(app.state.player("Aspas"))
        self.assertIsNotNone(app.state.player("valyn"))
        self.assertEqual(app.state.money, state.money - 3_000_000)
        self.assertEqual(app.state.pair_days[pair_key("Aspas", "valyn")], 201)
        self.assertEqual(self.store.load_or_create(), app.state)

    def test_pair_contract_partial_success_keeps_first_saved_contract(self):
        state = replace(self.state(), money=3_000_000, pair_days={pair_key("Aspas", "valyn"): 200})
        app = self.ui(state)
        app.sign_player_pair("Aspas", "valyn")
        self.assertIsNotNone(app.state.player("Aspas"))
        self.assertIsNone(app.state.player("valyn"))
        self.assertEqual(app.state.money, 1_500_000)
        self.assertIn("Aspasのみ契約成立", app.status.get())
        self.assertEqual(self.store.load_or_create(), app.state)

    def test_malformed_saved_pairs_are_rejected_instead_of_reset(self):
        self.store.save(self.state())
        original = json.loads(self.store.path.read_text(encoding="utf-8"))
        for rows in ([["Leo", "Boaster", 2]], [["Boaster", "Leo", 0]],
                     [["Boaster", "Leo", 2], ["Boaster", "Leo", 3]], "invalid"):
            data = {**original, "pair_days": rows}
            self.store.path.write_text(json.dumps(data), encoding="utf-8")
            before = self.store.path.read_bytes()
            with self.assertRaises(SeasonSaveError):
                self.store.load_or_create()
            self.assertEqual(self.store.path.read_bytes(), before)
        missing = dict(original)
        missing.pop("pair_days")
        self.store.path.write_text(json.dumps(missing), encoding="utf-8")
        with self.assertRaises(SeasonSaveError):
            self.store.load_or_create()


if __name__ == "__main__":
    unittest.main()
