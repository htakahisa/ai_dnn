"""World rating boundaries, scaled match inputs and monthly sponsor payments."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

import character_stats
import realtime_season_competitions as calendar
import realtime_season_teams as teams
import realtime_season_world_levels as settings
from realtime_season import SeasonSaveError, SeasonStore, new_season, validate_player
from run_realtime_season import RealtimeSeasonApp
from season_scrim import build_scrim_request
from season_series import build_series_request
from season_competitions import SeriesScore, next_match
from season_world_levels import WorldLevelError, configured_world_levels, world_level_for_rating


OWN = ("Leo", "Boaster", "Derke", "Chronicle", "Alfajer")
RIVALS = (("Aspas", "valyn", "trent", "leaf", "tex"),
          ("Boostio", "Ethan", "jawgemo", "C0M", "Demon1"),
          ("F0rsakeN", "Jinggg", "d4v41", "something", "PatMen"))
LEVELS = [
    {"レベル": 1, "必要レート": 0, "敵倍率": 1, "スポンサー資金": 7_500_000},
    {"レベル": 2, "必要レート": 1800, "敵倍率": 1.5, "スポンサー資金": 10_000_000},
    {"レベル": 3, "必要レート": 1900, "敵倍率": 2, "スポンサー資金": 20_000_000},
]
CUP = dict(id="cup", name="Cup", start_date="2026-03-01", team_count=4, format="single_elimination",
           prizes={}, normal_maps_to_win=1)


class WorldLevelTests(unittest.TestCase):
    def setUp(self):
        fixture = {name: replace(p, monthly_salary=100_000, loyalty=5)
                   for name, p in character_stats.CHARACTER_TABLE.items()}
        contexts = [patch.dict(character_stats.CHARACTER_TABLE, fixture),
                    patch.object(teams, "SEASON_TEAMS", [dict(name=f"Rival{i}", players=list(names))
                                                        for i, names in enumerate(RIVALS)]),
                    patch.object(settings, "WORLD_LEVELS", LEVELS),
                    patch.object(calendar, "START_DATE", "2026-01-01"),
                    patch.object(calendar, "TOURNAMENTS", [CUP])]
        for context in contexts:
            context.start()
            self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.store = SeasonStore(Path(directory.name) / "save.json")

    def state(self, rank=4):
        state = new_season(OWN).with_roster(OWN).with_confirmed_team()
        state = replace(state.with_selected_team(state.teams[0].id), money=10_000_000)
        rivals = {c.id: 1900 - index * 100 for index, c in enumerate(state.opponent_teams)}
        state = replace(state, ratings=tuple(replace(r, value=(1950, 1890, 1750, 1650)[rank - 1] if r.team_id == state.club_id
                                                    else rivals[r.team_id]) for r in state.ratings))
        state.validate()
        return state

    def test_rating_boundaries_include_threshold_and_use_highest_eligible_level(self):
        for rating, level in ((0, 1), (1799.9999, 1), (1800, 2), (1899.9999, 2), (1900, 3), (3000, 3)):
            with self.subTest(rating=rating):
                self.assertEqual(world_level_for_rating(rating).level, level)
        for rating in (-1, True, "1800", float("nan"), float("inf")):
            with self.subTest(rating=rating), self.assertRaises(WorldLevelError):
                world_level_for_rating(rating)

    def test_world_level_does_not_depend_on_other_teams_or_their_ratings(self):
        state = self.state(2)
        self.assertEqual(state.world_level, 2)
        without = state.with_opponent_teams(())
        self.assertEqual(without.world_rank, (1, 1))
        self.assertEqual(without.world_level, 2)
        stronger = replace(state, ratings=tuple(replace(r, value=3000) if r.team_id != state.club_id else r for r in state.ratings))
        self.assertEqual(stronger.world_rank, (4, 4))
        self.assertEqual(stronger.world_level, 2)

    def test_match_results_raise_and_lower_world_level_without_a_latch(self):
        state = self.state(2)
        # Place the two clubs close enough for one result to exchange their ranks.
        rival_id = state.opponent_teams[0].id
        state = replace(state, ratings=tuple(replace(r, value=1860) if r.team_id == rival_id else r for r in state.ratings))
        won = state.with_rated_result("win", state.club_id, rival_id, 1, 0)
        self.assertEqual((state.world_level, won.world_level), (2, 3))
        lost = won.with_rated_result("loss", won.club_id, rival_id, 0, 1)
        self.assertEqual(lost.world_level, 2)
        self.assertEqual(lost.monthly_sponsor_income, 10_000_000)

    def test_ties_and_removed_teams_follow_the_visible_ranking(self):
        state = self.state()
        tied = replace(state, ratings=tuple(replace(r, value=1500) for r in state.ratings))
        expected = next(i for i, r in enumerate(tied.rating_ranking, 1) if r.team_id == tied.club_id)
        self.assertEqual(tied.world_rank, (expected, 4))
        removed = tied.with_opponent_teams(tied.opponent_teams[:1])
        self.assertEqual(removed.world_rank[1], 2)
        self.assertEqual(len(removed.ratings), 4)

    def test_monthly_sponsor_uses_current_level_once_and_respects_disabled_contract(self):
        for rank, funds in ((1, 20_000_000), (2, 10_000_000), (4, 7_500_000)):
            with self.subTest(rank=rank):
                state = self.state(rank)
                self.assertEqual(state.monthly_sponsor_income, funds)
                self.assertEqual(state.advance_days(30).money, state.money)
                month = state.advance_days(31)
                self.assertEqual(month.money, state.money + funds - state.monthly_payroll)
                self.assertEqual(month.advance_days().money, month.money)
                disabled = state.with_sponsor_contract(False)
                self.assertEqual(disabled.monthly_sponsor_income, 0)
                self.assertEqual(disabled.advance_months().money, state.money - state.monthly_payroll)

    def test_monthly_sponsor_switches_to_the_lower_level_after_rank_drops(self):
        state = self.state(1).advance_months()
        state = replace(state, ratings=tuple(replace(r, value=100) if r.team_id == state.club_id else r for r in state.ratings))
        self.assertEqual(state.world_level, 1)
        self.assertEqual(state.advance_months().money, state.money + 7_500_000 - state.monthly_payroll)

    def test_enemy_scaling_caps_probabilities_and_preserves_economy_and_self_players(self):
        state = self.state(1)
        base = state.opponent_teams[0].players[0]
        effective = state.enemy_player(base)
        self.assertEqual(effective.iq, base.iq * 2)
        self.assertEqual(effective.reaction, base.reaction * 2)
        self.assertEqual(effective.influence, base.influence * 2)
        self.assertEqual(effective.hit_pct, base.hit_pct * 2)
        for key in ("hs_pct", "dodge_pct"):
            self.assertEqual(getattr(effective, key), min(1, getattr(base, key) * 2))
        self.assertEqual(effective.mental, min(10, base.mental * 2))
        self.assertEqual((effective.loyalty, effective.monthly_salary, effective.form_variance),
                         (base.loyalty, base.monthly_salary, base.form_variance))
        validate_player(effective)
        self.assertEqual(state.displayed_player(state.owned_players[0]), state.owned_players[0])
        self.assertEqual(state.displayed_player(base), effective)
        self.assertEqual(state.transfer_fee(base.name), base.monthly_salary * 12)

    @patch("realtime_season_pair_familiarity.pair_familiarity_enabled", False)
    def test_scrim_and_series_scale_raw_snapshots_once_at_each_match(self):
        state = self.state(1)
        club = state.opponent_teams[0]
        first = build_scrim_request(state, state.teams[0].id, club.id, render=False)
        second = build_scrim_request(state, state.teams[0].id, club.id, render=False)
        expected = [asdict(state.enemy_player(p)) for p in club.players]
        self.assertEqual(first["opponent"]["players"], expected)
        self.assertEqual(second["opponent"]["players"], expected)
        self.assertEqual(first["own"]["players"], [asdict(state.player(n)) for n in OWN])

        entered = state.with_tournament_entry("cup", state.selected_team_id).advance_days(59)
        request = build_series_request(entered, "cup", render=False)
        snapshot = next(t for t in entered.tournament("cup").entrants if t.id == request["right_id"])
        self.assertEqual(request["opponent"]["players"], [asdict(state.enemy_player(p)) for p in snapshot.players])
        lower = replace(entered, ratings=tuple(replace(r, value=100) if r.team_id == state.club_id else r for r in entered.ratings))
        self.assertEqual(build_series_request(lower, "cup")["opponent"]["players"],
                         [asdict(entered.enemy_player(p)) for p in snapshot.players])
        self.assertEqual(lower.world_level, entered.world_level)
        self.assertEqual(lower.tournament("cup").entrants, entered.tournament("cup").entrants)

    def test_world_level_locks_at_start_instead_of_early_registration(self):
        state = self.state(4)
        state = state.with_tournament_entry("cup", state.selected_team_id)
        self.assertIsNone(state.world_level_lock)
        raised = replace(state, ratings=tuple(replace(r, value=2500) if r.team_id == state.club_id else r
                                              for r in state.ratings))
        self.assertEqual(raised.world_level, 3)
        started = raised.advance_days(59)
        self.assertEqual(started.game_date, "2026-03-01")
        self.assertEqual(started.world_level_lock.level, 3)
        lowered = replace(started, ratings=tuple(replace(r, value=100) if r.team_id == state.club_id else r
                                                 for r in started.ratings))
        self.assertEqual(lowered.world_rank, (4, 4))
        self.assertEqual(lowered.world_level, 3)
        self.assertEqual(lowered.monthly_sponsor_income, 20_000_000)
        self.assertEqual(lowered.enemy_player(lowered.opponent_teams[0].players[0]).iq,
                         lowered.opponent_teams[0].players[0].iq * 2)
        self.store.save(lowered)
        self.assertEqual(self.store.load_or_create(), lowered)

    def test_tournament_finish_unlocks_level_after_final_rating_update(self):
        state = self.state(2)
        state = state.with_tournament_entry("cup", state.selected_team_id).advance_days(59)
        state = replace(state, ratings=tuple(replace(r, value=2500) if r.team_id == state.club_id else r
                                             for r in state.ratings))
        self.assertEqual(state.world_level, 2)
        while not state.tournament("cup").completed:
            match, _ = next_match(state.tournament_definition("cup"), state.tournament("cup"))
            own = state.club_id
            score = SeriesScore(match.id, match.left, match.right, 0 if match.right == own else match.maps_to_win,
                                match.maps_to_win if match.right == own else 0)
            state = state.with_tournament_result("cup", score)
            if not state.tournament("cup").completed:
                self.assertEqual(state.world_level, 2)
                state = state.advance_days()
        self.assertIsNone(state.world_level_lock)
        self.assertEqual(state.world_level, 3)
        self.assertEqual(state.monthly_sponsor_income, 20_000_000)
        self.store.save(state)
        self.assertEqual(self.store.load_or_create(), state)

    def test_legacy_active_tournament_locks_current_level_without_rewriting_save(self):
        state = self.state(1)
        state = state.with_tournament_entry("cup", state.selected_team_id).advance_days(59)
        self.store.save(state)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 16
        data.pop("world_level_lock")
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.store.path.read_bytes()
        self.assertEqual(self.store.load_or_create(), state)
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_recruitment_and_restart_keep_base_stats_without_compounding(self):
        state = self.state(1)
        base = state.opponent_teams[0].players[0]
        recruited = state.with_scouted_player(base.name, "short", 1)
        grown = replace(base, iq=base.iq + .1)
        self.assertEqual(recruited.player(base.name), grown)
        self.assertEqual(recruited.displayed_player(recruited.player(base.name)), grown)
        self.store.save(state)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, state)
        self.assertEqual(loaded.opponent_teams[0].players[0], base)
        self.assertEqual(loaded.enemy_player(base), state.enemy_player(base))
        with patch.object(settings, "WORLD_LEVELS", [LEVELS[0], {**LEVELS[2], "敵倍率": 3, "スポンサー資金": 30_000_000}]):
            loaded = self.store.load_or_create()
            self.assertEqual(loaded.monthly_sponsor_income, 30_000_000)
            self.assertEqual(loaded.enemy_player(base).iq, base.iq * 3)
            self.assertEqual(loaded.opponent_teams[0].players[0], base)

    def test_old_percentile_locks_keep_their_effects_until_tournament_completion(self):
        state = self.state(1)
        state = state.with_tournament_entry("cup", state.selected_team_id)
        state = state.advance_days(59)
        self.store.save(state)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 28
        locks = [data["world_level_lock"], *(c["world_level_lock"] for c in data["opponent_teams"])]
        for lock in locks:
            lock.pop("required_rating")
            lock["top_percent"] = 25
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.store.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.world_level, state.world_level)
        self.assertEqual(loaded.monthly_sponsor_income, state.monthly_sponsor_income)
        self.assertEqual(loaded.enemy_player(state.opponent_teams[0].players[0]),
                         state.enemy_player(state.opponent_teams[0].players[0]))
        self.assertEqual([c.world_level_lock.enemy_multiplier for c in loaded.opponent_teams],
                         [c.world_level_lock.enemy_multiplier for c in state.opponent_teams])
        self.assertEqual(self.store.path.read_bytes(), before)
        self.store.save(loaded)
        self.assertEqual(self.store.load_or_create(), loaded)

    @patch("realtime_season_pair_familiarity.pair_familiarity_enabled", False)
    def test_enemy_world_multiplier_reaches_actual_battle_characters(self):
        import game_core
        from run_game import VisualFPSBattle
        from season_scrim_worker import _play_scrim

        state = self.state(1)
        request = build_scrim_request(state, state.selected_team_id, state.opponent_teams[0].id, render=False, seed=17)
        observed = {}

        def finish_match(game):
            observed.update({c.name: c.base_iq for c in game.chars})
            game.attacker_wins, game.defender_wins = 13, 0
            game.match_over = True

        with patch.dict(character_stats.CHARACTER_TABLE), patch.object(game_core, "_character_stats", character_stats), \
                patch.object(VisualFPSBattle, "run", finish_match):
            result = _play_scrim(request)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(observed["Aspas"], state.opponent_teams[0].players[0].iq * 2)
        self.assertEqual(observed["Leo"], state.player("Leo").iq)

    def test_invalid_config_fails_before_overwriting_save(self):
        state = self.state()
        self.store.save(state)
        before = self.store.path.read_bytes()
        bad_tables = [[], [{**LEVELS[0], "必要レート": 90}], [LEVELS[0], LEVELS[0]],
                      [{**LEVELS[0], "敵倍率": float("nan")}], [{**LEVELS[0], "スポンサー資金": -1}],
                      [{**LEVELS[0], "スポンサー資金": 3.5}], [{**LEVELS[0], "レベル": True}],
                      [{**LEVELS[0], "必要レート": -1}], [{**LEVELS[0], "敵倍率": 0}],
                      [LEVELS[0], {**LEVELS[1], "スポンサー資金": 1}],
                      [LEVELS[0], {**LEVELS[1], "敵倍率": .5}], [{"レベル": 1}]]
        for table in bad_tables:
            with self.subTest(table=table), patch.object(settings, "WORLD_LEVELS", table):
                with self.assertRaises(WorldLevelError):
                    configured_world_levels()
                with self.assertRaisesRegex(SeasonSaveError, "世界レベル"):
                    self.store.save(state)
            self.assertEqual(self.store.path.read_bytes(), before)

    def test_scout_shows_and_sorts_base_stats_while_match_preparation_keeps_world_bonus(self):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        state = self.state(1)
        app = RealtimeSeasonApp(root, self.store, state)
        self.assertIn("世界レベル: 3", app.home_summary.get())
        self.assertIn("20,000,000", app.home_summary.get())
        app.show_screen("ratings")
        self.assertIn("世界レベル3", app.sponsor_summary.get())
        self.assertIn("必要レート1900以上", app.sponsor_summary.get())
        app.opponent_choice.set("Rival0")
        app.preview_preparation()
        self.assertEqual(float(app.opponent_roster.item(app.opponent_roster.get_children()[0], "values")[-1]),
                         state.opponent_teams[0].players[0].iq * 2)
        app.show_screen("scout")
        app.scout_filter.set("全選手")
        base = state.opponent_teams[0].players[0]
        self.assertEqual(float(app.scout_players.item(base.name, "values")[2]), base.iq)
        from season_player_stats import player_combat_power, player_duel_power
        self.assertEqual(app.scout_players.item(base.name, "values")[3:5],
                         (f"{player_combat_power(base):.2f}", f"{player_duel_power(base):.2f}"))
        higher_iq_lft = next(p for p in state.lft_players if base.iq < p.iq < base.iq * 2
                            and not app.offer_conditions("scout", p)[2])
        app.scout_sort.set("IQ")
        app.scout_sort_order.set("高い順")
        order = app.scout_players.get_children()
        self.assertLess(order.index(higher_iq_lft.name), order.index(base.name))
        app.scout_players.selection_set(base.name)
        app.refresh_offer("scout")
        self.assertIn(f"IQ: {base.iq:g} / 影響力: {base.influence:g}", app.scout_details.get())
        self.assertIn(f"HS率: {base.hs_pct:.1%} / 命中率: {base.hit_pct:.1%}", app.scout_details.get())
        scout_stats = app.scout_players.item(base.name, "values")[2:5]
        lower = replace(state, ratings=tuple(replace(r, value=100) if r.team_id == state.club_id else r for r in state.ratings))
        app.commit(lower, "順位変更")
        self.assertIn("世界レベルが3から1", app.status.get())
        self.assertIn("世界レベル: 1", app.home_summary.get())
        self.assertEqual(app.scout_players.item(base.name, "values")[2:5], scout_stats)
        self.assertEqual(self.store.load_or_create(), app.state)


if __name__ == "__main__":
    unittest.main()
