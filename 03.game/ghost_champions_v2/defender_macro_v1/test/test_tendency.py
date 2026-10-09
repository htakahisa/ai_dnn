from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as NS
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ghost_champions_v2.defender_macro_v1.analysis import AttackSiteAnalysis, Scenario, SiteModel, GCFeatureHistory
from ghost_champions_v2.defender_macro_v1.controller import GCDefenderMacro
from ghost_champions_v2.defender_macro_v1.tendency import summarize_tendency, allocation_for, scale_allocation
from ghost_champions_v2.defender_macro_v1.test.test_defender_macro import unit


class TendencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.scenario = Scenario()

    def analysis(self):
        return AttackSiteAnalysis(scenario=self.scenario, model=SiteModel(len(GCFeatureHistory(self.scenario).fields)))

    def test_one_round_does_not_commit_and_unknown_rounds_do_not_invent_attacks(self):
        for rows in ([dict(site="L")], [dict(site=None, contacts={})]*12):
            t = summarize_tendency(rows)
            self.assertEqual(t["allocation"], dict(A=2, Mid=1, B=2))
            self.assertFalse(t["biased"])

    def setup_plan(self, a):
        macro = GCDefenderMacro(a)
        baseline = dict(Xdll=(6, 6), SyouTa=(6, 12), Absol=(6, 17), eKo=(2, 35), SugarZ3ro=(7, 31))
        allies = [unit(n, (1, 18+i)) for i, n in enumerate(baseline)]
        state = dict(grid=self.scenario.grid, chars=allies, defender_setup_active=True,
                     defender_setup_ticks_remaining=20)
        planner = NS(round_initialized=True, assignments=dict(baseline))
        macro.configure_setup(planner, allies[0], state)
        return macro, planner, baseline, allies, state

    def test_repeated_A_attacks_change_one_learned_post_and_retain_Mid(self):
        a = self.analysis()
        a.previous_rounds = [dict(site="L", rush_site="A") for _ in range(4)]
        macro, planner, baseline, allies, state = self.setup_plan(a)
        self.assertEqual(macro.allocation, dict(A=3, Mid=1, B=1))
        self.assertEqual(list(macro.roles.values()).count("guard_A"), 3)
        self.assertEqual(macro.allocation_source, "round_history")
        self.assertEqual(sum(baseline[n] != planner.assignments[n] for n in baseline), 1)
        self.assertEqual(planner.assignments["Absol"], baseline["Absol"])
        self.assertEqual(a.frames, [])
        for goal in macro.goals.values():
            self.assertEqual(self.scenario.setup[goal], 0)

    def test_balanced_history_keeps_every_learned_post(self):
        a = self.analysis()
        a.previous_rounds = [dict(site=s) for s in ("L", "R")*3]
        macro, planner, baseline, *_ = self.setup_plan(a)
        self.assertEqual(planner.assignments, baseline)
        self.assertIsNone(macro.reinforcement)
        self.assertEqual(macro.allocation_source, "existing_gc")

    def test_small_historical_bias_does_not_trigger_a_cross_map_transfer(self):
        a = self.analysis()
        a.previous_rounds = [dict(site=s) for s in ("L", "R", "L", "L")]
        macro, planner, baseline, *_ = self.setup_plan(a)
        self.assertEqual(planner.assignments, baseline)
        self.assertIsNone(macro.reinforcement)

    def test_search_changes_at_most_the_reinforcement_and_retains_lower_actions(self):
        a = self.analysis()
        a.previous_rounds = [dict(site="R")]*6
        macro, _, baseline, allies, state = self.setup_plan(a)
        # Distinct GC marker assignments, including Mid and both sites.
        before = dict(zip(baseline, [(6, 6), (13, 6), (6, 25), (4, 41), (10, 38)]))
        search = NS(_assigned_positions={n: [p] for n, p in before.items()},
                    _assigned_dist_maps={}, _assigned_markers={},
                    _ensure_defense_assignment=lambda *args: None)
        macro.configure_search(search, allies[0], state)
        changed = [n for n in baseline if search._assigned_positions[n] != [before[n]]]
        self.assertEqual(changed, [macro.reinforcement])
        self.assertEqual(macro.posts.side(search._assigned_positions[macro.reinforcement][0]), "B")
        # A later contact or model reversal never rebuilds the squad.
        a.current_contacts["A"] = 3
        a.gate.side = "A"
        after = dict(search._assigned_positions)
        macro.configure_search(search, allies[0], state)
        self.assertEqual(search._assigned_positions, after)
        sentinel = ([1, 19], {"facing": "SW"})
        self.assertIs(macro.coordinate(allies[0], state, sentinel), sentinel)

    def test_live_patrol_redraw_cannot_remove_Mid_or_reverse_historical_allocation(self):
        a = self.analysis()
        a.previous_rounds = [dict(site="R")]*6
        macro, _, baseline, allies, state = self.setup_plan(a)
        before = dict(zip(baseline, [(6, 1), (4, 41), (14, 3), (17, 36), (6, 6)]))
        search = NS(_assigned_positions={n: [p] for n, p in before.items()},
                    _assigned_dist_maps={}, _assigned_markers={},
                    _ensure_defense_assignment=lambda *args: None)
        macro.configure_search(search, allies[0], state)
        counts = {s: sum(macro.posts.side(points[0]) == s for points in search._assigned_positions.values())
                  for s in ("A", "Mid", "B")}
        self.assertEqual(counts, macro.allocation)
        for name, goal in before.items():
            if macro.posts.side(goal) == macro.posts.side(macro.goals[name]):
                self.assertEqual(search._assigned_positions[name], [goal])

    def test_recent_B_rounds_reverse_old_A_tendency(self):
        rows = [dict(site="L", rush_site="A")]*4 + [dict(site="R", rush_site="B")]*8
        t = summarize_tendency(rows)
        self.assertEqual(t["preferred_site"], "B")
        self.assertGreater(t["allocation"]["B"], t["allocation"]["A"])

    def test_fresh_B_concentration_overrides_A_history(self):
        t = summarize_tendency([dict(site="L", rush_site="A")]*6)
        allocation, source, side = allocation_for(t, dict(A=0, B=3), "A")
        self.assertEqual(allocation, dict(A=1, Mid=1, B=3))
        self.assertEqual((source, side), ("current_contacts", "B"))

    def test_unobserved_neural_confidence_does_not_override_round_history(self):
        t = summarize_tendency([dict(site="L")]*4)
        allocation, source, _ = allocation_for(t, {}, "B")
        self.assertEqual(allocation["A"], 3)
        self.assertEqual(source, "round_history")
        allocation, source, _ = allocation_for(summarize_tendency([]), {}, "B")
        self.assertEqual(allocation, dict(A=2, Mid=1, B=2))
        self.assertEqual(source, "balanced")

    def test_rush_observations_survive_round_reset_without_future_plant_label(self):
        a = self.analysis()
        ally = unit("own", (8, 4))
        enemies = [unit(str(i), (10+i, 3), "A") for i in range(3)]
        state = dict(grid=self.scenario.grid, chars=[ally]+enemies, battle_tick=10, round_timer=90, is_planted=False)
        a.observe(ally, state, 1)
        self.assertIsNone(a.rush_ticks["A"])
        state["battle_tick"] = 11
        a.observe(ally, state, 1)
        self.assertEqual(a.rush_ticks["A"], 11)
        a.finish_round(winner="D", no_plant=True)
        a.reset_round()
        self.assertEqual(a.previous_rounds[-1]["rush_site"], "A")
        self.assertIsNone(a.previous_rounds[-1]["site"])
        self.assertTrue(a.previous_rounds[-1]["enemy_sightings"])
        self.assertEqual(a.history.tracks, {})
        self.assertEqual(a.current_contacts, dict(A=0, Mid=0, B=0))

    def test_setup_plan_is_invariant_to_hidden_enemy_positions(self):
        a = self.analysis()
        a.previous_rounds = [dict(site="L")]*4
        first, _, _, allies, state = self.setup_plan(a)
        enemy = unit("enemy", (8, 3), "A", False)
        state["chars"].append(enemy)
        baseline = dict(first.baseline_goals)
        second = GCDefenderMacro(a)
        enemy.pos = [7, 40]
        planner = NS(round_initialized=True, assignments=baseline)
        second.configure_setup(planner, allies[0], state)
        self.assertEqual(first.goals, second.goals)

    def test_alive_allocation_preserves_total_and_both_sites(self):
        for count in range(6):
            allocation = scale_allocation(dict(A=4, B=1, Mid=0), count)
            self.assertEqual(sum(allocation.values()), count)
            if count >= 2:
                self.assertGreaterEqual(min(allocation["A"], allocation["B"]), 1)

    def test_unknown_plant_location_does_not_become_a_no_plant_or_A_attack(self):
        a = self.analysis()
        ally = unit("own", (8, 4))
        state = dict(grid=self.scenario.grid, chars=[ally], battle_tick=30, round_timer=70,
                     is_planted=True, planted_pos=None)
        a.observe(ally, state, 1)
        a.finish_round(no_plant=True)
        self.assertIsNone(a.previous_rounds[-1]["site"])
        self.assertFalse(a.previous_rounds[-1]["no_plant"])
        self.assertEqual(summarize_tendency(a.previous_rounds)["known_rounds"], 0)

    def test_stale_or_dead_contacts_do_not_keep_overriding_history(self):
        a = self.analysis()
        ally = unit("own", (8, 4))
        enemy = unit("seen", (8, 3), "A")
        state = dict(grid=self.scenario.grid, chars=[ally, enemy], battle_tick=1, round_timer=99, is_planted=False)
        a.observe(ally, state, 1)
        self.assertEqual(a.current_contacts["A"], 1)
        enemy.position_known = False
        state["battle_tick"] = 10
        a.observe(ally, state, 1)
        self.assertEqual(a.current_contacts["A"], 0)
        enemy.position_known = True
        enemy.is_alive = False
        state["battle_tick"] = 11
        a.observe(ally, state, 1)
        self.assertEqual(a.current_contacts["A"], 0)

    def test_model_selection_rewards_defense_results_over_prediction_accuracy(self):
        from ghost_champions_v2.defender_macro_v1.train_defender_analysis_gc import rank
        weak = dict(defender_win_rate=.2, correct_all_plants=1., accuracy=1.,
                    opponents={"GG": dict(defender_win_rate=0.), "OMG": dict(defender_win_rate=.4)})
        stronger = dict(defender_win_rate=.5, correct_all_plants=.5, accuracy=.6,
                        opponents={"GG": dict(defender_win_rate=.4), "OMG": dict(defender_win_rate=.6)})
        self.assertGreater(rank(stronger), rank(weak))

    def _real_setup_for_history(self, side):
        import contextlib
        import io
        import tempfile
        from ghost_champions_v2.defender_macro_v1.analysis import schema
        from ghost_champions_v2.defender_macro_v1.tools.watch_defender_gc import build_match
        from simulation_runtime import cpu_inference
        a = self.analysis()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "analysis.pt"
            torch.save(dict(schema=schema(self.scenario), model=a.model.state_dict(), labeled_rounds=2), checkpoint)
            with contextlib.redirect_stdout(io.StringIO()), cpu_inference():
                game, defender = build_match("GG", checkpoint=checkpoint, seed=127, headless=True)
                defender.macro.analysis.previous_rounds = [dict(site="L" if side == "A" else "R", rush_site=side, winner=None)]*4
                for _ in range(100):
                    if not game.defender_setup_phase.active:
                        break
                    game.step_tick()
                self.assertFalse(game.defender_setup_phase.active)
        macro = defender.macro
        other = "B" if side == "A" else "A"
        self.assertEqual(macro.allocation, {side: 3, other: 1, "Mid": 1})
        self.assertEqual(macro.allocation_source, "round_history")
        self.assertEqual(macro.analysis.frames, [])
        actual = {s: sum(macro.posts.side(tuple(c.pos)) == s for c in game.chars if c.team == "D")
                  for s in ("A", "Mid", "B")}
        self.assertEqual(actual, macro.allocation)
        self.assertEqual(sum(macro.goals[n] != macro.baseline_goals[n] for n in macro.goals), 1)
        with contextlib.redirect_stdout(io.StringIO()), cpu_inference():
            game.step_tick()
        search_counts = {s: sum(macro.posts.side(points[0]) == s for points in defender.search._assigned_positions.values())
                         for s in ("A", "Mid", "B")}
        self.assertEqual(search_counts, macro.allocation)
        self.assertEqual(macro.search_goals, {n: tuple(points[0]) for n, points in defender.search._assigned_positions.items()})

    def test_real_engine_applies_A_history_during_setup_before_any_live_observation(self):
        self._real_setup_for_history("A")

    def test_real_engine_applies_B_history_and_retains_Mid_and_opposite_guard(self):
        self._real_setup_for_history("B")


if __name__ == "__main__":
    unittest.main()
