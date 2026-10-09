"""Attacker analysis: public information, legal detours, learning and loading."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import deque
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
import torch

from game_core import Character, PLANT_REQUIRED_TICKS
from frc_v1.perception import FrcPerceptionBuilder
from toruAI_v4.tv4_scenario import Scenario
from toruAI_v4.tv4_collect_attacker_analysis import AnalysisCollectorController
from toruAI_v4.tv4_learn_attacker_analysis import (
    AttackerEncoder, AttackerAnalysisModel, Route, candidate_routes, branch_sequence, shortest_path, load_analysis,
    optimize, TIME_MARGIN_TICKS,
)
from toruAI_v4.tv4_train_attacker_analysis import save_latest, prediction_metrics, best_rank, summarize, format_summary


def world():
    class Game(Scenario):
        def _smoke_cells(self):
            return set()

        def _ramp_blocks_movement(self, char):
            return False
    game = Game()
    cells = [tuple(map(int, p)) for p in np.argwhere(game.grid == 3)]
    game.chars = [Character(f"ally_{i}", "A", pos, "white", "red", has_spike=i == 2)
                  for i, pos in enumerate(cells)]
    game.chars += [Character(f"enemy_{i}", "D", post.watch, "white", "green")
                   for i, post in enumerate(game.posts)]
    game.current_round, game.battle_tick = 1, 0
    game.defender_setup_phase = SimpleNamespace(active=False)
    game.is_planted = False
    game.spike_pos = game.planted_pos = None
    game.round_timer, game.detonate_timer = 100, 55
    return game


class AttackerAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.scenario = Scenario()

    def model(self, encoder):
        return AttackerAnalysisModel(len(encoder.fields), len(encoder.route_fields), len(self.scenario.names))

    def test_routes_use_real_edges_no_repeated_cells_and_budget(self):
        origin = tuple(map(int, np.argwhere(self.scenario.grid == 3)[2]))
        routes = candidate_routes(self.scenario, origin, 100)
        self.assertEqual({r.site for r in routes}, {"L", "R"})
        self.assertGreater(len(routes), 2)
        for route in routes:
            self.assertEqual(len(route.cells), len(set(route.cells)))
            self.assertIn(route.cells[-1], self.scenario.sites[route.site])
            for a, b in zip(route.cells, route.cells[1:]):
                self.assertIn(b, set(self.scenario.neighbors(a)))
            self.assertLessEqual(len(route.cells) - 1 + PLANT_REQUIRED_TICKS + TIME_MARGIN_TICKS, 100)
        self.assertEqual(candidate_routes(self.scenario, origin, 3), [])
        self.assertTrue(any(len(r.cells) > len(shortest_path(self.scenario, origin,
                            self.scenario.sites[r.site])) for r in routes))

    def test_hidden_enemy_state_cannot_change_features_predictions_or_actions(self):
        game = world()
        encoder = AttackerEncoder(self.scenario)
        model = self.model(encoder)
        first = AnalysisCollectorController(self.scenario, model, np.random.default_rng(3))
        first.set_game(game)
        first.prepare_team_tick()
        x = first.frames[0]["observation"].copy()
        z = first.frames[0]["route_features"].copy()
        for i, enemy in enumerate(game.chars[5:]):
            enemy.pos = [2, 6 + i]
            enemy.has_spike = i == 0
        game.target_plant_pos = (7, 40)
        second = AnalysisCollectorController(self.scenario, model, np.random.default_rng(3))
        second.set_game(game)
        second.prepare_team_tick()
        np.testing.assert_array_equal(x, second.frames[0]["observation"])
        np.testing.assert_array_equal(z, second.frames[0]["route_features"])
        self.assertEqual(first.actions, second.actions)
        snapshot = second.snapshot
        prediction = model.analyze(second.encoder, snapshot, x)
        self.assertTrue(prediction["candidates"])
        for row in prediction["placement"].values():
            self.assertAlmostEqual(sum(row.values()), 1., places=5)

    def test_failure_rounds_train_without_fabricated_plant_time(self):
        game = world()
        encoder = AttackerEncoder(self.scenario)
        snapshot = FrcPerceptionBuilder("A").build(game)
        observation = encoder.observe(snapshot, [])
        route = candidate_routes(self.scenario, snapshot.allies[2].position, 100)[0]
        features = encoder.route_features(snapshot, route, "preserve")
        sample = {"observations": np.stack([observation]), "routes": np.stack([features]),
                  "placements": np.zeros((1, 5), dtype=np.int64),
                  "outcomes": np.asarray([[0., .5, .2, 0., 0.]], dtype=np.float32), "round": 1}
        model = self.model(encoder)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        before = model.outcome[-1].weight.detach().clone()
        loss = optimize(model, optimizer, [sample], np.random.default_rng(2), 2, 4)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, model.outcome[-1].weight))
        self.assertIsNone(prediction_metrics(model, [sample])["plant_ticks_mae"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest.pt"
            model.trained_rounds = 12
            save_latest(path, model, optimizer, deque([sample]), 1, "fnatic_v3", encoder.schema(), {}, np.random.default_rng(2))
            loaded, _, state = load_analysis(path, self.scenario, "fnatic_v3")
            self.assertEqual(loaded.trained_rounds, 12)
            self.assertEqual(state["completed_sets"], 1)
            with self.assertRaises(ValueError):
                load_analysis(path, self.scenario, "gc_v1")

    def test_guard_is_outside_controller(self):
        game = world()
        game.is_planted = True
        encoder = AttackerEncoder(self.scenario)
        controller = AnalysisCollectorController(self.scenario, self.model(encoder), np.random.default_rng(0))
        controller.set_game(game)
        controller.prepare_team_tick()
        self.assertEqual(controller.frames, [])
        self.assertEqual(controller.actions, {})

    def test_remaining_route_starts_at_current_holder_position(self):
        game = world()
        encoder = AttackerEncoder(self.scenario)
        controller = AnalysisCollectorController(self.scenario, self.model(encoder), np.random.default_rng(0))
        controller.set_game(game)
        controller.prepare_team_tick()
        holder = next(a for a in controller.snapshot.allies if a.has_spike)
        from dataclasses import replace
        moved = replace(holder, position=controller.route.cells[3])
        remaining = controller._remaining(moved)
        self.assertEqual(remaining.cells[0], moved.position)
        self.assertEqual(controller.cursors[holder.slot], 3)

    def run_synthetic_traffic(self, side, positions, route_cells, *, ticks=30, mode="preserve"):
        """Only apply nonconflicting allied moves/plant progress; no real game."""
        game = world()
        for i, char in enumerate(game.chars[:5]):
            char.pos = list(positions[i])
            char.has_spike = i == 0
        encoder = AttackerEncoder(self.scenario)
        controller = AnalysisCollectorController(self.scenario, self.model(encoder), np.random.default_rng(0))
        controller.set_game(game)
        controller.route = Route(side, tuple(route_cells), branch_sequence(self.scenario, route_cells))
        controller.mode = mode
        controller.combat_enabled = False  # Isolate movement/traffic from combat examples.
        # These tests isolate traffic on an explicitly supplied path. Dynamic
        # route selection is tested independently with changing observations.
        from toruAI_v4.tv4_attacker_route_planner import AttackPlan
        controller.route_planner.update = lambda *args, **kwargs: AttackPlan(
            controller.route, controller.mode, "entry", False, "traffic_fixture", controller.route.cells[-1], {})
        progress, planted = 0, False
        paths = {char.name: [tuple(char.pos)] for char in game.chars[:5]}
        for _ in range(ticks):
            controller.prepare_team_tick()
            occupied = {tuple(c.pos) for c in game.chars[:5]}
            destinations = []
            for char in game.chars[:5]:
                destination, action = controller.actions[char.name]
                destination = tuple(destination)
                if destination != tuple(char.pos):
                    self.assertNotIn(destination, occupied)
                    self.assertIn(destination, set(self.scenario.neighbors(tuple(char.pos))))
                destinations.append(destination)
                if char.has_spike and action == "PLANT":
                    self.assertEqual(self.scenario.grid[tuple(char.pos)], 2)
                    progress += 1
            self.assertEqual(len(set(destinations)), 5)
            for char, destination in zip(game.chars[:5], destinations):
                char.pos = list(destination)
                paths[char.name].append(destination)
            if progress >= PLANT_REQUIRED_TICKS:
                planted = True
                break
            game.battle_tick += 1
            game.round_timer -= 1
        return controller, planted, paths

    def test_left_site_queue_from_saved_timeout_clears_and_plants(self):
        # Observed omoko: carrier at (10,3), escorts at the entrance and behind.
        positions = [(10, 3), (9, 3), (11, 3), (8, 3), (12, 3)]
        cells = [(12, 3), (11, 3), (10, 3), (9, 3)]
        for mode in ("preserve", "supported"):
            controller, planted, paths = self.run_synthetic_traffic("L", positions, cells, mode=mode)
            self.assertTrue(planted)
            self.assertTrue(any(e["type"] == "site_clearance" and e["from"] == (8, 3) for e in controller.events))
            for char_name in ("ally_1", "ally_3"):
                # Neither the former entrance blocker nor a parked ally re-enters.
                entered = False
                for pos in paths[char_name]:
                    if pos not in cells:
                        entered = True
                    if entered:
                        self.assertNotIn(pos, cells)

    def test_right_site_queue_from_saved_timeout_clears_and_plants(self):
        positions = [(12, 40), (11, 40), (10, 40), (10, 41), (9, 40)]
        cells = [(12, 40), (11, 40), (10, 40)]
        controller, planted, _ = self.run_synthetic_traffic("R", positions, cells)
        self.assertTrue(planted)
        self.assertGreater(len(controller.events), 0)

    def test_site_clearance_never_uses_route_cells_or_colliding_destinations(self):
        controller = AnalysisCollectorController(self.scenario, self.model(AttackerEncoder(self.scenario)), np.random.default_rng(0))
        controller.route = Route("L", ((10, 3), (9, 3)), ())
        self.assertEqual(controller._site_clearance_step((9, 3), {(8, 3)}), (9, 3))
        first = controller._site_clearance_step((8, 3), set())
        self.assertNotIn(first, controller.route.cells)
        self.assertGreater(controller.site_depth[first], controller.site_depth[(8, 3)])
        self.assertEqual(controller._site_clearance_step((8, 3), {(7, 3), (8, 4)}), (8, 3))

    def test_unopposed_five_players_can_plant_on_every_candidate_route(self):
        positions = [tuple(char.pos) for char in world().chars[:5]]
        for route in candidate_routes(self.scenario, positions[0], 100):
            with self.subTest(site=route.site, route=route.key):
                _, planted, _ = self.run_synthetic_traffic(route.site, positions, route.cells, ticks=100)
                self.assertTrue(planted)

    def test_old_executor_checkpoint_cannot_silently_reuse_stalled_labels(self):
        encoder = AttackerEncoder(self.scenario)
        schema = {**encoder.schema(), "executor": "fixed_route_public_utility_v1"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old_best.pt"
            torch.save({"schema": schema, "model": self.model(encoder).state_dict(),
                        "opponent": "omoko_v1", "trained_rounds": 720}, path)
            with self.assertRaisesRegex(ValueError, "executor"):
                load_analysis(path, self.scenario, "omoko_v1")

    def test_no_route_still_returns_placement_and_respects_public_facts(self):
        from dataclasses import replace
        from frc_v1.perception import Sighting
        game = world()
        encoder = AttackerEncoder(self.scenario)
        snapshot = FrcPerceptionBuilder("A").build(game)
        enemies = list(snapshot.enemies)
        enemies[1] = replace(enemies[1], alive=False)
        pos = self.scenario.branches["e"][0]
        snapshot = replace(snapshot, round_timer=1, enemies=tuple(enemies), sightings=(Sighting(0, pos, "normal"),))
        observation = encoder.observe(snapshot, [])
        result = self.model(encoder).analyze(encoder, snapshot, observation)
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["placement"][0][self.scenario.names[int(self.scenario.region[pos])]], 1.)
        self.assertEqual(result["placement"][1]["dead"], 1.)
        self.assertEqual(result["placement"][2]["dead"], 0.)

    def test_best_prefers_success_before_resource_savings(self):
        base = {"selected": {"plant_rate": .5, "mean_alive": 4., "mean_damage": 50.,
                             "mean_ability_uses": 0., "mean_plant_ticks": 45.},
                "prediction": {"brier": .2}}
        better = {"selected": {**base["selected"], "plant_rate": .75, "mean_ability_uses": 4.},
                  "prediction": base["prediction"]}
        self.assertGreater(best_rank(better), best_rank(base))

    def test_log_rates_pool_counts_and_exclude_dead_players_unused_charges(self):
        rounds = [
            {"planted": True, "alive": 4, "enemy_alive": 3, "initial_alive": 5, "initial_enemy_alive": 5,
             "initial_abilities": 10, "remaining_abilities_alive": 4, "uses": 4, "damage": 100, "tick": 40, "reason": "planted"},
            {"planted": False, "alive": 0, "enemy_alive": 2, "initial_alive": 5, "initial_enemy_alive": 5,
             "initial_abilities": 5, "remaining_abilities_alive": 0, "uses": 1, "damage": 500, "tick": 100, "reason": "timeout"},
        ]
        result = summarize(rounds)
        self.assertEqual(result["plant_rate"], .5)
        self.assertEqual(result["ally_survival_rate"], .4)
        self.assertEqual(result["enemy_survival_rate"], .5)
        self.assertAlmostEqual(result["ability_use_rate"], 5 / 15)
        self.assertAlmostEqual(result["ability_reserve_rate"], 4 / 15)
        self.assertIn("敵生存率=50.0%", format_summary(result))
        rounds[0].update(initial_abilities=0, remaining_abilities_alive=0, uses=0)
        rounds[1].update(initial_abilities=0, remaining_abilities_alive=0, uses=0)
        result = summarize(rounds)
        self.assertIsNone(result["ability_use_rate"])
        self.assertIsNone(result["ability_reserve_rate"])
        self.assertIn("温存率=対象なし", format_summary(result))


if __name__ == "__main__":
    unittest.main()
