"""Aggregation/resume and fixed-route checks; no training or full evaluation."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ov1_evaluate_attacker_carry_routes as evaluation


class CarryEvaluationTests(unittest.TestCase):
    def test_defaults_and_updated_routes(self):
        self.assertEqual(set(evaluation.OPPONENTS), {
            "toru_ai_v4", "concon_v1", "touyama_gaming_v3", "fnatic_v3",
            "frc_v1", "ghost_champions_v2",
        })
        self.assertEqual(evaluation.ROUNDS_PER_SET, 12)
        self.assertEqual(evaluation.SETS_PER_ROUTE, 8)
        self.assertEqual(evaluation.ROUNDS_PER_ROUTE, 96)
        self.assertEqual(evaluation.PARALLEL_WORKERS, 6)
        routes = evaluation.read_routes()
        self.assertEqual(len(routes), len({r["id"] for r in routes}))
        self.assertTrue(all(r["id"] == evaluation.route_id(r["site"], r["waypoint"]) for r in routes))

    def test_fixed_factory_resolves_coordinate_instead_of_saved_index(self):
        route = {"site": "left", "waypoint": [9, 15]}
        carry = SimpleNamespace(waypoint_cells_by_site={"left": [(1, 1), (4, 11), (9, 15)]})
        factory = evaluation.fixed_attacker_factory(lambda: SimpleNamespace(carry_controller=carry), route)
        self.assertEqual(factory().carry_controller._choose_route(1), ("left", 2))
        self.assertEqual(carry._choose_route(20), ("left", 2))

    def test_resume_never_duplicates_trials_and_replaces_changed_scenario(self):
        routes = [{"id": "left:1,2", "site": "left", "waypoint": [1, 2]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            result = {"seed": 0, "rounds": [{"won": True, "planted": True} for _ in range(12)]}
            with patch.object(evaluation, "read_routes", return_value=routes), \
                    patch.object(evaluation, "scenario_manifest", return_value={"fingerprint": "same"}), \
                    patch.object(evaluation, "run_set", return_value=result) as run, \
                    patch.object(evaluation, "LOG_PATH", Path(directory) / "run.log"), \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluation.evaluate(["fnatic_v3"], rounds=24, output=path)
                evaluation.evaluate(["fnatic_v3"], rounds=36, output=path)
                self.assertEqual(run.call_count, 3)
                self.assertEqual([call.args[-1] for call in run.call_args_list],
                                 [evaluation.trial_seed(evaluation.BASE_SEED, "fnatic_v3", i) for i in range(3)])
                report = json.loads(path.read_text(encoding="utf-8"))
                row = report["opponents"]["fnatic_v3"]["routes"]["left:1,2"]
                self.assertEqual((row["sets_completed"], row["rounds"], row["wins"], row["win_rate"]),
                                 (3, 36, 36, 1.0))
                with patch.object(evaluation, "scenario_manifest", return_value={"fingerprint": "changed"}):
                    evaluation.evaluate(["fnatic_v3"], rounds=36, output=path)
                self.assertEqual(run.call_count, 6)
                replaced = json.loads(path.read_text(encoding="utf-8"))
                row = replaced["opponents"]["fnatic_v3"]["routes"]["left:1,2"]
                self.assertEqual((row["sets_completed"], row["rounds"]), (3, 36))

    def test_missing_model_is_error_not_zero_win_rate(self):
        routes = [{"id": "right:2,3", "site": "right", "waypoint": [2, 3]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            with patch.object(evaluation, "read_routes", return_value=routes), \
                    patch.object(evaluation, "scenario_manifest", return_value={"fingerprint": "same"}), \
                    patch.object(evaluation, "run_set", side_effect=FileNotFoundError("checkpoint")), \
                    patch.object(evaluation, "LOG_PATH", Path(directory) / "run.log"), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "incomplete.*FileNotFoundError: checkpoint"):
                    evaluation.evaluate(["fnatic_v3"], rounds=12, output=path)
            entry = json.loads(path.read_text(encoding="utf-8"))["opponents"]["fnatic_v3"]
            self.assertEqual(entry["status"], "error")
            self.assertIsNone(entry["routes"]["right:2,3"]["win_rate"])

    def test_same_game_controllers_score_and_ultimates_continue_for_twelve_rounds(self):
        carry = SimpleNamespace(waypoint_cells_by_site={"left": [(1, 2)]},
                                _active_site="left", _waypoint_index=0, _active_target=(3, 4))
        attacker_ai = SimpleNamespace(get_attacker_controller=lambda: SimpleNamespace(carry_controller=carry))

        class Game:
            def __init__(self):
                self.attacker_wins = self.defender_wins = 0
                self.current_round = 1
                self.round_over = self.match_over = self.sides_swapped = False
                self.is_planted = self.is_defused = False
                self.planted_pos = None
                self.battle_tick = 1
                self.chars = [SimpleNamespace(team="A", name="player", ultimate_points=0)]
                self.init_calls = 0

            def step_tick(self):
                if self.current_round % 2:
                    self.attacker_wins += 1
                else:
                    self.defender_wins += 1
                self.chars[0].ultimate_points += 1
                self.round_over = True
                return True

            def init_round(self):
                self.init_calls += 1
                self.round_over = False

        game = Game()
        result = evaluation.play_set(game, attacker_ai, {"site": "left", "waypoint": [1, 2]}, 123)
        self.assertEqual(game.init_calls, 11)
        self.assertEqual([r["round_number"] for r in result["rounds"]], list(range(1, 13)))
        self.assertEqual([r["start_ultimate_points"]["A:player"] for r in result["rounds"]], list(range(12)))
        self.assertEqual(sum(r["won"] for r in result["rounds"]), 6)
        self.assertEqual(result["rounds"][-1]["score_after"], [6, 6])

    def test_partial_set_is_replayed_from_its_seed_after_interrupt(self):
        routes = [{"id": "left:1,2", "site": "left", "waypoint": [1, 2]}]
        result = {"seed": 0, "rounds": [{"won": False, "planted": False} for _ in range(12)]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            with patch.object(evaluation, "read_routes", return_value=routes), \
                    patch.object(evaluation, "scenario_manifest", return_value={"fingerprint": "same"}), \
                    patch.object(evaluation, "LOG_PATH", Path(directory) / "run.log"), \
                    contextlib.redirect_stdout(io.StringIO()):
                with patch.object(evaluation, "run_set", side_effect=[result, KeyboardInterrupt()]):
                    with self.assertRaises(KeyboardInterrupt):
                        evaluation.evaluate(["fnatic_v3"], rounds=24, output=path)
                with patch.object(evaluation, "run_set", return_value=result) as run:
                    evaluation.evaluate(["fnatic_v3"], rounds=24, output=path)
                    self.assertEqual(run.call_count, 1)
                    self.assertEqual(run.call_args.args[-1],
                                     evaluation.trial_seed(evaluation.BASE_SEED, "fnatic_v3", 1))

    def test_rejects_incomplete_set_round_count(self):
        with self.assertRaisesRegex(ValueError, "multiple of 12"):
            evaluation.evaluate(["fnatic_v3"], rounds=100)

    def test_run_set_keeps_same_named_players_on_opposite_teams_separate(self):
        import run_game
        with patch.object(run_game, "VisualFPSBattle") as battle, \
                patch.object(run_game, "_build_team_ai",
                             return_value=SimpleNamespace(attacker_factory=lambda: None)), \
                patch.object(evaluation, "play_set", return_value={"rounds": []}):
            evaluation.run_set("toru_ai_v4", "Team Elites",
                               {"site": "left", "waypoint": [4, 11]}, 123)
        kwargs = battle.call_args.kwargs
        ours = next(name for name in kwargs["attacker_roster"] if str(name) == "おもこ")
        theirs = next(name for name in kwargs["defender_roster"] if str(name) == "おもこ")
        self.assertFalse(ours == theirs)
        self.assertEqual(len({ours: "attacker", theirs: "defender"}), 2)

    def test_models_loaded_via_bytes_are_fingerprinted_and_changes_rejected(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"value": 1}, path)
            models = evaluation.FrozenModels()
            with models.active():
                self.assertEqual(torch.load(io.BytesIO(path.read_bytes()), weights_only=False)["value"], 1)
                self.assertEqual(len(models.hashes), 1)
                torch.save({"value": 2, "extra": 3}, path)
                with self.assertRaisesRegex(ValueError, "changed"):
                    path.read_bytes()

    def test_parallel_merge_preserves_each_opponent_and_rejects_model_mismatch(self):
        report = {"schema_version": 2, "scenario": {"fingerprint": "same"},
                  "base_seed": 123, "opponents": {"frc_v1": {"status": "complete"}}}
        child = {"schema_version": 2, "scenario": {"fingerprint": "same"},
                 "base_seed": 123, "opponents": {"fnatic_v3": {"status": "running"}},
                 "model_hashes": {"carry.pt": "abc"}}
        models = evaluation.FrozenModels()
        evaluation.merge_worker_report(report, child, "fnatic_v3", models)
        self.assertEqual(report["opponents"]["frc_v1"]["status"], "complete")
        self.assertEqual(models.hashes, {"carry.pt": "abc"})
        child["model_hashes"]["carry.pt"] = "different"
        with self.assertRaisesRegex(ValueError, "different checkpoints"):
            evaluation.merge_worker_report(report, child, "fnatic_v3", models)
        self.assertEqual(models.hashes, {"carry.pt": "abc"})

    def test_parallel_merge_rejects_wrong_seed_or_opponent(self):
        report = {"schema_version": 2, "scenario": {"fingerprint": "same"},
                  "base_seed": 123, "opponents": {}}
        child = {**report, "base_seed": 456, "opponents": {"fnatic_v3": {}}}
        with self.assertRaisesRegex(ValueError, "differs"):
            evaluation.merge_worker_report(report, child, "fnatic_v3", evaluation.FrozenModels())
        child["base_seed"] = 123
        with self.assertRaisesRegex(ValueError, "differs"):
            evaluation.merge_worker_report(report, child, "frc_v1", evaluation.FrozenModels())

    def test_worker_stop_request_saves_without_starting_another_set(self):
        routes = [{"id": "left:1,2", "site": "left", "waypoint": [1, 2]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.json"
            with patch.object(evaluation, "read_routes", return_value=routes), \
                    patch.object(evaluation, "scenario_manifest", return_value={"fingerprint": "same"}), \
                    patch.object(evaluation, "run_set") as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = evaluation.evaluate(
                    ["fnatic_v3"], rounds=12, output=path,
                    log_path=Path(directory) / "worker.log",
                    stop_event=SimpleNamespace(is_set=lambda: True),
                )
            run.assert_not_called()
            self.assertEqual(result["opponents"]["fnatic_v3"]["status"], "interrupted")
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["opponents"]["fnatic_v3"]["routes"]["left:1,2"]["rounds"], 0)

    def test_windows_atomic_save_retries_transient_reader_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text('{"value": 0}', encoding="utf-8")
            original = Path.replace
            attempts = []

            def replace(source, target):
                attempts.append(source)
                if len(attempts) < 3:
                    raise PermissionError("transient Windows read lock")
                return original(source, target)

            with patch.object(Path, "replace", replace), patch.object(evaluation.time, "sleep"):
                evaluation.save_report(path, {"value": 1}, evaluation.FrozenModels())
            self.assertEqual(len(attempts), 3)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["value"], 1)

    def test_infrastructure_change_can_resume_but_gameplay_change_cannot(self):
        script = "omoko_v1/ov1_evaluate_attacker_carry_routes.py"
        saved = {"fingerprint": "old", "evaluation_version": "same-protocol",
                 "wall_los_version": "same-los", "rounds_per_set": 12, "routes": [],
                 "source_hashes": {script: "old-code", "battle_logic.py": "game"}}
        current = {**saved, "fingerprint": "new",
                   "source_hashes": {script: "new-code", "battle_logic.py": "game"}}
        self.assertTrue(evaluation.same_scenario(saved, current))
        current["source_hashes"]["battle_logic.py"] = "changed-game"
        self.assertFalse(evaluation.same_scenario(saved, current))

    def test_reviewed_prior_reader_does_not_invalidate_fixed_route_collection(self):
        carry = "omoko_v1/ov1_learning_attacker_carry.py"
        saved = {"fingerprint": "old", "evaluation_version": "same-protocol",
                 "wall_los_version": "same-los", "rounds_per_set": 12, "routes": [],
                 "source_hashes": {carry: "9b18eb7ed98a078266fb41ad6f323eed29e09f919ba524d0e408c381647e2db9"}}
        current = {**saved, "fingerprint": "new", "source_hashes": {
            carry: "c864b41a68811b859ab2fa56cf4d3fdc8ec38d0a211df8c00e054700f871c490",
            "omoko_v1/ov1_attacker_carry_route_priors.py": "new",
            "omoko_v1/ov1_summarize_attacker_carry_routes.py": "new",
        }}
        self.assertTrue(evaluation.same_scenario(saved, current))
        current["source_hashes"][carry] = "some-other-gameplay-change"
        self.assertFalse(evaluation.same_scenario(saved, current))


if __name__ == "__main__":
    unittest.main()
