"""Case restoration, opponent continuation, quotas and collection resumption."""

from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from concon_v1.co1_battle_training import OPPONENTS, _run_from_project_root
from concon_v1.co1_collect_defender_retake import create_game, collect, run_to_plant
from concon_v1.co1_defender_common import DefenderSearchDQN
from concon_v1.co1_defender_scenario import get_scenario
from concon_v1.co1_retake_cases import save_case, load_case, case_metadata
from iq_controller_adapter import IQAwareController


class RetakeCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = DefenderSearchDQN(get_scenario()).eval()

    def fixture(self, opponent="omoko_v1"):
        game = create_game(opponent, self.model, 17)
        game.defender_setup_phase.finish()
        game.is_planted, game.planted_pos, game.spike_pos = True, (7, 3), None
        game.detonate_timer = 54
        return game

    def test_all_opponents_restore_and_continue_with_iq(self):
        for opponent in OPPONENTS:
            with self.subTest(opponent=opponent), tempfile.TemporaryDirectory() as temporary:
                game = self.fixture(opponent)
                # Produce real IQ proxy views and nonempty controller state.
                with contextlib.redirect_stdout(io.StringIO()):
                    _run_from_project_root(lambda: game.step_tick())()
                metadata = case_metadata(game, opponent)
                original_states = [(c.name, c.pos[:], c.hp, c.facing) for c in game.chars]
                engine = game.current_defender_team_ai.perception_engine
                defender = next(c for c in game.chars if c.team == "D")
                engine._defuse_touched_viewers.add(id(defender))
                engine._memory_round = game.current_round
                path = Path(temporary) / "one.case.gz"
                save_case(path, game, metadata)
                restored, restored_metadata = load_case(path)
                self.assertEqual(restored_metadata, metadata)
                self.assertEqual([(c.name, c.pos, c.hp, c.facing) for c in restored.chars], original_states)
                self.assertIsInstance(restored.defender_controller, IQAwareController)
                if getattr(restored.attacker_controller, "handles_team_perception", False):
                    self.assertIs(restored.attacker_controller.game, restored)
                else:
                    self.assertIsInstance(restored.attacker_controller, IQAwareController)
                    self.assertIs(restored.attacker_controller.real_game, restored)
                self.assertIs(restored.defender_controller.real_game, restored)
                restored_defender = next(c for c in restored.chars if c.name == defender.name)
                self.assertIn(id(restored_defender), restored.current_defender_team_ai.perception_engine._defuse_touched_viewers)
                self.assertIs(restored.current_attacker_team_ai.get_attacker_controller(), restored.attacker_controller)
                before = restored.detonate_timer
                with contextlib.redirect_stdout(io.StringIO()):
                    _run_from_project_root(lambda: restored.step_tick())()
                self.assertEqual(restored.detonate_timer, before - 1)

    def test_tensor_dedup_rng_and_independent_repeated_loads(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            game = self.fixture()
            random.seed(41)
            np.random.seed(41)
            torch.manual_seed(41)
            save_case(directory / "a.case.gz", game, case_metadata(game, "omoko_v1"))
            tensors = sorted((directory / "tensors").iterdir())
            save_case(directory / "b.case.gz", game, case_metadata(game, "omoko_v1"))
            self.assertEqual(sorted((directory / "tensors").iterdir()), tensors)
            expected = random.random(), np.random.random(), torch.rand(1).item()
            cache = {}
            first, _ = load_case(directory / "a.case.gz", tensor_cache=cache)
            self.assertEqual((random.random(), np.random.random(), torch.rand(1).item()), expected)
            with patch("concon_v1.co1_retake_cases.torch.load", side_effect=AssertionError("cached tensor reread")):
                second, _ = load_case(directory / "a.case.gz", restore_rng=False, tensor_cache=cache)
            first.chars[0].hp = -123
            self.assertNotEqual(second.chars[0].hp, -123)
            p1 = next(first.defender_controller.inner.search_controller.model.parameters())
            p2 = next(second.defender_controller.inner.search_controller.model.parameters())
            self.assertNotEqual(p1.data_ptr(), p2.data_ptr())

    def test_saved_case_retake_env_uses_current_policy_without_search_reset(self):
        from concon_v1.co1_defender_retake_training import DefenderRetakeEnv
        from concon_v1.co1_retake_common import RetakeDQN, DEFUSE_ACTION
        from concon_v1.co1_retake_scenarios import get_scenario as retake_scenario
        for opponent in OPPONENTS:
            with self.subTest(opponent=opponent), tempfile.TemporaryDirectory() as temporary:
                game = self.fixture(opponent)
                defender = next(c for c in game.chars if c.team == "D")
                defender.pos = [7, 3]
                models = {site: RetakeDQN(retake_scenario(site)) for site in ("L", "R")}
                with torch.no_grad():
                    for model in models.values():
                        for parameter in model.parameters():
                            parameter.zero_()
                        model.head[2].bias[DEFUSE_ACTION] = 10
                        model.head[2].bias[32] = 1
                path = Path(temporary) / "case.case.gz"
                save_case(path, game, case_metadata(game, opponent))
                env = DefenderRetakeEnv(models, self.model, opponents=[opponent])
                with patch.object(env, "reset", side_effect=AssertionError("search round reset")):
                    env.reset_case(path)
                self.assertTrue(env.planted)
                self.assertEqual(env.elapsed_ticks, 0)
                self.assertEqual(env.site, "L")
                self.assertIs(env.retakes["L"].model, models["L"])
                self.assertEqual(env.game.detonate_timer, 54)
                self.assertIs(env.game.current_defender_team_ai.perception_engine,
                              env.game.defender_controller.perception_engine)
                attacker = env.game.attacker_controller
                transitions = []
                for _ in range(54):
                    produced, _, _ = env.step()
                    transitions.extend(produced)
                    if env.done:
                        break
                self.assertTrue(env.done)
                self.assertTrue(transitions)
                self.assertTrue(env.result()["retake_decisions"])
                self.assertEqual(env.result()["plant_defender_alive"], 5)
                self.assertIs(env.game.attacker_controller, attacker)

    def test_preplant_and_finished_rounds_are_rejected(self):
        game = self.fixture()
        game.is_planted = False
        with self.assertRaises(ValueError):
            case_metadata(game, "omoko_v1")
        game.is_planted, game.round_over = True, True
        with self.assertRaises(ValueError):
            case_metadata(game, "omoko_v1")

    def test_fnatic_smoke_use_memory_tracks_restored_smoke_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            game = self.fixture("fnatic_v3")
            smoke = dict(remaining_ticks=10, team="D", cells=[(7, 3)])
            game.smokes.append(smoke)
            controller = game.attacker_controller.inner
            name = game.chars[0].name
            controller.smoke_recon.smokes[id(smoke)] = smoke
            controller.smoke_recon.completed[name] = {id(smoke)}
            controller.smoke_recon.pending[name] = (id(smoke), 2, 5)
            controller.recon_gate.used[name] = {("smoke", id(smoke))}
            controller.recon_gate.pending[name] = (2, 5, {("smoke", id(smoke))})
            owner = controller.game
            controller.opponent_history.scope = (id(owner), "Fnatic2023")
            controller.opponent_history.round_key = (id(owner), 1, 1)
            path = Path(temporary) / "fnatic.case.gz"
            save_case(path, game, case_metadata(game, "fnatic_v3"))
            restored, _ = load_case(path)
            raw = restored.attacker_controller.inner
            key = id(restored.smokes[-1])
            self.assertEqual(raw.smoke_recon.completed[name], {key})
            self.assertEqual(raw.smoke_recon.pending[name][0], key)
            self.assertEqual(raw.recon_gate.used[name], {("smoke", key)})
            self.assertEqual(raw.recon_gate.pending[name][2], {("smoke", key)})
            self.assertEqual(raw.opponent_history.scope[0], id(raw.game))
            self.assertEqual(raw.opponent_history.round_key[0], id(raw.game))
    def test_quota_limits_and_resume_do_not_fabricate_missing_sites(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / "search.pt"
            checkpoint.write_bytes(b"fixture search checkpoint")
            dataset = directory / "cases"
            search = type("SearchFixture", (), {})()
            search.model_path, search.model = checkpoint, self.model
            game = self.fixture()
            def run_fixture(_game, _ticks):
                return 20, True  # The fixture only plants at L.
            with patch("concon_v1.co1_collect_defender_retake.ConconDefenderSearchController", return_value=search), \
                 patch("concon_v1.co1_collect_defender_retake.create_game", return_value=game), \
                 patch("concon_v1.co1_collect_defender_retake.run_to_plant", side_effect=run_fixture), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(collect(1, ["omoko_v1"], output_dir=dataset, max_attempts_per_team=2))
                records = [json.loads(line) for line in (dataset / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(Counter(r["site"] for r in records), Counter(L=1))
                self.assertFalse(collect(1, ["omoko_v1"], output_dir=dataset, max_attempts_per_team=3, resume=True))
                self.assertEqual(len((dataset / "cases.jsonl").read_text(encoding="utf-8").splitlines()), 1)
                rounds = [json.loads(line) for line in (dataset / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual([r["attempt"] for r in rounds], [1, 2, 3])
                with self.assertRaises(ValueError):
                    collect(1, ["omoko_v1"], output_dir=dataset)
                with self.assertRaises(ValueError):
                    collect(1, ["omoko_v1"], output_dir=dataset, seed=1, resume=True)

    def test_resume_adds_frc_without_replacing_existing_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / "search.pt"
            checkpoint.write_bytes(b"fixed search weights")
            search = type("Search", (), {})()
            search.model_path, search.model = checkpoint, self.model
            dataset = directory / "dataset"
            games = {opponent: self.fixture(opponent) for opponent in ("omoko_v1", "frc_v1")}
            with patch("concon_v1.co1_collect_defender_retake.ConconDefenderSearchController", return_value=search), \
                    patch("concon_v1.co1_collect_defender_retake.create_game", side_effect=lambda opponent, *_: games[opponent]) as create, \
                    patch("concon_v1.co1_collect_defender_retake.run_to_plant", return_value=(20, True)), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(collect(1, ["omoko_v1"], sites=("L",), output_dir=dataset))
                old_index = (dataset / "cases.jsonl").read_bytes()
                old_files = {path.name: path.read_bytes() for path in dataset.glob("*.case.gz")}
                create.reset_mock()
                self.assertTrue(collect(1, ["frc_v1"], sites=("L",), output_dir=dataset, resume=True))
                self.assertEqual(create.call_count, 1)
                self.assertEqual(create.call_args.args[0], "frc_v1")
                new_index = (dataset / "cases.jsonl").read_bytes()
                self.assertTrue(new_index.startswith(old_index))
                self.assertEqual(len(new_index.splitlines()), 2)
                for name, contents in old_files.items():
                    self.assertEqual((dataset / name).read_bytes(), contents)
                config = json.loads((dataset / "collection.json").read_text(encoding="utf-8"))
                self.assertEqual(config["provenance"]["opponents"], ["omoko_v1", "frc_v1"])
                create.reset_mock()
                self.assertTrue(collect(1, ["frc_v1"], sites=("L",), output_dir=dataset, resume=True))
                create.assert_not_called()
                self.assertEqual((dataset / "cases.jsonl").read_bytes(), new_index)
                checkpoint.write_bytes(b"changed search weights")
                with self.assertRaisesRegex(ValueError, "search weights"):
                    collect(1, ["frc_v1"], sites=("L",), output_dir=dataset, resume=True)
                self.assertEqual((dataset / "cases.jsonl").read_bytes(), new_index)

    def test_frc_collection_cycles_production_attack_plans(self):
        from frc_v1.baseline import plant_sites
        for round_number in range(1, 6):
            with self.subTest(round_number=round_number):
                game = create_game("frc_v1", self.model, 17)
                game.current_round = round_number
                game.attacker_controller.prepare_team_tick()
                controller = game.attacker_controller
                self.assertEqual(controller.snapshot.round_number, round_number)
                site_index = controller.actor._navigation_site
                cells = plant_sites(controller.snapshot.grid)[site_index]
                expected_left = round_number in (1, 3, 5)
                self.assertTrue(all((col < len(game.grid[0]) // 2) == expected_left for _, col in cells))

    def test_frc_collection_resume_keeps_rotating_after_missing_plants(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / "search.pt"
            checkpoint.write_bytes(b"fixed search weights")
            search = type("Search", (), {})()
            search.model_path, search.model = checkpoint, self.model
            dataset = directory / "dataset"
            game = self.fixture("frc_v1")
            rounds = []
            def rollout(game, _ticks):
                rounds.append(game.current_round)
                return 120, False
            with patch("concon_v1.co1_collect_defender_retake.ConconDefenderSearchController", return_value=search), \
                    patch("concon_v1.co1_collect_defender_retake.create_game", return_value=game), \
                    patch("concon_v1.co1_collect_defender_retake.run_to_plant", side_effect=rollout), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(collect(1, ["frc_v1"], output_dir=dataset, max_attempts_per_team=3))
                self.assertFalse(collect(1, ["frc_v1"], output_dir=dataset, max_attempts_per_team=7, resume=True))
            self.assertEqual(rounds, [1, 2, 3, 4, 5, 1, 2])
            records = [json.loads(line) for line in (dataset / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["opponent_attack_round"] for row in records], rounds)

    def test_rollout_stops_on_first_plant_tick_and_respects_limit(self):
        class GameFixture:
            is_planted = round_over = match_over = is_defused = False
            detonate_timer = 54
            chars = [type("Char", (), dict(team="D", is_alive=True))()]
            ticks = 0

            def step_tick(self):
                self.ticks += 1
                self.is_planted = self.ticks == 3
        game = GameFixture()
        self.assertEqual(run_to_plant(game, 10), (3, True))
        self.assertEqual(game.ticks, 3)
        game = GameFixture()
        self.assertEqual(run_to_plant(game, 2), (2, False))


if __name__ == "__main__":
    unittest.main()
