"""Selected sites isolate cases, optimizer updates and checkpoint writes."""

from collections import Counter
import contextlib
import io
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from concon_v1 import co1_train_defender_retake as battle
from concon_v1 import co1_train_defender_retake_base as base
from concon_v1.co1_defender_retake_training import RetakeTrainingAdapter
from concon_v1.co1_retake_case_training import RetakeCaseDataset, iter_case_training_windows
from concon_v1.co1_retake_training_schedule import iter_training_windows
from concon_v1.test.test_co1_retake_case_training import dataset_fixture, CaseEnvFixture
from concon_v1.test.test_co1_retake_training_schedule import BiasedSites


class SiteSelectionTests(unittest.TestCase):
    def test_foundation_flags_train_and_save_only_selected_site(self):
        for site in ("L", "R"):
            with self.subTest(site=site), tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(patch.object(base, "TRAIN_LEFT_SITE", site == "L"))
                stack.enter_context(patch.object(base, "TRAIN_RIGHT_SITE", site == "R"))
                stack.enter_context(patch.object(base, "validate_starts", return_value={}))
                stack.enter_context(patch.object(base, "RetakeDQN", return_value=Mock()))
                learn = stack.enter_context(patch.object(base, "learn_foundation", return_value=dict(training_states=1, loss=0.)))
                stack.enter_context(patch.object(base, "evaluate_foundation", return_value=dict(arrival_rate=1., defuse_rate=1., passed=True)))
                stack.enter_context(patch.object(base, "make_checkpoint", side_effect=lambda model, site, *args: dict(site=site)))
                save = stack.enter_context(patch.object(base.torch, "save"))
                self.assertEqual(set(base.train_foundation(save_dir=temporary)), {site})
                learn.assert_called_once()
                save.assert_called_once()
                self.assertEqual(save.call_args.args[0]["site"], site)

    def test_battle_flags_isolate_resume_optimizer_and_all_checkpoint_types(self):
        for site in ("L", "R"):
            with self.subTest(site=site), tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(patch.object(battle, "TRAIN_LEFT_SITE", site == "L"))
                stack.enter_context(patch.object(battle, "TRAIN_RIGHT_SITE", site == "R"))
                search = SimpleNamespace(model=torch.nn.Linear(1, 1), model_path=Path("search.pt"))
                stack.enter_context(patch.object(battle, "ConconDefenderSearchController", return_value=search))
                model = torch.nn.Linear(1, 1)
                load = stack.enter_context(patch.object(battle, "load_training_model", return_value=(model, dict(episode=10))))
                stack.enter_context(patch.object(battle, "DefenderRetakeEnv"))

                def schedule(env, opponents, episodes, interval, rng, epsilon_fn, on_step, on_progress, sites):
                    self.assertEqual(sites, (site,))
                    on_step([("L", (0,)), ("R", (0,))], 4)
                    yield dict(opponent="omoko_v1", site=site, planted=True, defused=True,
                               end_reason="defused", fire_decisions=0, moving_fire_decisions=0,
                               smoke_defuse_decisions=0, round=1,
                               training_episodes={"L": int(site == "L"), "R": int(site == "R")}), 1

                stack.enter_context(patch.object(battle, "iter_training_windows", side_effect=schedule))
                stack.enter_context(patch.object(battle, "make_checkpoint", side_effect=lambda model, site, episode, *args: dict(site=site, episode=episode)))
                metric = dict(retakes=1, mean_defuse_rate=1., min_defuse_rate=1., moving_fire_rate=0.)
                stack.enter_context(patch.object(battle, "evaluate", return_value={site: metric}))
                stack.enter_context(patch.object(battle, "print_evaluation_summary"))
                optimize = stack.enter_context(patch.object(battle, "optimize", return_value=None))
                save = stack.enter_context(patch.object(battle.torch, "save"))
                models = battle.train(episodes=1, checkpoint_interval=1, save_dir=temporary,
                                      opponents=["omoko_v1"], resume=True, force_save=True,
                                      epsilon_start=.05, epsilon_end=.05)
                self.assertEqual(set(models), {site})
                load.assert_called_once()
                self.assertEqual(load.call_args.args[0], site)
                self.assertEqual(load.call_args.args[3], Path(temporary))
                optimize.assert_called_once()
                self.assertIs(optimize.call_args.args[0], model)
                self.assertEqual(len(optimize.call_args.args[3]), 1)
                self.assertEqual(save.call_count, 3)  # latest, numbered, best
                self.assertTrue(all(call.args[0]["site"] == site for call in save.call_args_list))
                self.assertTrue(all(call.args[0]["episode"] == 11 for call in save.call_args_list))

    def test_unselected_cases_are_not_required_or_opened(self):
        for site in ("L", "R"):
            with self.subTest(site=site), tempfile.TemporaryDirectory() as temporary:
                other = "R" if site == "L" else "L"
                dataset_fixture(temporary, {("one", site): 3, ("one", other): 1})
                (Path(temporary) / f"one_{other}_0.case.gz").unlink()
                dataset = RetakeCaseDataset(temporary, ["one"], sites=(site,))
                self.assertEqual(dataset.size, 3)
                events = list(iter_case_training_windows(CaseEnvFixture(), dataset, 6, 2, random.Random(0), lambda n: 0))
                self.assertEqual(Counter(row["site"] for row, _ in events), {site: 6})
                self.assertEqual([boundary for _, boundary in events if boundary], [2, 4, 6])

    def test_live_quota_and_replay_ignore_other_site(self):
        for site in ("L", "R"):
            replay = []
            events = list(iter_training_windows(BiasedSites(), ["one"], 3, 3, random.Random(0),
                                               lambda n: .1, sites=(site,),
                                               on_step=lambda transitions, ticks: replay.extend(transitions)))
            counted = [row for row, _ in events if row["counted_episode"]]
            self.assertEqual([row["site"] for row in counted], [site] * 3)
            self.assertTrue(all(record_site == site for record_site, _ in replay))
            self.assertEqual(events[-1][1], 3)

    def test_evaluation_counts_only_selected_site(self):
        metric = dict(opponent="one", site="L", planted=True, defused=True,
                      fire_decisions=0, moving_fire_decisions=0, smoke_defuse_decisions=0)
        with contextlib.redirect_stdout(io.StringIO()), patch.object(battle, "DefenderRetakeEnv"), \
             patch.object(battle, "iter_retake_rounds", return_value=[metric]) as rounds:
            result = battle.evaluate({"L": Mock()}, Mock(), rounds=2, opponents=["one"])
            self.assertEqual(rounds.call_args.kwargs["site"], "L")
            self.assertEqual(result["R"]["required_retakes_per_opponent"], 0)

    def test_live_other_site_uses_production_controller_without_recording(self):
        search, selected, existing = Mock(), Mock(), Mock()
        adapter = RetakeTrainingAdapter(search, {"L": selected})
        game = Mock()
        adapter.set_game(game)
        with patch("concon_v1.co1_defender_retake_training.plant_site", return_value="R"), \
             patch("concon_v1.co1_defender_retake_training.ConconDefenderRetakeController", return_value=existing) as load:
            state = dict(is_planted=True, planted_pos=(0, 0))
            adapter.decide_move(Mock(), state)
            adapter.decide_move(Mock(), state)
            load.assert_called_once_with("R")
            existing.set_game.assert_called_once_with(game)
            self.assertEqual(existing.decide_move.call_count, 2)
            selected.decide_move.assert_not_called()

    def test_both_disabled_fail_before_loading_or_saving(self):
        for module, function in ((base, base.train_foundation), (battle, battle.train)):
            with patch.object(module, "TRAIN_LEFT_SITE", False), patch.object(module, "TRAIN_RIGHT_SITE", False):
                with self.assertRaisesRegex(ValueError, "at least one"):
                    function()


if __name__ == "__main__":
    unittest.main()
