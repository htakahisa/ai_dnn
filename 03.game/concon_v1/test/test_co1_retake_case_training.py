"""Dataset balancing, epoch coverage and saved-case CLI selection."""

from collections import Counter
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concon_v1.co1_attacker_scenarios import GAME_MAZE_STR
from concon_v1.co1_retake_case_training import RetakeCaseDataset, iter_case_training_windows
from concon_v1.co1_train_defender_retake import main, train, epsilon_by_episode
import torch


def dataset_fixture(directory, counts):
    directory = Path(directory)
    (directory / "search.pt").write_bytes(b"search")
    provenance = dict(version=1, map_sha256=hashlib.sha256(GAME_MAZE_STR.encode()).hexdigest(),
                      search_model_sha256=hashlib.sha256(b"search").hexdigest())
    (directory / "collection.json").write_text(json.dumps(dict(provenance=provenance)), encoding="utf-8")
    rows = []
    for (opponent, site), count in counts.items():
        for index in range(count):
            filename = f"{opponent}_{site}_{index}.case.gz"
            (directory / filename).write_bytes(b"fixture")
            rows.append(dict(version=1, opponent=opponent, site=site, file=filename))
    (directory / "cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class CaseEnvFixture:
    def reset_case(self, path, cache):
        self.opponent, self.site, _ = path.name.split("_")
        self.done = False

    def step(self, epsilon):
        self.done = True
        return [], [], True

    def result(self):
        return dict(opponent=self.opponent, site=self.site, planted=True, excluded_from_retake=False)


class CaseTrainingTests(unittest.TestCase):
    def test_each_epoch_covers_all_cases_and_each_window_balances_teams_and_sites(self):
        with tempfile.TemporaryDirectory() as temporary:
            counts = {(opponent, site): 3 for opponent in ("one", "two") for site in ("L", "R")}
            dataset_fixture(temporary, counts)
            dataset = RetakeCaseDataset(temporary, ["one", "two"])
            self.assertEqual(dataset.size, 12)
            dataset.validate_search(Path(temporary) / "search.pt")
            records = list(iter_case_training_windows(CaseEnvFixture(), dataset, 24, 8, random.Random(1), lambda n: .1))
            self.assertEqual([boundary for _, boundary in records if boundary], [8, 16, 24])
            self.assertEqual(Counter(row["case_file"] for row, _ in records),
                             Counter({row["file"]: 2 for rows in dataset.groups.values() for row in rows}))
            for start in (0, 8, 16):
                window = [row for row, _ in records[start:start + 8]]
                self.assertEqual(Counter((row["opponent"], row["site"]) for row in window),
                                 Counter({key: 2 for key in counts}))
            self.assertEqual(records[-1][0]["training_episodes"], dict(L=12, R=12))
            self.assertTrue(all(row["case_epoch"] == 1 for row, _ in records[:12]))
            self.assertTrue(all(row["case_epoch"] == 2 for row, _ in records[12:]))

    def test_partial_collection_balances_selected_cases_and_ignores_incomplete_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            counts = {("one", "L"): 4, ("one", "R"): 2}
            dataset_fixture(temporary, counts)
            with (Path(temporary) / "cases.jsonl").open("a", encoding="utf-8") as handle:
                handle.write('{"unfinished":')
            dataset = RetakeCaseDataset(temporary, ["one"])
            self.assertEqual(dataset.available_size, 6)
            self.assertEqual(dataset.size, 4)
            self.assertEqual(dataset.per_group, 2)
            with self.assertRaises(ValueError):
                RetakeCaseDataset(temporary, ["one", "missing"])
            (Path(temporary) / "search.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                dataset.validate_search(Path(temporary) / "search.pt")

    def test_invalid_window_and_cli_combinations_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset_fixture(temporary, {("one", "L"): 1, ("one", "R"): 1})
            dataset = RetakeCaseDataset(temporary, ["one"])
            with self.assertRaises(ValueError):
                list(iter_case_training_windows(CaseEnvFixture(), dataset, 4, 3, random.Random(1), lambda n: 0))
        with contextlib.redirect_stderr(io.StringIO()):
            with patch("concon_v1.co1_train_defender_retake.USE_COLLECTED_CASES", False):
                with self.assertRaises(SystemExit):
                    main(["--case-epochs", "10"])
            with self.assertRaises(SystemExit):
                main(["--cases-dir", "data/cases", "--case-epochs", "10", "--episodes", "100"])
        with patch("concon_v1.co1_train_defender_retake.train") as train:
            main(["--cases-dir", "data/cases", "--case-epochs", "10", "--resume"])
            self.assertEqual(train.call_args.kwargs["cases_dir"], Path("data/cases"))
            self.assertEqual(train.call_args.kwargs["case_epochs"], 10)
            self.assertIsNone(train.call_args.kwargs["episodes"])
            self.assertTrue(train.call_args.kwargs["resume"])

    def test_no_argument_launch_uses_source_and_resume_constants(self):
        with patch("concon_v1.co1_train_defender_retake.train") as train:
            with patch("concon_v1.co1_train_defender_retake.USE_COLLECTED_CASES", True), \
                 patch("concon_v1.co1_train_defender_retake.CASES_DIR", Path("data/example")), \
                 patch("concon_v1.co1_train_defender_retake.RESUME_TRAINING", True):
                main([])
                self.assertEqual(train.call_args.kwargs["cases_dir"], Path("data/example"))
                self.assertTrue(train.call_args.kwargs["resume"])
                self.assertIsNone(train.call_args.kwargs["episodes"])
                main(["--no-resume"])
                self.assertFalse(train.call_args.kwargs["resume"])
            with patch("concon_v1.co1_train_defender_retake.USE_COLLECTED_CASES", False), \
                 patch("concon_v1.co1_train_defender_retake.RESUME_TRAINING", False):
                main([])
                self.assertIsNone(train.call_args.kwargs["cases_dir"])
                self.assertFalse(train.call_args.kwargs["resume"])

    def test_evaluation_waits_for_low_epsilon_but_latest_is_saved_at_every_boundary(self):
        search = SimpleNamespace(model=torch.nn.Linear(1, 1), model_path=Path("search.pt"))
        metric = dict(retakes=1, mean_defuse_rate=.5, min_defuse_rate=.5, moving_fire_rate=0.)
        metrics = {site: dict(metric) for site in ("L", "R")}

        def schedule(env, opponents, episodes, interval, rng, epsilon_fn, on_step, on_progress):
            for offset in (100, 200, 300):
                # Only replay insertion; ticks=1 means no optimization executes.
                on_step([("L", (0,)), ("R", (0,))], 1)
                row = dict(opponent="omoko_v1", site="L", planted=True, defused=False,
                           excluded_from_retake=False, end_reason="detonated", fire_decisions=0,
                           moving_fire_decisions=0, smoke_defuse_decisions=0,
                           round=offset, training_episodes=dict(L=offset, R=0))
                yield row, offset

        def checkpoint(model, site, episode, *args):
            return dict(map_name=site, episode=episode)

        with tempfile.TemporaryDirectory() as temporary, \
             contextlib.redirect_stdout(io.StringIO()), \
             patch("concon_v1.co1_train_defender_retake.ConconDefenderSearchController", return_value=search), \
             patch("concon_v1.co1_train_defender_retake.load_training_model",
                   side_effect=lambda *a: (torch.nn.Linear(1, 1), {})), \
             patch("concon_v1.co1_train_defender_retake.DefenderRetakeEnv"), \
             patch("concon_v1.co1_train_defender_retake.iter_training_windows", side_effect=schedule), \
             patch("concon_v1.co1_train_defender_retake.make_checkpoint", side_effect=checkpoint), \
             patch("concon_v1.co1_train_defender_retake.evaluate", return_value=metrics) as evaluate, \
             patch("concon_v1.co1_train_defender_retake.print_evaluation_summary"), \
             patch("concon_v1.co1_train_defender_retake.optimize") as optimize, \
             patch("concon_v1.co1_train_defender_retake.torch.save") as save:
            train(episodes=300, checkpoint_interval=100, save_dir=temporary, opponents=["omoko_v1"])
            evaluate.assert_called_once()
            optimize.assert_not_called()
            latest = [call.args[0] for call in save.call_args_list if str(call.args[1]).endswith("latest.pt")]
            best = [call.args[0] for call in save.call_args_list if str(call.args[1]).endswith("best.pt")]
            self.assertEqual([c["episode"] for c in latest], [100, 100, 200, 200, 300, 300])
            self.assertTrue(all(c["evaluation"] is None for c in latest[:4]))
            self.assertTrue(all(c["evaluation_status"] == "waiting_for_epsilon" for c in latest[:4]))
            self.assertEqual([c["episode"] for c in best], [300, 300])
            self.assertTrue(all(c["evaluation_status"] == "completed" for c in latest[4:]))
        self.assertGreater(epsilon_by_episode(3499, 5000), .05)
        self.assertEqual(epsilon_by_episode(3500, 5000), .05)
if __name__ == "__main__":
    unittest.main()
