"""Training balance, site selection, and explicit engine opponent checks."""

from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concon_v1 import co1_train_guard as training
from concon_v1.co1_guard_battle_training import GuardBattleEnv, OPPONENTS, START_MODES


class GuardTrainingScheduleTests(unittest.TestCase):
    def test_every_prefix_and_curriculum_phase_balances_opponents_and_modes(self):
        for seed in (0, 1, 73):
            schedule = training.BalancedTrainingSchedule(OPPONENTS, seed)
            totals = Counter()
            phase_modes = None
            for episode in range(1, 1001):
                modes = training.curriculum_modes(episode, 1000)
                if modes != phase_modes:
                    phase_modes = modes
                    counts = {team: Counter() for team in OPPONENTS}
                team, mode = schedule.next_match(modes)
                totals[team] += 1
                counts[team][mode] += 1
                self.assertLessEqual(max(totals[t] for t in OPPONENTS)
                                     - min(totals[t] for t in OPPONENTS), 1)
                for mode_counts in counts.values():
                    values = [mode_counts[m] for m in modes]
                    self.assertLessEqual(max(values) - min(values), 1)

    def test_full_cycles_use_every_opponent_and_shuffle_reproducibly(self):
        schedules = [training.BalancedTrainingSchedule(OPPONENTS, 42) for _ in range(2)]
        sequences = [[schedule.next_match(START_MODES) for _ in range(180)] for schedule in schedules]
        self.assertEqual(*sequences)
        cycles = [tuple(team for team, _ in sequences[0][i:i + len(OPPONENTS)])
                  for i in range(0, 180, len(OPPONENTS))]
        self.assertTrue(all(set(cycle) == set(OPPONENTS) for cycle in cycles))
        self.assertGreater(len(set(cycles)), 1)
        pairs = Counter(sequences[0])
        self.assertTrue(all(pairs[team, mode] == 6 for team in OPPONENTS for mode in START_MODES))

    def test_engine_uses_explicit_opponent_and_rejects_unconfigured_team(self):
        teams = tuple(OPPONENTS)
        env = GuardBattleEnv(seed=4, opponents=teams[:2])
        with contextlib.redirect_stdout(io.StringIO()):
            initial = env.reset(start_mode="hold", opponent=teams[1])
        self.assertEqual(initial["opponent"], teams[1])
        self.assertEqual(env.opponent, teams[1])
        with self.assertRaisesRegex(ValueError, "configured guard opponents"):
            env.reset(opponent=teams[2])

    def test_training_loop_passes_balanced_matches_to_env_and_logs_them(self):
        class CompletedEnv:
            def reset(self, start_mode, opponent):
                self.done = True
                self.match = {"opponent": opponent, "start_mode": start_mode,
                              "winner": "A", "ticks": 1, "end_reason": "detonated"}
                return {}

            def result(self):
                return self.match

        evaluation = {"mean_win_rate": 1.0, "min_team_win_rate": 1.0}
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(training, "GuardBattleEnv", return_value=CompletedEnv()),
              patch.object(training, "prepare_positioning", return_value=({"passed": True}, {})),
              patch.object(training, "evaluate", return_value=evaluation),
              patch.object(training, "print_summary"),
              contextlib.redirect_stdout(io.StringIO())):
            training.train(episodes=31, save_dir=directory, checkpoint_interval=50)
            records = [json.loads(line) for line in
                       (Path(directory) / "training_log.jsonl").read_text(encoding="utf-8").splitlines()]
            counts = Counter(record["opponent"] for record in records)
            self.assertEqual(len(records), 31)
            self.assertEqual(set(counts), set(OPPONENTS))
            self.assertEqual(sorted(counts.values()), [5, 5, 5, 5, 5, 6])

    def run_main(self, left, right, arguments=(), resumes=None):
        with (patch.object(training, "TRAIN_LEFT_SITE", left),
              patch.object(training, "TRAIN_RIGHT_SITE", right),
              patch.object(training, "DEFAULT_RESUME_BY_MAP", resumes or {"L": None, "R": None}),
              patch.object(sys, "argv", ["co1_train_guard.py", *arguments]),
              patch.object(training, "train") as train,
              contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(io.StringIO())):
            training.main()
            return train.call_args_list

    def test_site_flags_select_both_left_or_right_with_no_options(self):
        for left, right, expected in ((True, True, ["L", "R"]),
                                      (True, False, ["L"]), (False, True, ["R"])):
            calls = self.run_main(left, right)
            self.assertEqual([call.args[1] for call in calls], expected)
            self.assertTrue(all(call.args[0] == training.DEFAULT_EPISODES for call in calls))
            self.assertTrue(all(call.args[3] is None for call in calls))
        with self.assertRaises(SystemExit):
            self.run_main(False, False)

    def test_map_override_and_separate_save_and_resume_paths(self):
        self.assertEqual([c.args[1] for c in self.run_main(True, True, ("--map", "R"))], ["R"])
        resumes = {"L": Path("data/left.pt"), "R": Path("data/right.pt")}
        calls = self.run_main(True, True, ("--save-dir", "data/trial"), resumes)
        for call, site in zip(calls, ("L", "R")):
            self.assertEqual(call.args[3], Path("data/trial") / f"guard_{site}_data")
            self.assertEqual(call.args[8], resumes[site])
        self.assertEqual(self.run_main(False, True, resumes=resumes)[0].args[8], resumes["R"])
        with self.assertRaises(SystemExit):
            self.run_main(True, True, ("--resume", "data/left.pt"))


if __name__ == "__main__":
    unittest.main()
