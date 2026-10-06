"""Equal team quotas with biased plant sites, without running learning."""

from collections import Counter
import contextlib
import io
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concon_v1.co1_retake_training_schedule import iter_training_windows
from concon_v1.co1_train_defender_retake import summarize
from concon_v1.co1_retake_logging import print_summary


class BiasedSites:
    def __init__(self):
        self.attempts, self.epsilons = Counter(), []

    def reset(self, opponent):
        self.opponent, self.done = opponent, False
        self.attempts[opponent] += 1

    def step(self, epsilon):
        self.done = True
        self.epsilons.append(float(epsilon))
        site = self.result()["site"]
        transitions = [(site, self.opponent)] if site is not None else []
        return transitions, [0.] * 5, True

    def result(self):
        kind = (self.attempts[self.opponent] - 1) % 6
        site = "L" if kind < 4 else "R" if kind == 5 else None
        return dict(opponent=self.opponent, site=site, planted=site is not None,
                    excluded_from_retake=site is None, defused=False, end_reason="detonated",
                    fire_decisions=0, moving_fire_decisions=0, smoke_defuse_decisions=0)


class ScheduleTests(unittest.TestCase):
    def test_two_hundred_retakes_means_forty_per_team_with_natural_sites(self):
        opponents = [f"team{index}" for index in range(5)]
        env, replay = BiasedSites(), Counter()
        def record_steps(transitions, tick):
            replay.update((site, opponent) for site, opponent in transitions)
        events = list(iter_training_windows(env, opponents, 200, 200, random.Random(0),
                                           lambda episode: .15, on_step=record_steps))
        boundaries = [(record, boundary) for record, boundary in events if boundary is not None]
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0][1], 200)
        self.assertEqual(boundaries[0][0]["training_episodes"], dict(L=160, R=40))
        records = [record for record, _ in events]
        self.assertEqual(sum(record["counted_episode"] for record in records), 200)
        summary = summarize(records, opponents, counted_only=True)
        self.assertEqual(summary["L"]["retakes"], 160)
        self.assertEqual(summary["R"]["retakes"], 40)
        for team in opponents:
            self.assertEqual(sum(summary[site]["opponents"][team]["retakes"] for site in ("L", "R")), 40)
            self.assertEqual(replay[("L", team)], 32)
            self.assertEqual(replay[("R", team)], 8)
        self.assertFalse(any(record["excluded_training_quota"] for record in records))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_summary(summary, "Training summary: episodes=1-200")
        text = output.getvalue()
        self.assertEqual(text.count("defender wins 0/40 (0.0%)"), 5)

    def test_each_window_and_tail_have_equal_team_quotas(self):
        opponents = ["one", "two", "three"]
        env = BiasedSites()
        events = list(iter_training_windows(env, opponents, 9, 6, random.Random(1), lambda episode: .1))
        self.assertEqual([boundary for _, boundary in events if boundary is not None], [6, 9])
        window = []
        for record, boundary in events:
            window.append(record)
            if boundary is None:
                continue
            counts = Counter(row["opponent"] for row in window if row["counted_episode"])
            self.assertEqual(counts, Counter(dict.fromkeys(opponents, 2 if boundary == 6 else 1)))
            window.clear()

    def test_epsilon_advances_by_combined_retakes(self):
        env = BiasedSites()
        events = list(iter_training_windows(env, ["one"], 5, 5, random.Random(0), lambda episode: episode / 100))
        self.assertEqual(env.epsilons[:3], [.01, .02, .03])
        self.assertEqual(env.epsilons[4:6], [.05, .05])
        self.assertEqual(events[-1][0]["training_episodes"], dict(L=4, R=1))

    def test_one_sided_attack_finishes_without_waiting_for_opposite_site(self):
        class LeftOnly(BiasedSites):
            def result(self):
                record = super().result()
                if record["site"] is not None:
                    record["site"] = "L"
                return record
        env = LeftOnly()
        events = list(iter_training_windows(env, ["one"], 40, 40, random.Random(0), lambda episode: .1))
        self.assertEqual(events[-1][1], 40)
        self.assertEqual(events[-1][0]["training_episodes"], dict(L=40, R=0))

    def test_unequal_team_quota_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            list(iter_training_windows(BiasedSites(), ["one", "two", "three"], 10, 6, random.Random(0), lambda episode: .1))


if __name__ == "__main__":
    unittest.main()
