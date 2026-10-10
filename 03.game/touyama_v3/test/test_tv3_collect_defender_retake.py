"""Missing sites stay incomplete; filled sites do not consume collection quota."""
from contextlib import redirect_stdout, nullcontext
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from touyama_v3 import tv3_collect_defender_retake as collector


class CollectionTests(unittest.TestCase):
    def run_collection(self, left, right, blocks, limit=20):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            best = directory / "best" / "gc_v1"
            best.mkdir(parents=True)
            model_path = best / "search_best.pt"
            model_path.write_bytes(b"mock model")
            (directory / "existing.case.gz").write_bytes(b"mock case")
            rows = [dict(opponent="gc_v1", site=side, block=1, file="existing.case.gz")
                    for side, count in (("L", left), ("R", right)) for _ in range(count)]
            (directory / "cases.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (directory / "blocks.jsonl").write_text(
                json.dumps(dict(opponent="gc_v1", block=blocks)) + "\n", encoding="utf-8")
            args = SimpleNamespace(opponents=["gc_v1"], sites=["L", "R"], train_presets=["Touyama Gaming"],
                                   best_dir=directory / "best", output_dir=directory, seed=42,
                                   resume=True, cases_per_site=50, max_blocks=limit, site_sampling="targeted")
            config = dict(format="touyama_v3_plant_cases_v1", scenario="mock scenario", seed=42,
                          presets=args.train_presets, opponents=args.opponents, sites=args.sites,
                          search_hashes={"gc_v1": collector.hashlib.sha256(model_path.read_bytes()).hexdigest()},
                          analysis_hashes={"gc_v1": "mock analysis"},
                          site_sampling="targeted", sampling_version=collector.SAMPLING_VERSION,
                          capture_boundary="end_of_first_plant_tick_before_retake_decisions")
            (directory / "collection.json").write_text(json.dumps(config), encoding="utf-8")
            output = io.StringIO()
            with patch.object(collector, "Scenario", return_value=SimpleNamespace(signature="mock scenario")), \
                 patch.object(collector, "load_analyses", return_value=({}, {"gc_v1": "mock analysis"})), \
                 patch.object(collector, "load_policy", return_value=(MagicMock(),
                              dict(opponent="gc_v1", analysis_hashes={"gc_v1": "mock analysis"}))), \
                 patch.object(collector, "eligible_presets", return_value=args.train_presets), \
                 patch.object(collector, "site_sampling", return_value=nullcontext()), \
                 patch.object(collector, "rollout", return_value=([dict(planted=True)], [])) as rollout, \
                 redirect_stdout(output):
                success = collector.collect(args)
            return success, rollout.call_count, output.getvalue()

    def test_resumed_left_only_collection_is_incomplete_at_limit(self):
        success, calls, output = self.run_collection(50, 0, 20)
        self.assertFalse(success)
        self.assertEqual(calls, 0)
        self.assertIn("missing={'gc_v1': {'R': 50}}", output)

    def test_zero_case_site_keeps_collecting_until_limit(self):
        success, calls, _ = self.run_collection(50, 0, 19, limit=22)
        self.assertFalse(success)
        self.assertEqual(calls, 3)

    def test_right_only_collection_also_supported(self):
        success, calls, output = self.run_collection(0, 50, 20)
        self.assertFalse(success)
        self.assertEqual(calls, 0)
        self.assertIn("missing={'gc_v1': {'L': 50}}", output)

    def test_partially_collected_site_remains_incomplete(self):
        success, _, output = self.run_collection(50, 1, 20)
        self.assertFalse(success)
        self.assertIn("missing={'gc_v1': {'R': 49}}", output)

    def test_both_empty_or_other_site_below_target_remain_incomplete(self):
        for left in (0, 49):
            with self.subTest(left=left):
                success, _, output = self.run_collection(left, 0, 20)
                self.assertFalse(success)
                self.assertIn("Incomplete", output)

    def test_both_full_need_no_more_games(self):
        success, calls, output = self.run_collection(50, 50, 1)
        self.assertTrue(success)
        self.assertEqual(calls, 0)
        self.assertIn("Complete", output)


if __name__ == "__main__":
    unittest.main()
