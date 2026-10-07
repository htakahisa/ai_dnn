import ast
import copy
import io
import json
import os
import tempfile
import unittest
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from series_data_validation import series_export_skip_reason


ROOT = Path(__file__).resolve().parent.parent
EXPORTERS = (
    ROOT / "run_competition_manager.py",
    ROOT / "gc_v1" / "evaluate_real_series_gc.py",
    ROOT / "gc_v1" / "train_attacker_carry_gc_real.py",
)


def series(scores):
    return {
        "team1": "Alpha", "team2": "Bravo", "team1_score": 2, "team2_score": 0,
        "maps": [
            {"number": number, "score1": left, "score2": right, "round_records": []}
            for number, (left, right) in enumerate(scores, 1)
        ],
    }


def export_function(source_path, export_root):
    """Run the actual export functions without importing GUI/model/training code."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    functions = {
        "generate_inference_input", "extract_team_metrics", "extract_tactic_metrics",
        "extract_entry_metrics", "extract_individual_efficiency", "calculate_side_outcomes",
    }
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in functions)
        or (isinstance(node, ast.ImportFrom) and node.module == "series_data_validation")
    ]
    namespace = {
        "json": json, "Path": Path, "datetime": datetime, "defaultdict": defaultdict,
        "ROOT": export_root,
        "os": SimpleNamespace(path=os.path, makedirs=os.makedirs, getcwd=lambda: str(export_root)),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["generate_inference_input"]


class SeriesValidationTests(unittest.TestCase):
    def test_completed_and_overtime_maps_are_accepted(self):
        self.assertIsNone(series_export_skip_reason(series([(13, 0), (0, 13), (13, 12), (12, 14), (16, 14)])))

    def test_one_incomplete_map_rejects_the_entire_series(self):
        for scores in (((13, 8), (0, 0)), ((13, 8), (12, 12)), ((5, 10),)):
            with self.subTest(scores=scores):
                self.assertIn("incomplete", series_export_skip_reason(series(scores)))
        reason = series_export_skip_reason(series([(13, 8), (0, 0)]))
        self.assertIn("Map 2", reason)
        self.assertIn("0-0", reason)

    def test_empty_or_invalid_map_scores_are_rejected(self):
        for value in (None, True, "13", -1, 12.5, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertIn("invalid scores", series_export_skip_reason(series([(13, value)])))
        for data in ({}, {"maps": []}, {"maps": None}, None, {"maps": [None]}):
            with self.subTest(data=data):
                self.assertIsNotNone(series_export_skip_reason(data))


class SeriesExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assertTrue(self.root.resolve().is_relative_to(ROOT.resolve()))

    def test_all_exporters_skip_before_creating_any_directory(self):
        for source in EXPORTERS:
            root = self.root / source.stem
            root.mkdir()
            export = export_function(source, root)
            for data in (series([(13, 8), (0, 0)]), series([(12, 12)]), series([(13, None)]), series([])):
                with self.subTest(source=source.name, maps=data["maps"]):
                    before = copy.deepcopy(data)
                    with patch("sys.stdout", new_callable=io.StringIO) as output:
                        self.assertIsNone(export(data))
                    self.assertIn("Skipped series_data export", output.getvalue())
                    self.assertFalse((root / "series_data").exists())
                    self.assertEqual(data, before)

    def test_all_exporters_still_save_complete_series(self):
        for source in EXPORTERS:
            with self.subTest(source=source.name):
                root = self.root / source.stem
                root.mkdir()
                export = export_function(source, root)
                data = series([(13, 8), (12, 14)])
                with patch("sys.stdout", new_callable=io.StringIO):
                    export(data)
                originals = list((root / "series_data").glob("*/*_original.json"))
                inference = list((root / "series_data").glob("*/*_inference.json"))
                self.assertEqual(len(originals), 1)
                self.assertEqual(len(inference), 1)
                self.assertEqual(json.loads(originals[0].read_text(encoding="utf-8")), data)

    def test_skip_leaves_existing_saved_series_untouched(self):
        for source in EXPORTERS:
            with self.subTest(source=source.name):
                root = self.root / source.stem
                saved = root / "series_data" / "previous"
                saved.mkdir(parents=True)
                sentinel = saved / "previous_original.json"
                sentinel.write_text("existing data", encoding="utf-8")
                with patch("sys.stdout", new_callable=io.StringIO):
                    export_function(source, root)(series([(13, 8), (5, 7)]))
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing data")
                self.assertEqual(list((root / "series_data").iterdir()), [saved])

    def test_custom_output_dirs_also_reject_incomplete_series(self):
        for source in EXPORTERS[1:]:
            with self.subTest(source=source.name):
                custom = self.root / source.stem
                with patch("sys.stdout", new_callable=io.StringIO):
                    result = export_function(source, self.root)(series([(3, 7)]), output_dir=custom)
                self.assertIsNone(result)
                self.assertFalse(custom.exists())


if __name__ == "__main__":
    unittest.main()
