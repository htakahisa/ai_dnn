import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analytics.analysis_branching import (
    AI_ANALYSIS_RULES,
    AnalysisUnavailableError,
    analyze_match_data,
    format_ai_decisions,
)


def decisions(**overrides):
    values = {
        rule["variable"]: False if rule["type"] == "bool" else "none"
        for rule in AI_ANALYSIS_RULES
    }
    values.update(overrides)
    return values


class AiAnalysisTests(unittest.TestCase):
    def test_bool_and_enum_are_mapped_to_fixed_output(self):
        result = format_ai_decisions(
            decisions(entry_coordination_needed=True, site_focus="B")
        )
        self.assertEqual(
            [(item["variable"], item["value"]) for item in result],
            [("entry_coordination_needed", True), ("site_focus", "B")],
        )
        self.assertEqual(
            result[0]["text"],
            next(
                rule["text"]
                for rule in AI_ANALYSIS_RULES
                if rule["variable"] == "entry_coordination_needed"
            ),
        )
        self.assertEqual(
            result[1]["text"],
            next(
                rule["text_map"]["B"]
                for rule in AI_ANALYSIS_RULES
                if rule["variable"] == "site_focus"
            ),
        )

    def test_invalid_types_and_missing_fields_are_rejected(self):
        for invalid in (
            decisions(entry_coordination_needed="true"),
            decisions(site_focus="C"),
            {"entry_coordination_needed": True},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(
                AnalysisUnavailableError
            ):
                format_ai_decisions(invalid)

    def test_missing_model_does_not_fall_back_to_rules(self):
        import analytics.analysis_branching as branching
        with patch.object(branching, "MODEL_PATH", Path("missing-analysis-model.pkl")):
            with self.assertRaisesRegex(AnalysisUnavailableError, "学習済みモデル"):
                analyze_match_data({}, "Alpha")


class MatchPageTests(unittest.TestCase):
    def test_page_passes_each_team_to_ai_and_uses_relative_match_link(self):
        from analytics.app import app
        from analytics import match_output

        original = {
            "team1": "Alpha",
            "team2": "Bravo",
            "team1_score": 1,
            "team2_score": 0,
            "maps": [
                {
                    "initial_attacker": "Alpha",
                    "score1": 1,
                    "score2": 0,
                    "player_stats": [],
                    "round_records": [
                        {
                            "round_number": 1,
                            "winner": "attacker",
                            "reason": "detonated",
                            "planted": True,
                            "tactic": {"final_attack_site": "A"},
                            "players": {},
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            series_dir = Path(temp_dir)
            self.assertTrue(series_dir.resolve().is_relative_to(Path.cwd().resolve()))
            match_dir = series_dir / "sample"
            match_dir.mkdir()
            (match_dir / "sample_original.json").write_text(
                json.dumps(original), encoding="utf-8"
            )
            with patch.object(match_output, "SERIES_DATA_DIR", series_dir), patch.object(
                match_output, "analyze_match_data", return_value=[]
            ) as ai_mock:
                client = app.test_client()
                index_response = client.get("/")
                self.assertEqual(index_response.status_code, 200)
                self.assertIn(b"/match/sample/sample_original.json", index_response.data)
                detail_response = client.get("/match/sample/sample_original.json")
                self.assertEqual(detail_response.status_code, 200)
                self.assertEqual(
                    json.loads((match_dir / "sample_original.json").read_text(encoding="utf-8")),
                    original,
                )
                self.assertFalse((match_dir / "sample_inference.json").exists())
                self.assertEqual(
                    [call.args[1] for call in ai_mock.call_args_list],
                    ["Alpha", "Bravo"],
                )
                for call in ai_mock.call_args_list:
                    self.assertEqual(
                        call.args[0]["round_features"][0]["attacker_team"],
                        "Alpha",
                    )


if __name__ == "__main__":
    unittest.main()
