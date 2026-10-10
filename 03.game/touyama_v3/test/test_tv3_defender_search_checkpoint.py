"""Best comparison regressions; no training or changes to saved user models."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import logging
import tempfile
import unittest
from unittest.mock import patch

import torch

from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_defender_policy import DefenderDQN, OBS_DIM
from touyama_v3.tv3_defender_controller import load_policy, policy_metadata
from touyama_v3.tv3_train_defender_search import load_search_comparison


class SearchBestCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = Scenario()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "search_best.pt"
        self.contract = dict(phase="search", opponent="fnatic_v3",
                             schema=policy_metadata(self.scenario),
                             analysis_hashes={"fnatic_v3": "current-analysis"})
        self.state = {**self.contract, "model": DefenderDQN(OBS_DIM).state_dict()}
        self.logger = logging.getLogger("search_checkpoint_test")

    def compare(self, resume=False):
        return load_search_comparison(self.path, self.scenario, self.contract,
                                      resume=resume, logger=self.logger)

    def test_fresh_skips_old_scenario_before_loading_weights(self):
        self.state["schema"] = {**self.contract["schema"], "scenario": "old-wall-los"}
        self.state["analysis_hashes"] = {"fnatic_v3": "old-analysis"}
        torch.save(self.state, self.path)
        original = self.path.read_bytes()
        with patch("touyama_v3.tv3_train_defender_search.load_policy") as load, \
                self.assertLogs(self.logger, level="INFO") as logs:
            self.assertEqual(self.compare(), (None, None))
        load.assert_not_called()
        self.assertIn("scenario", logs.output[0])
        self.assertIn("analysis_hashes", logs.output[0])
        self.assertEqual(self.path.read_bytes(), original)

    def test_fresh_skips_changed_analysis_even_with_matching_schema(self):
        self.state["analysis_hashes"] = {"fnatic_v3": "old-analysis"}
        torch.save(self.state, self.path)
        self.assertEqual(self.compare(), (None, None))

    def test_resume_rejects_scenario_or_analysis_changes(self):
        for key, value in (("schema", {**self.contract["schema"], "scenario": "old"}),
                           ("analysis_hashes", {"fnatic_v3": "old"})):
            with self.subTest(key=key):
                torch.save({**self.state, key: value}, self.path)
                with self.assertRaisesRegex(ValueError, "Best comparison conditions changed"):
                    self.compare(resume=True)

    def test_matching_best_loads_for_fresh_and_resume(self):
        torch.save(self.state, self.path)
        for resume in (False, True):
            with self.subTest(resume=resume):
                model, saved = self.compare(resume)
                self.assertFalse(model.training)
                self.assertEqual(saved["schema"], self.contract["schema"])
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, self.state["model"][name])

    def test_inference_still_rejects_old_scenario(self):
        self.state["schema"] = {**self.contract["schema"], "scenario": "old"}
        torch.save(self.state, self.path)
        with self.assertRaisesRegex(ValueError, "checkpoint/schema mismatch"):
            load_policy(self.path, "search", self.scenario)

    def test_matching_conditions_do_not_hide_invalid_weights(self):
        torch.save({**self.state, "model": {}}, self.path)
        with self.assertRaises(RuntimeError):
            self.compare()


if __name__ == "__main__":
    unittest.main()
