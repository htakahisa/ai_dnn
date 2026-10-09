"""Generic roster pools and independent evaluation plans; no live rollouts."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest
from unittest.mock import patch
from party_presets import get_preset
from toruAI_v4.tv4_scenario import OPPONENTS
from toruAI_v4.tv4_attacker_rosters import eligible_attacker_presets, source_presets
from toruAI_v4.tv4_train_attacker_plant import TRAINING_PRESETS, EVALUATION_PRESETS, parser
from toruAI_v4.tv4_train_attacker_plant import evaluate as evaluate_plant
from toruAI_v4.tv4_train_attacker_analysis import evaluate as evaluate_analysis


class RosterTests(unittest.TestCase):
    def test_default_rosters_are_multiple_disjoint_from_enemies_and_holdout(self):
        args = parser().parse_args([])
        self.assertGreater(len(args.train_presets), 1)
        self.assertGreater(len(args.eval_presets), 1)
        self.assertFalse(set(args.train_presets) & set(args.eval_presets))
        for opponent in OPPONENTS:
            enemy = set(get_preset(OPPONENTS[opponent][1]).players)
            training = eligible_attacker_presets(TRAINING_PRESETS, opponent)
            evaluation = eligible_attacker_presets(EVALUATION_PRESETS, opponent)
            self.assertGreater(len(training), 1)
            self.assertGreater(len(evaluation), 1)
            for preset in training + evaluation:
                self.assertFalse(set(get_preset(preset).players) & enemy)

    def test_legacy_metadata_readable_without_claiming_new_roster_conditions(self):
        old = source_presets({"attacker_preset": "Gorigons"}, "gc_v1")
        self.assertEqual(old, (["Gorigons"], ["Gorigons"]))
        new = source_presets({"train_presets": list(TRAINING_PRESETS), "eval_presets": list(EVALUATION_PRESETS)}, "gc_v1")
        self.assertGreater(len(new[0]), 1)
        self.assertFalse(set(new[0]) & set(new[1]))

    def test_plant_evaluation_rotates_holdout_presets_without_live_games(self):
        config = {"eval_presets": list(EVALUATION_PRESETS)}
        with patch("toruAI_v4.tv4_train_attacker_plant.rollout", return_value=([], [])) as rollout:
            result = evaluate_plant("gc_v1", None, None, None, config, [101, 102, 103])
            actual = [call.kwargs["preset_name"] for call in rollout.call_args_list]
            self.assertEqual(actual, list(EVALUATION_PRESETS))
            self.assertEqual(result["roster_plan"], [{"seed": s, "preset": p} for s, p in zip([101, 102, 103], EVALUATION_PRESETS)])

    def test_analysis_selected_and_random_use_same_holdout_roster(self):
        with patch("toruAI_v4.tv4_train_attacker_analysis.play_block", return_value=([], [])) as rollout:
            evaluate_analysis("gc_v1", None, None, {"eval_presets": list(EVALUATION_PRESETS)}, [101, 102, 103])
            actual = [call.kwargs["preset_name"] for call in rollout.call_args_list]
            self.assertEqual(actual, [p for preset in EVALUATION_PRESETS for p in (preset, preset)])

    def test_defender_analysis_uses_distinct_training_and_holdout_rosters(self):
        from unittest.mock import Mock
        from toruAI_v4.tv4_train_defender_analysis import parser as defender_parser, evaluate_seeds
        args = defender_parser().parse_args([])
        self.assertFalse(set(args.train_presets) & set(args.eval_presets))
        for opponent in OPPONENTS:
            self.assertGreater(len(eligible_attacker_presets(args.train_presets, opponent)), 1)
        identity = dict(opponent="gc_v1", phase="eval", set=10, trained_rounds=120, replay_rounds=0)
        with patch("toruAI_v4.tv4_train_defender_analysis.play_block", return_value=([], [])) as rollout:
            result = evaluate_seeds("gc_v1", None, None, args, identity, [101, 102, 103], Mock())
        self.assertEqual([c.args[3].defender_preset for c in rollout.call_args_list], args.eval_presets)
        self.assertEqual([r["preset"] for r in result["roster_plan"]], args.eval_presets)


if __name__ == "__main__":
    unittest.main()
