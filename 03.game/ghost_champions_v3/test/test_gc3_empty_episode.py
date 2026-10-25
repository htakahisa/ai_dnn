"""No-action post-plant results must not abort PPO or fabricate transitions."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ghost_champions_v3 import gc3_training as training
from ghost_champions_v3.tools.gc3_training_compat import (
    RESUME_SOURCE_MIGRATIONS, migrated_manifest, validate_resume_source,
)


def dead_entry(seed=18):
    return dict(phase="postplant", opponent="FRC", seed=seed, decisions=0,
                success=False, reason="defused", ticks=45,
                phase_start=dict(state=dict(chars=[
                    dict(team="A", is_alive=False, hp=0) for _ in range(5)])))


class EmptyEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(OPPONENT_CODES=("FRC", "FNC"), SEED=18, PHASE="postplant",
                           GAMMA=.99, GAE_LAMBDA=.95,
                           POSTPLANT_ENTRY_ATTACKER_CODES=("FRC",), POSTPLANT_ENTRY_MAX_ATTEMPTS=24)

    def test_dead_entry_is_logged_and_next_episode_supplies_the_batch(self):
        rows = [{"decision": 1}, {"decision": 2}]
        nonempty = dict(phase="postplant", opponent="FNC", seed=19, decisions=2, success=True)
        with TemporaryDirectory() as temp, patch.object(training, "play_phase", side_effect=[
                ([], dead_entry()), (rows, nonempty)]) as play:
            collected, episode = training.collect_training_batch(None, self.config, Path(temp), 0, 2)
            records = [json.loads(line) for line in (Path(temp)/"episodes.jsonl").read_text().splitlines()]
        self.assertEqual(collected, rows)
        self.assertEqual(episode, 2)
        self.assertEqual(records[0]["decisions"], 0)
        self.assertFalse(records[0]["success"])
        self.assertTrue(records[0]["training_skipped"])
        self.assertEqual(records[0]["training_skip_reason"], "no_alive_attacker_at_phase_start")
        self.assertEqual(records[1], nonempty)
        self.assertEqual([call.args[:2] for call in play.call_args_list], [("FRC", 18), ("FNC", 19)])

    def test_alive_attacker_and_missing_entry_do_not_hide_a_recording_failure(self):
        alive = dead_entry()
        alive["phase_start"]["state"]["chars"][0]["is_alive"] = True
        for stats in (alive, dict(phase="postplant", decisions=0)):
            with self.subTest(stats=stats), self.assertRaisesRegex(RuntimeError, "no policy decisions"):
                training.empty_episode_reason(stats, "postplant")
        with self.assertRaises(RuntimeError):
            training.empty_episode_reason(dead_entry(), "search")

    def test_repeated_no_action_entries_are_bounded(self):
        with TemporaryDirectory() as temp, patch.object(training, "MAX_CONSECUTIVE_EMPTY_EPISODES", 2), \
                patch.object(training, "play_phase", return_value=([], dead_entry())) as play:
            with self.assertRaisesRegex(RuntimeError, "2 consecutive"):
                training.collect_training_batch(None, self.config, Path(temp), 0, 1)
            self.assertEqual(play.call_count, 2)

    def test_resume_accepts_only_the_reviewed_pair_and_retains_source_history(self):
        previous, current = next(iter(RESUME_SOURCE_MIGRATIONS))
        original = dict(source_hash=previous, config=dict(SEED=18), initialization="scratch")
        migrated = migrated_manifest(original, current, 96191, 614)
        self.assertEqual(original["source_hash"], previous)
        self.assertEqual(migrated["source_hash"], current)
        self.assertEqual(migrated["config"], original["config"])
        self.assertEqual(migrated["source_migrations"][0]["checkpoint_step"], 96191)
        self.assertEqual(migrated_manifest(migrated, current, 96191, 614), migrated)
        for before, after in ((previous, "edited_maps"), ("other_old_sources", current)):
            with self.assertRaisesRegex(RuntimeError, "Source/maps/reward changed"):
                validate_resume_source(before, after)


if __name__ == "__main__":
    unittest.main()
