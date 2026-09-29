"""Task 16 replay review uses actor sightings rather than replay UI visibility."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from coach_v1.review_task16 import review


class ReviewTest(unittest.TestCase):
    def test_replay_visibility_does_not_become_actor_sighting(self):
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "match.json"
            rounds_path = Path(directory) / "match_rounds.json"
            chars = [
                {"name": "victim", "team": "A", "pos": [5, 4],
                 "facing": "E", "visible_to": ["A"]},
                {"name": "killer", "team": "D", "pos": [5, 5],
                 "facing": "W", "visible_to": ["A", "D"]},
            ]
            replay_path.write_text(json.dumps([
                {"round": 1, "tick": 10, "planted": False, "chars": chars},
                {"round": 1, "tick": 11, "planted": False, "chars": chars},
            ]), encoding="utf-8")
            rounds_path.write_text(json.dumps({
                "kill_events": [{"round": 1, "tick": 11, "victim": "victim",
                                 "victim_team": "A", "killer_position": [5, 5]}],
                "coach_decisions": [{"round_number": 1, "side": "attacker",
                                     "tick": 10, "sightings": []}],
            }), encoding="utf-8")
            candidate = {
                "kind": "repeated_kill_location", "opponents": ["one", "two"],
                "side": "attacker", "position": [5, 5], "phase": "pre_plant",
                "facing": "E", "count": 1, "covering_watch_point_ids": [],
                "uncovered_positions": [[5, 5]],
                "replay_refs": [{"id": str(replay_path), "frame": 1,
                                 "round": 1, "tick": 11}],
                "evidence": [{"nearby_enemy": "killer", "victim": "victim"}],
            }
            row = review({"candidates": [candidate]})[0]
            self.assertEqual(1, row["predeath_killer_visible"])
            self.assertEqual(0, row["predeath_actor_sighted"])
            self.assertIsNone(row["examples"][0]["actor_reported_position"])
            rounds = json.loads(rounds_path.read_text(encoding="utf-8"))
            rounds["coach_decisions"][0]["sightings"] = [["killer", [5, 4]]]
            rounds_path.write_text(json.dumps(rounds), encoding="utf-8")
            row = review({"candidates": [candidate]})[0]
            self.assertEqual(1, row["predeath_actor_sighted"])
            self.assertEqual([5, 4], row["examples"][0]["actor_reported_position"])


if __name__ == "__main__":
    unittest.main()
