"""Task 16 candidate extraction and replay privacy boundary."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest

from coach_v1.task16_hard_points import annotate_watch_coverage, candidate_report, collect_match


def frame(round_number, tick, *, dead=False, planted=False, enemy_position=(5, 5)):
    ours = [
        {"name": name, "team": "A", "pos": [5, 4] if i == 0 else [14 + i, 20],
         "alive": not dead if i == 0 else True, "facing": "E", "blind": 0,
         "revealed": False, "visible_to": ["A"]}
        for i, name in enumerate(("ごりまる", "ごんごん", "ごんた", "くんた", "くりまる"))
    ]
    enemies = [
        {"name": f"enemy{i}", "team": "D", "pos": list(enemy_position) if i == 0 else [20, 30 + i],
         "alive": True, "facing": "W", "blind": 0, "revealed": False,
         "visible_to": ["D"]}
        for i in range(5)
    ]
    return {"round": round_number, "tick": tick, "setup": False,
            "planted": planted, "planted_pos": [8, 8] if planted else None,
            "chars": ours + enemies}


def fixture(*, enemy_position=(5, 5), planted=False):
    replay = []
    records = []
    for number in (1, 2):
        replay.extend(frame(number, tick, dead=tick == 31, planted=planted,
                            enemy_position=enemy_position) for tick in range(32))
        records.append({"round_number": number, "winner": "defender", "reason": "attacker_wipe",
                        "planted": planted, "players": {
                            "ごりまる": {"team": "coach_v1", "side": "attacker"},
                            "enemy0": {"team": "opponent", "side": "defender"}}})
    action = {"round_number": 1, "tick": 1, "side": "attacker", "ability": "FLASH",
              "ability_target": (5, 5), "start": (14, 20), "facing": "E"}
    decisions = tuple({"round_number": number, "tick": 31, "side": "attacker",
                       "watch_point_ages": (("point", (5, 5), 31),)} for number in (1, 2))
    return SimpleNamespace(summary=SimpleNamespace(replay_ok=True), replay=tuple(replay),
                           round_records=tuple(records), coach_actions=(action,),
                           coach_decisions=decisions)


class HardPointTest(unittest.TestCase):
    def test_repeated_death_and_other_evidence_have_replay_refs(self):
        result = fixture(planted=True)
        report = candidate_report(((result, "opponent", "replay.json"),))
        self.assertEqual(1, report["matches"])
        by_kind = {c["kind"]: c for c in report["candidates"]}
        self.assertEqual(2, by_kind["death_near_enemy"]["count"])
        self.assertEqual([5, 5], by_kind["death_near_enemy"]["position"])
        self.assertEqual(2, by_kind["long_unseen_before_death"]["count"])
        self.assertEqual(2, by_kind["long_unconfirmed_watch_point"]["count"])
        self.assertEqual(2, by_kind["unseen_close_incursion"]["count"])
        self.assertEqual(2, by_kind["planted_round_loss"]["count"])
        self.assertIn("ability_no_observed_effect", by_kind)
        self.assertEqual("replay.json", by_kind["death_near_enemy"]["replay_refs"][0]["id"])
        self.assertFalse(by_kind["death_near_enemy"]["evidence"][0]["killer_identified"])

    def test_solo_entry_and_unseen_enemy_truth_stays_in_offline_report(self):
        result = fixture()
        before = deepcopy(result.replay)
        events = collect_match(result, opponent_id="opponent", replay_id="r")
        self.assertEqual(before, result.replay)
        self.assertTrue(any(e["kind"] == "solo_entry_death" for e in events))
        self.assertTrue(any(e["kind"] == "unseen_close_incursion" for e in events))
        # Referee truth is retained only in the offline event. No actor is
        # invoked and no observation or live game object is created here.
        self.assertFalse(any("observation" in event for event in events))

    def test_visible_enemy_is_not_reported_as_unseen_and_effect_suppresses_flag(self):
        result = fixture()
        replay = list(deepcopy(result.replay))
        for f in replay:
            f["chars"][5]["visible_to"] = ["A", "D"]
        replay[2]["chars"][5]["blind"] = 1
        result.replay = tuple(replay)
        kinds = {e["kind"] for e in collect_match(result, opponent_id="o", replay_id="r")}
        self.assertNotIn("unseen_close_incursion", kinds)
        self.assertNotIn("long_unseen_before_death", kinds)
        self.assertNotIn("ability_no_observed_effect", kinds)

    def test_shared_angle_is_counted_once_per_pair_per_round(self):
        result = fixture()
        replay = list(deepcopy(result.replay))
        for f in replay:
            f["chars"][1]["pos"] = [5, 3]
        result.replay = tuple(replay)
        candidates = candidate_report(((result, "o", "r"),))["candidates"]
        overlaps = [c for c in candidates if c["kind"] == "overlapping_facing"]
        self.assertEqual(1, len(overlaps))
        self.assertEqual(2, overlaps[0]["count"])

    def test_referee_kill_record_uses_actual_killer_location(self):
        result = fixture()
        result.kill_events = tuple({"round": number, "tick": 31,
                                    "victim": "ごりまる", "victim_team": "A",
                                    "killer": "enemy0", "killer_team": "D",
                                    "killer_position": [5, 5]} for number in (1, 2))
        report = candidate_report(((result, "o", "r"),))
        confirmed = [c for c in report["candidates"]
                     if c["kind"] == "repeated_kill_location"]
        self.assertEqual(1, len(confirmed))
        self.assertEqual(2, confirmed[0]["count"])
        self.assertTrue(confirmed[0]["evidence"][0]["killer_identified"])
        self.assertFalse(any(c["kind"] == "death_near_enemy" for c in report["candidates"]))

    def test_adjacent_kill_cells_form_one_review_location(self):
        result = fixture()
        result.kill_events = (
            {"round": 1, "tick": 31, "victim": "ごりまる", "victim_team": "A",
             "killer": "enemy0", "killer_team": "D", "killer_position": [5, 5]},
            {"round": 2, "tick": 31, "victim": "ごりまる", "victim_team": "A",
             "killer": "enemy0", "killer_team": "D", "killer_position": [5, 6]},
        )
        points = [c for c in candidate_report(((result, "o", "r"),))["candidates"]
                  if c["kind"] == "repeated_kill_location"]
        self.assertEqual(1, len(points))
        self.assertEqual(2, points[0]["count"])
        self.assertEqual({(5, 5), (5, 6)},
                         {tuple(pos) for pos in points[0]["member_positions"]})

    def test_inconsistent_rounds_are_rejected(self):
        result = fixture()
        result.round_records = result.round_records[:1]
        with self.assertRaises(ValueError):
            collect_match(result, opponent_id="o", replay_id="r")

    def test_mirrored_roster_uses_recorded_replay_side(self):
        result = fixture()
        result.round_records = tuple({**record, "players": {
            "ごりまる": {"team": "historical", "side": "defender"}}}
            for record in result.round_records)
        result.replay = tuple({**frame, "coach_side": "attacker"}
                              for frame in result.replay)
        events = collect_match(result, opponent_id="historical", replay_id="r")
        self.assertTrue(events)
        self.assertTrue(all(e["side"] == "attacker" for e in events))

    def test_coverage_uses_existing_watch_point_random_radius(self):
        report = {"candidates": [
            {"side": "attacker", "position": [6, 23], "member_positions": [[6, 23]]},
            {"side": "attacker", "position": [11, 31], "member_positions": [[11, 31]]},
            {"side": "defender", "position": [7, 34], "member_positions": [[7, 34]]},
        ]}
        annotated = annotate_watch_coverage(report)["candidates"]
        self.assertIn("watch_r06_c25", annotated[0]["covering_watch_point_ids"])
        self.assertEqual([], annotated[0]["uncovered_positions"])
        self.assertIn("watch_r11_c31", annotated[1]["covering_watch_point_ids"])
        self.assertEqual([], annotated[1]["uncovered_positions"])
        self.assertIn("watch_r07_c34", annotated[2]["covering_watch_point_ids"])
        self.assertEqual([], annotated[2]["uncovered_positions"])


if __name__ == "__main__":
    unittest.main()
