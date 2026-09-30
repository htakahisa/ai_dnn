"""Contract tests for paired Task 17 diagnostic evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import numpy as np

from coach_v1.common.types import Facing, MovementAction, ObjectiveAction, Side, TacticalIntent
from coach_v1.coordinator import ActionLog
from coach_v1.evaluate_task17 import (aggregate, configure_probe, match_metrics,
                                      _erase_channels, SingleViewerSensor)
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.observation.coach_encoder import COACH_GRID_CHANNELS
from coach_v1.test_coach_v1_task03_team_perception import (FakeCharacter, FakeGame,
                                                           fixed_team, floor_grid)


class ComparisonTest(unittest.TestCase):
    def test_watch_channel_removal_keeps_observation_metadata_and_source_immutable(self):
        grid = np.ones((len(COACH_GRID_CHANNELS), 26, 44), dtype=np.float32)
        grid.setflags(write=False)
        @dataclass(frozen=True)
        class Observation:
            grid: np.ndarray
            map_hash: str
            watch_points_hash: str

        source = Observation(grid, "map", "points")
        encoder = SimpleNamespace(encode=lambda: source)
        _erase_channels(encoder, ("watch_importance", "watch_confirmed"))
        changed = encoder.encode()
        self.assertIsNot(changed.grid, source.grid)
        self.assertFalse(changed.grid.flags.writeable)
        self.assertEqual(0, changed.grid[COACH_GRID_CHANNELS.index("watch_importance")].sum())
        self.assertEqual(26 * 44, source.grid[COACH_GRID_CHANNELS.index("watch_importance")].sum())
        self.assertEqual(("map", "points"), (changed.map_hash, changed.watch_points_hash))

    def test_tactical_intent_probe_changes_only_intent(self):
        instruction = CoachInstruction(MovementAction.MOVE_E, ObjectiveAction.NONE,
                                       TacticalIntent.ENTRY)
        coach = SimpleNamespace(act=lambda observation: (instruction,) * 5)
        controller = SimpleNamespace(coach=coach)
        team = SimpleNamespace(get_attacker_controller=lambda: controller,
                               get_defender_controller=lambda: controller)
        configure_probe(team, "no_tactical_intent")
        outputs = coach.act(object())
        self.assertEqual(5, len(outputs))
        self.assertTrue(all(item.movement is MovementAction.MOVE_E and
                            item.objective is ObjectiveAction.NONE and
                            item.intent is TacticalIntent.HOLD for item in outputs))

    def test_recurrent_probe_discards_hidden_state_before_every_action(self):
        seen = []
        coach = SimpleNamespace(hidden="previous")

        def act(observation):
            seen.append(coach.hidden)
            coach.hidden = "updated"
            return ()

        coach.act = act
        attacker = SimpleNamespace(coach=coach)
        defender = SimpleNamespace(coach=SimpleNamespace(hidden="previous", act=lambda o: ()))
        team = SimpleNamespace(get_attacker_controller=lambda: attacker,
                               get_defender_controller=lambda: defender)
        configure_probe(team, "no_recurrent_state")
        coach.act(object())
        coach.act(object())
        self.assertEqual([None, None], seen)

    def test_belief_probe_resets_before_each_legal_snapshot(self):
        events = []
        memory = SimpleNamespace(reset=lambda: events.append("reset"),
                                 update=lambda snapshot: events.append(snapshot))
        attacker = SimpleNamespace(memory=memory)
        defender = SimpleNamespace(memory=SimpleNamespace(reset=lambda: None,
                                                           update=lambda snapshot: None))
        team = SimpleNamespace(get_attacker_controller=lambda: attacker,
                               get_defender_controller=lambda: defender)
        configure_probe(team, "no_belief_memory")
        memory.update("safe_1")
        memory.update("safe_2")
        self.assertEqual(["reset", "safe_1", "reset", "safe_2"], events)

    def test_no_character_models_cannot_issue_ability_or_move(self):
        controller = SimpleNamespace(characters={i: object() for i in range(5)})
        team = SimpleNamespace(get_attacker_controller=lambda: controller,
                               get_defender_controller=lambda: controller)
        configure_probe(team, "no_character_models")
        for actor in controller.characters.values():
            action = actor.act(object())
            self.assertEqual(Facing.N, action.facing)
            self.assertFalse(action.use_ability)

    def test_single_viewer_sensor_does_not_share_another_slots_sighting(self):
        allies = fixed_team(all_alive=True, first_blind=3)
        enemy = FakeCharacter("seen_by_slot_1", "D", (3, 3), facing="W")
        game = FakeGame(floor_grid(), allies + [enemy])
        first = SingleViewerSensor(0).build(game=game, side=Side.ATTACKER)
        second = SingleViewerSensor(1).build(game=game, side=Side.ATTACKER)
        self.assertEqual((), first.sightings)
        self.assertEqual(1, second.sighting_for("seen_by_slot_1").viewer_slot)
        self.assertFalse(first.currently_visible[3][3])
        self.assertTrue(second.currently_visible[3][3])

    def test_single_viewer_sensor_never_exposes_unseen_enemy_truth(self):
        sensor = SingleViewerSensor(0)
        snapshots = [sensor.build(
            game=FakeGame(floor_grid(), fixed_team() + [
                FakeCharacter("unseen", "D", position)]),
            side=Side.ATTACKER,
        ) for position in ((3, 0), (0, 0))]
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual((), snapshots[0].sightings)
        self.assertFalse(hasattr(snapshots[0].enemies[0], "position"))

    def test_metrics_are_read_only_postmatch_and_keep_undefined_denominators(self):
        summary = SimpleNamespace(coach_score=1, opponent_score=0, rounds=1)
        frame = {"round": 1, "tick": 1, "setup": False, "coach_side": "attacker",
                 "chars": [{"team": "A", "alive": True, "facing": "N"},
                           {"team": "A", "alive": True, "facing": "E"}]}
        record = {"round_number": 1, "planted": True, "winner": "attacker"}
        result = SimpleNamespace(summary=summary, replay=(frame,),
                                 round_records=(record,), coach_actions=(),
                                 coach_decisions=(), ability_events=(), kill_events=())
        metrics = match_metrics(result)
        self.assertEqual((1, 1, 1), (metrics["wins"], metrics["plants"],
                                         metrics["multi_angle_frames"]))
        values = aggregate([{"metrics": metrics}])
        self.assertEqual(1.0, values["plant_rate"])
        self.assertIsNone(values["retake_rate"])
        self.assertIsNone(values["ability_effective_rate"])
        self.assertIsNone(values["solo_entry_rate"])

    def test_solo_entry_uses_postmatch_referee_positions_only(self):
        frame = {"round": 1, "tick": 5, "setup": False,
                 "coach_side": "attacker", "chars": [
                     {"name": "ごりまる", "team": "A", "alive": True,
                      "facing": "E", "pos": [10, 10]},
                     {"name": "ごんごん", "team": "A", "alive": True,
                      "facing": "N", "pos": [20, 20]},
                     {"name": "enemy", "team": "D", "alive": True,
                      "facing": "W", "pos": [10, 12]},
                 ]}
        action = ActionLog(1, "live", 5, Side.ATTACKER, 0, "MOVE",
                           (10, 9), (10, 10), "E", None, None)
        result = SimpleNamespace(
            summary=SimpleNamespace(coach_score=0, opponent_score=1, rounds=1),
            replay=(frame,), round_records=({"round_number": 1, "planted": False,
                                             "winner": "defender"},),
            coach_actions=(action,), coach_decisions=(), ability_events=(),
            kill_events=(),
        )
        metrics = match_metrics(result)
        self.assertEqual((1, 1), (metrics["solo_entries"], metrics["contested_entries"]))
        self.assertEqual(1.0, aggregate([{"metrics": metrics}])["solo_entry_rate"])


if __name__ == "__main__":
    unittest.main()
