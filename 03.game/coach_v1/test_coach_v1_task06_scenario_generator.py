import json
from collections import deque
from pathlib import Path
import tempfile
import unittest

import numpy as np

from map_data import NEW_MAZE_STR

from coach_v1.common.constants import FIXED_ROSTER, WATCH_POINTS_CONFIG_PATH
from coach_v1.common.types import Side
from coach_v1.common.watch_points import load_watch_points
from coach_v1.observation import CoachObservationEncoder
from coach_v1.perception import BeliefMemory
from coach_v1.perception.team_perception import TeamPerceptionBuilder
from coach_v1.test_coach_v1_task03_team_perception import FakeCharacter, FakeGame
from coach_v1.training import (
    ScenarioGenerationError,
    ScenarioGenerator,
    distribution_report,
    write_distribution_report,
    write_scenarios_jsonl,
)


MAP = tuple(NEW_MAZE_STR.strip().splitlines())
CONFIG = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
EMPTY_VISIBILITY = tuple(tuple(False for _ in row) for row in MAP)


def _distance(origin, destination):
    queue = deque(((origin, 0),))
    seen = {origin}
    while queue:
        (row, column), distance = queue.popleft()
        if (row, column) == destination:
            return distance
        for neighbor in ((row - 1, column), (row + 1, column),
                         (row, column - 1), (row, column + 1)):
            nr, nc = neighbor
            if (0 <= nr < len(MAP) and 0 <= nc < len(MAP[0])
                    and MAP[nr][nc] != "1" and neighbor not in seen):
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None


def _has_distinct_spawn_assignment(spawns, positions, elapsed_ticks):
    if not positions:
        return True
    first, *rest = positions
    return any(
        _distance(spawn, first) <= elapsed_ticks
        and _has_distinct_spawn_assignment(
            spawns[:index] + spawns[index + 1:], rest, elapsed_ticks
        )
        for index, spawn in enumerate(spawns)
    )


class CoachV1Task06ScenarioTest(unittest.TestCase):
    def test_seed_reproducibility_and_unique_legal_reachable_positions(self):
        first = ScenarioGenerator(2048)
        second = ScenarioGenerator(2048)
        for side, situation, enemy_spawn in ((Side.ATTACKER, "carry", "4"),
                                             (Side.DEFENDER, "search", "3")):
            for tick in (0, 2, 12, 40, 100):
                a = first.generate(actor_side=side, situation=situation,
                                   elapsed_ticks=tick, currently_visible=EMPTY_VISIBILITY)
                b = second.generate(actor_side=side, situation=situation,
                                    elapsed_ticks=tick, currently_visible=EMPTY_VISIBILITY)
                self.assertEqual(a, b)
                self.assertEqual(5, len(set(enemy.position for enemy in a.enemies)))
                spawns = [(r, c) for r, row in enumerate(MAP)
                          for c, value in enumerate(row) if value == enemy_spawn]
                self.assertTrue(_has_distinct_spawn_assignment(
                    spawns, [enemy.position for enemy in a.enemies], tick))
                for enemy in a.enemies:
                    r, c = enemy.position
                    self.assertNotEqual("1", MAP[r][c])
                    actual_distance = min(_distance(start, enemy.position) for start in spawns)
                    self.assertEqual(actual_distance, enemy.distance_from_spawn)
                    self.assertLessEqual(actual_distance, tick)
                    if tick == 0:
                        self.assertIn(enemy.position, spawns)

    def test_jitter_stays_one_to_three_walking_steps_from_its_point(self):
        generator = ScenarioGenerator(59)
        points = {point.point_id: point for point in CONFIG.points}
        scenarios = [generator.generate(actor_side="attacker", situation="carry",
                                        elapsed_ticks=100, enemy_count=1,
                                        currently_visible=EMPTY_VISIBILITY)
                     for _ in range(500)]
        jitter = [enemy for scenario in scenarios for enemy in scenario.enemies
                  if enemy.category == "jitter"]
        self.assertGreater(len(jitter), 50)
        for enemy in jitter:
            point = points[enemy.point_id]
            self.assertIn(Side.ATTACKER, point.sides)
            self.assertIn("carry", point.situations)
            self.assertLessEqual(1, _distance(point.position, enemy.position))
            self.assertLessEqual(_distance(point.position, enemy.position), point.random_radius)
            self.assertNotIn(enemy.position, {p.position for p in CONFIG.points})

    def test_distribution_is_close_to_initial_70_20_10(self):
        generator = ScenarioGenerator(71)
        scenarios = [generator.generate(actor_side="attacker", situation="carry",
                                        elapsed_ticks=100, enemy_count=5,
                                        currently_visible=EMPTY_VISIBILITY)
                     for _ in range(1000)]
        report = distribution_report(scenarios)
        self.assertEqual(5000, report["enemies"])
        for category, target in report["target_percent"].items():
            self.assertLess(abs(report["actual_percent"][category] - target), 3.0)

    def test_visibility_and_occupied_cells_excluded_and_inputs_checked(self):
        generator = ScenarioGenerator(7)
        visible = tuple(tuple(MAP[r][c] == "4" for c in range(len(MAP[0])))
                        for r in range(len(MAP)))
        with self.assertRaises(ScenarioGenerationError):
            generator.generate(actor_side="attacker", situation="carry",
                               elapsed_ticks=0, currently_visible=visible)
        excluded = {(r, c) for r, row in enumerate(MAP)
                    for c, value in enumerate(row) if value == "4"}
        scenario = generator.generate(actor_side="attacker", situation="carry",
                                      elapsed_ticks=40, enemy_count=5,
                                      currently_visible=visible)
        self.assertTrue(all(enemy.position not in excluded for enemy in scenario.enemies))
        scenario = generator.generate(actor_side="attacker", situation="carry",
                                      elapsed_ticks=0, enemy_count=1,
                                      currently_visible=EMPTY_VISIBILITY,
                                      occupied_positions=tuple(sorted(excluded)[:-1]))
        self.assertEqual(tuple(sorted(excluded)[-1]), scenario.enemies[0].position)
        for kwargs in (
            {"actor_side": "defender", "situation": "carry", "elapsed_ticks": 1},
            {"actor_side": "attacker", "situation": "carry", "elapsed_ticks": -1},
            {"actor_side": "attacker", "situation": "carry", "elapsed_ticks": 1,
             "currently_visible": None},
            {"actor_side": "attacker", "situation": "carry", "elapsed_ticks": 1,
             "currently_visible": ((False,),)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ScenarioGenerationError):
                generator.generate(**({"currently_visible": EMPTY_VISIBILITY} | kwargs))

    def test_logs_and_report_are_reproducible(self):
        generator = ScenarioGenerator(99)
        scenarios = [generator.generate(actor_side="defender", situation="retake",
                                        elapsed_ticks=100, enemy_count=2,
                                        currently_visible=EMPTY_VISIBILITY)
                     for _ in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "scenarios.jsonl"
            report_path = Path(directory) / "distribution.json"
            write_scenarios_jsonl(log, scenarios)
            write_distribution_report(report_path, scenarios)
            entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(10, len(entries))
            self.assertEqual(2, len(entries[0]["enemies"]))
            self.assertEqual(CONFIG.map_hash, entries[0]["map_hash"])
            self.assertEqual(20, json.loads(report_path.read_text(encoding="utf-8"))["enemies"])
            original = log.read_bytes()
            write_scenarios_jsonl(log, scenarios)
            self.assertEqual(original, log.read_bytes())

    def test_generated_hidden_enemy_truth_does_not_reach_actor(self):
        allies = [FakeCharacter(slot.character_name, "A", (23, 18 + index),
                                alive=index == 0)
                  for index, slot in enumerate(FIXED_ROSTER)]
        grid = np.asarray([[int(cell) for cell in row] for row in MAP], dtype=np.int8)
        builder = TeamPerceptionBuilder()
        baseline = builder.build(game=FakeGame(grid, allies), side=Side.ATTACKER)
        generator = ScenarioGenerator(13)
        scenarios = [generator.generate(actor_side="attacker", situation="carry",
                                        elapsed_ticks=100, enemy_count=1,
                                        currently_visible=baseline.currently_visible,
                                        occupied_positions=(ally.position for ally in baseline.allies))
                     for _ in range(2)]
        self.assertNotEqual(scenarios[0].enemies[0].position,
                            scenarios[1].enemies[0].position)
        observations = []
        for scenario in scenarios:
            enemy = FakeCharacter("hidden_enemy", "D", scenario.enemies[0].position)
            snapshot = builder.build(game=FakeGame(grid, allies + [enemy]), side=Side.ATTACKER)
            self.assertEqual((), snapshot.sightings)
            self.assertEqual(baseline.currently_visible, snapshot.currently_visible)
            belief = BeliefMemory(CONFIG.for_side(Side.ATTACKER)).update(snapshot)
            observations.append(CoachObservationEncoder().encode(
                snapshot, belief, situation="carry"))
        np.testing.assert_array_equal(observations[0].grid, observations[1].grid)
        np.testing.assert_array_equal(observations[0].vector, observations[1].vector)


if __name__ == "__main__":
    unittest.main()
