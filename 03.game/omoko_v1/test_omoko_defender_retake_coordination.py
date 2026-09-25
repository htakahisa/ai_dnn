"""Behavior checks for annotated, synchronized defender retake entry."""

import unittest
from types import SimpleNamespace

import numpy as np

from map_data import NEW_MAZE_STR
from omoko_v1.ov1_map_data_retake import retake_entry_pairs
from omoko_v1.ov1_retake_coordination import RetakeEntryCoordinator


GRID = np.array([[int(cell) for cell in row] for row in NEW_MAZE_STR.strip().splitlines()])
SPAWNS = list(zip(*np.where(GRID == 4)))


def defenders():
    return [
        SimpleNamespace(name=str(i), team="D", is_alive=True, pos=tuple(pos))
        for i, pos in enumerate(SPAWNS)
    ]


class RetakeCoordinationTests(unittest.TestCase):
    def test_annotations_are_walkable_and_paired(self):
        pairs = retake_entry_pairs(GRID)
        for side in ("left", "right"):
            self.assertEqual(len(pairs[side]), 4)
            for entry, watch in pairs[side]:
                self.assertNotEqual(GRID[entry], 1)
                self.assertNotEqual(GRID[watch], 1)

    def test_stages_and_releases_on_same_tick_for_both_sites(self):
        for spike in ((8, 3), (7, 42)):
            with self.subTest(spike=spike):
                team = defenders()
                coordinator = RetakeEntryCoordinator()
                first = coordinator.actions(GRID, team, spike, 55, 6)
                self.assertEqual(set(first), {unit.name for unit in team})
                assignments = coordinator.assignments
                self.assertIsNotNone(assignments)
                self.assertGreaterEqual(sum(not row[3] for row in assignments.values()), 2)

                for unit in team:
                    unit.pos = assignments[unit.name][0]
                release = coordinator.actions(GRID, team, spike, 40, 6)
                self.assertEqual(set(release), set(first))
                for unit in team:
                    stage, entry, watch, follower = assignments[unit.name]
                    if follower:
                        step = release[unit.name][0]
                        self.assertLessEqual(
                            abs(step[0] - stage[0]) + abs(step[1] - stage[1]), 1
                        )
                    else:
                        self.assertEqual(release[unit.name][0], entry)
                    self.assertEqual(release[unit.name][1], watch)

    def test_combat_or_short_timer_returns_to_model(self):
        team = defenders()
        coordinator = RetakeEntryCoordinator()
        self.assertEqual(coordinator.actions(GRID, team, (7, 42), 8, 6), {})
        self.assertTrue(coordinator.finished)

        team = defenders()
        coordinator = RetakeEntryCoordinator()
        coordinator.actions(GRID, team, (8, 3), 55, 6)
        team.append(SimpleNamespace(name="enemy", team="A", is_alive=True, pos=(2, 18)))
        self.assertEqual(coordinator.actions(GRID, team, (8, 3), 54, 6), {})
        self.assertTrue(coordinator.finished)

    def test_staging_does_not_block_deeper_lanes(self):
        for spike in ((8, 3), (7, 42)):
            with self.subTest(spike=spike):
                team = defenders()
                coordinator = RetakeEntryCoordinator()
                for tick in range(45):
                    decisions = coordinator.actions(GRID, team, spike, 55 - tick, 6)
                    if coordinator.released:
                        break
                    self.assertFalse(
                        coordinator.finished,
                        f"tick={tick}, positions={[tuple(unit.pos) for unit in team]}, "
                        f"stages={[coordinator.assignments[unit.name][0] for unit in team]}",
                    )
                    occupied = {tuple(unit.pos) for unit in team}
                    destinations = [decision[0] for decision in decisions.values()]
                    moving = [destinations[i] for i, unit in enumerate(team)
                              if destinations[i] != tuple(unit.pos)]
                    self.assertEqual(len(moving), len(set(moving)))
                    for unit in team:
                        destination = decisions[unit.name][0]
                        if destination not in occupied or destination == tuple(unit.pos):
                            unit.pos = destination
                self.assertTrue(coordinator.released)


if __name__ == "__main__":
    unittest.main()
