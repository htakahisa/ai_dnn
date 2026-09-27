"""Sample training-only enemy truth on the fixed map.

The generator has no actor or inference API. A caller must place these enemies
in the training game and then obtain actor input through TeamPerceptionBuilder.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Optional, Sequence, Tuple, Union

from map_data import NEW_MAZE_STR

from coach_v1.common.constants import WATCH_POINTS_CONFIG_PATH
from coach_v1.common.types import GridPosition, Side
from coach_v1.common.watch_points import (
    ATTACKER_SITUATIONS,
    DEFENDER_SITUATIONS,
    WatchPoint,
    load_watch_points,
)


_WEIGHTS = (("point", 0.70), ("jitter", 0.20), ("random", 0.10))
_NEIGHBORS = ((-1, 0), (0, 1), (1, 0), (0, -1))


class ScenarioGenerationError(ValueError):
    """Invalid context or no legal training placement."""


@dataclass(frozen=True)
class EnemyPlacement:
    """Private ground truth for a training game, never an actor observation."""

    position: GridPosition
    category: str
    point_id: Optional[str]
    distance_from_spawn: int


@dataclass(frozen=True)
class Scenario:
    actor_side: Side
    situation: str
    elapsed_ticks: int
    seed: int
    sequence: int
    map_hash: str
    watch_points_hash: str
    enemies: Tuple[EnemyPlacement, ...]


def _distances(rows: Sequence[str], origins: Iterable[GridPosition],
               maximum: Optional[int] = None) -> dict[GridPosition, int]:
    distances = {position: 0 for position in origins}
    queue = deque(distances)
    while queue:
        row, column = queue.popleft()
        depth = distances[(row, column)]
        if maximum is not None and depth >= maximum:
            continue
        for dr, dc in _NEIGHBORS:
            neighbor = row + dr, column + dc
            nr, nc = neighbor
            if (0 <= nr < len(rows) and 0 <= nc < len(rows[0])
                    and rows[nr][nc] != "1" and neighbor not in distances):
                distances[neighbor] = depth + 1
                queue.append(neighbor)
    return distances


class ScenarioGenerator:
    """Seeded 70/20/10 sampler with time, spawn, and visibility constraints."""

    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ScenarioGenerationError("seed must be an integer")
        self.seed = seed
        self._rng = random.Random(seed)
        self._sequence = 0
        self._rows = tuple(NEW_MAZE_STR.strip().splitlines())
        self._config = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
        self._spawn_distances_by_start = {
            Side.ATTACKER: tuple(_distances(self._rows, (cell,))
                                 for cell in self._spawn_cells("3")),
            Side.DEFENDER: tuple(_distances(self._rows, (cell,))
                                 for cell in self._spawn_cells("4")),
        }
        self._spawn_distances = {
            side: {cell: min(distances[cell] for distances in by_start
                             if cell in distances)
                   for cell in set().union(*(set(distances) for distances in by_start))}
            for side, by_start in self._spawn_distances_by_start.items()
        }
        self._all_spawn_reach_ticks = {
            side: max(max(distances.values()) for distances in by_start)
            for side, by_start in self._spawn_distances_by_start.items()
        }
        self._point_positions = frozenset(p.position for p in self._config.points)
        self._jitter_cells = {
            point.point_id: tuple(
                cell for cell, distance in _distances(
                    self._rows, (point.position,), point.random_radius
                ).items()
                if 1 <= distance <= point.random_radius
                and cell not in self._point_positions
            )
            for point in self._config.points
        }

    def _spawn_cells(self, marker: str) -> Tuple[GridPosition, ...]:
        cells = tuple((r, c) for r, row in enumerate(self._rows)
                      for c, value in enumerate(row) if value == marker)
        if len(cells) != 5:
            raise ScenarioGenerationError(f"expected five spawn cells for {marker}")
        return cells

    def generate(
        self,
        *,
        actor_side: Union[Side, str],
        situation: str,
        elapsed_ticks: int,
        enemy_count: int = 5,
        currently_visible: Sequence[Sequence[bool]],
        occupied_positions: Iterable[GridPosition] = (),
    ) -> Scenario:
        """Generate enemy truth for a training situation.

        `currently_visible` must be the actor team's legal current visibility
        mask. The caller supplies occupied allied cells to avoid overlap.
        At early ticks, unavailable categories are removed and remaining
        weights renormalized; the report records the actual category counts.
        """

        try:
            side = Side(actor_side)
        except (TypeError, ValueError) as exc:
            raise ScenarioGenerationError("invalid actor_side") from exc
        allowed = ATTACKER_SITUATIONS if side is Side.ATTACKER else DEFENDER_SITUATIONS
        if situation not in allowed:
            raise ScenarioGenerationError("situation does not match actor_side")
        if isinstance(elapsed_ticks, bool) or not isinstance(elapsed_ticks, int) or elapsed_ticks < 0:
            raise ScenarioGenerationError("elapsed_ticks must be a nonnegative integer")
        if isinstance(enemy_count, bool) or not isinstance(enemy_count, int) or not 1 <= enemy_count <= 5:
            raise ScenarioGenerationError("enemy_count must be in 1..5")

        visible = self._validate_visible(currently_visible)
        occupied = set()
        for position in occupied_positions:
            cell = self._validate_position(position)
            occupied.add(cell)

        enemy_side = Side.DEFENDER if side is Side.ATTACKER else Side.ATTACKER
        distances = self._spawn_distances[enemy_side]
        legal = {cell for cell, distance in distances.items()
                 if distance <= elapsed_ticks and cell not in visible and cell not in occupied}
        points = tuple(point for point in self._config.points
                       if point.supports_side(side) and point.supports_situation(situation))
        placements = []
        for _ in range(enemy_count):
            if elapsed_ticks < self._all_spawn_reach_ticks[enemy_side]:
                legal = {cell for cell in legal if self._spawn_assignment_exists(
                    tuple(enemy.position for enemy in placements) + (cell,),
                    self._spawn_distances_by_start[enemy_side], elapsed_ticks
                )}
            point_candidates = tuple(p for p in points if p.position in legal)
            jitter_candidates = tuple((p, tuple(c for c in self._jitter_cells[p.point_id]
                                                 if c in legal)) for p in points)
            jitter_candidates = tuple((p, cells) for p, cells in jitter_candidates if cells)
            categories = [(name, weight) for name, weight in _WEIGHTS
                          if (name == "point" and point_candidates)
                          or (name == "jitter" and jitter_candidates)
                          or (name == "random" and legal)]
            if not categories:
                raise ScenarioGenerationError("no legal enemy placement for this context")
            category = self._rng.choices(
                [name for name, _ in categories],
                weights=[weight for _, weight in categories], k=1
            )[0]
            if category == "point":
                point = self._rng.choices(point_candidates,
                                          weights=[p.importance for p in point_candidates], k=1)[0]
                position, point_id = point.position, point.point_id
            elif category == "jitter":
                point, cells = self._rng.choices(
                    jitter_candidates,
                    weights=[p.importance for p, _ in jitter_candidates], k=1
                )[0]
                position, point_id = self._rng.choice(cells), point.point_id
            else:
                position, point_id = self._rng.choice(sorted(legal)), None
            placements.append(EnemyPlacement(position, category, point_id, distances[position]))
            legal.remove(position)

        result = Scenario(side, situation, elapsed_ticks, self.seed, self._sequence,
                          self._config.map_hash, self._config.config_hash, tuple(placements))
        self._sequence += 1
        return result

    @staticmethod
    def _spawn_assignment_exists(
        positions: Tuple[GridPosition, ...],
        by_start: Tuple[dict[GridPosition, int], ...],
        elapsed_ticks: int,
    ) -> bool:
        """Each sampled enemy must be able to originate at a distinct spawn."""
        candidates = sorted(
            (tuple(index for index, distances in enumerate(by_start)
                   if distances.get(position, elapsed_ticks + 1) <= elapsed_ticks)
             for position in positions),
            key=len,
        )

        def assign(index: int, used: int) -> bool:
            if index == len(candidates):
                return True
            for origin in candidates[index]:
                bit = 1 << origin
                if not used & bit and assign(index + 1, used | bit):
                    return True
            return False

        return assign(0, 0)

    def _validate_visible(self, visible: Sequence[Sequence[bool]]) -> set[GridPosition]:
        if visible is None:
            raise ScenarioGenerationError("currently_visible is required")
        if len(visible) != len(self._rows) or any(
            len(row) != len(self._rows[0]) for row in visible
        ):
            raise ScenarioGenerationError("currently_visible shape does not match fixed map")
        cells = set()
        for row, values in enumerate(visible):
            for column, value in enumerate(values):
                if not isinstance(value, bool):
                    raise ScenarioGenerationError("currently_visible must contain bool values")
                if value:
                    cells.add((row, column))
        return cells

    def _validate_position(self, position: GridPosition) -> GridPosition:
        if (not isinstance(position, (tuple, list)) or len(position) != 2
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in position)):
            raise ScenarioGenerationError("occupied position must be (row, column)")
        row, column = position
        if not (0 <= row < len(self._rows) and 0 <= column < len(self._rows[0])):
            raise ScenarioGenerationError("occupied position is outside map")
        return row, column


def _scenario_record(scenario: Scenario) -> dict:
    return {
        "actor_side": scenario.actor_side.value,
        "situation": scenario.situation,
        "elapsed_ticks": scenario.elapsed_ticks,
        "seed": scenario.seed,
        "sequence": scenario.sequence,
        "map_hash": scenario.map_hash,
        "watch_points_hash": scenario.watch_points_hash,
        "enemies": [
            {"position": list(enemy.position), "category": enemy.category,
             "point_id": enemy.point_id, "distance_from_spawn": enemy.distance_from_spawn}
            for enemy in scenario.enemies
        ],
    }


def write_scenarios_jsonl(path: Union[str, Path], scenarios: Iterable[Scenario]) -> None:
    """Write private training truth to a caller-selected log file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for scenario in scenarios:
            stream.write(json.dumps(_scenario_record(scenario), ensure_ascii=False, sort_keys=True) + "\n")


def distribution_report(scenarios: Iterable[Scenario]) -> dict:
    counts = {name: 0 for name, _ in _WEIGHTS}
    contexts = {}
    scenario_count = 0
    for scenario in scenarios:
        scenario_count += 1
        context = (scenario.actor_side.value, scenario.situation,
                   scenario.elapsed_ticks, scenario.seed)
        contexts[context] = contexts.get(context, 0) + 1
        for enemy in scenario.enemies:
            counts[enemy.category] += 1
    total = sum(counts.values())
    return {
        "scenarios": scenario_count,
        "enemies": total,
        "target_percent": {name: int(weight * 100) for name, weight in _WEIGHTS},
        "counts": counts,
        "actual_percent": {name: (100 * count / total if total else 0.0)
                           for name, count in counts.items()},
        "contexts": [
            {"actor_side": side, "situation": situation,
             "elapsed_ticks": tick, "seed": seed, "scenarios": count}
            for (side, situation, tick, seed), count in sorted(contexts.items())
        ],
    }


def write_distribution_report(path: Union[str, Path], scenarios: Iterable[Scenario]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(distribution_report(scenarios), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
