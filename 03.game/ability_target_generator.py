"""Generate fixed tactical ability target candidates for an RL agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

Cell = Tuple[int, int]


@dataclass(frozen=True)
class AbilityTarget:
    name: str
    target: Cell
    valid: bool = True


class AbilityTargetGenerator:
    """Generate a stable list of target candidates for SMOKE/FLASH/RECON."""

    def __init__(self, env, candidate_count: int = 8) -> None:
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        self.env = env
        self.candidate_count = candidate_count

    def generate(self, player) -> list[AbilityTarget]:
        player_pos = self._as_cell(player.pos)
        enemies = sorted(
            (
                char
                for char in self.env.chars
                if getattr(char, "is_alive", False)
                and getattr(char, "team", None) != getattr(player, "team", None)
            ),
            key=lambda enemy: (
                self._manhattan(player_pos, self._as_cell(enemy.pos)),
                str(getattr(enemy, "name", "")),
            ),
        )

        targets: list[AbilityTarget] = [
            self._make("SELF_FRONT", self._front_cell(player)),
            self._enemy_target(enemies, 0),
            self._enemy_target(enemies, 1),
            self._named_env_cell("SITE_CENTER", ("site_center", "site_pos", "site_position")),
            self._named_env_cell("PLANT_POSITION", ("plant_pos", "plant_position", "spike_pos")),
            self._nearest_enemy_midpoint(player_pos, enemies),
            self._enemy_pair_midpoint(enemies),
            self._nearest_enemy_neighbor(enemies),
        ]

        targets = targets[: self.candidate_count]
        while len(targets) < self.candidate_count:
            targets.append(self._invalid("UNUSED", player_pos))
        return targets

    def valid_mask(self, player) -> list[bool]:
        return [candidate.valid for candidate in self.generate(player)]

    def _enemy_target(self, enemies: Sequence[object], index: int) -> AbilityTarget:
        if index >= len(enemies):
            return self._invalid(f"ENEMY_{index + 1}", (0, 0))
        enemy = enemies[index]
        return self._make(
            f"ENEMY_{index + 1}_{getattr(enemy, 'name', '')}",
            self._as_cell(enemy.pos),
        )

    def _nearest_enemy_midpoint(self, player_pos: Cell, enemies: Sequence[object]) -> AbilityTarget:
        if not enemies:
            return self._invalid("NEAREST_ENEMY_MIDPOINT", player_pos)
        return self._make(
            "NEAREST_ENEMY_MIDPOINT",
            self._midpoint(player_pos, self._as_cell(enemies[0].pos)),
        )

    def _enemy_pair_midpoint(self, enemies: Sequence[object]) -> AbilityTarget:
        if len(enemies) < 2:
            return self._invalid("ENEMY_PAIR_MIDPOINT", (0, 0))
        return self._make(
            "ENEMY_PAIR_MIDPOINT",
            self._midpoint(self._as_cell(enemies[0].pos), self._as_cell(enemies[1].pos)),
        )

    def _nearest_enemy_neighbor(self, enemies: Sequence[object]) -> AbilityTarget:
        if not enemies:
            return self._invalid("NEAREST_ENEMY_NEIGHBOR", (0, 0))

        enemy_pos = self._as_cell(enemies[0].pos)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            candidate = (enemy_pos[0] + dr, enemy_pos[1] + dc)
            if self._is_valid_target(candidate):
                return AbilityTarget("NEAREST_ENEMY_NEIGHBOR", candidate, True)
        return self._invalid("NEAREST_ENEMY_NEIGHBOR", enemy_pos)

    def _front_cell(self, player) -> Cell:
        player_pos = self._as_cell(player.pos)
        dr, dc = self._player_direction(player)
        if dr == 0 and dc == 0:
            return player_pos

        max_steps = max(int(self.env.height), int(self.env.width))
        last_valid = player_pos
        for step in range(1, max_steps + 1):
            candidate = (player_pos[0] + dr * step, player_pos[1] + dc * step)
            if not self._inside(candidate) or self._is_wall(candidate):
                break
            last_valid = candidate
        return last_valid

    def _player_direction(self, player) -> Cell:
        if hasattr(player, "dir_r") and hasattr(player, "dir_c"):
            return self._normalize_direction(int(player.dir_r), int(player.dir_c))

        direction = getattr(player, "direction", None)
        if isinstance(direction, (tuple, list)) and len(direction) == 2:
            return self._normalize_direction(int(direction[0]), int(direction[1]))

        facing = str(getattr(player, "facing", getattr(player, "direction_name", ""))).upper()
        return {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
        }.get(facing, (0, 0))

    def _named_env_cell(self, name: str, attribute_names: Iterable[str]) -> AbilityTarget:
        for attribute_name in attribute_names:
            value = getattr(self.env, attribute_name, None)
            if isinstance(value, (tuple, list)) and len(value) == 2:
                return self._make(name, self._as_cell(value))

        for attribute_name in ("site_cells", "plant_cells", "spike_site_cells"):
            cells = getattr(self.env, attribute_name, None)
            if cells:
                normalized = [
                    self._as_cell(cell)
                    for cell in cells
                    if isinstance(cell, (tuple, list)) and len(cell) == 2
                ]
                if normalized:
                    center = (
                        round(sum(cell[0] for cell in normalized) / len(normalized)),
                        round(sum(cell[1] for cell in normalized) / len(normalized)),
                    )
                    return self._make(name, center)

        return self._invalid(name, (0, 0))

    def _make(self, name: str, cell: Cell) -> AbilityTarget:
        normalized = self._as_cell(cell)
        if not self._is_valid_target(normalized):
            nearest = self._nearest_valid_cell(normalized)
            if nearest is None:
                return self._invalid(name, normalized)
            normalized = nearest
        return AbilityTarget(name, normalized, True)

    def _nearest_valid_cell(self, origin: Cell) -> Optional[Cell]:
        if self._is_valid_target(origin):
            return origin

        max_radius = max(int(self.env.height), int(self.env.width))
        for radius in range(1, max_radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in (-radius, radius):
                    cell = (origin[0] + dr, origin[1] + dc)
                    if self._is_valid_target(cell):
                        return cell
            for dc in range(-radius + 1, radius):
                for dr in (-radius, radius):
                    cell = (origin[0] + dr, origin[1] + dc)
                    if self._is_valid_target(cell):
                        return cell
        return None

    def _is_valid_target(self, cell: Cell) -> bool:
        return self._inside(cell) and not self._is_wall(cell)

    def _inside(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < int(self.env.height) and 0 <= c < int(self.env.width)

    def _is_wall(self, cell: Cell) -> bool:
        r, c = cell
        return int(self.env.grid[r, c]) == 1

    @staticmethod
    def _invalid(name: str, target: Cell) -> AbilityTarget:
        return AbilityTarget(name, target, False)

    @staticmethod
    def _as_cell(value) -> Cell:
        return int(value[0]), int(value[1])

    @staticmethod
    def _midpoint(first: Cell, second: Cell) -> Cell:
        return ((first[0] + second[0]) // 2, (first[1] + second[1]) // 2)

    @staticmethod
    def _manhattan(first: Cell, second: Cell) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    @staticmethod
    def _normalize_direction(dr: int, dc: int) -> Cell:
        return (
            0 if dr == 0 else (1 if dr > 0 else -1),
            0 if dc == 0 else (1 if dc > 0 else -1),
        )
