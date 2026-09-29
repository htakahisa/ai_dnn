"""Versioned Task 15 opponent pool and deterministic opponent sampling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

from party_presets import get_preset

from coach_v1.common.constants import CONFIG_DIR, PACKAGE_DIR


OPPONENT_POOL_SCHEMA = "coach-opponent-pool-v1"
DEFAULT_POOL_PATH = CONFIG_DIR / "opponent_pool.json"
REQUIRED_EXISTING_OPPONENTS = frozenset({"omoko_v1", "touyama_v2", "gc_v1"})


class OpponentKind(str, Enum):
    EXISTING_AI = "existing_ai"
    HISTORICAL_COACH = "historical_coach"


@dataclass(frozen=True)
class OpponentSpec:
    opponent_id: str
    kind: OpponentKind
    preset: str
    difficulty: int
    weight: float
    ai_key: str | None = None
    attacker_checkpoint: Path | None = None
    attacker_sha256: str | None = None
    defender_checkpoint: Path | None = None
    defender_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.opponent_id or not self.opponent_id.replace("_", "").isalnum():
            raise ValueError("opponent id must contain letters, digits, or underscores")
        if not 1 <= self.difficulty <= 3:
            raise ValueError("opponent difficulty must be between 1 and 3")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("opponent weight must be positive and finite")
        if get_preset(self.preset) is None:
            raise ValueError(f"unknown opponent preset: {self.preset}")
        if self.kind is OpponentKind.EXISTING_AI:
            if not self.ai_key or any(
                value is not None for value in (
                    self.attacker_checkpoint, self.attacker_sha256,
                    self.defender_checkpoint, self.defender_sha256,
                )
            ):
                raise ValueError("existing AI opponent requires only ai_key")
            return
        if self.ai_key is not None:
            raise ValueError("historical coach opponent cannot have ai_key")
        for name, path, digest in (
            ("attacker", self.attacker_checkpoint, self.attacker_sha256),
            ("defender", self.defender_checkpoint, self.defender_sha256),
        ):
            if path is None or digest is None:
                raise ValueError(f"historical coach requires {name} checkpoint and hash")
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"invalid {name} SHA-256")

    @property
    def mirrored_roster(self) -> bool:
        return self.kind is OpponentKind.HISTORICAL_COACH

    def verify_checkpoints(self) -> None:
        if self.kind is not OpponentKind.HISTORICAL_COACH:
            return
        for name, path, expected in (
            ("attacker", self.attacker_checkpoint, self.attacker_sha256),
            ("defender", self.defender_checkpoint, self.defender_sha256),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"missing historical {name} checkpoint: {path}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(
                    f"historical {name} checkpoint changed: expected {expected}, got {actual}"
                )


class OpponentPool:
    def __init__(self, opponents: Sequence[OpponentSpec], *, require_coverage: bool = True):
        self.opponents = tuple(opponents)
        ids = [item.opponent_id for item in self.opponents]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("opponent pool must contain unique opponents")
        if require_coverage:
            actual_existing = {
                item.opponent_id for item in self.opponents
                if item.kind is OpponentKind.EXISTING_AI
            }
            missing = REQUIRED_EXISTING_OPPONENTS - actual_existing
            if missing:
                raise ValueError(f"opponent pool is missing required AIs: {sorted(missing)}")
            if not any(item.kind is OpponentKind.HISTORICAL_COACH
                       for item in self.opponents):
                raise ValueError("opponent pool requires a historical coach checkpoint")

    def eligible(self, *, max_difficulty: int = 3) -> tuple[OpponentSpec, ...]:
        if not 1 <= max_difficulty <= 3:
            raise ValueError("max_difficulty must be between 1 and 3")
        result = tuple(item for item in self.opponents
                       if item.difficulty <= max_difficulty)
        if not result:
            raise ValueError("difficulty filter removed every opponent")
        return result

    def sample(
        self,
        *,
        seed: int,
        count: int,
        max_difficulty: int = 3,
        point_rates: Mapping[str, float] | None = None,
    ) -> tuple[OpponentSpec, ...]:
        """Sample reproducibly, increasing exposure to current weaknesses.

        ``point_rates`` is actor evaluation data (rounds won / rounds played),
        not live game truth. A 0% matchup gets 1.5x its configured weight and a
        100% matchup gets 0.5x, which adjusts curriculum pressure without ever
        dropping an opponent from the pool.
        """

        if count <= 0:
            raise ValueError("sample count must be positive")
        opponents = self.eligible(max_difficulty=max_difficulty)
        rates = dict(point_rates or {})
        weights = []
        for item in opponents:
            rate = float(rates.get(item.opponent_id, 0.5))
            if not 0 <= rate <= 1 or not math.isfinite(rate):
                raise ValueError("point rates must be finite values from zero to one")
            weights.append(item.weight * (1.5 - rate))
        rng = random.Random(seed)
        return tuple(rng.choices(opponents, weights=weights, k=count))

    def with_weights(self, point_rates: Mapping[str, float]) -> "OpponentPool":
        """Freeze weakness-adjusted weights for logging or a later train run."""

        adjusted = []
        for item in self.opponents:
            rate = float(point_rates.get(item.opponent_id, 0.5))
            if not 0 <= rate <= 1 or not math.isfinite(rate):
                raise ValueError("point rates must be finite values from zero to one")
            adjusted.append(replace(item, weight=item.weight * (1.5 - rate)))
        return OpponentPool(adjusted, require_coverage=False)


def _resolve_checkpoint(value: str, *, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_opponent_pool(
    path: Path = DEFAULT_POOL_PATH,
    *,
    require_coverage: bool = True,
) -> OpponentPool:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"schema_version", "opponents"}:
        raise ValueError("invalid opponent pool top-level keys")
    if payload["schema_version"] != OPPONENT_POOL_SCHEMA:
        raise ValueError("unsupported opponent pool schema")
    opponents = []
    for raw in payload["opponents"]:
        kind = OpponentKind(raw["kind"])
        opponents.append(OpponentSpec(
            opponent_id=str(raw["id"]),
            kind=kind,
            preset=str(raw["preset"]),
            difficulty=int(raw["difficulty"]),
            weight=float(raw["weight"]),
            ai_key=raw.get("ai_key"),
            attacker_checkpoint=(
                _resolve_checkpoint(raw["attacker_checkpoint"], base=PACKAGE_DIR)
                if "attacker_checkpoint" in raw else None
            ),
            attacker_sha256=raw.get("attacker_sha256"),
            defender_checkpoint=(
                _resolve_checkpoint(raw["defender_checkpoint"], base=PACKAGE_DIR)
                if "defender_checkpoint" in raw else None
            ),
            defender_sha256=raw.get("defender_sha256"),
        ))
    pool = OpponentPool(opponents, require_coverage=require_coverage)
    for item in pool.opponents:
        item.verify_checkpoints()
    return pool
