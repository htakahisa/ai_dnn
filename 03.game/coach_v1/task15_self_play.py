"""Task 15 pool evaluation, self-play construction, and promotion gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from party_presets import get_preset

from coach_v1.common.constants import CHECKPOINTS_DIR, CHARACTER_CHECKPOINT_IDS
from coach_v1.common.watch_point_versions import LEGACY_WATCH_POINTS_PATH
from coach_v1.common.types import Side
from coach_v1.full_match import FullMatchResult, run_headless_full_match
from coach_v1.opponent_pool import OpponentKind, OpponentPool, OpponentSpec
from coach_v1.team_ai import build_coach_v1_team


@dataclass(frozen=True)
class PoolMatchRecord:
    opponent_id: str
    opponent_kind: str
    difficulty: int
    seed: int
    coach_started_as: str
    coach_score: int
    opponent_score: int
    rounds: int
    won: bool
    memory_reset_ok: bool
    side_checkpoint_switch_ok: bool
    replay_ok: bool


@dataclass(frozen=True)
class PoolEvaluation:
    attacker_checkpoint: str
    attacker_sha256: str
    defender_checkpoint: str
    defender_sha256: str
    matches: tuple[PoolMatchRecord, ...]
    recommended_weights: Mapping[str, float]

    def aggregates(self) -> dict[str, dict[str, float | int]]:
        result = {}
        for opponent_id in sorted({item.opponent_id for item in self.matches}):
            records = [item for item in self.matches if item.opponent_id == opponent_id]
            coach_rounds = sum(item.coach_score for item in records)
            all_rounds = sum(item.coach_score + item.opponent_score for item in records)
            result[opponent_id] = {
                "matches": len(records),
                "wins": sum(item.won for item in records),
                "win_rate": sum(item.won for item in records) / len(records),
                "round_point_rate": coach_rounds / max(1, all_rounds),
                "score_margin_per_match": (
                    sum(item.coach_score - item.opponent_score for item in records)
                    / len(records)
                ),
            }
        return result

    def overall(self) -> dict[str, float | int]:
        if not self.matches:
            raise ValueError("evaluation contains no matches")
        coach_rounds = sum(item.coach_score for item in self.matches)
        all_rounds = sum(item.coach_score + item.opponent_score for item in self.matches)
        return {
            "matches": len(self.matches),
            "wins": sum(item.won for item in self.matches),
            "win_rate": sum(item.won for item in self.matches) / len(self.matches),
            "round_point_rate": coach_rounds / max(1, all_rounds),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": {
                "attacker_checkpoint": self.attacker_checkpoint,
                "attacker_sha256": self.attacker_sha256,
                "defender_checkpoint": self.defender_checkpoint,
                "defender_sha256": self.defender_sha256,
            },
            "overall": self.overall(),
            "opponents": self.aggregates(),
            "recommended_weights": dict(self.recommended_weights),
            "matches": [asdict(item) for item in self.matches],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PoolEvaluation":
        candidate = value["candidate"]
        matches = tuple(PoolMatchRecord(**item) for item in value["matches"])
        loaded = cls(
            attacker_checkpoint=str(candidate["attacker_checkpoint"]),
            attacker_sha256=str(candidate["attacker_sha256"]),
            defender_checkpoint=str(candidate["defender_checkpoint"]),
            defender_sha256=str(candidate["defender_sha256"]),
            matches=matches,
            recommended_weights=dict(value.get("recommended_weights", {})),
        )
        # Stored summaries are redundant by design; reject hand-edited or
        # partially written reports before using one as a promotion baseline.
        if "overall" in value and dict(value["overall"]) != loaded.overall():
            raise ValueError("stored overall metrics do not match match records")
        if "opponents" in value and dict(value["opponents"]) != loaded.aggregates():
            raise ValueError("stored opponent metrics do not match match records")
        return loaded


@dataclass(frozen=True)
class PromotionCriteria:
    minimum_matches_per_opponent: int = 2
    maximum_overall_point_rate_drop: float = 0.0
    maximum_opponent_point_rate_drop: float = 0.05
    minimum_historical_win_rate: float = 0.5

    def __post_init__(self) -> None:
        if self.minimum_matches_per_opponent <= 0:
            raise ValueError("minimum matches must be positive")
        for value in (
            self.maximum_overall_point_rate_drop,
            self.maximum_opponent_point_rate_drop,
            self.minimum_historical_win_rate,
        ):
            if not 0 <= value <= 1:
                raise ValueError("promotion rate criteria must be between zero and one")


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reasons: tuple[str, ...]
    candidate_overall: Mapping[str, float | int]
    incumbent_overall: Mapping[str, float | int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "reasons": list(self.reasons),
            "candidate_overall": dict(self.candidate_overall),
            "incumbent_overall": dict(self.incumbent_overall),
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_opponent_team(spec: OpponentSpec, *, device: str = "cpu"):
    """Build a fresh opponent so recurrent and round state never leaks."""

    if spec.kind is OpponentKind.HISTORICAL_COACH:
        spec.verify_checkpoints()
        archived = CHECKPOINTS_DIR / "experiments" / "task16_watch_prechange" / "characters"
        return build_coach_v1_team(
            name=spec.opponent_id,
            attacker_checkpoint=spec.attacker_checkpoint,
            defender_checkpoint=spec.defender_checkpoint,
            character_checkpoints={name: archived / name / "best.pt"
                                   for name in CHARACTER_CHECKPOINT_IDS},
            gongon_defender_checkpoint=archived / "gongon" / "defender_best.pt",
            watch_points_config_path=LEGACY_WATCH_POINTS_PATH,
            device=device,
        )
    # Import lazily because run_game imports every legacy team implementation.
    from run_game import _build_team_ai

    return _build_team_ai(spec.ai_key)


def _record(spec: OpponentSpec, result: FullMatchResult) -> PoolMatchRecord:
    summary = result.summary
    return PoolMatchRecord(
        opponent_id=spec.opponent_id,
        opponent_kind=spec.kind.value,
        difficulty=spec.difficulty,
        seed=summary.seed,
        coach_started_as=summary.coach_started_as,
        coach_score=summary.coach_score,
        opponent_score=summary.opponent_score,
        rounds=summary.rounds,
        won=summary.coach_score > summary.opponent_score,
        memory_reset_ok=summary.memory_reset_ok,
        side_checkpoint_switch_ok=summary.side_checkpoint_switch_ok,
        replay_ok=summary.replay_ok,
    )


def evaluate_pool(
    *,
    attacker_checkpoint: Path,
    defender_checkpoint: Path,
    pool: OpponentPool,
    seeds: Sequence[int],
    max_difficulty: int = 3,
    device: str = "cpu",
    match_runner: Callable[..., FullMatchResult] = run_headless_full_match,
) -> PoolEvaluation:
    """Evaluate every eligible opponent on the same alternating-side seeds."""

    attacker_checkpoint = Path(attacker_checkpoint)
    defender_checkpoint = Path(defender_checkpoint)
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("evaluation seeds must be nonempty and unique")
    matches = []
    for spec in pool.eligible(max_difficulty=max_difficulty):
        preset = get_preset(spec.preset)
        for index, seed in enumerate(seeds):
            start_side = Side.ATTACKER if index % 2 == 0 else Side.DEFENDER
            result = match_runner(
                coach_team_ai=build_coach_v1_team(
                    attacker_checkpoint=attacker_checkpoint,
                    defender_checkpoint=defender_checkpoint,
                    device=device,
                ),
                opponent_team_ai=build_opponent_team(spec, device=device),
                opponent_roster=preset.players,
                opponent_spike_holder=preset.spike_holder,
                opponent_igl=preset.igl,
                coach_starts_as=start_side,
                seed=seed,
                opponent_team_name=spec.opponent_id,
                allow_mirrored_roster=spec.mirrored_roster,
            )
            matches.append(_record(spec, result))
    provisional = PoolEvaluation(
        attacker_checkpoint=str(attacker_checkpoint),
        attacker_sha256=_sha256(attacker_checkpoint),
        defender_checkpoint=str(defender_checkpoint),
        defender_sha256=_sha256(defender_checkpoint),
        matches=tuple(matches),
        recommended_weights={},
    )
    rates = {
        opponent_id: float(values["round_point_rate"])
        for opponent_id, values in provisional.aggregates().items()
    }
    adjusted = pool.with_weights(rates)
    return PoolEvaluation(
        attacker_checkpoint=provisional.attacker_checkpoint,
        attacker_sha256=provisional.attacker_sha256,
        defender_checkpoint=provisional.defender_checkpoint,
        defender_sha256=provisional.defender_sha256,
        matches=provisional.matches,
        recommended_weights={item.opponent_id: item.weight
                             for item in adjusted.opponents},
    )


def decide_promotion(
    candidate: PoolEvaluation,
    incumbent: PoolEvaluation,
    *,
    historical_opponents: Sequence[str],
    criteria: PromotionCriteria = PromotionCriteria(),
) -> PromotionDecision:
    """Reject regressions using paired schedules and per-opponent checks."""

    candidate_keys = tuple(
        (item.opponent_id, item.seed, item.coach_started_as)
        for item in candidate.matches
    )
    incumbent_keys = tuple(
        (item.opponent_id, item.seed, item.coach_started_as)
        for item in incumbent.matches
    )
    if candidate_keys != incumbent_keys:
        raise ValueError("candidate and incumbent evaluations must use one paired schedule")
    candidate_by = candidate.aggregates()
    incumbent_by = incumbent.aggregates()
    reasons = []
    for opponent_id, candidate_metrics in candidate_by.items():
        if int(candidate_metrics["matches"]) < criteria.minimum_matches_per_opponent:
            reasons.append(f"{opponent_id}: insufficient matches")
        drop = (float(incumbent_by[opponent_id]["round_point_rate"])
                - float(candidate_metrics["round_point_rate"]))
        if drop > criteria.maximum_opponent_point_rate_drop:
            reasons.append(f"{opponent_id}: round point rate regressed by {drop:.3f}")
    candidate_overall = candidate.overall()
    incumbent_overall = incumbent.overall()
    overall_drop = (float(incumbent_overall["round_point_rate"])
                    - float(candidate_overall["round_point_rate"]))
    if overall_drop > criteria.maximum_overall_point_rate_drop:
        reasons.append(f"overall round point rate regressed by {overall_drop:.3f}")
    if float(candidate_overall["win_rate"]) < float(incumbent_overall["win_rate"]):
        reasons.append("overall match win rate regressed")
    for opponent_id in historical_opponents:
        if opponent_id not in candidate_by:
            reasons.append(f"missing historical opponent: {opponent_id}")
        elif (float(candidate_by[opponent_id]["win_rate"])
              < criteria.minimum_historical_win_rate):
            reasons.append(f"{opponent_id}: historical win rate below threshold")
    return PromotionDecision(
        promoted=not reasons,
        reasons=tuple(reasons) if reasons else ("all promotion gates passed",),
        candidate_overall=candidate_overall,
        incumbent_overall=incumbent_overall,
    )
