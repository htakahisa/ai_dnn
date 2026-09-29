"""Headless full-match execution and integration auditing for coach_v1."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np

from coach_v1.common.constants import FIXED_ROSTER_NAMES
from coach_v1.common.types import Side
from coach_v1.coordinator import TeamExecutionCoordinator
from map_data import NEW_MAZE_STR


@dataclass(frozen=True)
class FullMatchSummary:
    seed: int
    coach_started_as: str
    coach_score: int
    opponent_score: int
    winner: str
    rounds: int
    overtime: bool
    replay_frames: int
    round_reasons: dict[str, int]
    planted_rounds: int
    defused_rounds: int
    detonation_rounds: int
    elimination_rounds: int
    timeout_rounds: int
    setup_rounds: int
    live_rounds: int
    attacker_decisions: int
    defender_decisions: int
    attacker_actions: int
    defender_actions: int
    memory_reset_ok: bool
    side_checkpoint_switch_ok: bool
    replay_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FullMatchResult:
    summary: FullMatchSummary
    replay: tuple[dict[str, Any], ...]
    round_records: tuple[dict[str, Any], ...]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_headless_full_match(
    *,
    coach_team_ai,
    opponent_team_ai,
    opponent_roster: Iterable[str],
    opponent_spike_holder: str,
    opponent_igl: str,
    coach_starts_as: Side = Side.ATTACKER,
    seed: int = 14,
    coach_team_name: str = "coach_v1",
    opponent_team_name: str = "opponent",
) -> FullMatchResult:
    """Run one complete map through the unmodified game lifecycle."""

    if not isinstance(coach_starts_as, Side):
        raise ValueError("coach_starts_as must be attacker or defender")
    opponent_roster = tuple(str(name) for name in opponent_roster)
    if len(opponent_roster) != len(FIXED_ROSTER_NAMES):
        raise ValueError("opponent roster must contain five players")
    if len(set(opponent_roster)) != len(opponent_roster):
        raise ValueError("opponent roster contains duplicate players")
    if set(opponent_roster) & set(FIXED_ROSTER_NAMES):
        raise ValueError("opponent roster must not overlap the fixed Gorigons roster")
    if opponent_spike_holder not in opponent_roster or opponent_igl not in opponent_roster:
        raise ValueError("opponent spike holder and IGL must belong to its roster")

    # Importing here keeps checkpoint-only users independent of the Tk entry
    # point until they explicitly request a match.
    from run_game import VisualFPSBattle

    seed_all(seed)
    coach_roster = tuple(FIXED_ROSTER_NAMES)
    coach_holder = "ごんた"
    coach_igl = "ごりまる"
    if coach_starts_as is Side.ATTACKER:
        attacker_ai, defender_ai = coach_team_ai, opponent_team_ai
        attacker_roster, defender_roster = coach_roster, opponent_roster
        attacker_holder, defender_holder = coach_holder, opponent_spike_holder
        attacker_igl, defender_igl = coach_igl, opponent_igl
        attacker_name, defender_name = coach_team_name, opponent_team_name
    else:
        attacker_ai, defender_ai = opponent_team_ai, coach_team_ai
        attacker_roster, defender_roster = opponent_roster, coach_roster
        attacker_holder, defender_holder = opponent_spike_holder, coach_holder
        attacker_igl, defender_igl = opponent_igl, coach_igl
        attacker_name, defender_name = opponent_team_name, coach_team_name

    game = VisualFPSBattle(
        NEW_MAZE_STR,
        attacker_ai,
        defender_ai,
        headless=True,
        attacker_roster=list(attacker_roster),
        defender_roster=list(defender_roster),
        spike_holder_name=attacker_holder,
        defender_spike_holder_name=defender_holder,
        attacker_igl_name=attacker_igl,
        defender_igl_name=defender_igl,
        attacker_team_name=attacker_name,
        defender_team_name=defender_name,
        disable_side_swap=False,
    )
    game.run()
    return audit_completed_match(
        game,
        coach_team_ai=coach_team_ai,
        coach_started_as=coach_starts_as,
        seed=seed,
        coach_team_name=coach_team_name,
        opponent_team_name=opponent_team_name,
    )


def audit_completed_match(
    game,
    *,
    coach_team_ai,
    coach_started_as: Side,
    seed: int,
    coach_team_name: str,
    opponent_team_name: str,
) -> FullMatchResult:
    """Validate side switching, round memory, terminal state, and replay."""

    if not bool(getattr(game, "match_over", False)):
        raise RuntimeError("headless match did not reach its terminal state")
    replay = tuple(getattr(game, "replay_frames", ()))
    records = tuple(getattr(game.analytics_tracker, "round_records", ()))
    if not replay or not records:
        raise RuntimeError("completed match is missing replay or round records")

    current_attackers = {str(name) for name in (game.attacker_roster or ())}
    coach_is_attacker = current_attackers == set(FIXED_ROSTER_NAMES)
    coach_score = int(game.attacker_wins if coach_is_attacker else game.defender_wins)
    opponent_score = int(game.defender_wins if coach_is_attacker else game.attacker_wins)
    reasons = Counter(str(record.get("reason", "unknown")) for record in records)
    setup_rounds = len({int(frame["round"]) for frame in replay if frame.get("setup")})
    live_rounds = len({int(frame["round"]) for frame in replay if not frame.get("setup")})
    replay_ok = _validate_replay(replay, len(records))

    attacker = getattr(coach_team_ai, "_attacker_controller", None)
    defender = getattr(coach_team_ai, "_defender_controller", None)
    coordinators = tuple(
        item for item in (attacker, defender)
        if isinstance(item, TeamExecutionCoordinator)
    )
    audits = tuple(audit for item in coordinators for audit in item.decision_audit)
    first_by_round_side = {}
    for audit in audits:
        first_by_round_side.setdefault((audit.round_number, audit.side), audit)
    memory_reset_ok = bool(first_by_round_side) and all(
        audit.memory_tick == 0 for audit in first_by_round_side.values()
    )
    side_checkpoint_switch_ok = (
        isinstance(attacker, TeamExecutionCoordinator)
        and attacker.side is Side.ATTACKER
        and isinstance(defender, TeamExecutionCoordinator)
        and defender.side is Side.DEFENDER
        and bool(attacker.decision_audit)
        and bool(defender.decision_audit)
    )
    summary = FullMatchSummary(
        seed=int(seed),
        coach_started_as=coach_started_as.value,
        coach_score=coach_score,
        opponent_score=opponent_score,
        winner=coach_team_name if coach_score > opponent_score else opponent_team_name,
        rounds=len(records),
        overtime=bool(getattr(game, "overtime", False)),
        replay_frames=len(replay),
        round_reasons=dict(sorted(reasons.items())),
        planted_rounds=sum(bool(record.get("planted")) for record in records),
        defused_rounds=reasons["defused"],
        detonation_rounds=reasons["detonated"],
        elimination_rounds=(reasons["attacker_wipe"] + reasons["defender_wipe"]),
        timeout_rounds=reasons["time_expired"],
        setup_rounds=setup_rounds,
        live_rounds=live_rounds,
        attacker_decisions=(len(attacker.decision_audit)
                            if isinstance(attacker, TeamExecutionCoordinator) else 0),
        defender_decisions=(len(defender.decision_audit)
                            if isinstance(defender, TeamExecutionCoordinator) else 0),
        attacker_actions=(len(attacker.action_log)
                          if isinstance(attacker, TeamExecutionCoordinator) else 0),
        defender_actions=(len(defender.action_log)
                          if isinstance(defender, TeamExecutionCoordinator) else 0),
        memory_reset_ok=memory_reset_ok,
        side_checkpoint_switch_ok=side_checkpoint_switch_ok,
        replay_ok=replay_ok,
    )
    if not (summary.memory_reset_ok and summary.side_checkpoint_switch_ok
            and summary.replay_ok):
        raise RuntimeError(f"coach_v1 full-match audit failed: {summary}")
    return FullMatchResult(summary, replay, records)


def _validate_replay(replay: tuple[dict[str, Any], ...], rounds: int) -> bool:
    seen_rounds = {int(frame.get("round", 0)) for frame in replay}
    if seen_rounds != set(range(1, rounds + 1)):
        return False
    for frame in replay:
        chars = frame.get("chars")
        if not isinstance(chars, list) or len(chars) != 10:
            return False
        for char in chars:
            visible_to = char.get("visible_to")
            if not isinstance(visible_to, list) or any(
                team not in {"A", "D"} for team in visible_to
            ):
                return False
    return True


def write_match_artifacts(
    result: FullMatchResult,
    *,
    report_path: Path,
    replay_path: Path | None = None,
) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = result.summary.to_dict()
    report["round_records"] = list(result.round_records)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if replay_path is not None:
        replay_path = Path(replay_path)
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(
            json.dumps(list(result.replay), ensure_ascii=False), encoding="utf-8"
        )
