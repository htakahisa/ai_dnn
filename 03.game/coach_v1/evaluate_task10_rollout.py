"""Reproducible real-game smoke test of Task 09 character checkpoints.

This evaluates a fixed STAY coach in the actual game loop. It measures
runtime application and sightline coverage, not supervised facing accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Mapping

from map_data import NEW_MAZE_STR
from run_game import VisualFPSBattle, _build_team_ai
from team_ai import DualRoleTeamAI

from coach_v1.common.constants import CHECKPOINTS_DIR, FIXED_ROSTER
from coach_v1.common.types import Facing, MovementAction, ObjectiveAction, Side, TacticalIntent
from coach_v1.common.constants import FACING_DELTAS
from coach_v1.coordinator import TeamExecutionCoordinator
from coach_v1.learning_character_base import CharacterPolicy
from coach_v1.observation.character_encoder import CoachInstruction


class StayCoach:
    def __init__(self):
        self.calls = 0

    def act(self, observation):
        self.calls += 1
        return (CoachInstruction(MovementAction.STAY, ObjectiveAction.NONE,
                                 TacticalIntent.HOLD),) * 5


def _charges(character):
    return sum(int(getattr(character, name, 0)) for name in
               ("smoke_charges", "flash_charges", "recon_charges"))


def _front_los(game, character) -> tuple[bool, bool]:
    """Evaluator-only ground truth; never passed to either actor."""
    origin = tuple(character.pos)
    facing = FACING_DELTAS[Facing(character.facing)]
    has_los = False
    covered = False
    for enemy in game.chars:
        if enemy.team == character.team or not enemy.is_alive:
            continue
        target = tuple(enemy.pos)
        if not game.check_cell_line_of_sight(origin, target, block_smoke=True):
            continue
        has_los = True
        delta = target[0] - origin[0], target[1] - origin[1]
        if delta[0] * facing[0] + delta[1] * facing[1] >= 0:
            covered = True
    return has_los, covered


def _aligned_with_shared_sighting(character, sightings) -> bool:
    origin = tuple(character.pos)
    dr, dc = FACING_DELTAS[Facing(character.facing)]
    for sighting in sightings:
        target_dr = sighting.reported_position[0] - origin[0]
        target_dc = sighting.reported_position[1] - origin[1]
        distance = math.hypot(target_dr, target_dc)
        if distance == 0 or (dr * target_dr + dc * target_dc) / (
            math.hypot(dr, dc) * distance
        ) >= math.cos(math.pi / 4):
            return True
    return False


def evaluate(side: Side, *, ticks: int, seed: int, near: bool = False,
             variant: str = "best",
             checkpoint_overrides: Mapping[int, Path] | None = None,
             trace_slot: int | None = None) -> dict:
    random.seed(seed)
    coach = StayCoach()
    if variant not in {"best", "facing_best"}:
        raise ValueError("unknown checkpoint variant")
    def policy(slot):
        if checkpoint_overrides and slot in checkpoint_overrides:
            return CharacterPolicy(slot, checkpoint_overrides[slot])
        if variant == "best":
            return CharacterPolicy(slot)
        path = (CHECKPOINTS_DIR / "experiments" / "task09_data360"
                / FIXED_ROSTER[slot].checkpoint_id / "facing_best.pt")
        return CharacterPolicy(slot, path)
    coordinator = TeamExecutionCoordinator(
        side, coach, {slot: policy(slot) for slot in range(5)},
    )
    team = DualRoleTeamAI(
        "coach_v1_eval", lambda: coordinator, lambda: coordinator,
    )
    opponent = _build_team_ai("default")
    roster = [item.character_name for item in FIXED_ROSTER]
    game = VisualFPSBattle(
        NEW_MAZE_STR,
        team if side is Side.ATTACKER else opponent,
        team if side is Side.DEFENDER else opponent,
        headless=True,
        attacker_roster=roster if side is Side.ATTACKER else None,
        defender_roster=roster if side is Side.DEFENDER else None,
        disable_side_swap=True,
    )
    team_code = "A" if side is Side.ATTACKER else "D"
    effects = {"flash_enemy_hits": 0, "recon_enemy_hits": 0}
    original_record = game.analytics_tracker.record_contribution
    def record_contribution(assister, victim, tick, method):
        if (assister is not None and victim is not None
                and assister.team == team_code and victim.team != team_code
                and method in ("flash", "recon")):
            effects[f"{method}_enemy_hits"] += 1
        return original_record(assister, victim, tick, method)
    game.analytics_tracker.record_contribution = record_contribution
    initial_round = game.current_round
    while game.defender_setup_phase.active:
        game._run_defender_setup_tick()
    if near:
        # Evaluation-only encounter: place opponents on legal, open cells
        # close to the fixed roster. Actors still see them only via sensors.
        enemy_row = 21 if side is Side.ATTACKER else 2
        opponents = [char for char in game.chars if char.team != team_code]
        for index, opponent_char in enumerate(opponents):
            opponent_char.pos = [enemy_row, 18 + index]
    setup_coach_calls = coach.calls
    applied = 0
    facing_actions = 0
    forced_facing_actions = 0
    unforced_applied = 0
    ability_requests = 0
    ability_successes = 0
    ability_activated_by_type = {"SMOKE": 0, "FLASH": 0, "RECON": 0}
    sightline_opportunities = 0
    sightline_covered = 0
    sighting_actions = 0
    sighting_aligned_45 = 0
    unforced_sighting_actions = 0
    unforced_aligned_45 = 0
    per_slot = {str(slot): {"actions": 0, "sighting_actions": 0,
                           "aligned_45": 0, "unforced_sighting_actions": 0,
                           "unforced_aligned_45": 0, "ability_requests": 0,
                           "ability_successes": 0} for slot in range(5)}
    sighting_trace = []
    tick_count = 0
    for _ in range(ticks):
        if game.current_round != initial_round or game.match_over:
            break
        game._build_occupancy_counts()
        try:
            for character in game._move_order():
                if not character.is_alive:
                    continue
                before_log = len(coordinator.action_log)
                charges_before = _charges(character)
                game.move_character(character)
                if character.team != team_code or len(coordinator.action_log) == before_log:
                    continue
                action = coordinator.action_log[-1]
                facing_actions += 1
                forced = bool(getattr(character, "facing_forced_this_tick", False))
                forced_facing_actions += forced
                applied += character.facing == action.facing
                unforced_applied += not forced and character.facing == action.facing
                slot_metrics = per_slot[str(action.slot)]
                slot_metrics["actions"] += 1
                sightings = coordinator._snapshot.sightings
                if sightings:
                    sighting_actions += 1
                    slot_metrics["sighting_actions"] += 1
                    aligned = _aligned_with_shared_sighting(character, sightings)
                    if action.slot == trace_slot:
                        origin = tuple(int(value) for value in character.pos)
                        sighting_trace.append({
                            "tick": tick_count,
                            "position": origin,
                            "facing": str(character.facing),
                            "requested_facing": str(action.facing),
                            "forced": forced,
                            "aligned_45": aligned,
                            "reported_offsets": [
                                (int(item.reported_position[0]) - origin[0],
                                 int(item.reported_position[1]) - origin[1])
                                for item in sightings
                            ],
                        })
                    sighting_aligned_45 += aligned
                    slot_metrics["aligned_45"] += aligned
                    if not forced:
                        unforced_sighting_actions += 1
                        unforced_aligned_45 += aligned
                        slot_metrics["unforced_sighting_actions"] += 1
                        slot_metrics["unforced_aligned_45"] += aligned
                opportunity, covered = _front_los(game, character)
                sightline_opportunities += opportunity
                sightline_covered += covered
                if action.action == "ABILITY":
                    ability_requests += 1
                    slot_metrics["ability_requests"] += 1
                    succeeded = _charges(character) < charges_before
                    ability_successes += succeeded
                    slot_metrics["ability_successes"] += succeeded
                    if succeeded:
                        ability_activated_by_type[action.ability] += 1
        finally:
            game._clear_occupancy_counts()
        game.process_battle()
        tick_count += 1
    return {
        "side": side.value,
        "seed": seed,
        "checkpoint_variant": variant if not checkpoint_overrides else "custom",
        "coach": "fixed STAY/HOLD dummy",
        "opponent": "default",
        "encounter": "near" if near else "natural",
        "live_ticks": tick_count,
        "setup_coach_calls": setup_coach_calls,
        "live_coach_calls": coach.calls - setup_coach_calls,
        "logged_actions": len(coordinator.action_log),
        "facing_applied": applied,
        "facing_actions": facing_actions,
        "forced_facing_actions": forced_facing_actions,
        "unforced_applied": unforced_applied,
        "ability_requests": ability_requests,
        "ability_successes": ability_successes,
        "ability_activated_by_type": ability_activated_by_type,
        **effects,
        "sightline_opportunities": sightline_opportunities,
        "sightline_covered": sightline_covered,
        "sighting_actions": sighting_actions,
        "sighting_aligned_45": sighting_aligned_45,
        "unforced_sighting_actions": unforced_sighting_actions,
        "unforced_aligned_45": unforced_aligned_45,
        "per_slot": per_slot,
        **({"sighting_trace": sighting_trace} if trace_slot is not None else {}),
        "round_finished": game.current_round != initial_round,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--near", action="store_true")
    parser.add_argument("--variant", choices=("best", "facing_best"), default="best")
    parser.add_argument("--gorimaru-checkpoint", type=Path)
    parser.add_argument("--output")
    args = parser.parse_args()
    overrides = {0: args.gorimaru_checkpoint} if args.gorimaru_checkpoint else None
    results = [evaluate(side, ticks=args.ticks, seed=seed, near=args.near,
                        variant=args.variant, checkpoint_overrides=overrides)
               for seed in range(args.seed, args.seed + args.seeds)
               for side in (Side.ATTACKER, Side.DEFENDER)]
    encoded = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
