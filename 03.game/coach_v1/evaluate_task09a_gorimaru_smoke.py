"""Evaluate Gorimaru smoke in real games without changing actor inputs.

The evaluator temporarily removes only Gorimaru's smoke while querying the
game's existing shot LOS function, then restores the original smoke list.
This is a same-state shot-lane comparison, not a counterfactual match result.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from game_core import SHOOTING_SITE_DIGREE
from map_data import NEW_MAZE_STR
from run_game import VisualFPSBattle, _build_team_ai
from team_ai import DualRoleTeamAI

from coach_v1.common.constants import CHECKPOINTS_DIR, FIXED_ROSTER, REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.coordinator import TeamExecutionCoordinator
from coach_v1.evaluate_task10_rollout import StayCoach
from coach_v1.learning_character_base import CharacterPolicy
from coach_v1.training.gorimaru_rollout import _stage_encounter


CHECKPOINTS = {
    "prior_official": (CHECKPOINTS_DIR / "experiments" / "task09a_gorimaru_rollout"
                       / "epoch70_latest.pt"),
    "selected": (CHECKPOINTS_DIR / "experiments" / "task09a_gorimaru_attacker_adjustment"
                 / "near_priority" / "epoch90_latest.pt"),
}
ENCOUNTERS = ("near", "west", "east", "crossfire", "natural")


class _RecordedGorimaru:
    def __init__(self, checkpoint: Path) -> None:
        self.policy = CharacterPolicy(0, checkpoint)
        self.observation = None

    def act(self, observation):
        self.observation = observation
        return self.policy.act(observation)


def _smoke_cells_at(game, target: tuple[int, int]) -> set[tuple[int, int]]:
    row, column = target
    return {(r, c) for r in range(row - 1, row + 2)
            for c in range(column - 1, column + 2)
            if 0 <= r < game.height and 0 <= c < game.width
            and game.grid[r, c] != 1}


def _shot_lanes(game, own_team: str) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Potential directed human shot lanes at this instant, by shooter's team."""
    allies = set()
    enemies = set()
    alive = [character for character in game.chars if character.is_alive]
    for shooter in alive:
        if (getattr(shooter, "plant_timer", 0) > 0
                or getattr(shooter, "defuse_timer", 0) > 0
                or getattr(shooter, "collecting_orb_this_tick", False)):
            continue
        for target in alive:
            if shooter.team == target.team:
                continue
            if game._facing_angle_diff(shooter, target) > SHOOTING_SITE_DIGREE:
                continue
            if not game.check_shot_line_of_sight(shooter, target):
                continue
            pair = id(shooter), id(target)
            (allies if shooter.team == own_team else enemies).add(pair)
    return allies, enemies


def marginal_shot_lanes(game, own_team: str, smoke: dict) -> dict[str, int]:
    """Count lanes removed by this smoke, preserving all other live smokes."""
    if not any(item is smoke for item in game.smokes):
        raise ValueError("evaluated smoke is not active")
    actual_smokes = game.smokes
    with_ally, with_enemy = _shot_lanes(game, own_team)
    try:
        game.smokes = [item for item in actual_smokes if item is not smoke]
        without_ally, without_enemy = _shot_lanes(game, own_team)
    finally:
        game.smokes = actual_smokes
    if not with_ally <= without_ally or not with_enemy <= without_enemy:
        raise AssertionError("adding smoke unexpectedly opened a shot lane")
    return {
        "ally_lanes_without": len(without_ally),
        "enemy_lanes_without": len(without_enemy),
        "ally_lanes_blocked": len(without_ally - with_ally),
        "enemy_lanes_blocked": len(without_enemy - with_enemy),
    }


def _new_game(side: Side, checkpoint: Path, seed: int, encounter: str):
    random.seed(seed)
    coach = StayCoach()
    recorder = _RecordedGorimaru(checkpoint)
    coordinator = TeamExecutionCoordinator(
        side, coach,
        {slot: recorder if slot == 0 else CharacterPolicy(slot)
         for slot in range(5)},
    )
    team = DualRoleTeamAI("coach_v1_smoke_eval", lambda: coordinator,
                          lambda: coordinator)
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
    while game.defender_setup_phase.active:
        game._run_defender_setup_tick()
    own_team = "A" if side is Side.ATTACKER else "D"
    if encounter == "near":
        row = 21 if side is Side.ATTACKER else 2
        for index, enemy in enumerate(c for c in game.chars if c.team != own_team):
            enemy.pos = [row, 18 + index]
    elif encounter in {"west", "east", "crossfire"}:
        _stage_encounter(game, own_team, encounter)
    elif encounter != "natural":
        raise ValueError(f"unknown encounter: {encounter}")
    return game, coordinator, own_team, recorder


def evaluate_smoke(*, side: Side, seed: int, encounter: str,
                   checkpoint: Path, ticks: int = 20,
                   shadow_checkpoint: Path | None = None) -> dict:
    """Run one real round; evaluative truth is never given to an actor."""
    game, coordinator, own_team, recorder = _new_game(side, checkpoint, seed, encounter)
    shadow_policy = (CharacterPolicy(0, shadow_checkpoint)
                     if shadow_checkpoint is not None else None)
    initial_round = game.current_round
    requests = successes = 0
    placement = None
    smoke = None
    pair_ticks = {key: 0 for key in (
        "ally_lanes_without", "enemy_lanes_without",
        "ally_lanes_blocked", "enemy_lanes_blocked",
    )}
    active_ticks = 0
    shadow_smoke = None
    shadow_pair_ticks = pair_ticks.copy()
    for _ in range(ticks):
        if game.current_round != initial_round or game.match_over:
            break
        game._build_occupancy_counts()
        try:
            for character in game._move_order():
                if not character.is_alive:
                    continue
                before_log = len(coordinator.action_log)
                charges_before = int(getattr(character, "smoke_charges", 0))
                game.move_character(character)
                if (character.team != own_team or len(coordinator.action_log) == before_log
                        or coordinator.action_log[-1].slot != 0):
                    continue
                action = coordinator.action_log[-1]
                if action.action != "ABILITY" or action.ability != "SMOKE":
                    continue
                requests += 1
                if int(character.smoke_charges) >= charges_before:
                    continue
                successes += 1
                smoke = next(item for item in reversed(game.smokes)
                             if item.get("owner") == character.name)
                cells = set(smoke["cells"])
                shadow_action = (shadow_policy.act(recorder.observation)
                                 if shadow_policy is not None else None)
                placement = {
                    "tick": int(game.battle_tick),
                    "target": tuple(map(int, action.ability_target)),
                    "cell_count": len(cells),
                    "allies_in_smoke": sum(c.is_alive and c.team == own_team
                                           and tuple(c.pos) in cells for c in game.chars),
                    "enemies_in_smoke": sum(c.is_alive and c.team != own_team
                                            and tuple(c.pos) in cells for c in game.chars),
                    "legal_shared_sightings": len(coordinator._snapshot.sightings),
                    "marginal_shot_lanes": marginal_shot_lanes(game, own_team, smoke),
                }
                if shadow_action is not None:
                    placement["shadow_requested_smoke"] = bool(shadow_action.use_ability)
                    if shadow_action.use_ability and shadow_action.target is not None:
                        shadow_target = tuple(map(int, shadow_action.target))
                        shadow_smoke = {
                            "cells": _smoke_cells_at(game, shadow_target),
                            "remaining_ticks": smoke["remaining_ticks"],
                            "owner": character.name,
                        }
                        placement["shadow_target"] = shadow_target
                        actual_smokes = game.smokes
                        try:
                            game.smokes = [item for item in actual_smokes if item is not smoke]
                            game.smokes.append(shadow_smoke)
                            placement["shadow_marginal_shot_lanes"] = marginal_shot_lanes(
                                game, own_team, shadow_smoke)
                        finally:
                            game.smokes = actual_smokes
        finally:
            game._clear_occupancy_counts()
        if smoke is not None and any(item is smoke for item in game.smokes):
            impact = marginal_shot_lanes(game, own_team, smoke)
            for key, value in impact.items():
                pair_ticks[key] += value
            if shadow_smoke is not None:
                actual_smokes = game.smokes
                try:
                    game.smokes = [item for item in actual_smokes if item is not smoke]
                    game.smokes.append(shadow_smoke)
                    shadow_impact = marginal_shot_lanes(game, own_team, shadow_smoke)
                finally:
                    game.smokes = actual_smokes
                for key, value in shadow_impact.items():
                    shadow_pair_ticks[key] += value
            active_ticks += 1
        game.process_battle()
    return {
        "side": side.value, "seed": seed, "encounter": encounter,
        "live_ticks": int(game.battle_tick),
        "smoke_requests": requests, "smoke_successes": successes,
        "placement": placement,
        "smoke_active_evaluated_ticks": active_ticks,
        "marginal_shot_lane_pair_ticks": pair_ticks,
        **({"shadow_marginal_shot_lane_pair_ticks": shadow_pair_ticks}
           if shadow_policy is not None else {}),
    }


def summarize(games: list[dict]) -> dict:
    keys = ("ally_lanes_without", "enemy_lanes_without",
            "ally_lanes_blocked", "enemy_lanes_blocked")
    totals = {key: sum(game["marginal_shot_lane_pair_ticks"][key] for game in games)
              for key in keys}
    summary = {
        "rounds": len(games),
        "smoke_requests": sum(game["smoke_requests"] for game in games),
        "smoke_successes": sum(game["smoke_successes"] for game in games),
        "rounds_with_immediate_enemy_lane_block": sum(
            bool(game["placement"] and game["placement"]["marginal_shot_lanes"][
                "enemy_lanes_blocked"])
            for game in games),
        "rounds_with_immediate_ally_lane_block": sum(
            bool(game["placement"] and game["placement"]["marginal_shot_lanes"][
                "ally_lanes_blocked"])
            for game in games),
        "smoke_active_evaluated_ticks": sum(game["smoke_active_evaluated_ticks"]
                                            for game in games),
        "allies_inside_at_throw": sum(game["placement"]["allies_in_smoke"]
                                      for game in games if game["placement"]),
        "enemies_inside_at_throw": sum(game["placement"]["enemies_in_smoke"]
                                       for game in games if game["placement"]),
        **totals,
    }
    if any("shadow_marginal_shot_lane_pair_ticks" in game for game in games):
        summary["paired_shadow_uses"] = sum(
            bool(game["placement"] and game["placement"].get("shadow_target") is not None)
            for game in games)
        summary["paired_different_targets"] = sum(
            bool(game["placement"] and game["placement"].get("shadow_target") is not None
                 and tuple(game["placement"]["target"]) !=
                 tuple(game["placement"]["shadow_target"]))
            for game in games)
        summary["shadow_pair_ticks"] = {
            key: sum(game["shadow_marginal_shot_lane_pair_ticks"][key]
                     for game in games if "shadow_marginal_shot_lane_pair_ticks" in game)
            for key in keys
        }
    return summary


def run(*, start: int, stop: int, ticks: int,
        encounters: tuple[str, ...], output: Path,
        shadow_prior: bool = False) -> dict:
    torch.set_num_threads(1)
    result = {"seeds": [start, stop - 1], "ticks_per_round": ticks,
              "encounters": encounters, "models": {}}
    for name, checkpoint in CHECKPOINTS.items():
        if shadow_prior and name != "selected":
            continue
        games = [evaluate_smoke(side=side, seed=seed, encounter=encounter,
                                checkpoint=checkpoint, ticks=ticks,
                                shadow_checkpoint=(CHECKPOINTS["prior_official"]
                                                   if shadow_prior else None))
                 for seed in range(start, stop)
                 for side in (Side.ATTACKER, Side.DEFENDER)
                 for encounter in encounters]
        result["models"][name] = {
            "checkpoint": str(checkpoint), "all": summarize(games),
            "by_side_encounter": {
                f"{side.value}_{encounter}": summarize([
                    game for game in games if game["side"] == side.value
                    and game["encounter"] == encounter
                ])
                for side in (Side.ATTACKER, Side.DEFENDER)
                for encounter in encounters
            },
            "games": games,
        }
        print(name, result["models"][name]["all"], flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=60)
    parser.add_argument("--stop", type=int, default=70)
    parser.add_argument("--ticks", type=int, default=20)
    parser.add_argument("--encounter", choices=ENCOUNTERS, action="append")
    parser.add_argument("--shadow-prior", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=REPORTS_DIR / "task09a_gorimaru_smoke.json")
    args = parser.parse_args()
    run(start=args.start, stop=args.stop, ticks=args.ticks,
        encounters=tuple(args.encounter) if args.encounter else ENCOUNTERS,
        output=args.output, shadow_prior=args.shadow_prior)
