"""Paired fixed-checkpoint comparisons and diagnostic feature removal.

These interventions run only in the evaluator. They do not change the game,
training distribution, checkpoint, or production inference path.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from party_presets import get_preset

from coach_v1.ability_effect_audit import summarize_ability_events
from coach_v1.common.constants import (CHARACTER_CHECKPOINT_IDS, FIXED_ROSTER, REPORTS_DIR,
                                       WATCH_POINTS_CONFIG_PATH)
from coach_v1.common.types import Facing, Side, TacticalIntent
from coach_v1.full_match import run_headless_full_match
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.observation.coach_encoder import COACH_GRID_CHANNELS
from coach_v1.opponent_pool import OpponentKind, load_opponent_pool
from coach_v1.perception.belief_memory import BeliefMemory
from coach_v1.perception.team_perception import TeamPerceptionBuilder
from coach_v1.common.watch_points import load_watch_points
from coach_v1.task15_self_play import build_opponent_team
from coach_v1.team_ai import build_coach_v1_team
from coach_v1.training.character_environment import CharacterAction
from map_data import NEW_MAZE_STR


BASELINE = "baseline"
PROBES = ("no_belief_memory", "no_watch_point_input", "no_tactical_intent",
          "no_character_models", "no_recurrent_state", "no_team_shared_vision")
TRAINING_ONLY = "no_random_placement"
CHARACTER_ROOT = (Path(__file__).resolve().parent / "checkpoints" /
                  "experiments" / "task16_watch_added" / "characters")


def _erase_channels(encoder, names):
    original = encoder.encode
    indices = tuple(COACH_GRID_CHANNELS.index(name) for name in names)

    def encode(*args, **kwargs):
        observation = original(*args, **kwargs)
        grid = observation.grid.copy()
        grid[list(indices)] = 0
        grid.setflags(write=False)
        return replace(observation, grid=grid)

    encoder.encode = encode


class SingleViewerSensor(TeamPerceptionBuilder):
    """Evaluation sensor restricted to one legally observing roster slot."""

    def __init__(self, slot: int):
        super().__init__()
        self.slot = slot

    def _visible_cells(self, *, viewer, **kwargs):
        if viewer.slot != self.slot:
            return frozenset()
        return super()._visible_cells(viewer=viewer, **kwargs)

    def _sighting_for(self, *, viewers, **kwargs):
        own = tuple(viewer for viewer in viewers if viewer.slot == self.slot)
        return super()._sighting_for(viewers=own, **kwargs)


def configure_probe(team, variant: str) -> None:
    """Attach a probe to a fresh team before the match starts."""
    if variant not in (BASELINE, *PROBES):
        raise ValueError(f"unknown inference probe: {variant}")
    if variant == BASELINE:
        return
    for controller in (team.get_attacker_controller(), team.get_defender_controller()):
        if variant == "no_belief_memory":
            original = controller.memory.update

            def update(snapshot, *, memory=controller.memory, fn=original):
                memory.reset()
                return fn(snapshot)

            controller.memory.update = update
        elif variant == "no_watch_point_input":
            names = tuple(name for name in COACH_GRID_CHANNELS if name.startswith("watch_"))
            _erase_channels(controller.encoder, names)
            _erase_channels(controller.character_environment.encoder._coach_encoder, names)
        elif variant == "no_team_shared_vision":
            controller.sensor = SingleViewerSensor(0)
            sensors = {slot: SingleViewerSensor(slot) for slot in controller.characters}
            config = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
            memories = {slot: BeliefMemory(config.for_side(controller.side))
                        for slot in controller.characters}
            original_prepare = controller.character_environment.prepare

            def prepare(snapshot, belief, *, situation, slot, instruction,
                        sensors=sensors, memories=memories, fn=original_prepare,
                        owner=controller):
                own = sensors[slot].build(game=owner.game, side=owner.side)
                own_belief = memories[slot].update(own)
                return fn(own, own_belief, situation=situation, slot=slot,
                          instruction=instruction)

            controller.character_environment.prepare = prepare
        elif variant == "no_tactical_intent":
            original = controller.coach.act

            def act(observation, *, fn=original):
                return tuple(CoachInstruction(item.movement, item.objective,
                                              TacticalIntent.HOLD)
                             for item in fn(observation))

            controller.coach.act = act
        elif variant == "no_character_models":
            class NoCharacterModel:
                def act(self, observation):
                    return CharacterAction(Facing.N)

            controller.characters = {slot: NoCharacterModel()
                                     for slot in controller.characters}
        elif variant == "no_recurrent_state":
            original = controller.coach.act

            def act(observation, *, coach=controller.coach, fn=original):
                coach.hidden = None
                return fn(observation)

            controller.coach.act = act


def _coach_side_by_round(result):
    return {int(frame["round"]): frame["coach_side"] for frame in result.replay
            if "coach_side" in frame}


def match_metrics(result) -> dict:
    """Use referee/replay records only after a match; never feed them to actors."""
    summary = result.summary
    sides = _coach_side_by_round(result)
    records = result.round_records
    attacker_rounds = [r for r in records if sides.get(int(r["round_number"])) == "attacker"]
    defender_plants = [r for r in records if sides.get(int(r["round_number"])) == "defender"
                       and r.get("planted")]
    finals = {}
    angle_frames = 0
    live_frames = 0
    for frame in result.replay:
        if not frame.get("setup"):
            live_frames += 1
            own_team = "A" if frame.get("coach_side") == "attacker" else "D"
            alive = [char for char in frame["chars"] if char["team"] == own_team
                     and char["alive"]]
            if len({char["facing"] for char in alive}) >= 2:
                angle_frames += 1
        finals[int(frame["round"])] = frame
    survived = []
    for frame in finals.values():
        own_team = "A" if frame.get("coach_side") == "attacker" else "D"
        survived.append(sum(char["team"] == own_team and char["alive"]
                            for char in frame["chars"]))
    # Two teammates asking for one destination in the same tick is a potential
    # movement collision, independent of the game's sequential resolution.
    actions = {}
    for item in result.coach_actions:
        if item.action == "MOVE" and item.start != item.requested_position:
            key = (item.round_number, item.phase, item.tick, item.side.value)
            actions.setdefault(key, []).append(item.requested_position)
    collision_ticks = sum(len(positions) != len(set(positions))
                          for positions in actions.values())
    ages = [age for decision in result.coach_decisions
            for _, _, age in decision.watch_point_ages]
    frames_by_tick = {(int(frame["round"]), int(frame["tick"])): frame
                      for frame in result.replay if not frame.get("setup")}
    contested_entries = 0
    solo_entries = 0
    for item in result.coach_actions:
        if (item.side is not Side.ATTACKER or item.phase != "live"
                or item.action != "MOVE" or item.start == item.requested_position):
            continue
        frame = frames_by_tick.get((item.round_number, item.tick))
        if frame is None or frame.get("coach_side") != "attacker":
            continue
        enemies = [char for char in frame["chars"] if char["team"] == "D"
                   and char["alive"]]
        distance = lambda pos: abs(pos[0] - item.requested_position[0]) + abs(
            pos[1] - item.requested_position[1])
        if not enemies or min(distance(char["pos"]) for char in enemies) > 2:
            continue
        contested_entries += 1
        allies = [char for char in frame["chars"] if char["team"] == "A"
                  and char["alive"] and
                  str(char["name"]) != FIXED_ROSTER[item.slot].character_name]
        if not allies or min(distance(char["pos"]) for char in allies) > 3:
            solo_entries += 1
    ability = summarize_ability_events(result.ability_events)
    resolved = sum(value["resolved_casts"] for value in ability.values())
    effective = sum(value["effective_casts"] for value in ability.values())
    kills = result.kill_events
    trades = 0
    eligible_deaths = 0
    for death in kills:
        if death["victim_team"] != ("A" if sides.get(death["round"]) == "attacker" else "D"):
            continue
        eligible_deaths += 1
        if any(other["round"] == death["round"] and
               death["tick"] <= other["tick"] <= death["tick"] + 3 and
               other["killer_team"] == death["victim_team"] and
               other["victim_team"] == death["killer_team"]
               for other in kills):
            trades += 1
    return {
        "wins": int(summary.coach_score > summary.opponent_score),
        "coach_rounds_won": summary.coach_score,
        "rounds": summary.rounds,
        "attacker_rounds": len(attacker_rounds),
        "plants": sum(bool(r.get("planted")) for r in attacker_rounds),
        "defender_plants": len(defender_plants),
        "retakes": sum(r.get("winner") == "defender" for r in defender_plants),
        "survivors_sum": sum(survived), "survivor_rounds": len(survived),
        "multi_angle_frames": angle_frames, "live_frames": live_frames,
        "collision_ticks": collision_ticks, "movement_ticks": len(actions),
        "watch_age_sum": sum(ages), "watch_age_count": len(ages),
        "effective_abilities": effective, "resolved_abilities": resolved,
        "trades": trades, "eligible_deaths": eligible_deaths,
        "solo_entries": solo_entries, "contested_entries": contested_entries,
    }


def aggregate(records: list[dict]) -> dict:
    sums = {key: sum(item["metrics"][key] for item in records)
            for key in records[0]["metrics"]}

    def rate(num, den):
        return sums[num] / sums[den] if sums[den] else None

    return {
        "matches": len(records), "win_rate": sums["wins"] / len(records),
        "round_win_rate": rate("coach_rounds_won", "rounds"),
        "plant_rate": rate("plants", "attacker_rounds"),
        "retake_rate": rate("retakes", "defender_plants"),
        "average_survivors": rate("survivors_sum", "survivor_rounds"),
        "trade_rate": rate("trades", "eligible_deaths"),
        "ability_effective_rate": rate("effective_abilities", "resolved_abilities"),
        "multi_angle_rate": rate("multi_angle_frames", "live_frames"),
        "movement_collision_rate": rate("collision_ticks", "movement_ticks"),
        "mean_watch_confirmation_age": rate("watch_age_sum", "watch_age_count"),
        "solo_entry_rate": rate("solo_entries", "contested_entries"),
    }


def evaluate(*, seed: int, matches_per_opponent: int, variants: tuple[str, ...],
             opponents: tuple[str, ...], attacker_checkpoint: Path,
             defender_checkpoint: Path) -> dict:
    if matches_per_opponent < 2 or matches_per_opponent % 2:
        raise ValueError("matches_per_opponent must be even and at least two")
    if (not variants or len(set(variants)) != len(variants)
            or any(v not in (BASELINE, *PROBES) for v in variants)):
        raise ValueError("invalid variants")
    if not opponents or len(set(opponents)) != len(opponents):
        raise ValueError("opponents must be nonempty and unique")
    pool = load_opponent_pool()
    specs = [spec for spec in pool.opponents if spec.opponent_id in opponents
             and spec.kind is OpponentKind.EXISTING_AI]
    if len(specs) != len(opponents):
        raise ValueError("unknown or non-existing-AI opponent")
    records = []
    for variant in variants:
        for spec in specs:
            preset = get_preset(spec.preset)
            for offset in range(matches_per_opponent):
                team = build_coach_v1_team(
                    attacker_checkpoint=attacker_checkpoint,
                    defender_checkpoint=defender_checkpoint,
                    character_checkpoints={name: CHARACTER_ROOT / name / "best.pt"
                                           for name in CHARACTER_CHECKPOINT_IDS},
                    gongon_defender_checkpoint=(CHARACTER_ROOT / "gongon_defender" /
                                                "best.pt"),
                )
                configure_probe(team, variant)
                result = run_headless_full_match(
                    coach_team_ai=team, opponent_team_ai=build_opponent_team(spec),
                    opponent_roster=preset.players,
                    opponent_spike_holder=preset.spike_holder,
                    opponent_igl=preset.igl,
                    coach_starts_as=(Side.ATTACKER if offset % 2 == 0 else Side.DEFENDER),
                    seed=seed + offset, opponent_team_name=spec.opponent_id,
                    capture_referee_events=True, capture_ability_events=True,
                )
                records.append({"variant": variant, "opponent": spec.opponent_id,
                                "seed": seed + offset,
                                "started_as": result.summary.coach_started_as,
                                "metrics": match_metrics(result)})
    by_variant = {}
    for variant in variants:
        selected = [r for r in records if r["variant"] == variant]
        by_variant[variant] = {
            "overall": aggregate(selected),
            "opponents": {name: aggregate([r for r in selected if r["opponent"] == name])
                          for name in opponents},
        }
    return {
        "schema_version": "coach-task17-evaluation-v1",
        "method": "paired frozen-checkpoint inference probes",
        "checkpoint_sha256": {
            "attacker": hashlib.sha256(attacker_checkpoint.read_bytes()).hexdigest(),
            "defender": hashlib.sha256(defender_checkpoint.read_bytes()).hexdigest(),
            **{f"character_{name}": hashlib.sha256(
                (CHARACTER_ROOT / name / "best.pt").read_bytes()).hexdigest()
               for name in CHARACTER_CHECKPOINT_IDS},
            "character_gongon_defender": hashlib.sha256(
                (CHARACTER_ROOT / "gongon_defender" / "best.pt").read_bytes()).hexdigest(),
            "watch_points": hashlib.sha256(WATCH_POINTS_CONFIG_PATH.read_bytes()).hexdigest(),
        },
        "schedule": {"seed": seed, "matches_per_opponent": matches_per_opponent,
                     "opponents": list(opponents)},
        "training_only_ablation": {TRAINING_ONLY: "requires a separately retrained checkpoint"},
        "definitions": {
            "no_recurrent_state": "clear recurrent hidden state each tick; weights remain recurrent",
            "no_team_shared_vision": "coach sees slot 0 only; each character sees its own legal vision and maintains its own belief",
            "no_watch_point_input": "zero watch-point grid channels; checkpoint still trained with points",
            "no_character_models": "fixed north facing and no normal ability",
            "solo_entry_rate": "postmatch referee proxy: attacker moves within Manhattan distance two of a living enemy with no living teammate within distance three; denominator is such contested moves",
            "mean_watch_confirmation_age": "mean age in legal belief audit; never-confirmed points use elapsed round tick",
            "movement_collision_rate": "fraction of movement ticks with duplicate requested destinations",
            "ability_effective_rate": "successful resolved SMOKE/FLASH/RECON casts with measured effect",
        },
        "variants": by_variant, "matches": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--attacker-checkpoint", type=Path,
                        default=root / "checkpoints/experiments/task16_watch_added/coach/attacker/latest.pt")
    parser.add_argument("--defender-checkpoint", type=Path,
                        default=root / "checkpoints/experiments/task16_watch_added/coach/defender/latest.pt")
    parser.add_argument("--seed", type=int, default=1700)
    parser.add_argument("--matches-per-opponent", type=int, default=2)
    parser.add_argument("--variants", nargs="+", choices=(BASELINE, *PROBES),
                        default=(BASELINE, *PROBES))
    parser.add_argument("--opponents", nargs="+",
                        default=("omoko_v1", "touyama_v2", "gc_v1"))
    parser.add_argument("--output", type=Path,
                        default=REPORTS_DIR / "task17_comparison.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = evaluate(seed=args.seed, matches_per_opponent=args.matches_per_opponent,
                      variants=tuple(args.variants), opponents=tuple(args.opponents),
                      attacker_checkpoint=args.attacker_checkpoint,
                      defender_checkpoint=args.defender_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({name: value["overall"] for name, value in report["variants"].items()},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
