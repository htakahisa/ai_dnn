"""Post-match, referee-only trace for Issue 001. Never used by an actor."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import torch

from party_presets import get_preset
from coach_v1.common.constants import CHARACTER_CHECKPOINT_IDS
from coach_v1.common.types import Side
from coach_v1.evaluate_task17 import CHARACTER_ROOT
from coach_v1.full_match import run_headless_full_match
from coach_v1.opponent_pool import load_opponent_pool
from coach_v1.task15_self_play import build_opponent_team
from coach_v1.team_ai import build_coach_v1_team
from coach_v1.models.coach_model import legal_action_mask
from coach_v1.observation.coach_encoder import CoachObservationEncoder
from coach_v1.training.attacker_teacher import teacher_actions as attacker_teacher_actions
from coach_v1.training.defender_teacher import teacher_actions as defender_teacher_actions
from coach_v1.training.scenario_generator import _distances
from map_data import NEW_MAZE_STR


ROOT = Path(__file__).resolve().parent
_ROWS = tuple(NEW_MAZE_STR.strip().splitlines())
_PLANT_CELLS = tuple((r, c) for r, row in enumerate(_ROWS)
                     for c, tile in enumerate(row) if tile == "2")
_SITE_DISTANCE = _distances(_ROWS, _PLANT_CELLS)


class DiagnosticTeacher:
    """Evaluation-only coach using the same public observation as an actor."""

    def __init__(self, side: Side) -> None:
        self.encoder = CoachObservationEncoder()
        self.teacher = (attacker_teacher_actions if side is Side.ATTACKER
                        else defender_teacher_actions)

    def act(self, observation):
        return self.teacher(observation, legal_action_mask(observation))


def trace(result) -> dict:
    frames = defaultdict(list)
    for frame in result.replay:
        if not frame.get("setup"):
            frames[int(frame["round"])].append(frame)
    actions = defaultdict(list)
    for action in result.coach_actions:
        actions[action.round_number].append(action)
    decisions = defaultdict(list)
    for decision in result.coach_decisions:
        decisions[decision.round_number].append(decision)
    records = {int(record["round_number"]): record for record in result.round_records}
    rounds = []
    for number, sequence in sorted(frames.items()):
        side = sequence[0].get("coach_side")
        planted = next((frame for frame in sequence if frame["planted"]), None)
        carrier_frames = []
        if side == "attacker":
            for frame in sequence:
                holder = next((char for char in frame["chars"] if char["team"] == "A"
                               and char["has_spike"]), None)
                if holder:
                    carrier_frames.append((frame["tick"], holder["name"], holder["pos"]))
        own_actions = [a for a in actions[number] if a.side.value == side]
        by_tick = {int(frame["tick"]): frame for frame in sequence}
        blocked = 0
        for action in own_actions:
            if action.action != "MOVE" or action.start == action.requested_position:
                continue
            # Replay tick t is captured before that tick's movement. The
            # resulting position first appears in frame t+1.
            frame = by_tick.get(action.tick + 1)
            if frame is None:
                continue
            character = next((char for char in frame["chars"]
                              if char["team"] == ("A" if side == "attacker" else "D")
                              and char["name"] == ("ごりまる", "ごんごん", "ごんた", "くんた", "くりまる")[action.slot]), None)
            if character is not None and tuple(character["pos"]) == action.start:
                blocked += 1
        count = Counter(a.action for a in own_actions)
        item = {
            "round": number, "side": side,
            "winner": records.get(number, {}).get("winner"),
            "reason": records.get(number, {}).get("reason"),
            "plant_position": planted.get("planted_pos") if planted else None,
            "first_plant_tick": planted["tick"] if planted else None,
            "actions": dict(count), "decision_count": len(decisions[number]),
            "blocked_move_requests": blocked,
            "last_tick": sequence[-1]["tick"],
        }
        if side == "attacker":
            item["carrier_start"] = carrier_frames[0] if carrier_frames else None
            item["carrier_last"] = carrier_frames[-1] if carrier_frames else None
            item["carrier_unique_positions"] = len({tuple(pos) for _, _, pos in carrier_frames})
            distances = [_SITE_DISTANCE.get(tuple(pos)) for _, _, pos in carrier_frames]
            item["carrier_start_site_distance"] = distances[0] if distances else None
            item["carrier_last_site_distance"] = distances[-1] if distances else None
            item["carrier_nearest_site_distance"] = min(distances) if distances else None
            item["carrier_reached_plantable"] = any(distance == 0 for distance in distances)
            item["carrier_stalled_ticks"] = max((sum(1 for _ in group)
                                                  for _, group in groupby(
                                                      carrier_frames,
                                                      key=lambda sample: (sample[1], tuple(sample[2])))),
                                                 default=0)
            oscillation = 2
            for index in range(len(carrier_frames) - 1, 1, -1):
                current, previous, earlier = (carrier_frames[index],
                                              carrier_frames[index - 1],
                                              carrier_frames[index - 2])
                if (current[0] != previous[0] + 1
                        or current[1] != previous[1] or current[1] != earlier[1]
                        or current[2] != earlier[2] or current[2] == previous[2]):
                    break
                oscillation += 1
            item["carrier_two_cell_oscillation_tail_ticks"] = (
                oscillation if oscillation >= 4 else 0)
            item["carrier_action_counts"] = dict(Counter(
                a.action for a in own_actions if a.slot == 2))
            item["carrier_last_actions"] = [
                {"tick": a.tick, "action": a.action, "start": a.start,
                 "requested": a.requested_position}
                for a in own_actions if a.slot == 2
            ][-12:]
            item["last_positions"] = [
                {"name": char["name"], "pos": char["pos"], "alive": char["alive"],
                 "has_spike": char["has_spike"]}
                for char in sequence[-1]["chars"] if char["team"] == "A"
            ]
        elif planted:
            after = [a for a in own_actions if a.tick >= planted["tick"]]
            item["postplant_actions"] = dict(Counter(a.action for a in after))
            item["defuse_attempts"] = sum(a.action == "DEFUSE" for a in after)
            item["defuse_actions"] = [
                {"tick": a.tick, "slot": a.slot, "start": a.start}
                for a in after if a.action == "DEFUSE"
            ]
            first_decision = next((d for d in decisions[number]
                                   if d.tick >= planted["tick"] and d.side.value == side), None)
            item["first_postplant_decision"] = (
                {"tick": first_decision.tick,
                 "observed_spike": first_decision.spike_position,
                 "instructions": first_decision.instructions}
                if first_decision else None)
            item["first_postplant_actions"] = [
                {"slot": a.slot, "start": a.start, "requested": a.requested_position,
                 "action": a.action} for a in after if a.tick == planted["tick"]
            ]
            item["last_positions"] = [
                char["pos"] for char in sequence[-1]["chars"] if char["team"] == "D"
            ]
        rounds.append(item)
    return {"summary": result.summary.to_dict(), "rounds": rounds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1700)
    parser.add_argument("--opponent", default="omoko_v1")
    parser.add_argument("--starts-as", choices=("attacker", "defender"), default="attacker")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--candidate", action="store_true")
    policy.add_argument("--teacher", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/issue001_baseline.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec = next(spec for spec in load_opponent_pool().opponents
                if spec.opponent_id == args.opponent)
    preset = get_preset(spec.preset)
    checkpoint_root = ROOT / ("checkpoints/experiments/issue001_distance/coach"
                              if args.candidate else
                              "checkpoints/experiments/task16_watch_added/coach")
    attacker_checkpoint = checkpoint_root / "attacker/latest.pt"
    defender_checkpoint = checkpoint_root / "defender/latest.pt"
    team = build_coach_v1_team(
        attacker_checkpoint=attacker_checkpoint,
        defender_checkpoint=defender_checkpoint,
        character_checkpoints={name: CHARACTER_ROOT / name / "best.pt"
                               for name in CHARACTER_CHECKPOINT_IDS},
        gongon_defender_checkpoint=CHARACTER_ROOT / "gongon_defender/best.pt",
        attacker_coach=DiagnosticTeacher(Side.ATTACKER) if args.teacher else None,
        defender_coach=DiagnosticTeacher(Side.DEFENDER) if args.teacher else None,
    )
    result = run_headless_full_match(
        coach_team_ai=team, opponent_team_ai=build_opponent_team(spec),
        opponent_roster=preset.players, opponent_spike_holder=preset.spike_holder,
        opponent_igl=preset.igl, coach_starts_as=Side(args.starts_as),
        seed=args.seed, opponent_team_name=spec.opponent_id,
        capture_referee_events=True,
    )
    report = trace(result)
    report["seed"] = args.seed
    report["opponent"] = args.opponent
    report["candidate"] = args.candidate
    report["policy"] = ("teacher" if args.teacher else
                        "candidate" if args.candidate else "baseline")
    report["policy_sha256"] = (
        {"attacker_teacher": hashlib.sha256(
            (ROOT / "training/attacker_teacher.py").read_bytes()).hexdigest(),
         "defender_teacher": hashlib.sha256(
            (ROOT / "training/defender_teacher.py").read_bytes()).hexdigest()}
        if args.teacher else
        {"attacker": hashlib.sha256(attacker_checkpoint.read_bytes()).hexdigest(),
         "defender": hashlib.sha256(defender_checkpoint.read_bytes()).hexdigest()})
    if not args.teacher:
        report["checkpoint_sha256"] = report["policy_sha256"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"score": [result.summary.coach_score,
                                result.summary.opponent_score],
                      "plants": sum(r["plant_position"] is not None
                                    for r in report["rounds"] if r["side"] == "attacker"),
                      "defuse_attempts": sum(r.get("defuse_attempts", 0)
                                             for r in report["rounds"]),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
