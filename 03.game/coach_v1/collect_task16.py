"""Run fixed-map matches and save Task 16 review candidates with replays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from party_presets import get_preset

from coach_v1.common.constants import REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.full_match import run_headless_full_match, write_match_artifacts
from coach_v1.learning_coach_attacker import DEFAULT_DIRECTORY as ATTACKER_DIRECTORY
from coach_v1.learning_coach_defender import DEFAULT_DIRECTORY as DEFENDER_DIRECTORY
from coach_v1.opponent_pool import DEFAULT_POOL_PATH, load_opponent_pool
from coach_v1.task15_self_play import build_opponent_team
from coach_v1.task16_hard_points import candidate_report
from coach_v1.team_ai import build_coach_v1_team


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attacker-checkpoint", type=Path,
                        default=ATTACKER_DIRECTORY / "latest.pt")
    parser.add_argument("--defender-checkpoint", type=Path,
                        default=DEFENDER_DIRECTORY / "latest.pt")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument("--seed", type=int, default=1600)
    parser.add_argument("--matches-per-opponent", type=int, default=2)
    parser.add_argument("--max-difficulty", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path,
                        default=REPORTS_DIR / "task16_candidates.json")
    args = parser.parse_args()
    if args.matches_per_opponent < 1:
        parser.error("matches-per-opponent must be positive")
    torch.set_num_threads(1)
    matches = []
    match_summaries = []
    replay_dir = args.output.parent / (args.output.stem + "_replays")
    for spec in load_opponent_pool(args.pool).eligible(max_difficulty=args.max_difficulty):
        preset = get_preset(spec.preset)
        for offset in range(args.matches_per_opponent):
            seed = args.seed + offset
            start = Side.ATTACKER if offset % 2 == 0 else Side.DEFENDER
            result = run_headless_full_match(
                coach_team_ai=build_coach_v1_team(
                    attacker_checkpoint=args.attacker_checkpoint,
                    defender_checkpoint=args.defender_checkpoint, device=args.device),
                opponent_team_ai=build_opponent_team(spec, device=args.device),
                opponent_roster=preset.players,
                opponent_spike_holder=preset.spike_holder,
                opponent_igl=preset.igl,
                coach_starts_as=start, seed=seed,
                opponent_team_name=spec.opponent_id,
                allow_mirrored_roster=spec.mirrored_roster,
                capture_referee_events=True,
            )
            stem = f"{spec.opponent_id}_{seed}_{start.value}"
            replay_path = replay_dir / f"{stem}.json"
            write_match_artifacts(result, report_path=replay_dir / f"{stem}_rounds.json",
                                  replay_path=replay_path)
            matches.append((result, spec.opponent_id, str(replay_path)))
            match_summaries.append({"opponent": spec.opponent_id, "seed": seed,
                                    "coach_started_as": start.value,
                                    "coach_score": result.summary.coach_score,
                                    "opponent_score": result.summary.opponent_score})
    report = candidate_report(matches)
    report["source"] = {
        "pool": str(args.pool), "seed": args.seed,
        "matches_per_opponent": args.matches_per_opponent,
        "max_difficulty": args.max_difficulty,
        "attacker_checkpoint_sha256": hashlib.sha256(args.attacker_checkpoint.read_bytes()).hexdigest(),
        "defender_checkpoint_sha256": hashlib.sha256(args.defender_checkpoint.read_bytes()).hexdigest(),
        "match_summaries": match_summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"matches": report["matches"],
                      "candidate_count": report["candidate_count"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
