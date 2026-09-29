"""CLI for one trained coach_v1 headless match and optional replay export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from party_presets import get_preset
from run_game import _build_team_ai

from coach_v1.common.constants import REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.full_match import run_headless_full_match, write_match_artifacts
from coach_v1.team_ai import build_coach_v1_team


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent-ai", default="default")
    parser.add_argument("--opponent-preset", default="Ghost Champions")
    parser.add_argument("--coach-start-side", choices=[side.value for side in Side],
                        default=Side.ATTACKER.value)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path,
                        default=REPORTS_DIR / "task14_full_match.json")
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()

    preset = get_preset(args.opponent_preset)
    if preset is None:
        raise ValueError(f"unknown opponent preset: {args.opponent_preset}")
    result = run_headless_full_match(
        coach_team_ai=build_coach_v1_team(device=args.device),
        opponent_team_ai=_build_team_ai(args.opponent_ai),
        opponent_roster=preset.players,
        opponent_spike_holder=preset.spike_holder,
        opponent_igl=preset.igl,
        coach_starts_as=Side(args.coach_start_side),
        seed=args.seed,
        opponent_team_name=preset.name,
    )
    write_match_artifacts(result, report_path=args.report, replay_path=args.replay)
    print(json.dumps(result.summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
