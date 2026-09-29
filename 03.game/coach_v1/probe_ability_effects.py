"""Check normal-ability effectiveness measurement on real fixed-map matches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from party_presets import get_preset
from run_game import _build_team_ai

from coach_v1.ability_effect_audit import summarize_ability_events
from coach_v1.common.constants import REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.full_match import run_headless_full_match
from coach_v1.team_ai import build_coach_v1_team


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1610)
    parser.add_argument("--matches", type=int, default=2)
    parser.add_argument("--opponent-ai", default="default")
    parser.add_argument("--opponent-preset", default="Ghost Champions")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path,
                        default=REPORTS_DIR / "task17_ability_readiness.json")
    args = parser.parse_args()
    if args.matches < 1:
        parser.error("matches must be positive")
    preset = get_preset(args.opponent_preset)
    if preset is None:
        parser.error("unknown opponent preset")
    torch.set_num_threads(1)
    events = []
    matches = []
    for offset in range(args.matches):
        seed = args.seed + offset
        start = Side.ATTACKER if offset % 2 == 0 else Side.DEFENDER
        result = run_headless_full_match(
            coach_team_ai=build_coach_v1_team(device=args.device),
            opponent_team_ai=_build_team_ai(args.opponent_ai),
            opponent_roster=preset.players,
            opponent_spike_holder=preset.spike_holder,
            opponent_igl=preset.igl,
            coach_starts_as=start, seed=seed,
            opponent_team_name=preset.name,
            capture_ability_events=True,
        )
        matches.append({"seed": seed, "start_side": start.value,
                        "coach_score": result.summary.coach_score,
                        "opponent_score": result.summary.opponent_score,
                        "ability_events": len(result.ability_events)})
        events.extend({**event, "match_seed": seed} for event in result.ability_events)
    payload = {"definition": {
        "denominator": "successful and resolved normal-ability casts",
        "FLASH": "at least one enemy gains blind duration at this projectile's impact",
        "RECON": "at least one enemy gains reveal duration at this projectile's impact",
        "SMOKE": "more enemy than ally directed shot-lane pair-ticks blocked during active replay frames",
        "HUNT": "excluded: this slot has no normal-ability cast in the v1 action mask",
        "unresolved": "excluded from denominator and reported separately",
    }, "matches": matches, "summary": summarize_ability_events(events), "events": events}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"matches": matches, "summary": payload["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
