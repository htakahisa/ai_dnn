"""Evaluate GC against selectable full-engine AI teams and rosters.

Each trial plays two independent maps with the team under test on both sides:
map 1 has team1 attacking and map 2 has team1 defending.  This avoids judging
the defender only from the simplified Retake simulator.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _roster_preset(team_name: str, roster_name: str | None):
    from party_presets import get_preset

    team = get_preset(team_name)
    if team is None:
        raise ValueError(f"unknown team preset: {team_name}")
    if not roster_name:
        return team
    roster = get_preset(roster_name)
    if roster is None:
        raise ValueError(f"unknown roster preset: {roster_name}")
    # Keep the requested team identity/controller while borrowing the complete
    # player/IGL/spike-holder configuration from another named roster.
    return replace(
        roster,
        name=team.name,
        short_name=team.short_name,
        description=f"roster={roster.name}; controller team={team.name}",
    )


def _sha256(path: Path | None):
    if path is None:
        return None
    path = path.resolve()
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        if path.exists()
        else None,
    }


def _configure_gc_models(controller_key, search_model, retake_model):
    """Override Ghost Champions' defender checkpoints for this process only."""
    if not (search_model or retake_model):
        return
    if "ghost_champions" not in str(controller_key).lower() and str(controller_key).lower() not in {"gc", "gc_v1"}:
        raise ValueError(
            "--our-search-model/--our-retake-model require a Ghost Champions controller"
        )
    import ghost_champions_v1 as gc

    if search_model:
        gc.SEARCH = (Path(search_model).resolve(),)
    if retake_model:
        gc.RETAKE = (Path(retake_model).resolve(),)


def evaluate(args):
    from party_presets import get_preset
    from run_competition_manager import play_map

    our = _roster_preset(args.our_team, args.our_roster)
    opponent = _roster_preset(args.opponent_team, args.opponent_roster)
    _configure_gc_models(
        args.our_controller,
        args.our_search_model,
        args.our_retake_model,
    )

    maps = []
    pair_rows = []
    start = time.monotonic()
    for pair_index in range(args.pairs):
        seed_base = args.base_seed + pair_index * 2
        pair_maps = []
        # Map 1: our team attacks. Map 2: our team defends.
        for map_number, seed in ((1, seed_base), (2, seed_base + 1)):
            result = play_map(
                our,
                opponent,
                map_number,
                seed,
                False,
                args.our_controller,
                args.opponent_controller,
            )
            row = {
                "pair": pair_index + 1,
                "map_number": map_number,
                "seed": seed,
                "our_side": "attacker" if map_number == 1 else "defender",
                "score_our": result.score1,
                "score_opponent": result.score2,
                "winner": "our" if result.score1 > result.score2 else "opponent",
                "initial_attacker": result.initial_attacker,
                "overtime": result.overtime,
                "total_rounds": result.total_rounds,
            }
            maps.append(row)
            pair_maps.append(row)
        our_map_wins = sum(row["winner"] == "our" for row in pair_maps)
        opponent_map_wins = sum(row["winner"] == "opponent" for row in pair_maps)
        pair_rows.append(
            {
                "pair": pair_index + 1,
                "seed_base": seed_base,
                "our_map_wins": our_map_wins,
                "opponent_map_wins": opponent_map_wins,
                "tie": our_map_wins == opponent_map_wins,
                "our_rounds": sum(row["score_our"] for row in pair_maps),
                "opponent_rounds": sum(row["score_opponent"] for row in pair_maps),
            }
        )

    total_maps = len(maps)
    our_map_wins = sum(row["winner"] == "our" for row in maps)
    opponent_map_wins = sum(row["winner"] == "opponent" for row in maps)
    attacker_maps = [row for row in maps if row["our_side"] == "attacker"]
    defender_maps = [row for row in maps if row["our_side"] == "defender"]
    our_rounds = sum(row["score_our"] for row in maps)
    opponent_rounds = sum(row["score_opponent"] for row in maps)
    summary = {
        "evaluation": "full_engine_gc_matchup",
        "pairs": args.pairs,
        "maps": total_maps,
        "base_seed": args.base_seed,
        "our_team": our.name,
        "opponent_team": opponent.name,
        "our_controller": args.our_controller,
        "opponent_controller": args.opponent_controller,
        "our_roster_preset": args.our_roster or args.our_team,
        "opponent_roster_preset": args.opponent_roster or args.opponent_team,
        "our_search_model": _sha256(args.our_search_model),
        "our_retake_model": _sha256(args.our_retake_model),
        "our_map_wins": our_map_wins,
        "opponent_map_wins": opponent_map_wins,
        "map_win_rate": our_map_wins / max(1, total_maps),
        "rounds_our": our_rounds,
        "rounds_opponent": opponent_rounds,
        "round_win_rate": our_rounds / max(1, our_rounds + opponent_rounds),
        "attacker_map_win_rate": sum(r["winner"] == "our" for r in attacker_maps)
        / max(1, len(attacker_maps)),
        "defender_map_win_rate": sum(r["winner"] == "our" for r in defender_maps)
        / max(1, len(defender_maps)),
        "pair_win_rate": sum(
            row["our_map_wins"] > row["opponent_map_wins"] for row in pair_rows
        )
        / max(1, len(pair_rows)),
        "pair_tie_rate": sum(row["tie"] for row in pair_rows)
        / max(1, len(pair_rows)),
        "seconds": round(time.monotonic() - start, 1),
        "maps_detail": maps,
        "pairs_detail": pair_rows,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GC against selectable full-engine AI/controller teams"
    )
    parser.add_argument("--our-team", default="Ghost Champions")
    parser.add_argument("--opponent-team", default="Touyama Gaming")
    parser.add_argument("--our-controller", default="ghost_champions_v1")
    parser.add_argument("--opponent-controller", default="touyama_gaming_v2")
    parser.add_argument("--our-roster", help="optional roster preset independent of team identity")
    parser.add_argument("--opponent-roster", help="optional roster preset independent of team identity")
    parser.add_argument("--our-search-model", type=Path)
    parser.add_argument("--our-retake-model", type=Path)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=3026091700)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be at least 1")
    summary = evaluate(args)
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("detail") and k != "maps_detail"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
