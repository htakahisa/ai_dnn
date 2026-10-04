"""Run one season scrim using the normal run_game engine in a child process."""

import argparse
import json
from pathlib import Path
import random
import traceback

from season_scrim import validate_scrim_request


def play_scrim(request):
    validate_scrim_request(request)
    from simulation_runtime import cpu_inference
    with cpu_inference(enabled=not request["render"]):
        return _play_scrim(request)


def _play_scrim(request):
    # The worker is isolated: saved individual abilities apply only to this
    # match and never replace the catalog used by the season home or other modes.
    import character_stats
    import game_core
    import numpy as np
    import torch
    from run_game import VisualFPSBattle, _build_team_ai
    from map_data import NEW_MAZE_STR

    random.seed(request["seed"])
    np.random.seed(request["seed"])
    torch.manual_seed(request["seed"])
    for key in ("own", "opponent"):
        for row in request[key]["players"]:
            character_stats.CHARACTER_TABLE[row["name"]] = character_stats.CharacterStats(**row)
    # game_core ordinarily loads its own copy of the character module.
    # Point its stat getter at this worker's saved snapshots as well.
    game_core._character_stats = character_stats
    own, opponent = request["own"], request["opponent"]
    attacker, defender = (own, opponent) if request["initial_side"] == "A" else (opponent, own)
    game = VisualFPSBattle(
        NEW_MAZE_STR, _build_team_ai(attacker["ai"]), _build_team_ai(defender["ai"]),
        headless=not request["render"],
        attacker_roster=[p["name"] for p in attacker["players"]],
        defender_roster=[p["name"] for p in defender["players"]],
        attacker_igl_name=attacker["igl"], defender_igl_name=defender["igl"],
        spike_holder_name=attacker["spike_holder"], defender_spike_holder_name=defender["spike_holder"],
        attacker_team_name=attacker["name"], defender_team_name=defender["name"],
        tick_time_ms=request["tick_time_ms"],
    )
    game.record_replay = False
    callback_errors = []
    if request["render"]:
        def report_callback_error(error_type, error, stack):
            traceback.print_exception(error_type, error, stack)
            callback_errors.append(error)
            game._close_match_window()
        game.root.report_callback_exception = report_callback_error
        def watch_end():
            if game.match_over:
                game.root.after(1500, game._close_match_window)
            else:
                game.root.after(100, watch_end)
        game.root.after(100, watch_end)
    game.run()
    if callback_errors:
        raise RuntimeError(f"試合画面の実行に失敗しました: {callback_errors[0]}") from callback_errors[0]
    if game.attacker_team_name == own["name"]:
        own_score, opponent_score = game.attacker_wins, game.defender_wins
    else:
        own_score, opponent_score = game.defender_wins, game.attacker_wins
    return {
        "status": "completed" if game.match_over else "cancelled",
        "own_team": own["name"], "opponent_team": opponent["name"],
        "own_score": int(own_score), "opponent_score": int(opponent_score),
        "winner": (own["name"] if own_score > opponent_score else opponent["name"]) if game.match_over else None,
        "seed": request["seed"],
        "player_stats": {str(name): {"kills": int(stats.get("kills", 0)), "deaths": int(stats.get("deaths", 0))}
                         for name, stats in game.match_stats.items()},
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args(argv)
    code = 0
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        if "maps_to_win" in request:
            from season_series import play_series
            result = play_series(request)
        else:
            result = play_scrim(request)
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "error", "message": str(exc)}
        code = 1
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
