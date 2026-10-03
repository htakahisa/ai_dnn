"""Read-only timing/profiling of one saved season map request."""

import argparse
import contextlib
import cProfile
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import time
from unittest.mock import patch


def preset_request(name):
    """Catalog-only fixtures, independent of a player's current save."""
    from dataclasses import asdict
    from character_stats import CHARACTER_TABLE
    from realtime_season_teams import SEASON_TEAMS
    teams = {team["name"]: team for team in SEASON_TEAMS}
    own = {"name": "Benchmark Toru", "players": ["Lysoar", "Smoggy", "まーやまくん", "Flashback", "valyn"],
           "igl": "valyn", "carrier": "Lysoar", "ai": "toru_ai_v3.1"}
    opponent = teams["Furina Party"]
    if name == "fnatic-toru":
        own, opponent = teams["Fnatic"], teams["Evil Geniuses"]
    elif name == "ghost-toru":
        own, opponent = teams["Ghost Champions"], teams["Evil Geniuses"]
    def pack(team):
        return {"name": team["name"], "players": [asdict(CHARACTER_TABLE[n]) for n in team["players"][:5]],
                "igl": team["igl"], "spike_holder": team["carrier"], "ai": team["ai"]}
    return {"own": pack(own), "opponent": pack(opponent), "render": False, "initial_side": "A",
            "tick_time_ms": 100, "seed": 1505420757}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--preset", choices=("toru-frc", "fnatic-toru", "ghost-toru"), default="toru-frc")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--own-ai")
    parser.add_argument("--opponent-ai")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--max-ticks", type=int)
    parser.add_argument("--reference", action="store_true", help="Use uncached navigation and scalar visibility with original CPU threads")
    args = parser.parse_args(argv)
    with contextlib.redirect_stdout(io.StringIO()):
        request = json.loads(args.request.read_text(encoding="utf-8")) if args.request else preset_request(args.preset)
    request["render"] = False
    for key in ("own", "opponent"):
        value = getattr(args, f"{key}_ai")
        if value is not None:
            request[key]["ai"] = value
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        import torch
        import run_game
        from season_scrim_worker import play_scrim
    imports = time.perf_counter() - started
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    original = run_game.VisualFPSBattle
    digest = hashlib.sha256()
    counts = {"ticks": 0}

    def create(*a, **kw):
        counts["inference_threads"] = torch.get_num_threads()
        game = original(*a, **kw)
        simulate = game._simulate_tick
        def tick():
            simulate()
            counts["ticks"] += 1
            state = (game.current_round, game.battle_tick, game.round_timer, game.detonate_timer,
                     game.attacker_wins, game.defender_wins, game.is_planted,
                     [(str(c.name), c.team, list(c.pos), c.hp, c.is_alive, c.facing, c.has_spike,
                       c.kills, c.deaths, c.ultimate_points) for c in game.chars])
            digest.update(repr(state).encode("utf-8"))
            if args.max_ticks is not None and counts["ticks"] >= args.max_ticks:
                game.match_over = True
        game._simulate_tick = tick
        return game

    profile = cProfile.Profile() if args.profile else None
    started = time.perf_counter()
    with contextlib.ExitStack() as overrides:
        if args.reference:
            import simulation_runtime
            from frc_v1 import baseline, navigation, perception
            from frc_v1.model import FrcPolicy
            from abilities_los import AbilityLosMixin
            import numpy as np
            def scalar_visible(grid, viewers, smoke):
                geometry = type("Geometry", (AbilityLosMixin,), {})()
                geometry.grid, geometry.smokes = np.asarray(grid), [{"cells": smoke}]
                viewers = tuple(viewers)
                return {(r, c) for r, row in enumerate(grid) for c, cell in enumerate(row)
                        if cell != 1 and (r, c) not in smoke and any(
                            direction[0] * (r - origin[0]) + direction[1] * (c - origin[1]) >= -1e-9
                            and geometry.check_cell_line_of_sight(origin, (r, c)) for origin, direction in viewers)}
            overrides.enter_context(patch.object(simulation_runtime, "cpu_inference", lambda **kw: contextlib.nullcontext()))
            overrides.enter_context(patch.object(perception, "visible_cells", scalar_visible))
            original_sample = FrcPolicy.sample
            def full_sample(policy, *a, **kw):
                with patch.object(policy, "collect_statistics", True):
                    return original_sample(policy, *a, **kw)
            overrides.enter_context(patch.object(FrcPolicy, "sample", full_sample))
            for module, name in ((baseline, "_cached_plant_sites"), (baseline, "_cached_route_step"),
                                 (navigation, "_cached_walk_distances")):
                original = baseline._bfs_route_step if name == "_cached_route_step" else getattr(module, name).__wrapped__
                overrides.enter_context(patch.object(module, name, original))
        with patch.object(run_game, "VisualFPSBattle", side_effect=create), contextlib.redirect_stdout(io.StringIO()):
            if profile:
                profile.enable()
            result = play_scrim(request)
            if profile:
                profile.disable()
    elapsed = time.perf_counter() - started
    print(json.dumps({"ai": [request[k]["ai"] for k in ("own", "opponent")],
                      "import_seconds": round(imports, 3), "map_seconds": round(elapsed, 3),
                      "torch_threads": torch.get_num_threads(), "ticks": counts["ticks"],
                      "inference_threads": counts["inference_threads"], "reference": args.reference,
                      "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                      "tick_digest": digest.hexdigest(), "partial": args.max_ticks is not None,
                      "result": result}, ensure_ascii=True))
    if profile:
        pstats.Stats(profile).strip_dirs().sort_stats("cumtime").print_stats(35)


if __name__ == "__main__":
    main()
