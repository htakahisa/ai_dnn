"""Watch GC defending from round one with visible attack-site predictions."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from datetime import datetime
import hashlib
import json
import os
import random

from ghost_champions_v2.defender_macro_v1.analysis import AttackSiteAnalysis
from ghost_champions_v2.defender_macro_v1.train_defender_analysis_gc import OPPONENTS, seed_all, seed_defender_opening, build_seeded_opponent
from ghost_champions_v2.defender_macro_v1.controller import GhostChampionsV2DefenderController
from ghost_champions_v2.defender_macro_v1.status import format_defender_status


def build_match(opponent, *, checkpoint=None, seed=42, tick_ms=150, headless=False):
    from party_presets import get_preset
    from run_game import VisualFPSBattle, _build_team_ai
    from team_ai import DualRoleTeamAI
    from controllers import DefaultAttackerController
    seed_all(seed)
    analysis = AttackSiteAnalysis(checkpoint=checkpoint)
    defender = GhostChampionsV2DefenderController(analysis=analysis)
    seed_defender_opening(defender, seed)
    team = DualRoleTeamAI("Ghost Champions v2", DefaultAttackerController, lambda: defender)
    key, name = OPPONENTS[opponent]
    attacking, defending = get_preset(name), get_preset("Ghost Champions")
    game = VisualFPSBattle(analysis.scenario.maze, build_seeded_opponent(opponent, seed), team, headless=headless,
        attacker_roster=list(attacking.players), defender_roster=list(defending.players),
        spike_holder_name=attacking.spike_holder, defender_spike_holder_name=defending.spike_holder,
        attacker_igl_name=attacking.igl, defender_igl_name=defending.igl,
        attacker_team_name=attacking.name, defender_team_name=defending.name,
        disable_side_swap=True, tick_time_ms=tick_ms)
    return game, defender


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--opponent", choices=OPPONENTS, default="OMG")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--tick-ms", type=int, default=150)
    p.add_argument("--seed", type=int)
    p.add_argument("--autoplay", action="store_true")
    args = p.parse_args(argv)
    if args.tick_ms < 1:
        p.error("--tick-ms must be positive")
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2 ** 32)
    directory = HERE / "logs" / "live"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{args.opponent}_{seed}.json"
    old_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        import tkinter as tk
        import torch
        from simulation_runtime import cpu_inference
        with cpu_inference(), torch.no_grad():
            game, defender = build_match(args.opponent, checkpoint=checkpoint, seed=seed, tick_ms=args.tick_ms)
            from toruAI_v4.tv4_train_analysis import relocate_debug_logs
            relocate_debug_logs(game.attacker_controller, directory)
            relocate_debug_logs(game.defender_controller, directory)
            model = defender.macro.analysis.checkpoint
            metadata = dict(opponent=args.opponent, seed=seed, checkpoint=str(model),
                checkpoint_sha256=hashlib.sha256(model.read_bytes()).hexdigest() if model.is_file() else None,
                runtime_hashes={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                for name in ("analysis.py", "controller.py", "tendency.py", "deployment.py", "combat.py", "retake.py", "status.py")},
                python_hash_seed=os.environ.get("PYTHONHASHSEED"), opening_rng="match_seed_v1",
                opponent_rng="match_seed_concon_routes_guards_v1")
            decisions = []
            last_key = None
            def save():
                payload = dict(metadata, current_round=game.current_round, match_over=game.match_over,
                    decisions=decisions, round_records=game.analytics_tracker.round_records,
                    replay_frames=game.replay_frames)
                temp = path.with_suffix(".tmp")
                temp.write_text(json.dumps(payload, ensure_ascii=False, default=lambda value: value.item()), encoding="utf-8")
                temp.replace(path)
            def close():
                try:
                    save()
                finally:
                    game._close_match_window()
            caption = tk.StringVar(game.root)
            game.root.title(f"GC v2 防衛マクロ vs {args.opponent}")
            tk.Label(game.root, textvariable=caption, anchor="w", padx=8, pady=4).pack(fill="x", before=game.canvas)
            def update_caption():
                nonlocal last_key
                snapshot = defender.defender_snapshot()
                caption.set(f"Round {game.current_round}  GC防衛  " + format_defender_status(snapshot))
                key = game.current_round, game.battle_tick, game.is_planted
                if key != last_key:
                    decisions.append(dict(round=game.current_round, tick=game.battle_tick, **snapshot))
                    if last_key is None or key[0] != last_key[0] or game.match_over:
                        save()
                    last_key = key
                if not game.match_over:
                    game.root.after(150, update_caption)
            game.root.protocol("WM_DELETE_WINDOW", close)
            if not args.autoplay:
                game.toggle_pause()
            update_caption()
            game.root.update_idletasks()
            game.root.lift()
            print(f"GC defender vs {args.opponent}; seed={seed}; model={model}; log={path}; 再開で開始", flush=True)
            game.run()
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
