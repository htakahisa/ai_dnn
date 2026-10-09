"""Watch a new match with the trained GC attacker. Run from ghost_champions_v2/."""
import argparse
from datetime import datetime
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random
import sys

AI_ROOT=Path(__file__).resolve().parents[1]
ROOT=AI_ROOT.parent
sys.path.insert(0,str(ROOT))
OPPONENTS={"TYG":"Touyama Gaming","OMG":"Omoko Gaming","FRC":"Furina Classic",
           "FNC":"Fnatic2023","GG":"Gorigons","SPS":"SUPES"}
OPENING_LABELS={"DEFAULT":"両側索敵", "RUSH":"ラッシュ", "SPLIT":"スプリット"}


def save_match_log(game,path,metadata):
    """Persist actual watched frames for diagnosing a reported round."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(metadata,current_round=game.current_round,match_over=game.match_over,
                 round_records=game.analytics_tracker.round_records,
                 replay_frames=game.replay_frames)
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload,ensure_ascii=False,default=lambda value:value.item()),encoding="utf-8")
    temporary.replace(path)


def build_match(checkpoint,opponent,*,seed,tick_ms=150,opening="random",attack_only=False,headless=False,
                legacy_checkpoint=False):
    import numpy as np
    import torch
    from party_presets import get_preset
    from map_data import NEW_MAZE_STR
    from run_game import VisualFPSBattle,_build_team_ai
    from team_ai import DualRoleTeamAI
    from ghost_champions_v2.defender_macro_v1.controller import GhostChampionsV2DefenderController
    from ghost_champions_v2.config import load_config
    from ghost_champions_v2.rl.controller import LearnedAttackerController
    policy=None
    if legacy_checkpoint:
        from ghost_champions_v2.tools.upgrade_observation_checkpoint import upgraded_policy
        policy,_=upgraded_policy(checkpoint)
    else:
        from ghost_champions_v2.rl.policy import ResidualPolicy
        policy,payload=ResidualPolicy.load(checkpoint)
        if payload.get("hazard_trained") is False:
            print("WARNING: this checkpoint has padded hazard inputs; avoidance has NOT been learned.",flush=True)

    config=load_config()
    if opening!="random":
        config["opening"]["weights"]={key:int(key==opening) for key in OPENING_LABELS}
    random.seed(seed)
    np.random.seed(seed%(2**32))
    torch.manual_seed(seed)
    gc,enemy=get_preset("Ghost Champions"),get_preset(OPPONENTS[opponent])
    team=DualRoleTeamAI(name="Ghost Champions v2",
        attacker_factory=partial(LearnedAttackerController,checkpoint=str(checkpoint),policy=policy,
                                 config=config,stage="entry",mode="greedy"),
        defender_factory=GhostChampionsV2DefenderController)
    return VisualFPSBattle(NEW_MAZE_STR,team,_build_team_ai(enemy.default_ai),headless=headless,
        attacker_roster=list(gc.players),defender_roster=list(enemy.players),
        spike_holder_name=gc.spike_holder,defender_spike_holder_name=enemy.spike_holder,
        attacker_igl_name=gc.igl,defender_igl_name=enemy.igl,
        attacker_team_name="Ghost Champions v2",defender_team_name=enemy.name,
        disable_side_swap=attack_only,tick_time_ms=tick_ms)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="attacker_rl_defuse_20261007_eval2")
    parser.add_argument("--checkpoint",type=Path,help="Override the trained checkpoint")
    parser.add_argument("--opponent",choices=OPPONENTS,default="OMG")
    parser.add_argument("--seed",type=int,help="Repeat a match seed; omitted means a new draw")
    parser.add_argument("--tick-ms",type=int,default=150)
    parser.add_argument("--opening",choices=("random",*OPENING_LABELS),default="random",
                        help="Draw every round, or watch one chosen strategy")
    parser.add_argument("--attack-only",action="store_true",help="Keep GC attacking across rounds")
    parser.add_argument("--autoplay",action="store_true",help="Start moving as soon as the window opens")
    parser.add_argument("--legacy-checkpoint",action="store_true",
                        help="Explicitly pad a v2 checkpoint for viewing; hazard avoidance is untrained")
    args=parser.parse_args()
    if Path(args.run).name!=args.run or args.run in (".",".."):
        parser.error("run must be a directory name")
    checkpoint=(args.checkpoint or AI_ROOT/"data"/args.run/"last_passed_guardrail.pt").resolve()
    if not checkpoint.is_file():
        parser.error(f"Trained checkpoint not found: {checkpoint}")
    if args.tick_ms<1:
        parser.error("tick-ms must be positive")
    seed=args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    print(f"Live match: GC v2 vs {args.opponent}; seed={seed}\nCheckpoint: {checkpoint}",flush=True)
    if args.legacy_checkpoint:
        print("WARNING: legacy input padding; hazard avoidance has NOT been learned. Checkpoint stays unchanged.",flush=True)
    # Older shared controllers resolve model paths from the game directory.
    original_cwd=Path.cwd()
    os.chdir(ROOT)
    try:
        from simulation_runtime import cpu_inference
        import torch
        import tkinter as tk
        with cpu_inference(),torch.no_grad():
            game=build_match(checkpoint,args.opponent,seed=seed,tick_ms=args.tick_ms,
                             opening=args.opening,attack_only=args.attack_only,legacy_checkpoint=args.legacy_checkpoint)
            game.root.title(f"GC v2 vs {args.opponent} — 実況観戦")
            log_path=AI_ROOT/"logs/live"/f"{datetime.now():%Y%m%d_%H%M%S_%f}_{args.opponent}_{seed}.json"
            metadata=dict(opponent=args.opponent,seed=seed,checkpoint=str(checkpoint),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                opening=args.opening,attack_only=args.attack_only,
                legacy_input_padding=args.legacy_checkpoint,
                config=game.attacker_controller.inner.config,
                source_hashes={name:hashlib.sha256((AI_ROOT/name).read_bytes()).hexdigest()
                               for name in ("tactics.py","hazards.py","controller.py","rl/actions.py","rl/controller.py","rl/observation.py")},
                perception_source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                               for name in ("public_effects.py","grid_visibility.py","iq_controller_adapter.py","team_ai.py")})
            save_match_log(game,log_path,metadata)
            print(f"Live log: {log_path}",flush=True)
            def close():
                try:
                    save_match_log(game,log_path,metadata)
                finally:
                    game._close_match_window()
            game.root.protocol("WM_DELETE_WINDOW",close)
            caption=tk.StringVar(game.root)
            tk.Label(game.root,textvariable=caption,font=("Arial",11),anchor="w",padx=8,pady=4
                     ).pack(fill="x",before=game.canvas)
            last_opening=None
            last_saved_rounds=0
            def update_caption():
                nonlocal last_opening,last_saved_rounds
                completed=len(game.analytics_tracker.round_records)
                if completed!=last_saved_rounds:
                    save_match_log(game,log_path,metadata)
                    last_saved_rounds=completed
                snapshot=game._attacker_rule_snapshot() or {}
                strategy=snapshot.get("opening_strategy")
                if strategy:
                    text=(f"Round {game.current_round}  作戦: {OPENING_LABELS.get(strategy,strategy)}"
                          f"  初期: {snapshot.get('initial_site') or '…'}"
                          f"  現在: {snapshot.get('selected_site') or '…'}")
                    key=(game.current_round,strategy,snapshot.get("initial_site"))
                    if key!=last_opening:
                        print(text,flush=True)
                        last_opening=key
                elif game.current_attacker_team_ai is game.initial_attacker_team_ai:
                    text=f"Round {game.current_round}  GC攻撃・初動抽選待ち"
                else:
                    defender=game.defender_controller
                    from ghost_champions_v2.defender_macro_v1.status import format_defender_status
                    text=f"Round {game.current_round}  GC防衛  " + format_defender_status(defender.defender_snapshot())
                caption.set(text)
                game.root.after(150,update_caption)
            if not args.autoplay:
                game.toggle_pause()
            update_caption()
            game.root.update_idletasks()
            game.root.lift()
            print(f"LIVE_READY window={game.root.winfo_id()} paused={game.paused}; 再開で試合開始",flush=True)
            game.run()
    finally:
        os.chdir(original_cwd)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
