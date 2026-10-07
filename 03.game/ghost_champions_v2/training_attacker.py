"""Run from ghost_champions_v2/: py training_attacker.py --run <name>."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

AI_ROOT=Path(__file__).resolve().parent
ROOT=AI_ROOT.parent
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--config",type=Path,default=AI_ROOT/"rl/training_config.json")
    parser.add_argument("--runtime-snapshot",type=Path,default=AI_ROOT/"data/runtime_fixed_20261007")
    parser.add_argument("--phase",choices=("all","collect","bc","rl"),default="all")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--verification",action="store_true",help="Short functional run; never marks a model as verified")
    parser.add_argument("--steps",type=int)
    parser.add_argument("--bc-rounds",type=int)
    parser.add_argument("--bc-epochs",type=int)
    parser.add_argument("--evaluation-workers",type=int)
    args=parser.parse_args()
    if Path(args.run).name!=args.run or args.run in (".",".."):
        parser.error("run must be one directory name")
    config=json.loads(args.config.read_text(encoding="utf-8"))
    for name,value in (("steps",args.steps),("bc_rounds_per_opponent",args.bc_rounds),
                       ("bc_epochs",args.bc_epochs),("evaluation_workers",args.evaluation_workers)):
        if value is not None:
            config[name]=value
    if args.verification:
        config.update(steps=min(config["steps"],2000),bc_rounds_per_opponent=min(config["bc_rounds_per_opponent"],2),
                      evaluation_series=1,initial_evaluation_series=1,final_evaluation_series=1)
    for name in ("steps","bc_rounds_per_opponent","bc_epochs","batch_size","evaluate_every_steps","evaluation_series","initial_evaluation_series","final_evaluation_series","evaluation_workers","shaping_decay_steps","ppo_epochs"):
        if not isinstance(config[name],int) or config[name]<1:
            parser.error(name+" must be a positive integer")
    if config["ability_bc_floor"]<.3:
        parser.error("ability_bc_floor must be at least 0.3")
    if config.get("schema_version")!=1:
        parser.error("unsupported training config schema")
    for name in ("gamma","gae_lambda","ppo_clip","tyg_minimum_win_rate"):
        if not math.isfinite(config[name]) or not 0<config[name]<=1:
            parser.error(name+" must be finite and in (0, 1]")
    for name in ("learning_rate","ability_bc_floor","max_gradient_norm","value_coefficient"):
        if not math.isfinite(config[name]) or config[name]<=0:
            parser.error(name+" must be finite and positive")
    for name in ("recon_new_enemy_reward","entropy_coefficient"):
        if not math.isfinite(config[name]) or config[name]<0:
            parser.error(name+" must be finite and nonnegative")
    os.environ["GC_V2_STAGE"]="entry"
    from tools.run_eval import runtime_snapshot
    snapshot=runtime_snapshot(args.runtime_snapshot)
    os.environ["GC_RUNTIME_SNAPSHOT"]=str(snapshot)
    sys.path.insert(0,str(snapshot))
    # Activate the snapshot before importing any game/model observation modules.
    from ghost_champions_v2.rl.training import train
    try:
        train(args.run,config,snapshot,phase=args.phase,resume=args.resume,verification=args.verification)
    except Exception as exc:
        status_path=AI_ROOT/"logs"/args.run/"status.json"
        if status_path.exists():
            status=json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("process_id")==os.getpid():
                status.update(phase="failed",complete=False,error=f"{type(exc).__name__}: {exc}")
                status_path.write_text(json.dumps(status,indent=2),encoding="utf-8")
        raise


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
