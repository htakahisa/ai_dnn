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
    parser.add_argument("--runtime-snapshot",type=Path)
    parser.add_argument("--initial-evaluation-from",type=Path,
                        help="Completed evaluation folder or run name to reuse as a historical initial baseline")
    parser.add_argument("--prepare-only",action="store_true",help="Prepare snapshot and initial evaluation reference without training")
    parser.add_argument("--phase",choices=("all","collect","bc","rl"),default="all")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--verification",action="store_true",help="Short functional run; never marks a model as verified")
    parser.add_argument("--steps",type=int)
    parser.add_argument("--bc-rounds",type=int)
    parser.add_argument("--bc-epochs",type=int)
    parser.add_argument("--evaluation-workers",type=int)
    args=parser.parse_args()
    if args.verification and args.initial_evaluation_from:
        parser.error("historical initial evaluations are for production runs only")
    if Path(args.run).name!=args.run or args.run in (".",".."):
        parser.error("run must be one directory name")
    config=json.loads(args.config.read_text(encoding="utf-8"))
    config.setdefault("initial_evaluation_series",config["evaluation_series"])
    config.setdefault("final_evaluation_series",config["evaluation_series"])
    config.setdefault("tyg_confirmation_series",config["evaluation_series"])
    config.setdefault("hazard_shaping_weight",.05)
    config.setdefault("hazard_hit_penalty",.03)
    config.setdefault("tyg_periodic_action","stop")
    if config["tyg_periodic_action"] not in ("stop","warn"):
        parser.error("tyg_periodic_action must be stop or warn")
    for name,value in (("steps",args.steps),("bc_rounds_per_opponent",args.bc_rounds),
                       ("bc_epochs",args.bc_epochs),("evaluation_workers",args.evaluation_workers)):
        if value is not None:
            config[name]=value
    if args.verification:
        config.update(steps=min(config["steps"],2000),bc_rounds_per_opponent=min(config["bc_rounds_per_opponent"],2),
                      evaluation_series=1,initial_evaluation_series=1,final_evaluation_series=1,tyg_confirmation_series=1)
    for name in ("steps","bc_rounds_per_opponent","bc_epochs","batch_size","evaluate_every_steps","evaluation_series","initial_evaluation_series","final_evaluation_series","tyg_confirmation_series","evaluation_workers","shaping_decay_steps","ppo_epochs"):
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
    for name in ("recon_new_enemy_reward","entropy_coefficient","hazard_shaping_weight","hazard_hit_penalty"):
        if not math.isfinite(config[name]) or config[name]<0:
            parser.error(name+" must be finite and nonnegative")
    os.environ["GC_V2_STAGE"]="entry"
    continuation_path=AI_ROOT/"data"/args.run/"continuation.json"
    continuation=json.loads(continuation_path.read_text(encoding="utf-8")) if continuation_path.exists() else {}
    from tools.run_eval import runtime_snapshot
    snapshot=runtime_snapshot(args.runtime_snapshot or continuation.get("runtime_snapshot")
                              or AI_ROOT/"data/runtime_fixed_20261008_unitstatus")
    os.environ["GC_RUNTIME_SNAPSHOT"]=str(snapshot)
    sys.path.insert(0,str(snapshot))
    if continuation:
        opponent_snapshot=continuation.get("opponent_snapshot")
        if opponent_snapshot:
            os.environ["GC_OPPONENT_SNAPSHOT"]=opponent_snapshot
            sys.path.insert(1,opponent_snapshot)
    # Activate the snapshot before importing any game/model observation modules.
    from ghost_champions_v2.rl.training import train
    data_dir=AI_ROOT/"data"/args.run
    if args.initial_evaluation_from:
        from ghost_champions_v2.rl.initial_evaluation import bind_initial_evaluation
        source=args.initial_evaluation_from
        if not source.is_dir():
            source=AI_ROOT/"logs"/source/"evaluation_bc"
        bind_initial_evaluation(data_dir,source)
    if args.prepare_only:
        from ghost_champions_v2.rl.initial_evaluation import load_initial_evaluation
        if (data_dir/"manifest.json").exists():
            parser.error("training run already started; preparation cannot replace its status")
        reference=load_initial_evaluation(data_dir)
        log_dir=AI_ROOT/"logs"/args.run
        log_dir.mkdir(parents=True,exist_ok=True)
        status=dict(phase="ready_for_bc_retraining",preparation_complete=True,complete=False,process_id=None,
                    step=0,rl_auto_start=False,runtime_snapshot=str(snapshot),
                    initial_evaluation_reused=reference is not None,
                    initial_evaluation_source=reference["folder"] if reference else None,
                    initial_evaluation_kind="historical_baseline" if reference else None,
                    initial_evaluation_rates=reference["rates"] if reference else None,
                    initial_evaluates_new_bc=False,evaluation_workers=config["evaluation_workers"])
        (log_dir/"status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
        print(f"Prepared {args.run}. Training and evaluation have not started.")
        return
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
