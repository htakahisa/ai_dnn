"""Continue saved BC/PPO with an evaluation-only change; never starts training.

Run from ghost_champions_v2/: py tools/fork_evaluation_run.py --source OLD --run NEW
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

AI_ROOT=Path(__file__).resolve().parents[1]
ROOT=AI_ROOT.parent
sys.path.insert(0,str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reuse_initial_series(source,target,checkpoint,series):
    """Reuse matching completed series while keeping the old evaluation intact."""
    from ghost_champions_v2.rl.training import EVALUATION_SOURCE_PATHS,evaluation_minimum
    from tools.run_eval import RUNTIME_FILES
    manifest_path=source/"manifest.json"
    if not manifest_path.exists():
        return None
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["controller"]!="ghost_champions_v2" or manifest["stage"]!="entry"
            or manifest["base_seed"]!=20261007 or manifest["python"]!=sys.version
            or manifest["residual_checkpoint"]["sha256"]!=digest(checkpoint)):
        raise ValueError("Partial initial evaluation does not match the BC model/runtime")
    runtime=Path(manifest.get("runtime_snapshot",ROOT))
    if manifest["runtime_data_hashes"]!={name:digest(runtime/name) for name in RUNTIME_FILES}:
        raise ValueError("Partial evaluation runtime changed")
    for name,old in manifest["source_hashes"].items():
        path=ROOT/name
        current=digest(path)
        if current!=old and str(path.resolve()) not in EVALUATION_SOURCE_PATHS:
            raise ValueError("Partial evaluation policy source changed: "+name)
        manifest["source_hashes"][name]=current
    manifest["series_count"]=series
    minimum=evaluation_minimum(series)
    if minimum!=100:
        manifest["minimum_attack_rounds"]=minimum
    else:
        manifest.pop("minimum_attack_rounds",None)
    manifest["residual_checkpoint"]["path"]=str(checkpoint.resolve())
    target.mkdir(parents=True,exist_ok=True)
    hashes={}
    for code in manifest["opponents"]:
        for index in range(series):
            name=f"series_{code}_{index:03d}.json"
            if (source/name).is_file():
                shutil.copy2(source/name,target/name)
                hashes[name]=digest(target/name)
    (target/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return dict(folder=str(target.resolve()),hashes=hashes,source_folder=str(source.resolve()),
                source_manifest_sha256=digest(manifest_path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    parser.add_argument("--run",required=True)
    parser.add_argument("--opponent-snapshot",type=Path)
    parser.add_argument("--reuse-completed-initial",action="store_true",
                        help="Reuse completed BC evaluation as a starting baseline, even below the training target")
    parser.add_argument("--reuse-completed-evaluation",action="store_true",
                        help="Reuse the completed PPO evaluation at the saved step instead of running it again")
    args=parser.parse_args()
    if any(Path(name).name!=name or name in (".","..") for name in (args.source,args.run)):
        parser.error("source and run must be directory names")
    source=AI_ROOT/"data"/args.source
    target=AI_ROOT/"data"/args.run
    if target.exists():
        parser.error("target run already exists")
    parent=json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    snapshots={Path(name).parent for name in parent["frozen_inputs"] if Path(name).name=="game_core.py"}
    if len(snapshots)!=1:
        parser.error("parent must identify exactly one frozen game runtime")
    snapshot=snapshots.pop()
    sys.path.insert(0,str(snapshot))
    os.environ["GC_V2_STAGE"]="entry"
    os.environ["GC_RUNTIME_SNAPSHOT"]=str(snapshot)
    if args.opponent_snapshot:
        args.opponent_snapshot=args.opponent_snapshot.resolve()
        os.environ["GC_OPPONENT_SNAPSHOT"]=str(args.opponent_snapshot)
        sys.path.insert(1,str(args.opponent_snapshot))
    from ghost_champions_v2.rl.training import fingerprint,validate_evaluation_change,load_continuation
    from ghost_champions_v2.rl.policy import ResidualPolicy
    from ghost_champions_v2.rl.initial_evaluation import evaluation_reference
    config=json.loads((AI_ROOT/"rl/training_config.json").read_text(encoding="utf-8"))
    config["evaluation_workers"]=parent["config"]["evaluation_workers"]
    frozen=fingerprint(snapshot)
    validate_evaluation_change(parent,config,frozen)
    parent_continuation=None
    if (source/"continuation.json").exists():
        if digest(source/"continuation.json")!=parent.get("continuation_sha256"):
            parser.error("parent continuation metadata changed")
        parent_continuation=load_continuation(source,parent["config"],frozen)
    bc_frozen=json.loads((source/"demonstrations.json").read_text(encoding="utf-8"))["frozen_inputs"]
    validate_evaluation_change(dict(parent,frozen_inputs=bc_frozen),config,frozen)
    bc_only=not (source/"latest.pt").is_file()
    if args.reuse_completed_initial and not bc_only:
        parser.error("initial-baseline recovery is only for runs that have not started PPO")
    if args.reuse_completed_evaluation and bc_only:
        parser.error("completed PPO evaluation reuse requires a saved PPO checkpoint")
    _,payload=ResidualPolicy.load(source/("bc.pt" if bc_only else "latest.pt"))
    initial=AI_ROOT/"logs"/args.source/"evaluation_bc"
    initial_evaluation=None
    initial_as_baseline=bool(args.reuse_completed_initial or
                             (parent_continuation and parent_continuation.get("initial_evaluation_as_baseline")))
    if args.reuse_completed_initial:
        reference=evaluation_reference(initial)
        if reference["source_checkpoint"]["sha256"]!=digest(source/"bc.pt"):
            parser.error("completed initial evaluation does not match the saved BC model")
        initial_evaluation=dict(folder=reference["folder"],hashes=reference["hashes"])
    elif not bc_only:
        if parent_continuation and parent_continuation.get("initial_evaluation"):
            initial=Path(parent_continuation["initial_evaluation"]["folder"])
        sample=json.loads((initial/"status.json").read_text(encoding="utf-8"))
        summary=json.loads((initial/"summary.json").read_text(encoding="utf-8"))
        if not sample["complete"]:
            parser.error("parent BC evaluation is incomplete")
        if not initial_as_baseline and config.get("tyg_periodic_action","stop")=="stop" and summary["Touyama Gaming"]["attack_rate"]<config["tyg_minimum_win_rate"]:
            parser.error("parent BC evaluation fails TYG guardrail")
        initial_reference=evaluation_reference(initial)
        initial_evaluation=dict(folder=initial_reference["folder"],hashes=initial_reference["hashes"])
    step,episode=payload.get("step",0),payload.get("episode",0)
    evaluated=f"evaluated_{step:07d}.pt"
    pending_evaluation=None
    if args.reuse_completed_evaluation:
        reference=evaluation_reference(AI_ROOT/"logs"/args.source/f"evaluation_{step:07d}")
        if reference["source_checkpoint"]["sha256"]!=digest(source/evaluated):
            parser.error("completed PPO evaluation does not match the saved evaluation checkpoint")
        evaluated_policy,_=ResidualPolicy.load(source/evaluated)
        import torch
        saved_policy,_=ResidualPolicy.load(source/"latest.pt")
        if any(not torch.equal(value,evaluated_policy.state_dict()[name]) for name,value in saved_policy.state_dict().items()):
            parser.error("completed PPO evaluation does not match the latest model weights")
        pending_evaluation=dict(step=step,reference=reference)
    copies={name:name for name in ("demonstrations.npz","demonstrations.json","bc.pt")}
    if not bc_only:
        copies["continuation_source.pt"]="latest.pt"
    if (source/evaluated).exists():
        copies[evaluated]=evaluated
    target.mkdir(parents=True)
    for name,original in copies.items():
        shutil.copy2(source/original,target/name)
    if not bc_only:
        shutil.copy2(source/"latest.pt",target/"latest.pt")
    continuation=dict(version=1,parent_run=args.source,parent_manifest=parent,
        reason=("User requested warning-only periodic TYG checks and reuse of the completed PPO evaluation"
                if args.reuse_completed_evaluation else "Reuse completed BC and initial evaluation as a starting baseline"
                if args.reuse_completed_initial else "User requested evaluation orchestration changes"),
        artifact_hashes={name:digest(target/name) for name in copies},
        initial_evaluation=initial_evaluation,runtime_snapshot=str(snapshot),bc_frozen_inputs=bc_frozen)
    if initial_as_baseline:
        continuation["initial_evaluation_as_baseline"]=True
    if pending_evaluation:
        continuation["pending_evaluation"]=pending_evaluation
    log_dir=AI_ROOT/"logs"/args.run
    if bc_only and not args.reuse_completed_initial:
        reused=reuse_initial_series(initial,log_dir/"evaluation_bc",target/"bc.pt",config["initial_evaluation_series"])
        if reused:
            continuation["reused_initial_evaluation"]=reused
    if args.opponent_snapshot:
        continuation["opponent_snapshot"]=str(args.opponent_snapshot)
    (target/"continuation.json").write_text(json.dumps(continuation,indent=2),encoding="utf-8")
    load_continuation(target,config,frozen)
    manifest=dict(schema_hash=parent["schema_hash"],config=config,frozen_inputs=frozen,verification=False,
        continuation_sha256=digest(target/"continuation.json"))
    (target/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    log_dir.mkdir(parents=True,exist_ok=True)
    status=dict(phase="ready_for_initial_evaluation" if bc_only and not initial_evaluation else "ready_for_rl",requested_phase="rl",complete=False,preparation_complete=True,
        process_id=None,step=step,episode=episode,
        last_evaluated_step=payload.get("last_evaluated_step",0),continued_from=args.source,
        evaluation_series=config["evaluation_series"],initial_evaluation_series=config["initial_evaluation_series"],
        final_evaluation_series=config["final_evaluation_series"],
        tyg_periodic_action=config.get("tyg_periodic_action","stop"),
        rl_auto_start=False,updated_at=datetime.now(timezone.utc).isoformat())
    if args.reuse_completed_initial:
        status.update(initial_evaluation_reused=True,initial_evaluation_source=str(initial),
                      initial_evaluation_as_baseline=True,initial_guardrail_passed=False,
                      initial_evaluation_rates=reference["rates"],bc_checkpoint=str(target/"bc.pt"))
    if pending_evaluation:
        status.update(completed_ppo_evaluation_source=pending_evaluation["reference"]["folder"],
                      completed_ppo_evaluation_rates=pending_evaluation["reference"]["rates"],
                      completed_ppo_evaluation_will_be_reused=True)
    (log_dir/"status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    print(f"Prepared {args.run} at {step} decisions. Training has not started.")
    if pending_evaluation:
        print(f"Completed evaluation at {step} will be reused. Periodic TYG action: {status['tyg_periodic_action']}.")
    if args.reuse_completed_initial:
        print("Saved BC and all completed initial series reused. PPO starts without repeating BC or initial evaluation.")
    elif bc_only:
        print(f"Reused {len(continuation.get('reused_initial_evaluation',{}).get('hashes',{}))} completed initial series.")
    print(f"py training_attacker.py --run {args.run} --evaluation-workers {config['evaluation_workers']} --phase rl --resume")


if __name__=="__main__":
    main()
