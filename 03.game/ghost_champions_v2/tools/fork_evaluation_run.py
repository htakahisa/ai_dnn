"""Continue saved PPO state with an evaluation-only change; never starts training.

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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    parser.add_argument("--run",required=True)
    args=parser.parse_args()
    if any(Path(name).name!=name or name in (".","..") for name in (args.source,args.run)):
        parser.error("source and run must be directory names")
    source=AI_ROOT/"data"/args.source
    target=AI_ROOT/"data"/args.run
    if target.exists():
        parser.error("target run already exists")
    parent=json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    snapshot=AI_ROOT/"data/runtime_fixed_20261007"
    sys.path.insert(0,str(snapshot))
    os.environ["GC_V2_STAGE"]="entry"
    os.environ["GC_RUNTIME_SNAPSHOT"]=str(snapshot)
    from ghost_champions_v2.rl.training import fingerprint,validate_evaluation_change,load_continuation
    from ghost_champions_v2.rl.policy import ResidualPolicy
    config=json.loads((AI_ROOT/"rl/training_config.json").read_text(encoding="utf-8"))
    config["evaluation_workers"]=parent["config"]["evaluation_workers"]
    frozen=fingerprint(snapshot)
    validate_evaluation_change(parent,config,frozen)
    _,payload=ResidualPolicy.load(source/"latest.pt")
    initial=AI_ROOT/"logs"/args.source/"evaluation_bc"
    sample=json.loads((initial/"status.json").read_text(encoding="utf-8"))
    summary=json.loads((initial/"summary.json").read_text(encoding="utf-8"))
    if not sample["complete"] or sample.get("opponents_below_100_attack_rounds"):
        parser.error("parent BC evaluation is incomplete")
    if summary["Touyama Gaming"]["attack_rate"]<config["tyg_minimum_win_rate"]:
        parser.error("parent BC evaluation fails TYG guardrail")
    evaluated=f"evaluated_{payload['step']:07d}.pt"
    copies={name:name for name in ("demonstrations.npz","demonstrations.json","bc.pt")}
    copies["continuation_source.pt"]="latest.pt"
    if (source/evaluated).exists():
        copies[evaluated]=evaluated
    target.mkdir(parents=True)
    for name,original in copies.items():
        shutil.copy2(source/original,target/name)
    shutil.copy2(source/"latest.pt",target/"latest.pt")
    continuation=dict(version=1,parent_run=args.source,parent_manifest=parent,
        reason="User requested two BO3 series per opponent for periodic evaluation; final evaluation remains ten",
        artifact_hashes={name:digest(target/name) for name in copies},
        initial_evaluation=dict(folder=str(initial),hashes={name:digest(initial/name)
            for name in ("manifest.json","summary.json","status.json")}))
    (target/"continuation.json").write_text(json.dumps(continuation,indent=2),encoding="utf-8")
    load_continuation(target,config,frozen)
    manifest=dict(schema_hash=parent["schema_hash"],config=config,frozen_inputs=frozen,verification=False,
        continuation_sha256=digest(target/"continuation.json"))
    (target/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    log_dir=AI_ROOT/"logs"/args.run
    log_dir.mkdir(parents=True,exist_ok=True)
    status=dict(phase="ready_for_rl",requested_phase="rl",complete=False,preparation_complete=True,
        process_id=None,step=payload["step"],episode=payload["episode"],
        last_evaluated_step=payload.get("last_evaluated_step",0),continued_from=args.source,
        evaluation_series=config["evaluation_series"],final_evaluation_series=config["final_evaluation_series"],
        rl_auto_start=False,updated_at=datetime.now(timezone.utc).isoformat())
    (log_dir/"status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    print(f"Prepared {args.run} at {payload['step']} decisions. Training has not started.")
    print(f"py training_attacker.py --run {args.run} --evaluation-workers {config['evaluation_workers']} --phase rl --resume")


if __name__=="__main__":
    main()
