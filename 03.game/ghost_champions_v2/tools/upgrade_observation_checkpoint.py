"""Explicit v2/v3 -> current input padding for viewing; no new behavior is trained."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

AI_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(AI_ROOT.parent))
from ghost_champions_v2.rl.observation import OBS_DIM,LEGACY_V2_BLOCKS,LEGACY_V3_BLOCKS
from ghost_champions_v2.rl.actions import ACTIONS
from ghost_champions_v2.rl.policy import ResidualPolicy


def upgraded_policy(source):
    source=Path(source).resolve()
    payload=torch.load(source,map_location="cpu",weights_only=True)
    legacy_schemas=(dict(version=2,dim=448,blocks=LEGACY_V2_BLOCKS,enemy_roster_version=1),
                    dict(version=3,dim=811,blocks=LEGACY_V3_BLOCKS,enemy_roster_version=1,public_effect_version=1))
    legacy=next((schema for schema in legacy_schemas if payload.get("schema_hash")==
                 hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest()),None)
    if legacy is None or tuple(payload.get("actions",()))!=ACTIONS:
        raise ValueError("Only exact 448-input v2 or 811-input v3 schemas can be padded")
    weights=dict(payload["model_state_dict"])
    first=weights["features.0.weight"]
    padded=torch.zeros((payload["hidden"],OBS_DIM),dtype=first.dtype)
    padded[:,:legacy["dim"]]=first
    weights["features.0.weight"]=padded
    with torch.random.fork_rng(devices=[]):
        model=ResidualPolicy(payload["hidden"])
        model.load_state_dict(weights)
    model.eval()
    metadata=dict(training_stage="legacy_input_upgrade_untrained",verified=False,
               hazard_trained=bool(payload.get("hazard_trained",False)) if legacy["version"]==3 else False,
               unit_status_trained=False,source_schema_version=legacy["version"],source_checkpoint=str(source),
               source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    return model,metadata


def upgrade(source,destination):
    destination=Path(destination).resolve()
    if destination.exists():
        raise ValueError("Destination already exists; choose a new output path")
    model,metadata=upgraded_policy(source)
    model.save(destination,**metadata)
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=upgrade(args.source,args.output)
    print(f"Saved: {result}\nNew input weights are zero; the added inputs have NOT been learned. "
          "This output is for legacy viewing only.")


if __name__=="__main__":
    main()
