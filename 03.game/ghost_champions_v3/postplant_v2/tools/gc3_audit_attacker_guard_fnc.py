"""FNC audit across historical failures and a separate fixed seed sample.

Run from postplant_v2: python tools/gc3_audit_attacker_guard_fnc.py
True positions are used for audit counters only, never for policy decisions.
"""
from pathlib import Path
import sys

GUARD_ROOT=Path(__file__).resolve().parents[1]
RUN_NAME='attacker_postplant_antidefuse_v1'
CHECKPOINT_NAME='latest.pt'
OPPONENT='FNC'
HISTORICAL_FAILURE_SEEDS=(20261014,20261040,20261053,20261091,20261110,20261214,20261217)
FIXED_SAMPLE_SEEDS=tuple(range(20361034,20361043))
TORCH_THREADS=1
OUTPUT_PATH=GUARD_ROOT/'logs/validation/gc3_guard_fnc_multi_seed.json'
sys.path.insert(0,str(GUARD_ROOT.parents[1]))


def summarize(rounds):
    total=len(rounds)
    opportunities=sum(r['adjacent_unrestricted_opportunities'] for r in rounds)
    behind=sum(r['adjacent_behind_decisions'] for r in rounds)
    return dict(rounds=total,wins=sum(r['success'] for r in rounds),defused=sum(r['reason']=='defused' for r in rounds),
        rounds_with_adjacent_opportunity=sum(r['adjacent_unrestricted_opportunities']>0 for r in rounds),
        rounds_with_adjacent_behind=sum(r['adjacent_behind_decisions']>0 for r in rounds),
        rounds_with_repeated_adjacent_behind=sum(r['adjacent_repeated_behind'] for r in rounds),
        adjacent_unrestricted_opportunities=opportunities,adjacent_behind_decisions=behind,
        adjacent_behind_rate=behind/opportunities if opportunities else None,
        missing_notifications=sum(r['missing_notifications'] for r in rounds))


def main():
    import torch
    torch.set_num_threads(TORCH_THREADS)
    from ghost_champions_v3.postplant_v2 import gc3_train_attacker_guard as settings
    from ghost_champions_v3.postplant_v2.gc3_policy_attacker_guard import GuardPolicy
    from ghost_champions_v3.postplant_v2.gc3_rollout_attacker_guard import load_entry
    from ghost_champions_v3.postplant_v2.tools.gc3_audit_attacker_guard import audit_round
    from ghost_champions_v3.preplant.gc3_data_attacker_preplant import file_hash
    from ghost_champions_v3.gc3_io import write_json
    config={k:v for k,v in vars(settings).items() if k.isupper()}
    checkpoint=GUARD_ROOT/'data'/RUN_NAME/CHECKPOINT_NAME
    policy,saved=GuardPolicy.load(checkpoint);entry=load_entry(config);rounds=[]
    for group,seeds in (('historical_failures',HISTORICAL_FAILURE_SEEDS),('fixed_sample',FIXED_SAMPLE_SEEDS)):
        for seed in seeds:
            result=audit_round(policy,entry,config,OPPONENT,seed,direction_probabilities=False)
            result['sample_group']=group;rounds.append(result)
            summary={name:summarize([r for r in rounds if r['sample_group']==name]) for name in ('historical_failures','fixed_sample')}
            write_json(OUTPUT_PATH,dict(run=RUN_NAME,checkpoint=CHECKPOINT_NAME,step=saved['step'],checkpoint_sha256=file_hash(checkpoint),
                summary=summary,rounds=rounds,training_modified=False,
                definition='An adjacent living active defuser with a clear shot line after the decision, while the attacker can turn, is unblinded and survives; facing outside the shooting cone counts as behind.',
                sampling='Historical failures are a targeted diagnostic sample; the fixed sample is reported separately. Counts are not a population frequency estimate.'))
            print(dict(group=group,seed=seed,reason=result['reason'],opportunities=result['adjacent_unrestricted_opportunities'],
                behind=result['adjacent_behind_decisions'],repeated=result['adjacent_repeated_behind']),flush=True)
    print(summary,flush=True)


if __name__=='__main__':main()
