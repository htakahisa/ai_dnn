"""Count FNC facing choices in the original saved guard rollouts.

Run from postplant_v2: python tools/gc3_audit_attacker_guard_records.py
These are intended choices near the perceived spike, not true-defuser distances.
"""
from pathlib import Path
import json
import sys
import numpy as np

GUARD_ROOT=Path(__file__).resolve().parents[1]
RUN_NAME='attacker_postplant_antidefuse_v1'
OPPONENT='FNC'
NEARBY_DISTANCE=2
OUTPUT_PATH=GUARD_ROOT/'logs/validation/gc3_guard_fnc_historical_choices.json'
sys.path.insert(0,str(GUARD_ROOT.parents[1]))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    from game_core import FACING_VECTORS
    from ghost_champions_v3.gc3_observation import ARENA_SHAPE,CELLS,CONTEXT_WIDTH,BOARD_CHANNELS,UNIT_WIDTH
    from ghost_champions_v3.gc3_engine_actions import FACING_DIRS,facing_towards
    from ghost_champions_v3.gc3_io import write_json
    logs=GUARD_ROOT/'logs'/RUN_NAME;data=GUARD_ROOT/'data'/RUN_NAME
    rounds=read_jsonl(logs/'episodes.jsonl');updates=read_jsonl(logs/'ppo_updates.jsonl')
    previous_episode=0;report=[]
    for update in updates:
        paths=list((data/'rollouts').glob(f"batch_{update['start_step']:06d}_*.npz"))
        if len(paths)!=1:raise RuntimeError(f'Ambiguous saved batch at {update["start_step"]}')
        batch_rounds=rounds[previous_episode:update['episode']];previous_episode=update['episode']
        with np.load(paths[0],allow_pickle=False) as saved:
            obs=saved['obs'];metadata=json.loads(str(saved['metadata']));offset=0
            if sum(r['sampled_decisions'] for r in batch_rounds)!=len(metadata):
                raise RuntimeError('Saved rounds do not match the complete rollout batch')
            for stats in batch_rounds:
                count=stats['sampled_decisions'];entries=[]
                if stats['opponent']==OPPONENT:
                    for index in range(offset,offset+count):
                        x=obs[index];row=metadata[index]
                        if x[-32+3]<=0:continue
                        origin=tuple(int(round(x[k]*ARENA_SHAPE[k])) for k in (0,1))
                        spike=tuple(int(round(x[9+k]*ARENA_SHAPE[k])) for k in (0,1))
                        distance=max(abs(origin[k]-spike[k]) for k in (0,1))
                        start=CONTEXT_WIDTH+BOARD_CHANNELS*CELLS
                        units=x[start:start+10*UNIT_WIDTH].reshape(10,UNIT_WIDTH)
                        own=next(u for u in units if u[1]>.5);facing=FACING_DIRS[int(own[-8:].argmax())]
                        selections=row['selections'];mode=selections['mode'];destination=origin
                        if mode in (0,2):destination=tuple(map(int,np.unravel_index(selections['position'],ARENA_SHAPE)))
                        if mode in (1,2) or (mode==4 and row['ultimate'] in ('RAID','TUNNEL')):
                            facing=facing_towards(destination,np.unravel_index(selections['watch'],ARENA_SHAPE),facing)
                        elif mode==0:
                            dr,dc=destination[0]-origin[0],destination[1]-origin[1]
                            if dr:facing='S' if dr>0 else 'N'
                            elif dc:facing='E' if dc>0 else 'W'
                        dr,dc=spike[0]-destination[0],spike[1]-destination[1];fx,fy=FACING_VECTORS[facing]
                        entries.append(dict(distance=distance,away=fx*dc+fy*dr<0,at_spike=origin==spike))
                    nearby=[e for e in entries if e['distance']<=NEARBY_DISTANCE and not e['at_spike']]
                    report.append(dict(seed=stats['seed'],success=stats['success'],defused=stats['defused'],
                        alerted_decisions=len(entries),nearby_alerted_decisions=len(nearby),
                        nearby_away_choices=sum(e['away'] for e in nearby)))
                offset+=count
    summary=dict(rounds=len(report),alerted_decisions=sum(r['alerted_decisions'] for r in report),
        nearby_alerted_decisions=sum(r['nearby_alerted_decisions'] for r in report),
        nearby_away_choices=sum(r['nearby_away_choices'] for r in report),
        rounds_with_nearby_away_choices=sum(r['nearby_away_choices']>0 for r in report))
    write_json(OUTPUT_PATH,dict(run=RUN_NAME,opponent=OPPONENT,nearby_distance=NEARBY_DISTANCE,summary=summary,rounds=report,
        note='Diagnostic of intended commands near the perceived public spike; not true-defuser distance, executable turning, or a causal claim. Original records omit facing-lock flags. Different rounds use different training updates.'))
    print(summary,flush=True)


if __name__=='__main__':main()
