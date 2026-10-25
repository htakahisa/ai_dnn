"""Trace public defuse notifications and facing choices in real postplant rounds.

Run from postplant_v2: python tools/gc3_audit_attacker_guard.py
Only diagnostics read true enemy positions. They never enter policy inputs.
"""
from collections import Counter
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys

GUARD_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = 'attacker_postplant_antidefuse_v1'
CHECKPOINT_NAMES = ('initial.pt', 'latest.pt')
OPPONENT = 'FNC'
SEEDS = (20361030, 20361014)
OUTPUT_PATH = GUARD_ROOT/'logs'/'validation'/'gc3_guard_facing_audit.json'
TORCH_THREADS = 1
MAX_TICKS = 400
sys.path.insert(0, str(GUARD_ROOT.parents[1]))


def audit_round(policy, entry, config, opponent, seed):
    import numpy as np
    import torch
    from game_core import SHOOTING_SITE_DEGREE
    from ghost_champions_v3.gc3_engine_actions import MODES, FACING_DIRS, decode_command
    from ghost_champions_v3.gc3_rollout import engine_directory, module_digest
    from ghost_champions_v3.postplant_v2.gc3_features_attacker_guard import tap_state
    from ghost_champions_v3.postplant_v2.gc3_rollout_attacker_guard import prepare_entry, handoff
    from simulation_runtime import cpu_inference

    with engine_directory(), redirect_stdout(io.StringIO()), torch.no_grad(), cpu_inference():
        game, raw, start = prepare_entry(opponent, seed, entry, config)
        handoff(game, raw, policy)
        original_select = policy.select
        traces = []
        def traced_select(char, state):
            result, row = original_select(char, state)
            actual = [c for c in game.chars if c.team == 'D' and c.is_alive and c.defuse_timer > 0]
            if not actual:
                return result, row
            active, progress, remaining, _ = tap_state(state)
            context = policy.contexts([row])[0]
            mode_probs = policy.distribution(context, row, 'mode').probs
            record = dict(tick=game.battle_tick, name=str(char.name), pos=list(char.pos), facing=char.facing,
                public_notification=active, progress=progress, remaining=remaining,
                can_turn=not bool(getattr(char, 'facing_forced_this_tick', False) or getattr(char, 'movement_disabled_remaining', 0)),
                blind=float(getattr(char, 'blind_remaining', 0)), mode=MODES[row['selections']['mode']],
                selections=row['selections'], command=result,
                mode_probabilities={name:float(mode_probs[i]) for i,name in enumerate(MODES)},
                visible_enemies=[str(c.name) for c in state['chars'] if c.team!='A' and getattr(c,'position_known',False)],
                actual_defusers=[dict(name=str(c.name),pos=list(c.pos),progress=int(c.defuse_timer),
                    distance=max(abs(c.pos[i]-char.pos[i]) for i in (0,1))) for c in actual])
            # Integrate each legal movement and watch choice into the engine's
            # eight directions; do not sample or consume any RNG for auditing.
            directions={f:0. for f in FACING_DIRS}
            for mode, kinds in ((0,('position',)),(1,('watch',)),(2,('position','watch'))):
                if not float(mode_probs[mode]): continue
                position = policy.distribution(context,row,'position').probs if 'position' in kinds else None
                watch = policy.distribution(context,row,'watch').probs if 'watch' in kinds else None
                positions = np.flatnonzero(row['full_masks']['position']) if position is not None else [None]
                watches = np.flatnonzero(row['full_masks']['watch']) if watch is not None else [None]
                for p in positions:
                    for w in watches:
                        selections={'mode':mode}
                        if p is not None: selections['position']=int(p)
                        if w is not None: selections['watch']=int(w)
                        destination, command=decode_command(char,state,selections)
                        if isinstance(command,dict): facing=command['facing']
                        else: facing=game._facing_from_delta(destination[0]-char.pos[0],destination[1]-char.pos[1],char.facing)
                        mass=float(mode_probs[mode])*(float(position[p]) if p is not None else 1.)*(float(watch[w]) if w is not None else 1.)
                        directions[facing]+=mass
            record['move_watch_facing_probabilities']=directions
            traces.append(record)
            return result,row
        policy.select=traced_select
        before=module_digest([policy])
        try:
            for tick in range(MAX_TICKS):
                offset=len(traces)
                game._simulate_tick()
                for record in traces[offset:]:
                    char=next(c for c in game.chars if str(c.name)==record['name'])
                    record.update(actual_facing_after=char.facing,actual_pos_after=list(char.pos),
                        shot=any(str(s['shooter'].name)==char.name for s in (game.last_shots or [])),
                        still_alive=bool(char.is_alive))
                if game.round_over: break
            else: raise RuntimeError('Audit round exceeded limit')
        finally:
            policy.select=original_select
        assert module_digest([policy])==before
        near=[r for r in traces if min(d['distance'] for d in r['actual_defusers'])<=1]
        return dict(seed=seed,entry_seed=start['seed'],opponent=opponent,success=bool(game.attacker_wins),
            reason=game.analytics_tracker.round_records[-1]['reason'],ticks=tick+1,
            notified_decisions=len(traces),near_defuser_decisions=len(near),
            near_defuser_shot_decisions=sum(r['shot'] for r in near),
            missing_notifications=sum(not r['public_notification'] for r in traces),
            near_modes=dict(Counter(r['mode'] for r in near)),traces=traces)


def main():
    import torch
    torch.set_num_threads(TORCH_THREADS)
    from ghost_champions_v3.postplant_v2 import gc3_train_attacker_guard as settings
    from ghost_champions_v3.postplant_v2.gc3_policy_attacker_guard import GuardPolicy
    from ghost_champions_v3.postplant_v2.gc3_rollout_attacker_guard import load_entry
    from ghost_champions_v3.gc3_io import write_json
    config={k:v for k,v in vars(settings).items() if k.isupper()}
    entry=load_entry(config);report=[]
    for name in CHECKPOINT_NAMES:
        policy,saved=GuardPolicy.load(GUARD_ROOT/'data'/RUN_NAME/name)
        for seed in SEEDS:
            result=audit_round(policy,entry,config,OPPONENT,seed)
            result.update(checkpoint=name,step=saved['step'])
            report.append(result)
            print({k:v for k,v in result.items() if k!='traces'},flush=True)
            write_json(OUTPUT_PATH,dict(rounds=report,training_started=False))


if __name__=='__main__': main()
