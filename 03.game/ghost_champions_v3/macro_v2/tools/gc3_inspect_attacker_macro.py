"""Read saved macro records and probe disposable copies; never save model weights."""
from copy import deepcopy
import json
from pathlib import Path
import sys

MACRO_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = 'attacker_macro_three_outputs_v1'
BATCH_DECISIONS = 512
PROBE_ROWS = 64
TEACHER_PROBE_ROWS = 256
PROBE_SEED = 23
TORCH_THREADS = 1
VERIFY_FROZEN_ROUND = True
REPORT_PATH = MACRO_ROOT/'logs/validation/gc3_three_output_macro_inspection.json'
sys.path.insert(0, str(MACRO_ROOT.parents[1]))

import numpy as np
import torch
from ghost_champions_v3.gc3_observation import ARENA_SHAPE
from ghost_champions_v3.preplant.gc3_data_attacker_preplant import file_hash
from ghost_champions_v3.preplant.gc3_observation_attacker_history import MatchHistory
from ghost_champions_v3.macro_v2.gc3_policy_attacker_macro import MacroPolicy
from ghost_champions_v3.macro_v2.gc3_candidates_attacker_macro import MacroCandidates
from ghost_champions_v3.macro_v2.gc3_directives_attacker_macro import INSTRUCTIONS
from ghost_champions_v3.macro_v2.gc3_bc_attacker_macro import load_teacher


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def reconstruct(history, candidates):
    rows=[];previous=None;round_id=None
    for index,event in enumerate(history.events):
        if event['name']!='macro':continue
        if event['round_id']!=round_id:
            previous=None;round_id=event['round_id']
        plan=event['macro_plan'];active=np.array([p is not None for p in plan['destinations']],dtype=bool)
        masks=dict(destination=np.tile(candidates.destination_mask,(5,1)),
            instruction=np.ones((5,len(INSTRUCTIONS)),dtype=bool),plant=candidates.plant_mask.copy())
        for slot in np.flatnonzero(~active):
            for key in ('destination','instruction'):
                masks[key][slot]=False;masks[key][slot,0]=True
        cell=lambda p:int(np.ravel_multi_index(p,ARENA_SHAPE))
        if not event['obs'][2]:
            rows.append(dict(obs=event['obs'],history=history,history_index=index,current_plan=deepcopy(previous),
                active=active,masks=masks,phase='macro',selections=dict(
                    destination=[cell(p) if p is not None else 0 for p in plan['destinations']],
                    instruction=[INSTRUCTIONS.index(v) if v is not None else 0 for v in plan['instructions']],
                    plant=cell(plan['plant']))))
        previous=plan
    return rows


def distributions(policy,rows):
    masks={k:np.stack([r['masks'][k] for r in rows]) for k in ('destination','instruction','plant')}
    return policy.distributions(policy.contexts(rows),masks)


def drift(reference,policy,rows):
    with torch.no_grad():
        before,after=distributions(reference,rows),distributions(policy,rows)
        active=torch.from_numpy(np.stack([r['active'] for r in rows]))
        result={}
        for key in before:
            term=torch.distributions.kl_divergence(before[key],after[key])
            if key!='plant':term=(term*active).sum(-1)
            result[key+'_kl_mean']=float(term.mean())
        result['joint_kl_mean']=sum(result.values())
        return result


def gradient_probe(policy,rows):
    logp,entropy,_=policy.evaluate_records(rows)
    # Deliberately reproduce the former shared-gradient value path for the v1 audit.
    value=policy.critic(policy.contexts(rows)).squeeze(-1)
    advantage=torch.tensor([r['advantage'] for r in rows]);advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    ratio=(logp-torch.tensor([r['logp'] for r in rows])).exp()
    losses=dict(actor=-(ratio*advantage).mean(),critic=.5*(value-torch.tensor([r['return'] for r in rows])).square().mean(),
        entropy=-.01*entropy.mean())
    parameters=[p for name,p in policy.named_parameters() if not name.startswith(('destination.','instruction.','plant.','critic.'))]
    gradients={name:torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True) for name,loss in losses.items()}
    norms={name:float(sum(g.square().sum() for g in gs if g is not None).sqrt()) for name,gs in gradients.items()}
    dot=sum((a*b).sum() for a,b in zip(gradients['actor'],gradients['critic']) if a is not None and b is not None)
    return dict(losses={k:float(v.detach()) for k,v in losses.items()},shared_gradient_norms=norms,
        critic_to_actor_gradient=norms['critic']/max(norms['actor'],1e-12),
        actor_critic_gradient_cosine=float(dot)/(norms['actor']*norms['critic']+1e-12))


def update_probe(reference,rows,config,kind):
    policy=deepcopy(reference)
    optimizer=torch.optim.Adam(policy.parameters(),lr=config['LEARNING_RATE'])
    advantages=np.array([r['advantage'] for r in rows],dtype=np.float32)
    advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
    rng=np.random.RandomState(PROBE_SEED);trace=[]
    probe=rows[:PROBE_ROWS]
    for epoch in range(config['PPO_EPOCHS']):
        indices=rng.permutation(len(rows));clipped=[]
        for offset in range(0,len(rows),config['BATCH_SIZE']):
            slots=indices[offset:offset+config['BATCH_SIZE']];batch=[rows[i] for i in slots]
            logp,entropy,_=policy.evaluate_records(batch)
            value=policy.critic(policy.contexts(batch)).squeeze(-1)
            ratio=(logp-torch.tensor([r['logp'] for r in batch])).exp()
            adv=torch.from_numpy(advantages[slots]);clip=config['PPO_CLIP']
            actor=-torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()
            critic=.5*(value-torch.tensor([r['return'] for r in batch])).square().mean()
            loss=dict(critic_only=critic,actor_only=actor-config['ENTROPY_WEIGHT']*entropy.mean(),
                full_ppo=actor+critic-config['ENTROPY_WEIGHT']*entropy.mean())[kind]
            clipped.extend(((ratio-1).abs()>clip).detach().tolist())
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        row=dict(epoch=epoch+1,clip_fraction_during_update=float(np.mean(clipped)),**drift(reference,policy,probe))
        trace.append(row);print(kind,row,flush=True)
    return dict(kind=kind,trace=trace,optimizer_updates=config['PPO_EPOCHS']*((len(rows)+config['BATCH_SIZE']-1)//config['BATCH_SIZE']),
        note='Disposable copy, stored first batch, fixed diagnostic shuffle; not a replay of the original shuffle.')


def main():
    torch.set_num_threads(TORCH_THREADS)
    data=MACRO_ROOT/'data'/RUN_NAME;logs=MACRO_ROOT/'logs'/RUN_NAME
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'));config=manifest['config']
    paths=[data/'best.pt',data/'latest.pt',*[Path(p) for p in (config['PREPLANT_CHECKPOINT'],config['POSTPLANT_CHECKPOINT'])]]
    hashes={str(p):file_hash(p) for p in paths}
    best,best_saved=MacroPolicy.load(data/'best.pt');latest,latest_saved=MacroPolicy.load(data/'latest.pt')
    if best_saved['step']!=0:raise ValueError('This audit requires the preserved BC best at ST0')
    from map_data import NEW_MAZE_STR
    candidates=MacroCandidates.configured(np.array([[int(c) for c in line] for line in NEW_MAZE_STR.strip().splitlines()]))
    history=MatchHistory.load(data/'histories/match_000000.npz')
    rows=reconstruct(history,candidates)[:BATCH_DECISIONS]
    transitions=read_lines(logs/'rollout_transitions.jsonl')[:BATCH_DECISIONS]
    assert len(rows)==len(transitions)==BATCH_DECISIONS
    value_error=0.
    with torch.no_grad():
        for offset in range(0,len(rows),config['BATCH_SIZE']):
            batch=rows[offset:offset+config['BATCH_SIZE']];logp,_,values=best.evaluate_records(batch)
            for i,r in enumerate(batch):
                saved=transitions[offset+i];r.update(saved,logp=float(logp[i]))
                value_error=max(value_error,abs(float(values[i])-r['value']))
    if value_error>2e-5:raise ValueError(f'Reconstructed pre-update values differ: {value_error}')
    report=dict(run=RUN_NAME,best_step=best_saved['step'],latest_step=latest_saved['step'],
        first_batch_replay_value_max_error=value_error,
        gradient_first_minibatch=gradient_probe(best,rows[:PROBE_ROWS]),
        gradient_spaced_batch=gradient_probe(best,rows[::len(rows)//PROBE_ROWS][:PROBE_ROWS]))
    from ghost_champions_v3.gc3_rollout import module_digest
    initial,_=MacroPolicy.load(data/'initial.pt')
    report['bc_critic_head_never_updated']=module_digest([initial.critic])==module_digest([best.critic])
    print('gradients',report,flush=True)
    teacher=load_teacher(data/'bc_teacher',candidates);probe=teacher[::max(1,len(teacher)//TEACHER_PROBE_ROWS)][:TEACHER_PROBE_ROWS]
    report['teacher_distribution_drift']=drift(best,latest,probe)
    print('saved policy drift',report['teacher_distribution_drift'],flush=True)
    report['update_probes']=[update_probe(best,rows,config,kind) for kind in ('full_ppo','critic_only','actor_only')]
    if VERIFY_FROZEN_ROUND:
        from ghost_champions_v3.macro_v2.gc3_rollout_attacker_macro import load_lower_policies,play_macro
        from ghost_champions_v3.preplant.gc3_rollout_attacker_preplant import choose_roster
        pre,guard=load_lower_policies(config['PREPLANT_CHECKPOINT'],config['POSTPLANT_CHECKPOINT'])
        code=choose_roster(config['SEED'],0,config['OPPONENT_CODES'])
        frozen,stats,_=play_macro(code,config['SEED'],policy=best,preplant=pre,guard=guard,candidates=candidates,
            gamma=config['GAMMA'],gae_lambda=config['GAE_LAMBDA'])
        with torch.no_grad():
            logp,_,_=best.evaluate_records(frozen)
        logp_error=float((logp-torch.tensor([r['logp'] for r in frozen])).abs().max())
        reward_error=max(abs(r['reward']-transitions[i]['reward']) for i,r in enumerate(frozen))
        if logp_error>2e-5 or reward_error>1e-6:raise ValueError('Frozen first-round replay differs from stored records')
        report['original_first_round_frozen_replay']=dict(decisions=len(frozen),planted=stats['planted'],win=stats['success'],
            cached_logp_max_error=logp_error,reward_log_max_error=reward_error,lower_weights_unchanged=stats['lower_weights_unchanged'])
    evaluations=[json.loads(p.read_text(encoding='utf-8')) for p in [logs/'initial_evaluation.json',*logs.glob('evaluation_*.json')]]
    report['evaluations']=[{k:r[k] for k in ('step','success_rate','plant_rate','reward_mean','plant_target_changes_mean','instructions')}
        for r in sorted(evaluations,key=lambda r:r['step'])]
    report['reward_breakdowns_match']=all(np.isclose(sum(r['reward_components'].values()),r['reward']) for r in read_lines(logs/'rollout_transitions.jsonl'))
    episodes=read_lines(logs/'episodes.jsonl')
    report['simulated_rounds']=len(episodes);report['discarded_rounds']=sum(r['decisions']==0 for r in episodes)
    report['sampled_decisions']=sum(r['sampled_decisions'] for r in episodes);report['used_decisions']=sum(r['decisions'] for r in episodes)
    report['checkpoint_hashes_unchanged']=all(file_hash(p)==hashes[str(p)] for p in paths)
    if not report['checkpoint_hashes_unchanged']:raise RuntimeError('Checkpoints changed during read-only diagnosis')
    report['checkpoint_hashes']=hashes;report['training_started']=False;report['saved_weight_updates']=0
    REPORT_PATH.parent.mkdir(parents=True,exist_ok=True);REPORT_PATH.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(REPORT_PATH,flush=True)


if __name__=='__main__':main()
