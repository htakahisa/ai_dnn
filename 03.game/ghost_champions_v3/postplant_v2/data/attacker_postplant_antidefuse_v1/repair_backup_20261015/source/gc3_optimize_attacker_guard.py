"""Exact mode-conditional joint KL, isolated critic and conservative postplant PPO."""
from copy import deepcopy
import numpy as np
import torch
from ..macro_v2.gc3_optimize_attacker_macro import fit_value,actor_optimizer,value_optimizer

OPTIMIZER_VERSION='gc3-objective-guard-conditional-kl-v1'


def probabilities(policy,rows,batch_size):
    parts={}
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            for key,value in policy.all_probabilities(rows[start:start+batch_size]).items():parts.setdefault(key,[]).append(value.detach().clone())
    return {key:torch.cat(values) for key,values in parts.items()}


def conditional_kl(reference,current,rows):
    def kl(key):
        old,new=reference[key],current[key];tiny=torch.finfo(old.dtype).tiny
        return (old*(old.clamp_min(tiny).log()-new.clamp_min(tiny).log())).sum(-1)
    mode=reference['mode'];result=kl('mode')
    # MOVE=0, WATCH=1, MOVE_WATCH=2, ABILITY=3, ULTIMATE=4.
    result+=(mode[:,0]+mode[:,2])*kl('position')
    ult_watch=torch.tensor([r['ultimate'] in ('RAID','TUNNEL') for r in rows])
    ult_target=torch.tensor([r['ultimate'] in ('NEON','ESCAPE') for r in rows])
    ally=torch.tensor([r['ability']=='DANCE' for r in rows])
    ability_target=torch.tensor([r['ability'] not in ('DANCE','RAMP') for r in rows])
    result+=(mode[:,1]+mode[:,2]+mode[:,4]*ult_watch)*kl('watch')
    result+=mode[:,3]*ability_target*kl('target')+mode[:,3]*ally*kl('ally')
    result+=mode[:,4]*ult_target*kl('ultimate_target')
    return result


def joint_kl(policy,rows,reference,batch_size):
    total=0.
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size]
            old={key:values[start:start+len(batch)] for key,values in reference.items()}
            total+=float(conditional_kl(old,policy.all_probabilities(batch),batch).sum())
    result=max(0.,total/len(rows))
    if not np.isfinite(result):raise RuntimeError('Nonfinite guard KL')
    return result


def optimize(policy,actor,critic,rows,config,warmup=False):
    bs=config['BATCH_SIZE'];reference=probabilities(policy,rows,bs)
    indices=np.linspace(0,len(rows)-1,min(len(rows),config['KL_MONITOR_ROWS']),dtype=int)
    monitor=[rows[i] for i in indices];old={key:values[indices] for key,values in reference.items()}
    initial_optimizer=deepcopy(actor.state_dict())
    warm=fit_value(policy,critic,rows,config,config['VALUE_WARMUP_EPOCHS']) if warmup else {}
    initial=deepcopy(policy.state_dict());accepted=rejected=0;stop=None;losses=[]
    advantage=np.asarray([r['advantage'] for r in rows],dtype=np.float32)
    advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
    for _ in range(config['PPO_EPOCHS']):
        order=np.random.permutation(len(rows))
        for start in range(0,len(rows),bs):
            slots=order[start:start+bs];batch=[rows[i] for i in slots]
            logp,entropy,_=policy.evaluate_records(batch)
            ratio=(logp-torch.tensor([r['logp'] for r in batch])).exp();adv=torch.from_numpy(advantage[slots])
            actor_loss=-torch.minimum(ratio*adv,ratio.clamp(1-config['PPO_CLIP'],1+config['PPO_CLIP'])*adv).mean()
            loss=actor_loss-config['ENTROPY_WEIGHT']*entropy.mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite guard loss')
            saved=deepcopy(policy.state_dict());saved_opt=deepcopy(actor.state_dict())
            actor.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.actor_parameters(),1.,error_if_nonfinite=True);actor.step()
            change=joint_kl(policy,monitor,old,bs)
            if change>config['MAX_KL']:
                policy.load_state_dict(saved);actor.load_state_dict(saved_opt);rejected+=1;stop='max_kl_rollback';break
            accepted+=1;losses.append(float(loss.detach()))
            if change>=config['TARGET_KL']:stop='target_kl';break
        if stop:break
    full=joint_kl(policy,rows,reference,bs)
    if full>config['MAX_KL']:
        policy.load_state_dict(initial);actor.load_state_dict(initial_optimizer);rejected+=accepted;accepted=0
        full=joint_kl(policy,rows,reference,bs);stop='full_batch_rollback'
    value=fit_value(policy,critic,rows,config,config['VALUE_EPOCHS'])
    return dict(optimizer_version=OPTIMIZER_VERSION,actor_updates=accepted,rejected_actor_updates=rejected,
        actor_loss_mean=float(np.mean(losses)) if losses else 0.,joint_kl=full,target_kl=config['TARGET_KL'],max_kl=config['MAX_KL'],
        stop_reason=stop,value_warmup=warm,**value)
