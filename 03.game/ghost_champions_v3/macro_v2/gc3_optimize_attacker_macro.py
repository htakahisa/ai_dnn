"""Detached value fitting and exact joint-KL checks for the three-output actor."""
from copy import deepcopy
import numpy as np
import torch

OPTIMIZER_VERSION='macro-detached-value-joint-kl-bc-replay-v1'


def actor_optimizer(policy,config):
    return torch.optim.Adam(policy.actor_parameters(),lr=config['LEARNING_RATE'])


def value_optimizer(policy,config):
    return torch.optim.Adam(policy.value_parameters(),lr=config.get('VALUE_LEARNING_RATE',.001))


def record_distributions(policy,rows):
    masks={key:np.stack([r['masks'][key] for r in rows]) for key in ('destination','instruction','plant')}
    return policy.distributions(policy.contexts(rows),masks)


def probabilities(policy,rows,batch_size):
    collected={key:[] for key in ('destination','instruction','plant')}
    with torch.no_grad():
        for offset in range(0,len(rows),batch_size):
            distributions=record_distributions(policy,rows[offset:offset+batch_size])
            for key,dist in distributions.items():collected[key].append(dist.probs.detach().clone())
    return {key:torch.cat(parts) for key,parts in collected.items()}


def joint_kl(policy,rows,reference,batch_size):
    total=0.
    with torch.no_grad():
        for offset in range(0,len(rows),batch_size):
            batch=rows[offset:offset+batch_size];distributions=record_distributions(policy,batch)
            active=torch.from_numpy(np.stack([r['active'] for r in batch]))
            terms=torch.zeros(len(batch))
            for key,dist in distributions.items():
                old=reference[key][offset:offset+len(batch)]
                # Avoid renormalizing saved probabilities a second time: the
                # identical policy must have exactly zero KL after rollback.
                tiny=torch.finfo(old.dtype).tiny
                term=(old*(old.clamp_min(tiny).log()-dist.probs.clamp_min(tiny).log())).sum(-1)
                if key!='plant':term=(term*active).sum(-1)
                terms+=term
            total+=float(terms.sum())
    result=max(0.,total/len(rows))
    if not np.isfinite(result):raise RuntimeError('Nonfinite macro joint KL')
    return result


def fit_value(policy,optimizer,rows,config,epochs):
    # Cache the current actor representation once. Only the critic sees grads.
    contexts=[];batch_size=config['BATCH_SIZE']
    with torch.no_grad():
        for offset in range(0,len(rows),batch_size):contexts.append(policy.contexts(rows[offset:offset+batch_size]).detach())
    contexts=torch.cat(contexts);returns=torch.tensor([r['return'] for r in rows],dtype=torch.float32)
    def mse():
        with torch.no_grad():return float((policy.critic(contexts).squeeze(-1)-returns).square().mean())
    before=mse();losses=[]
    for _ in range(epochs):
        order=np.random.permutation(len(rows))
        for offset in range(0,len(rows),batch_size):
            slots=order[offset:offset+batch_size]
            loss=.5*(policy.critic(contexts[slots]).squeeze(-1)-returns[slots]).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite macro value loss')
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.value_parameters(),1.,error_if_nonfinite=True)
            optimizer.step();losses.append(float(loss.detach()))
    return dict(value_mse_before=before,value_mse_after=mse(),value_updates=len(losses))


def optimize_macro(policy,optimizer,critic_optimizer,rows,config,*,teacher_rows=None,warmup=False):
    batch_size=config['BATCH_SIZE'];target_kl=config.get('TARGET_KL',.005);max_kl=config.get('MAX_KL',.01)
    if not 0<target_kl<=max_kl:raise ValueError('Require 0 < TARGET_KL <= MAX_KL')
    replay_weight=config.get('BC_REPLAY_WEIGHT',0.)
    if replay_weight<0 or replay_weight and not teacher_rows:raise ValueError('BC replay needs saved teacher data')
    policy.eval();reference=probabilities(policy,rows,batch_size)
    monitor_count=min(len(rows),config.get('KL_MONITOR_ROWS',128))
    if monitor_count<1:raise ValueError('KL_MONITOR_ROWS must be positive')
    monitor_indices=np.linspace(0,len(rows)-1,monitor_count,dtype=int)
    monitor=[rows[i] for i in monitor_indices]
    monitor_reference={key:values[monitor_indices] for key,values in reference.items()}
    before_optimizer=deepcopy(optimizer.state_dict())
    warmup_stats={}
    if warmup and config.get('VALUE_WARMUP_EPOCHS',10):
        warmup_stats=fit_value(policy,critic_optimizer,rows,config,config.get('VALUE_WARMUP_EPOCHS',10))
    # Snapshot AFTER value warmup so actor rollback never erases value fitting.
    before_policy=deepcopy(policy.state_dict())
    advantages=np.asarray([r['advantage'] for r in rows],dtype=np.float32)
    advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
    losses=[];ratios=[];bc_losses=[];accepted=rejected=0;stop_reason=None;monitor_kl=0.
    policy.train()
    for epoch in range(config['PPO_EPOCHS']):
        order=np.random.permutation(len(rows))
        for offset in range(0,len(rows),batch_size):
            slots=order[offset:offset+batch_size];batch=[rows[i] for i in slots]
            logp,entropy,_=policy.evaluate_records(batch)
            ratio=(logp-torch.tensor([r['logp'] for r in batch],dtype=torch.float32)).exp()
            adv=torch.from_numpy(advantages[slots]);clip=config['PPO_CLIP']
            actor=-torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()
            loss=actor-config['ENTROPY_WEIGHT']*entropy.mean();bc_loss=loss.new_zeros(())
            if replay_weight:
                count=min(len(teacher_rows),config.get('BC_REPLAY_BATCH_SIZE',16))
                # Shuffle chronological blocks so a tiny BC minibatch does
                # not re-encode every match's full ledger at once.
                first=int(np.random.randint((len(teacher_rows)+count-1)//count))*count
                labels=teacher_rows[first:first+count]
                bc_logp,_,_=policy.evaluate_records(labels);bc_loss=-bc_logp.mean()
                loss=loss+replay_weight*bc_loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite macro actor loss')
            snapshot=deepcopy(policy.state_dict());optim_snapshot=deepcopy(optimizer.state_dict())
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor_parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            candidate_kl=joint_kl(policy,monitor,monitor_reference,batch_size)
            if candidate_kl>max_kl:
                policy.load_state_dict(snapshot);optimizer.load_state_dict(optim_snapshot)
                rejected+=1;stop_reason='max_kl_rollback';break
            accepted+=1;monitor_kl=candidate_kl;losses.append(float(loss.detach()));bc_losses.append(float(bc_loss.detach()))
            ratios.extend(((ratio.detach()-1).abs()>clip).tolist())
            if candidate_kl>=target_kl:stop_reason='target_kl';break
        if stop_reason:break
    policy.eval()
    full_kl=joint_kl(policy,rows,reference,batch_size)
    if full_kl>max_kl:
        policy.load_state_dict(before_policy);optimizer.load_state_dict(before_optimizer)
        rejected+=accepted;accepted=0;monitor_kl=0.;full_kl=joint_kl(policy,rows,reference,batch_size)
        stop_reason='full_batch_max_kl_rollback'
    value_stats=fit_value(policy,critic_optimizer,rows,config,config.get('VALUE_EPOCHS',5))
    return dict(optimizer_version=OPTIMIZER_VERSION,actor_loss_mean=float(np.mean(losses)) if losses else 0.,
        bc_replay_loss_mean=float(np.mean(bc_losses)) if bc_losses else 0.,actor_updates=accepted,rejected_actor_updates=rejected,
        joint_kl=full_kl,monitor_joint_kl=monitor_kl,target_kl=target_kl,max_kl=max_kl,
        clip_fraction=float(np.mean(ratios)) if ratios else 0.,stop_reason=stop_reason,
        value_warmup=warmup_stats,**value_stats)


def optimize(policy,optimizer,rows,config):
    """Compatibility entry point for small tests; production keeps both optimizers."""
    stats=optimize_macro(policy,optimizer,value_optimizer(policy,config),rows,config)
    return stats['actor_loss_mean']
