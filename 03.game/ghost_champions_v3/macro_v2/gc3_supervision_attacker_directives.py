"""Training-only public command supervision; inference retains every legal action."""
import random
import numpy as np
import torch
from ..gc3_engine_actions import MODES
from ..gc3_observation import ARENA_SHAPE
from ..preplant.gc3_features_attacker_navigation import public_layout,walking_distances
from .gc3_directives_attacker_macro import INSTRUCTIONS


def instruction_options(row):
    """All legal improving moves, or holding at the target; never one forced route."""
    cached=row.get('_instruction_options')
    if cached is not None:return cached
    obs=row['obs'];directive=row['macro_directive'];order=directive['instruction']
    if order not in INSTRUCTIONS:raise ValueError('Unknown instruction')
    board,units=public_layout(obs)
    origin=tuple(np.rint(obs[:2]*ARENA_SHAPE).astype(int))
    legal=np.asarray(row['masks']['position'],dtype=bool).reshape(ARENA_SHAPE)
    modes=np.asarray(row['masks']['mode'],dtype=bool)
    hold=order=='WAIT'
    options=np.zeros(ARENA_SHAPE,dtype=bool)
    distance=None
    if not hold:
        target=directive['retreat_target'] if order=='RETREAT' else directive['destination']
        walls=np.asarray(board[0]>.5,dtype=bool)
        targets=np.zeros(ARENA_SHAPE,dtype=bool);targets[tuple(target)]=True
        distances=walking_distances(walls.tobytes(),targets.tobytes())
        distance=int(distances[origin]);hold=distance==0
        if distance>0:options=legal & (distances>=0) & (distances<distance)
    if hold:options[origin]=legal[origin]
    # Do not teach a forced cancellation of planting/orb collection, nor demand
    # progress when all improving neighbours are occupied/physically unavailable.
    objective=bool(modes[MODES.index('PLANT')] or modes[MODES.index('COLLECT_ORB')] or obs[12]>0 or obs[14]>0)
    can_move=bool(modes[0] or modes[2])
    eligible=bool(options.any() and (hold or can_move) and not objective)
    combat=bool(((units[5:,0]>.5)&(units[5:,2]>.5)&(units[5:,3]>.5)).any())
    result=dict(instruction=order,positions=options.reshape(-1),hold=hold,eligible=eligible,
                origin=origin,distance=distance,combat=combat)
    row['_instruction_options']=result
    return result


def instruction_loss(policy,rows,contexts,config):
    """Increase joint command probability, balanced across the three instructions."""
    mode_record=dict(phase='standard',obs=np.stack([r['obs'] for r in rows]),
        hints=np.stack([r['hints'] for r in rows]),masks={'mode':np.stack([r['masks']['mode'] for r in rows])})
    position_record=dict(mode_record,masks={'position':np.stack([r['masks']['position'] for r in rows])})
    mode=policy.distribution(contexts,mode_record,'mode').probs
    position=policy.distribution(contexts,position_record,'position').probs
    options=[instruction_options(r) for r in rows]
    allowed=torch.as_tensor(np.stack([o['positions'] for o in options]),device=contexts.device)
    positional=(position*allowed).sum(-1)
    move=mode[:,0]+mode[:,2]
    # WATCH/ABILITY/PLANT/DEFUSE/COLLECT_ORB hold position. ULTIMATE is omitted
    # because ESCAPE/NEON may move; runtime remains free to use it.
    stationary=mode[:,[1,3,5,6,7]].sum(-1)
    hold=torch.tensor([o['hold'] for o in options],device=contexts.device)
    probability=move*positional+torch.where(hold,stationary,torch.zeros_like(stationary))
    losses=[];report={}
    for order in INSTRUCTIONS:
        ids=[i for i,o in enumerate(options) if o['eligible'] and o['instruction']==order]
        report[order+'_samples']=len(ids)
        if not ids:continue
        weights=contexts.new_tensor([config['INSTRUCTION_COMBAT_WEIGHT'] if options[i]['combat'] else 1. for i in ids])
        losses.append((-probability[ids].clamp_min(1e-8).log()*weights).mean())
        report[order+'_probability']=float(probability[ids].detach().mean())
    loss=torch.stack(losses).mean() if losses else contexts.sum()*0.
    return loss,report


def optimize_directives(policy,anchor,optimizer,rows,config):
    """Causal PPO plus instruction probability loss, only for the input adapter."""
    advantages=np.array([r['advantage'] for r in rows],dtype=np.float32)
    advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
    reports=[];policy.train()
    for _ in range(config['PPO_EPOCHS']):
        chunks=list(range(0,len(rows),config['BATCH_SIZE']));random.shuffle(chunks)
        for offset in chunks:
            batch=rows[offset:offset+config['BATCH_SIZE']]
            context=policy.contexts(batch)
            logp,entropy,values=policy.evaluate_contexts(batch,context)
            old=context.new_tensor([r['logp'] for r in batch])
            adv=torch.from_numpy(advantages[offset:offset+len(batch)])
            ratio=(logp-old).exp()
            actor=-torch.minimum(ratio*adv,ratio.clamp(1-config['PPO_CLIP'],1+config['PPO_CLIP'])*adv).mean()
            with torch.no_grad():anchor_context=anchor.contexts(batch)
            divergence=context.new_zeros(())
            for kind in ('mode','position','watch','target','ally','site'):
                ids=[i for i,r in enumerate(batch) if kind in r['selections']]
                if not ids:continue
                record=dict(phase='standard',obs=np.stack([batch[i]['obs'] for i in ids]),
                    goal_site=[batch[i].get('goal_site') for i in ids],
                    masks={kind:np.stack([batch[i]['masks'][kind] for i in ids])},
                    hints=np.stack([batch[i]['hints'] for i in ids]))
                if kind=='site':
                    from ..preplant.gc3_features_attacker_navigation import navigation_features
                    record.update(navigation_features=np.stack([navigation_features(batch[i],batch[i].get('goal_site')) for i in ids]),
                        neutral_navigation_features=np.stack([navigation_features(batch[i],None) for i in ids]))
                baseline=anchor.distribution(anchor_context[ids],record,kind)
                current=policy.distribution(context[ids],record,kind)
                divergence+=torch.distributions.kl_divergence(baseline,current).sum()/len(batch)
            supervised,report=instruction_loss(policy,batch,context,config)
            # The original critic is frozen. Do not bend the command adapter to
            # fit new returns through a critic that cannot itself be updated.
            loss=actor-config['ENTROPY_WEIGHT']*entropy.mean()+config['BC_KL_WEIGHT']*divergence+config['INSTRUCTION_LOSS_WEIGHT']*supervised
            if not torch.isfinite(loss):raise RuntimeError('Non-finite instruction PPO loss')
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            reports.append(dict(loss=float(loss.detach()),instruction_loss=float(supervised.detach()),**report))
    policy.eval()
    return float(np.mean([r['loss'] for r in reports])),reports
