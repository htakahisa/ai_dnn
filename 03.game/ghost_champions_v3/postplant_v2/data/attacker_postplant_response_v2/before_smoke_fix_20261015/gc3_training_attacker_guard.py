"""Separate postplant run; original macro, micro and guard checkpoints remain intact."""
import hashlib
import json
from pathlib import Path
import uuid
import numpy as np
import torch
from ..gc3_io import write_json
from ..gc3_policy import FreshPolicy
from ..preplant.gc3_data_attacker_preplant import file_hash
from ..preplant.gc3_train_attacker_pipeline import run_directories,serial_config,seed_all,rng_state,restore_rng
from ..preplant.gc3_observation_attacker_history import MatchHistory
from ..preplant.gc3_rollout_attacker_preplant import choose_roster
from ..tools.gc3_training_metrics import record_training_metrics,reconcile_resume_logs
from ..macro_v2.gc3_training_attacker_directives import source_signature as entry_signature
from ..macro_v2.gc3_training_attacker_macro import select_round_rows
from .gc3_policy_attacker_guard import GuardPolicy
from .gc3_rollout_attacker_guard import load_entry,play_guard
from .gc3_reward_attacker_guard import reward_settings,REWARD_VERSION
from .gc3_optimize_attacker_guard import actor_optimizer,value_optimizer,optimize,OPTIMIZER_VERSION


def source_signature():
    digest=hashlib.sha256(entry_signature().encode())
    for path in sorted(Path(__file__).parent.glob('*.py')):
        if path.name.startswith('gc3_train_'):continue
        digest.update(path.name.encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def initialize(config,data):
    inputs={key:file_hash(config[key]) for key in ('INITIAL_CHECKPOINT','ENTRY_MACRO_CHECKPOINT','ENTRY_MICRO_CHECKPOINT')}
    if config.get('START_CHECKPOINT') is not None:inputs['START_CHECKPOINT']=file_hash(config['START_CHECKPOINT'])
    manifest=dict(source_hash=source_signature(),config=serial_config(config),inputs=inputs,reward_settings=reward_settings(),
        initialization='current_trained_guard_plus_zero_objective_residual',scope='postplant_only',entry_models_frozen=True,
        optimizer_version=OPTIMIZER_VERSION)
    saved=None
    if config['RESUME']:
        if json.loads((data/'manifest.json').read_text(encoding='utf-8'))!=manifest:raise RuntimeError('Guard source/settings/entry changed; choose a new RUN_NAME')
        policy,saved=GuardPolicy.load(data/'latest.pt')
    else:
        seed_all(config['SEED'])
        if config.get('START_CHECKPOINT') is not None:
            policy,previous=GuardPolicy.load(config['START_CHECKPOINT'])
        else:
            base,previous=FreshPolicy.load(config['INITIAL_CHECKPOINT'],role='attacker');policy=GuardPolicy.from_guard(base)
        if not previous.get('trained'):raise ValueError('Supply the trained current postplant guard')
        policy.feature_version=2
        write_json(data/'manifest.json',manifest)
        policy.save(data/'initial.pt',trained=True,stage='guard_rl',step=0,inherited_step=previous['step'],inputs=inputs)
    policy.collect_response_targets=config.get('RESPONSE_LOSS_WEIGHT',0.)>0
    actor=actor_optimizer(policy,config);critic=value_optimizer(policy,config)
    if saved:
        if saved.get('optimizer_version')!=OPTIMIZER_VERSION:raise ValueError('Wrong guard optimizer version')
        actor.load_state_dict(saved['optimizer']);critic.load_state_dict(saved['value_optimizer']);restore_rng(saved)
    return policy,actor,critic,saved,inputs


def evaluate(policy,entry,config):
    state=rng_state();rounds=[];collect=policy.collect_response_targets;policy.collect_response_targets=False
    try:
        for i,code in enumerate(config['OPPONENT_CODES']):
            history=MatchHistory();view=None
            for repeat in range(config['EVALUATION_ROUNDS_PER_OPPONENT']):
                _,stats,view=play_guard(code,config['SEED']+100000+i*1000+repeat,policy=policy,entry=entry,
                    config=config,history=history,view=view)
                rounds.append(stats)
        n=len(rounds);started=[r for r in rounds if r['defuse_started']]
        return dict(rounds=rounds,success_rate=sum(r['success'] for r in rounds)/n,defused_rate=sum(r['defused'] for r in rounds)/n,
            defuse_start_rate=len(started)/n,defuse_started_rounds=len(started),
            defuse_started_win_rate=sum(r['success'] for r in started)/len(started) if started else None,
            reward_mean=float(np.mean([sum(r['reward_component_sums'].values()) for r in rounds])),
            entry_models_frozen=True,sampled=True,opponent_ai='default')
    finally:restore_rng(state);policy.collect_response_targets=collect


def select_response_rows(rows,stats_list,target,fraction):
    """Retain rare public alerts while still sampling the other completed rounds."""
    if not fraction:return select_round_rows(rows,stats_list,target)
    starts=np.cumsum([0]+[stats['decisions'] for stats in stats_list])
    if int(starts[-1])!=len(rows):raise ValueError('Round counts do not match guard records')
    selected=set()
    if target>=len(stats_list):
        selected.update(int(np.random.randint(a,b)) for a,b in zip(starts[:-1],starts[1:]) if b>a)
    alerted=[i for i,r in enumerate(rows) if r.get('response_targets') is not None and i not in selected]
    count=min(len(alerted),max(0,int(target*fraction)-sum(rows[i].get('response_targets') is not None for i in selected)),target-len(selected))
    if count:selected.update(map(int,np.random.choice(alerted,count,replace=False)))
    pool=np.asarray([i for i in range(len(rows)) if i not in selected])
    count=min(target-len(selected),len(pool))
    if count:selected.update(map(int,np.random.choice(pool,count,replace=False)))
    selected=np.asarray(sorted(selected))
    for stats,a,b in zip(stats_list,starts[:-1],starts[1:]):
        stats.update(sampled_decisions=stats['decisions'],decisions=int(((selected>=a)&(selected<b)).sum()))
    return [rows[i] for i in selected]


def save_records(path,rows):
    # Every collected public observation and action is retained, including rows
    # not selected for this PPO update. No future outcome enters its input.
    metadata=[dict({key:row[key] for key in ('ability','ultimate','phase','selections','reward','return','advantage','value','logp','age','discount','reward_components')},
                   response_targets=row.get('response_targets')) for row in rows]
    with path.open('xb') as handle:
        np.savez_compressed(handle,obs=np.stack([r['obs'] for r in rows]),hints=np.stack([r['hints'] for r in rows]),
            metadata=np.asarray(json.dumps(metadata,default=lambda v:v.tolist())),
            **{'mask_'+key:np.stack([r['masks'][key] for r in rows]) for key in rows[0]['masks']},
            **{'full_mask_'+key:np.stack([r['full_masks'][key] for r in rows]) for key in rows[0]['full_masks']})


def train(config):
    if config['FIXED_OPPONENT_AI']!='default':raise ValueError('Keep logic opponents for this stage')
    if not 0<config['TARGET_KL']<=config['MAX_KL'] or min(config['VALUE_EPOCHS'],config['ENTRY_MAX_ATTEMPTS'],config['KL_MONITOR_ROWS'])<1:
        raise ValueError('Invalid guard training settings')
    if not 0<=config.get('RESPONSE_SAMPLE_FRACTION',0.)<=1 or min(config.get('RESPONSE_LOSS_WEIGHT',0.),config.get('ANCHOR_KL_WEIGHT',0.))<0:
        raise ValueError('Invalid response training settings')
    torch.set_num_threads(config['TORCH_THREADS']);entry=load_entry(config)
    with run_directories(config) as (data,logs):
        policy,actor,critic,saved,inputs=initialize(config,data)
        anchor=None
        if config.get('ANCHOR_KL_WEIGHT',0.):
            anchor,_=GuardPolicy.load(data/'initial.pt');anchor.requires_grad_(False).eval()
        step,episode,next_eval,best_rank=(saved['step'],saved['episode'],saved['next_eval'],tuple(saved['best_rank'])) if saved else (
            0,0,config['EVALUATE_EVERY_STEPS'],(-1.,-1.,-1e30))
        if saved:reconcile_resume_logs(logs,step,episode)
        if saved and (logs/'ppo_updates.jsonl').exists():
            updates=[json.loads(line) for line in (logs/'ppo_updates.jsonl').read_text(encoding='utf-8').splitlines() if line]
            with (logs/'ppo_updates.jsonl').open('w',encoding='utf-8') as f:
                for row in updates:
                    if row['step']<=step:f.write(json.dumps(row)+'\n')
        last_update=saved.get('last_update') if saved else None
        evaluations_without_improvement=saved.get('evaluations_without_improvement',0) if saved else 0
        stopped_early=False
        def save(complete=False):
            policy.save(data/'latest.pt',trained=True,stage='guard_rl',step=step,episode=episode,next_eval=next_eval,
                best_rank=list(best_rank),optimizer=actor.state_dict(),value_optimizer=critic.state_dict(),last_update=last_update,
                evaluations_without_improvement=evaluations_without_improvement,
                optimizer_version=OPTIMIZER_VERSION,reward_version=REWARD_VERSION,inputs=inputs,**rng_state())
            write_json(logs/'status.json',dict(step=step,episode=episode,complete=complete,stage='guard_rl',last_update=last_update,
                stopped_early=stopped_early,evaluations_without_improvement=evaluations_without_improvement,
                entry_macro_step=entry['macro_step'],entry_micro_step=entry['micro_step'],entry_models_frozen=True,best_score=best_rank[0]))
        def evaluation(initial=False):
            nonlocal best_rank,next_eval,evaluations_without_improvement
            report=evaluate(policy,entry,config);report['step']=step
            write_json(logs/('initial_evaluation.json' if initial else f'evaluation_{step}.json'),report)
            rank=(report['success_rate'],report['defuse_started_win_rate'] if report['defuse_started_win_rate'] is not None else -1.,report['reward_mean'])
            if rank>best_rank:
                evaluations_without_improvement=0
                best_rank=rank;policy.save(data/'best.pt',trained=True,stage='guard_rl',step=step,evaluation=report,inputs=inputs,reward_version=REWARD_VERSION)
            elif not initial:evaluations_without_improvement+=1
            while next_eval<=step:next_eval+=config['EVALUATE_EVERY_STEPS']
            save();print(f"Guard ST={step} win={report['success_rate']:.1%} defused={report['defused_rate']:.1%}",flush=True)
        save()
        if not saved:evaluation(True)
        while step<config['TRAINING_STEPS']:
            rows=[];stats_list=[];target=min(config['ROLLOUT_DECISIONS'],config['TRAINING_STEPS']-step)
            while len(rows)<target:
                history=MatchHistory();view=None;first=episode
                for _ in range(config['ROUNDS_PER_MATCH']):
                    code=choose_roster(config['SEED'],episode,config['OPPONENT_CODES'])
                    part,stats,view=play_guard(code,config['SEED']+episode,policy=policy,entry=entry,config=config,history=history,view=view)
                    if not part:raise RuntimeError('Live plant supplied no guard decisions')
                    rows.extend(part);stats_list.append(stats);episode+=1
                root=data/'history';root.mkdir(exist_ok=True)
                path=root/f'match_{first:06d}_{uuid.uuid4().hex}.npz';view.save(path)
            root=data/'rollouts';root.mkdir(exist_ok=True)
            save_records(root/f'batch_{step:06d}_{uuid.uuid4().hex}.npz',rows)
            rows=select_response_rows(rows,stats_list,target,config.get('RESPONSE_SAMPLE_FRACTION',0.))
            for stats in stats_list:
                with (logs/'episodes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(stats)+'\n')
            record_training_metrics(logs,step,episode,policy,rows,config['BATCH_SIZE'])
            last_update=optimize(policy,actor,critic,rows,config,warmup=step==0,anchor=anchor)
            last_update.update(step=step+len(rows),start_step=step,episode=episode)
            with (logs/'ppo_updates.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(last_update)+'\n')
            step+=len(rows);save()
            print(f"Guard ST={step} KL={last_update['joint_kl']:.5f} value_MSE={last_update['value_mse_after']:.3f}",flush=True)
            if step>=next_eval or step>=config['TRAINING_STEPS']:
                evaluation()
                if config.get('EVALUATION_PATIENCE',0) and evaluations_without_improvement>=config['EVALUATION_PATIENCE']:
                    stopped_early=True;save();break
        best,metadata=GuardPolicy.load(data/'best.pt');report=evaluate(best,entry,config);report['step']=metadata['step']
        write_json(logs/'final_evaluation.json',report);save(not stopped_early)
