"""Fresh three-output macro BC/PPO; frozen lower policies and exact step accounting."""
import json
from pathlib import Path
import uuid
import numpy as np
import torch
from ..gc3_io import write_json
from ..gc3_rollout import module_digest
from ..preplant.gc3_data_attacker_preplant import file_hash
from ..preplant.gc3_observation_attacker_history import MatchHistory
from ..preplant.gc3_rollout_attacker_preplant import choose_roster
from ..preplant.gc3_train_attacker_pipeline import (
    run_directories,serial_config,seed_all,rng_state,restore_rng,check_preplant_runtime,
)
from ..tools.gc3_training_metrics import record_training_metrics,reconcile_resume_logs
from .gc3_optimize_attacker_macro import optimize, optimize_macro, actor_optimizer, value_optimizer, OPTIMIZER_VERSION
from .gc3_training_attacker_directives import source_signature
from .gc3_policy_attacker_macro import MacroPolicy
from .gc3_candidates_attacker_macro import MacroCandidates
from .gc3_rollout_attacker_macro import load_lower_policies,play_macro
from .gc3_reward_attacker_macro import reward_settings,REWARD_VERSION
from .gc3_bc_attacker_macro import MacroTeacher,save_teacher,load_teacher,optimize_bc


def initialize(config,data,candidates):
    inputs={k:file_hash(config[k]) for k in ('PREPLANT_CHECKPOINT','POSTPLANT_CHECKPOINT')}
    initial_path=config.get('INITIAL_CHECKPOINT')
    if initial_path is not None:inputs['INITIAL_CHECKPOINT']=file_hash(initial_path)
    replay_path=config.get('BC_REPLAY_PATH')
    if replay_path is not None:
        replay_path=Path(replay_path)
        replay_files=sorted([*replay_path.glob('*.pt'),*replay_path.glob('*.npz')])
        if not replay_files:raise ValueError('Saved macro BC replay data is missing')
        inputs['BC_REPLAY_FILES']={p.name:file_hash(p) for p in replay_files}
    settings=serial_config(config);settings.pop('TRAINING_STEPS',None)
    manifest=dict(source_hash=source_signature(),input_hashes=inputs,config=settings,candidate_signature=candidates.signature,
        reward_settings=reward_settings(),lower_models_frozen=True,optimizer_version=OPTIMIZER_VERSION,
        initialization='saved_bc_macro_actor' if initial_path is not None else 'fresh_macro_with_optional_simple_logic_bc',
        outputs='five player destinations, team plant coordinate, five WAIT/ADVANCE/RETREAT instructions',
        learning_scope='preplant macro decisions only; actual full-round reward including fixed guard')
    saved=None
    if config['RESUME']:
        if json.loads((data/'manifest.json').read_text(encoding='utf-8'))!=manifest:
            raise RuntimeError('Macro source/settings/map/lower models changed; choose a new RUN_NAME')
        policy,saved=MacroPolicy.load(data/'latest.pt')
        if saved.get('candidate_signature')!=candidates.signature or saved.get('lower_checkpoint_hashes')!={
                k:inputs[k] for k in ('PREPLANT_CHECKPOINT','POSTPLANT_CHECKPOINT')}:
            raise ValueError('Macro checkpoint map/lower model mismatch')
        if saved.get('stage') not in ('macro_bc','macro_rl'):raise ValueError('Wrong macro training stage')
    else:
        seed_all(config['SEED'])
        initial=None
        if initial_path is None:policy=MacroPolicy.fresh('attacker',config['SEED']).eval()
        else:
            policy,initial=MacroPolicy.load(initial_path)
            if not initial.get('trained') or not initial.get('bc_epoch') or initial.get('step')!=0:
                raise ValueError('Recovery requires the preserved trained macro BC best at ST0')
            if initial.get('candidate_signature')!=candidates.signature or initial.get('lower_checkpoint_hashes')!={
                    k:inputs[k] for k in ('PREPLANT_CHECKPOINT','POSTPLANT_CHECKPOINT')}:
                raise ValueError('Macro BC best requires its original map and lower models')
            policy._initial_bc_epoch=int(initial['bc_epoch'])
        write_json(data/'manifest.json',manifest)
        policy.save(data/'initial.pt',stage='macro_bc',trained=initial is not None,step=0,
            bc_epoch=getattr(policy,'_initial_bc_epoch',0),optimizer_version=OPTIMIZER_VERSION,
            candidate_signature=candidates.signature,lower_checkpoint_hashes={k:inputs[k] for k in ('PREPLANT_CHECKPOINT','POSTPLANT_CHECKPOINT')})
    optimizer=actor_optimizer(policy,config)
    if saved:optimizer.load_state_dict(saved['optimizer']);restore_rng(saved)
    else:
        # Collection may be interrupted before the first BC epoch. Keep a
        # resumable fresh model while reusing committed teacher shards.
        policy.save(data/'latest.pt',stage='macro_bc',trained=bool(getattr(policy,'_initial_bc_epoch',0)),step=0,
            bc_epoch=getattr(policy,'_initial_bc_epoch',0),optimizer_version=OPTIMIZER_VERSION,
            candidate_signature=candidates.signature,lower_checkpoint_hashes={k:inputs[k] for k in ('PREPLANT_CHECKPOINT','POSTPLANT_CHECKPOINT')},
            optimizer=optimizer.state_dict(),**rng_state())
    return policy,optimizer,saved,inputs


def evaluate(policy,preplant,guard,candidates,config):
    state=rng_state()
    try:
        rounds=[]
        for i,code in enumerate(config['OPPONENT_CODES']):
            history=MatchHistory();view=None
            for repeat in range(config['EVALUATION_ROUNDS_PER_OPPONENT']):
                _,stats,view=play_macro(code,config['SEED']+100000+i*1000+repeat,policy=policy,
                    preplant=preplant,guard=guard,candidates=candidates,history=history,macro_history=view,
                    gamma=config['GAMMA'],gae_lambda=config['GAE_LAMBDA'])
                rounds.append(stats)
        n=len(rounds)
        return dict(rounds=rounds,success_rate=sum(r['success'] for r in rounds)/n,
            actual_round_win_rate=sum(r['success'] for r in rounds)/n,plant_rate=sum(r['planted'] for r in rounds)/n,
            reward_mean=float(np.mean([r['environment_reward'] for r in rounds])),
            no_defuse_bonus_rate=sum(r['reward_component_sums'].get('no_defuse_start',0)>0 for r in rounds)/n,
            destination_changes_mean=float(np.mean([r['destination_changes'] for r in rounds])),
            plant_target_changes_mean=float(np.mean([r['plant_target_changes'] for r in rounds])),
            instructions={key:sum(r['instructions'].get(key,0) for r in rounds) for key in ('WAIT','ADVANCE','RETREAT')},
            macro_only_learning=True,lower_models_frozen=True,sampled=True,opponent_ai='default')
    finally:restore_rng(state)


def warmup(policy,optimizer,saved,preplant,guard,candidates,config,data,logs,identity):
    if saved and saved['stage']=='macro_rl':return optimizer,int(saved.get('bc_epoch',0))
    if config.get('INITIAL_CHECKPOINT'):
        return optimizer,int(saved['bc_epoch']) if saved else policy._initial_bc_epoch
    if not config['BC_WARMUP_STEPS']:return optimizer,0
    root=data/'bc_teacher';root.mkdir(exist_ok=True)
    rows=load_teacher(root,candidates)
    match_index=len(list(root.glob('*.pt')))
    while len(rows)<config['BC_WARMUP_STEPS']:
        history=MatchHistory();view=None;batch=[]
        for offset in range(config['ROUNDS_PER_MATCH']):
            episode=match_index*config['ROUNDS_PER_MATCH']+offset;seed=config['SEED']+500000+episode
            teacher=MacroTeacher(policy,config,seed)
            code=choose_roster(config['SEED'],episode,config['OPPONENT_CODES'])
            part,stats,view=play_macro(code,seed,policy=teacher,preplant=preplant,guard=guard,candidates=candidates,
                history=history,macro_history=view,gamma=config['GAMMA'],gae_lambda=config['GAE_LAMBDA'])
            batch.extend(part)
            with (logs/'bc_teacher_rounds.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(stats)+'\n')
        if not batch:raise RuntimeError('Macro teacher supplied no preplant decisions')
        path=root/f'match_{match_index:06d}.pt'
        if path.with_suffix('.npz').exists():path=path.with_name(path.stem+'_'+uuid.uuid4().hex+'.pt')
        batch=batch[:config['BC_WARMUP_STEPS']-len(rows)]
        save_teacher(path,batch,view,candidates);rows.extend(batch);match_index+=1
        write_json(logs/'status.json',dict(stage='macro_bc',step=0,bc_epoch=0,teacher_steps=len(rows),
            teacher_target=config['BC_WARMUP_STEPS'],complete=False))
        print(f"Macro teacher labels={len(rows)}/{config['BC_WARMUP_STEPS']}",flush=True)
    for group in optimizer.param_groups:group['lr']=config['BC_LEARNING_RATE']
    epoch=int(saved.get('bc_epoch',0)) if saved else 0
    for epoch in range(epoch+1,config['BC_WARMUP_EPOCHS']+1):
        loss=optimize_bc(policy,optimizer,rows,config)
        policy.save(data/'latest.pt',stage='macro_bc',trained=True,step=0,bc_epoch=epoch,
            optimizer=optimizer.state_dict(),**identity,**rng_state())
        write_json(logs/'status.json',dict(stage='macro_bc',step=0,bc_epoch=epoch,bc_loss=loss,
            teacher_steps=len(rows),complete=False))
        with (logs/'bc_metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(epoch=epoch,step=0,train_loss=loss))+'\n')
        print(f'Macro BC epoch={epoch} loss={loss:.4f} labels={len(rows)}',flush=True)
    return actor_optimizer(policy,config),epoch


def select_round_rows(rows,stats_list,target):
    """Cover every completed round when possible instead of cutting the match tail."""
    starts=np.cumsum([0]+[stats['decisions'] for stats in stats_list])
    if int(starts[-1])!=len(rows):raise ValueError('Macro round counts do not match records')
    if target>=len(rows):selected=np.arange(len(rows))
    else:
        required=[]
        if target>=len(stats_list):
            required=[int(np.random.randint(start,end)) for start,end in zip(starts[:-1],starts[1:]) if end>start]
        pool=np.setdiff1d(np.arange(len(rows)),np.asarray(required,dtype=int))
        selected=np.sort(np.concatenate((np.asarray(required,dtype=int),np.random.choice(pool,target-len(required),replace=False))))
    for stats,start,end in zip(stats_list,starts[:-1],starts[1:]):
        sampled=stats['decisions']
        stats.update(sampled_decisions=sampled,decisions=int(((selected>=start)&(selected<end)).sum()))
    return [rows[i] for i in selected]


def train(config):
    if config['BC_WARMUP_STEPS']<0 or config['BC_WARMUP_EPOCHS']<1:raise ValueError('Invalid macro BC warmup settings')
    if config.get('VALUE_EPOCHS',5)<1 or config.get('VALUE_WARMUP_EPOCHS',10)<0:raise ValueError('Invalid value learning epochs')
    if config.get('BC_REPLAY_BATCH_SIZE',16)<1:raise ValueError('Invalid BC replay batch size')
    if config.get('FIXED_OPPONENT_AI')!='default':raise ValueError('Use fixed default logic for this stage')
    torch.set_num_threads(config['TORCH_THREADS'])
    from map_data import NEW_MAZE_STR
    candidates=MacroCandidates.configured(np.array([[int(v) for v in row] for row in NEW_MAZE_STR.strip().splitlines()]))
    check_preplant_runtime()
    preplant,guard=load_lower_policies(config['PREPLANT_CHECKPOINT'],config['POSTPLANT_CHECKPOINT'])
    lower_before=module_digest([preplant,guard])
    with run_directories(config) as (data,logs):
        policy,optimizer,saved,inputs=initialize(config,data,candidates)
        identity=dict(candidate_signature=candidates.signature,
            lower_checkpoint_hashes={k:inputs[k] for k in ('PREPLANT_CHECKPOINT','POSTPLANT_CHECKPOINT')},
            reward_version=REWARD_VERSION,optimizer_version=OPTIMIZER_VERSION)
        optimizer,bc_epoch=warmup(policy,optimizer,saved,preplant,guard,candidates,config,data,logs,identity)
        resumed=bool(saved and saved['stage']=='macro_rl')
        critic_optimizer=value_optimizer(policy,config)
        if resumed:
            if saved.get('optimizer_version')!=OPTIMIZER_VERSION or 'value_optimizer' not in saved:
                raise ValueError('Resume requires the corrected split macro optimizers')
            critic_optimizer.load_state_dict(saved['value_optimizer'])
        replay_rows=[]
        if config.get('BC_REPLAY_WEIGHT',0.):
            root=Path(config.get('BC_REPLAY_PATH') or data/'bc_teacher')
            replay_rows=load_teacher(root,candidates)
            if not replay_rows:raise ValueError('BC replay requires saved teacher labels')
        step,episode,next_eval,best_rank=(saved['step'],saved['episode'],saved['next_eval'],tuple(saved['best_rank'])) if resumed else (
            0,0,config['EVALUATE_EVERY_STEPS'],(-1e30,-1.,-1.))
        if resumed:reconcile_resume_logs(logs,step,episode)
        if resumed and (logs/'ppo_updates.jsonl').exists():
            records=[json.loads(line) for line in (logs/'ppo_updates.jsonl').read_text(encoding='utf-8').splitlines() if line]
            with (logs/'ppo_updates.jsonl').open('w',encoding='utf-8') as f:
                for row in records:
                    if row['step']<=step:f.write(json.dumps(row)+'\n')
        last_update=saved.get('last_update') if resumed else None

        def save(complete=False):
            policy.save(data/'latest.pt',stage='macro_rl',trained=bool(step or bc_epoch),step=step,episode=episode,
                bc_epoch=bc_epoch,next_eval=next_eval,best_rank=list(best_rank),optimizer=optimizer.state_dict(),
                value_optimizer=critic_optimizer.state_dict(),last_update=last_update,**identity,**rng_state())
            write_json(logs/'status.json',dict(stage='macro_rl',step=step,episode=episode,bc_epoch=bc_epoch,
                best_score=best_rank[0],complete=complete,lower_models_frozen=True,last_update=last_update))

        def evaluation(initial=False):
            nonlocal best_rank,next_eval
            report=evaluate(policy,preplant,guard,candidates,config);report['step']=step
            write_json(logs/('initial_evaluation.json' if initial else f'evaluation_{step}.json'),report)
            rank=report['reward_mean'],report['actual_round_win_rate'],report['plant_rate']
            if (step or bc_epoch) and rank>best_rank:
                best_rank=rank
                policy.save(data/'best.pt',stage='macro_rl',trained=True,step=step,bc_epoch=bc_epoch,evaluation=report,**identity)
            while next_eval<=step:next_eval+=config['EVALUATE_EVERY_STEPS']
            save()
            print(f"Macro evaluation ST={step} reward={report['reward_mean']:.2f} plant={report['plant_rate']:.1%} win={report['success_rate']:.1%}",flush=True)

        if not resumed:evaluation(initial=True)
        while step<config['TRAINING_STEPS']:
            rows=[];stats_list=[];target=min(config['ROLLOUT_DECISIONS'],config['TRAINING_STEPS']-step)
            while len(rows)<target:
                history=MatchHistory();view=None;first=episode
                for _ in range(config['ROUNDS_PER_MATCH']):
                    code=choose_roster(config['SEED'],episode,config['OPPONENT_CODES'])
                    part,stats,view=play_macro(code,config['SEED']+episode,policy=policy,preplant=preplant,guard=guard,
                        candidates=candidates,history=history,macro_history=view,gamma=config['GAMMA'],gae_lambda=config['GAE_LAMBDA'])
                    if not part:raise RuntimeError('Macro episode supplied no decisions')
                    rows.extend(part);stats_list.append(stats);episode+=1
                path=data/'histories'/f'match_{first:06d}.npz'
                if path.exists():path=path.with_name(path.stem+'_'+uuid.uuid4().hex+'.npz')
                view.save(path)
            rows=select_round_rows(rows,stats_list,target)
            for stats in stats_list:
                with (logs/'episodes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(stats)+'\n')
            record_training_metrics(logs,step,episode,policy,rows,config['BATCH_SIZE'])
            last_update=optimize_macro(policy,optimizer,critic_optimizer,rows,config,teacher_rows=replay_rows,warmup=step==0)
            last_update.update(step=step+len(rows),start_step=step,episode=episode)
            with (logs/'ppo_updates.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(last_update)+'\n')
            if module_digest([preplant,guard])!=lower_before:raise RuntimeError('Macro learning changed lower model weights')
            step+=len(rows);save()
            print(f"Macro ST={step} actor={last_update['actor_loss_mean']:.4f} KL={last_update['joint_kl']:.5f} "
                f"value_MSE={last_update['value_mse_after']:.4f} stop={last_update['stop_reason']}",flush=True)
            if step>=next_eval or step>=config['TRAINING_STEPS']:evaluation()
        best,metadata=MacroPolicy.load(data/'best.pt')
        report=evaluate(best,preplant,guard,candidates,config);report['step']=metadata['step']
        write_json(logs/'final_evaluation.json',report);save(complete=True)
