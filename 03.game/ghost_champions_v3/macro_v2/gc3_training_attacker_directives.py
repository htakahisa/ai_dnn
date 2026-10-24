"""Explicit warm-start PPO for the new micro command inputs only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from ..gc3_policy import FreshPolicy
from ..gc3_io import write_json
from ..preplant.gc3_data_attacker_preplant import file_hash,source_signature as base_signature
from ..preplant.gc3_policy_attacker_history import load_preplant_policy
from ..preplant.gc3_policy_attacker_site import RETRIEVAL_SCHEMA_HASH
from ..preplant.gc3_train_attacker_pipeline import (
    run_directories,serial_config,seed_all,rng_state,restore_rng,check_preplant_runtime,
)
from ..preplant.gc3_observation_attacker_history import MatchHistory
from ..preplant.gc3_rollout_attacker_preplant import choose_roster
from ..tools.gc3_training_metrics import record_training_metrics,reconcile_resume_logs
from .gc3_candidates_attacker_macro import MacroCandidates
from .gc3_directives_attacker_macro import DirectedMicroPolicy
from .gc3_curriculum_attacker_directives import directive_policy
from .gc3_reward_attacker_directives import reward_settings,REWARD_VERSION
from .gc3_rollout_attacker_directives import play_directives
from .gc3_supervision_attacker_directives import optimize_directives
from .gc3_directives_attacker_macro import INSTRUCTIONS


def source_signature():
    digest=hashlib.sha256(base_signature().encode())
    root=Path(__file__).parent
    for folder in (root,root.parent/'macro_v1'):
        for path in sorted(folder.glob('*.py')):
            if path.name.startswith('gc3_train_'):continue
            digest.update(str(path.relative_to(root.parent)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def trainable_directives(policy):
    policy.requires_grad_(False)
    for parameter in policy.directive.parameters():parameter.requires_grad_(True)
    return list(policy.directive.parameters())


def base_digest(policy):
    digest=hashlib.sha256()
    for name,value in sorted(policy.state_dict().items()):
        if name.startswith('directive.'):continue
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def initialize(config,data,candidates):
    settings=serial_config(config);settings.pop('ADDITIONAL_TRAINING_STEPS',None)
    inputs={key:file_hash(config[key]) for key in ('INITIAL_CHECKPOINT','POSTPLANT_CHECKPOINT')}
    anchor,initial=load_preplant_policy(config['INITIAL_CHECKPOINT'])
    if anchor.schema_hash!=RETRIEVAL_SCHEMA_HASH or initial.get('stage')!='rl' or not initial.get('trained'):
        raise ValueError('Additional instruction learning requires the completed retrieval RL model')
    anchor.requires_grad_(False).eval()
    inherited=int(initial['step'])
    directive_path=config.get('DIRECTIVE_CHECKPOINT')
    transferred=None
    if directive_path is not None:
        inputs['DIRECTIVE_CHECKPOINT']=file_hash(directive_path)
        transferred,metadata=DirectedMicroPolicy.load(directive_path)
        if not metadata.get('trained') or metadata.get('stage')!='directive_rl' or not transferred.directives_trained:
            raise ValueError('Use a trained instruction checkpoint for further adjustment')
        previous_manifest=Path(directive_path).parent/'manifest.json'
        if previous_manifest.exists() and json.loads(previous_manifest.read_text(encoding='utf-8'))['candidate_signature']!=candidates.signature:
            raise ValueError('Instruction source candidate map changed')
        if base_digest(transferred)!=base_digest(anchor):raise ValueError('Instruction source has a different frozen base micro')
        inherited=int(metadata['step'])
    manifest=dict(source_hash=source_signature(),input_hashes=inputs,config=settings,candidate_signature=candidates.signature,
        inherited_step=inherited,initialization='trained_command_inputs' if transferred is not None else 'completed_micro_weights_plus_zero_command_input',
        training_scope='directive input layer only; base micro and guard frozen',reward_settings=reward_settings(config),
        optimization='PPO plus balanced public instruction probability loss; no frozen-critic regression',
        best_selection=('win rate, plant rate, tested instruction minimum and mean' if config.get('MACRO_DIRECTIVE_MODE')=='simple'
            else 'minimum applicable instruction rate, mean instruction rate, win rate, plant rate'))
    saved=None
    if config['RESUME']:
        previous=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
        if previous!=manifest:raise RuntimeError('Instruction source/settings/map/source models changed; choose a new RUN_NAME')
        policy,saved=DirectedMicroPolicy.load(data/'latest.pt')
        if saved.get('stage')!='directive_rl' or saved.get('inherited_step')!=inherited:
            raise ValueError('Wrong instruction stage or inherited source step')
    else:
        seed_all(config['SEED'])
        policy=transferred if transferred is not None else DirectedMicroPolicy.from_micro(anchor)
        write_json(data/'manifest.json',manifest)
        policy.save(data/'initial.pt',stage='directive_rl',trained=False,step=inherited,additional_step=0,inherited_step=inherited)
    if base_digest(policy)!=base_digest(anchor):raise ValueError('Instruction checkpoint changed the frozen base micro weights')
    optimizer=torch.optim.Adam(trainable_directives(policy),lr=config['LEARNING_RATE'])
    if saved:
        optimizer.load_state_dict(saved['optimizer']);restore_rng(saved)
    return policy,anchor,optimizer,saved,inherited


def evaluate(policy,guard,candidates,config):
    state=rng_state()
    try:
        rounds=[]
        for i,code in enumerate(config['OPPONENT_CODES']):
            history=MatchHistory()
            for repeat in range(config['EVALUATION_ROUNDS_PER_OPPONENT']):
                _,stats=play_directives(code,config['SEED']+100000+i*1000+repeat,policy=policy,guard=guard,
                    candidates=candidates,config=config,history=history)
                rounds.append(stats)
        n=len(rounds)
        plant_distances=[r['planned_plant_distance'] for r in rounds if r['planned_plant_distance'] is not None]
        measures={key:sum(r['instruction_measures'].get(key,0) for r in rounds)
            for key in {key for r in rounds for key in r['instruction_measures']}}
        compliance={key:measures.get(key+'_compliant',0)/max(1,measures.get(key+'_applicable',0)) for key in INSTRUCTIONS}
        ready=all(measures.get(key+'_applicable',0)>0 and compliance[key]>=config['MIN_INSTRUCTION_RATE'] for key in INSTRUCTIONS)
        tested=[key for key in INSTRUCTIONS if measures.get(key+'_applicable',0)>0]
        required=list(INSTRUCTIONS if config.get('MACRO_DIRECTIVE_MODE')!='simple' or config.get('SIMPLE_RETREAT_TICKS',0)>0 else ('WAIT','ADVANCE'))
        return dict(rounds=rounds,success_rate=sum(r['success'] for r in rounds)/n,
            actual_round_win_rate=sum(r['success'] for r in rounds)/n,plant_rate=sum(r['planted'] for r in rounds)/n,
            orb_collections=sum(r['orb_collections'] for r in rounds),
            planned_plant_distance_mean=float(np.mean(plant_distances)) if plant_distances else None,
            planned_plant_exact_rate=sum(d==0 for d in plant_distances)/len(plant_distances) if plant_distances else None,
            instructions={key:sum(r['instructions'].get(key,0) for r in rounds) for key in ('WAIT','ADVANCE','RETREAT')},
            instruction_measures=measures,
            instruction_rates={key:measures.get(key+'_satisfied',0)/max(1,measures.get(key+'_total',0))
                for key in ('WAIT','ADVANCE','RETREAT')},
            instruction_compliance_rates=compliance,instruction_ready=ready,
            tested_instructions=tested,required_instructions=required,
            scenario_instruction_ready=all(measures.get(key+'_applicable',0)>0 and compliance[key]>=config['MIN_INSTRUCTION_RATE'] for key in required),
            macro_directive_mode=config.get('MACRO_DIRECTIVE_MODE','random'),
            minimum_instruction_rate=config['MIN_INSTRUCTION_RATE'],
            compliance_definition='WAIT=hold; ADVANCE/RETREAT=approach when a legal improving move exists, outside the target and without a plant/orb opportunity',
            sampled=True,phase='directed_preplant',opponent_ai='default')
    finally:restore_rng(state)


def evaluation_rank(report,config=None):
    rates=[report.get('instruction_compliance_rates',{}).get(key,0.) for key in INSTRUCTIONS]
    if config and config.get('MACRO_DIRECTIVE_MODE')=='simple':
        tested=report.get('required_instructions',report.get('tested_instructions',[]))
        rates=[report.get('instruction_compliance_rates',{}).get(key,0.) for key in tested] or [0.]
        return report['actual_round_win_rate'],report['plant_rate'],min(rates),float(np.mean(rates))
    return min(rates),float(np.mean(rates)),report['actual_round_win_rate'],report['plant_rate']


def train(config):
    torch.set_num_threads(config['TORCH_THREADS'])
    if config['ADDITIONAL_TRAINING_STEPS']<1:raise ValueError('ADDITIONAL_TRAINING_STEPS must be positive')
    if min(config['ORDER_APPROACH_WEIGHT'],config['WAIT_HOLD_REWARD'],config['WAIT_MOVE_PENALTY'],
           config['ORDER_AWAY_PENALTY'],config['ORDER_ARRIVAL_REWARD'],config['INSTRUCTION_LOSS_WEIGHT'],config['BC_KL_WEIGHT'])<0:
        raise ValueError('Instruction rewards/penalty magnitude must be nonnegative')
    if not 0<=config['INSTRUCTION_COMBAT_WEIGHT']<=1 or not 0<config['MIN_INSTRUCTION_RATE']<=1:
        raise ValueError('Instruction combat weight/readiness rate must be within valid ranges')
    directive_policy(config,config['SEED'])
    from map_data import NEW_MAZE_STR
    arena=np.array([[int(v) for v in row] for row in NEW_MAZE_STR.strip().splitlines()])
    # Fail on an unconfigured map before loading/creating/updating any model.
    candidates=MacroCandidates.configured(arena)
    check_preplant_runtime()
    guard,metadata=FreshPolicy.load(config['POSTPLANT_CHECKPOINT'],role='attacker')
    if not metadata.get('trained'):raise ValueError('Use the trained frozen postplant guard')
    guard.requires_grad_(False).eval()
    with run_directories(config) as (data,logs):
        policy,anchor,optimizer,saved,inherited=initialize(config,data,candidates)
        initial_base=base_digest(policy)
        additional,episode,next_eval,best_rank=(saved['additional_step'],saved['episode'],saved['next_eval'],tuple(saved['best_rank'])) if saved else (
            0,0,config['EVALUATE_EVERY_STEPS'],(-1.,-1.,-1.,-1.))
        if saved:reconcile_resume_logs(logs,inherited+additional,episode)

        def save(complete=False):
            policy.save(data/'latest.pt',stage='directive_rl',trained=additional>0,step=inherited+additional,
                additional_step=additional,inherited_step=inherited,episode=episode,next_eval=next_eval,
                best_rank=list(best_rank),optimizer=optimizer.state_dict(),reward_version=REWARD_VERSION,**rng_state())
            write_json(logs/'status.json',dict(stage='directive_rl',step=inherited+additional,additional_step=additional,
                inherited_step=inherited,episode=episode,complete=complete,
                best_score=best_rank[0 if config.get('MACRO_DIRECTIVE_MODE')=='simple' else 2],
                best_instruction_rate=best_rank[2 if config.get('MACRO_DIRECTIVE_MODE')=='simple' else 0],
                directives_trained=policy.directives_trained,base_micro_frozen=True))

        def evaluation(initial=False):
            nonlocal best_rank,next_eval
            report=evaluate(policy,guard,candidates,config)
            report.update(step=inherited+additional,additional_step=additional)
            write_json(logs/('initial_evaluation.json' if initial else f'evaluation_{inherited+additional}.json'),report)
            rank=evaluation_rank(report,config)
            # Keep best among the new trained input layers, never silently choose
            # the initial zero-input adapter as if instruction learning had run.
            if not initial and rank>best_rank:
                best_rank=rank
                policy.save(data/'best.pt',stage='directive_rl',trained=True,step=inherited+additional,
                    additional_step=additional,inherited_step=inherited,evaluation=report,reward_version=REWARD_VERSION)
            while next_eval<=additional:next_eval+=config['EVALUATE_EVERY_STEPS']
            save()
            rates=report.get('instruction_compliance_rates',{})
            print(f"Evaluation ST={inherited+additional} plant={report['plant_rate']:.1%} win={report['actual_round_win_rate']:.1%} "
                f"WAIT={rates.get('WAIT',0.):.1%} ADVANCE={rates.get('ADVANCE',0.):.1%} RETREAT={rates.get('RETREAT',0.):.1%} "
                f"instruction_ready={report.get('instruction_ready',False)}",flush=True)

        if not saved:evaluation(initial=True)
        while additional<config['ADDITIONAL_TRAINING_STEPS']:
            rows=[];episode_stats=[]
            target=min(config['ROLLOUT_DECISIONS'],config['ADDITIONAL_TRAINING_STEPS']-additional)
            while len(rows)<target:
                history=MatchHistory();first_episode=episode
                for _ in range(config['ROUNDS_PER_MATCH']):
                    code=choose_roster(config['SEED'],episode,config['OPPONENT_CODES'])
                    batch,stats=play_directives(code,config['SEED']+episode,policy=policy,guard=guard,
                        candidates=candidates,config=config,history=history)
                    if not batch:raise RuntimeError('Instruction episode supplied no policy decisions')
                    rows.extend(batch)
                    episode_stats.append(stats)
                    episode+=1
                path=data/'histories'/f'match_{first_episode:06d}.npz'
                if path.exists():
                    import uuid
                    path=path.with_name(path.stem+'_'+uuid.uuid4().hex+'.npz')
                history.save(path)
            # Count precisely the transitions used by PPO; simulated look-ahead
            # still supplies their full-round rewards and causal histories.
            rows=rows[:target]
            remaining=target
            for stats in episode_stats:
                sampled=stats['decisions']
                stats['sampled_decisions']=sampled
                stats['decisions']=min(sampled,remaining)
                remaining-=stats['decisions']
                with (logs/'episodes.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(stats,ensure_ascii=False)+'\n')
            record_training_metrics(logs,inherited+additional,episode,policy,rows,config['BATCH_SIZE'])
            loss,optimization=optimize_directives(policy,anchor,optimizer,rows,config)
            with (logs/'instruction_optimization.jsonl').open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(dict(step=inherited+additional+len(rows),batches=optimization))+'\n')
            if base_digest(policy)!=initial_base:raise RuntimeError('Instruction PPO updated the frozen base micro')
            additional+=len(rows);policy.directives_trained=True;save()
            print(f'Directive RL ST={inherited+additional} added={additional} loss={loss:.4f}',flush=True)
            if additional>=next_eval or additional>=config['ADDITIONAL_TRAINING_STEPS']:evaluation()
        best,best_metadata=DirectedMicroPolicy.load(data/'best.pt')
        final=evaluate(best,guard,candidates,config)
        final.update(step=best_metadata['step'])
        write_json(logs/'final_evaluation.json',final)
        save(complete=True)
