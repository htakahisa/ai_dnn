"""Round-dependent deployment predictions from public inputs; labels never control actions."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))

OPPONENT='frc_v1'
BEST_DIRECTORY=HERE/'data'/'best'
LATEST_PATH=HERE/'data'/'attacker_analysis'/OPPONENT/'latest.pt'
OUTPUT_PATH=HERE/'logs'/'attacker_route_risk_review'/'round_pattern.json'
LIVE_SEED=954000731
RUN_LIVE_BLOCK=True
LIVE_START_ROUNDS=(1,13)
TORCH_THREADS=1

import json
import hashlib
import numpy as np
import torch
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_guard_runtime import load_sources
from touyama_v3.tv3_learn_attacker_analysis import load_analysis
from touyama_v3.tv3_collect_attacker_analysis import play_block


def region_right_weights(scenario):
    return np.asarray([np.mean([scenario.site_of(p)=='R' for p in scenario.branches[n]])
                       for n in scenario.names],dtype=np.float32)


def predict_initial(model,encoder,observation,route,labels):
    """Labels are used only to score the frozen prediction, never as model inputs."""
    x=torch.tensor(np.asarray(observation,np.float32)).unsqueeze(0)
    z=torch.tensor(np.asarray(route,np.float32)).unsqueeze(0)
    with torch.inference_mode():
        logits,outcome=model(x,z)
        probabilities=logits[0,:,:model.regions].softmax(-1).numpy()
        success=float(outcome.sigmoid()[0,0])
    for enemy_id in range(5):
        if observation[encoder.fields.index(f'enemy_{enemy_id}_alive')]<.5:
            probabilities[enemy_id]=0.
        elif observation[encoder.fields.index(f'enemy_{enemy_id}_current')]>.5:
            probabilities[enemy_id]=np.asarray([observation[encoder.fields.index(f'enemy_{enemy_id}_{n}')]
                                              for n in encoder.scenario.names])
    weights=region_right_weights(encoder.scenario)
    alive=labels<model.regions
    truth=float(weights[labels[alive]].sum())
    return dict(expected_right_region_mass=float((probabilities*weights).sum()),
        true_right_region_mass=truth,
        native_region_accuracy=float((probabilities.argmax(-1)[alive]==labels[alive]).mean()),
        ambiguous_true_regions=[int(i) for i in labels[alive] if 0<weights[i]<1],
        saved_route_success=success,
        known_enemies=int(sum(observation[encoder.fields.index(f'enemy_{i}_current')]>.5 for i in range(5))))


def summarize_rows(rows):
    groups={}
    for parity in ('odd','even'):
        selected=[r for r in rows if (r['round']%2==1)==(parity=='odd')]
        groups[parity]={name:dict(rounds=len(selected),
            predicted_right_mean=float(np.mean([r['models'][name]['expected_right_region_mass'] for r in selected])),
            true_right_mean=float(np.mean([r['models'][name]['true_right_region_mass'] for r in selected])),
            count_mae=float(np.mean([abs(r['models'][name]['expected_right_region_mass']-
                                       r['models'][name]['true_right_region_mass']) for r in selected])),
            native_accuracy=float(np.mean([r['models'][name]['native_region_accuracy'] for r in selected])))
            for name in selected[0]['models']} if selected else {}
    return groups


def main():
    torch.set_num_threads(TORCH_THREADS)
    scenario=Scenario()
    source=load_sources(scenario,(OPPONENT,),BEST_DIRECTORY,BEST_DIRECTORY)[OPPONENT]
    latest,encoder,state=load_analysis(LATEST_PATH,scenario,OPPONENT)
    if state['config'].get('collection_source_hashes')!=source['hashes']:
        raise ValueError('Latest collection executor differs')
    from touyama_v3.tv3_learn_attacker_analysis import AttackerEncoder
    original_encoder=AttackerEncoder(scenario,source['analysis'].execution_version)
    if encoder.schema()!=original_encoder.schema():
        raise ValueError('Candidate input contracts differ')
    models=dict(adopted=source['analysis'],latest=latest)
    replay=[]
    for sample in state['replay']:
        obs=sample['observations'][0].numpy().astype(np.float32)
        route=sample['routes'][0].numpy().astype(np.float32)
        labels=sample['placements'][0].numpy()
        if obs[encoder.fields.index('tick')]!=0 or np.argmax(obs[:12])+1!=sample['round']:
            raise ValueError('Replay frame is not the expected round-start observation')
        replay.append(dict(round=sample['round'],models={n:predict_initial(m,encoder,obs,route,labels) for n,m in models.items()}))
    output=dict(scope='round_start_public_predictions_private_labels_only',optimizer_updates=0,
        source_hashes=source['hashes'],latest_hash=hashlib.sha256(LATEST_PATH.read_bytes()).hexdigest(),
        replay_scope='latest_training_replay_not_holdout',replay=replay,replay_summary=summarize_rows(replay))
    if RUN_LIVE_BLOCK:
        live=[]
        seen=set()
        def observe(controller):
            snap=controller.snapshot
            if snap.phase=='setup' or snap.round_number in seen or not controller.frames:
                return
            frame=controller.frames[-1]
            if frame['tick']!=0:
                raise RuntimeError('First live observation was missed')
            seen.add(snap.round_number)
            obs=frame['observation']; route=frame['route_features']
            # The collector has already frozen public inputs/actions before it supplies labels.
            labels=frame['placement']
            evaluations={n:predict_initial(m,controller.analysis_encoder,obs,route,labels) for n,m in models.items()}
            zero_round=obs.copy(); zero_round[:12]=0.
            no_round={n:predict_initial(m,controller.analysis_encoder,zero_round,route,labels) for n,m in models.items()}
            rankings={}
            for name,model in models.items():
                result=model.analyze(controller.analysis_encoder,snap,obs)
                rankings[name]=[dict(site=c['route'].site,branches=c['route'].branches,
                    success=c['plant_probability'],score=c['score']) for c in result['candidates'][:1]]
            enemies=[c for c in controller.game.chars if c.team=='D' and c.is_alive]
            live.append(dict(round=snap.round_number,models=evaluations,rankings=rankings,
                round_code=[float(v) for v in obs[:12]],
                no_round_code=no_round,
                actual_right_half=sum(scenario.site_of(tuple(c.pos))=='R' for c in enemies),
                planned_driver_site=controller.route.site))
        config=dict(max_round_steps=400,collection_executor='fixed_plant',combat_diagnostics=False)
        rounds=[]
        for start in LIVE_START_ROUNDS:
            block,_=play_block(OPPONENT,scenario,source['analysis'],config,LIVE_SEED,0.,
                preset_name='Touyama Gaming',plant_policy=source['plant'],tick_callback=observe,
                start_round=start)
            rounds.extend(block)
        if sorted(seen)!=[i for start in LIVE_START_ROUNDS for i in range(start,start+12)]:
            raise RuntimeError('Not all twelve rounds observed')
        output.update(live_seed=LIVE_SEED,live=live,live_summary=summarize_rows(live),rounds=rounds)
        output['half_summary']={str(start):dict(
            predictions=summarize_rows([r for r in live if start<=r['round']<start+12]),
            plants=sum(r['planted'] for r in rounds if start<=r['round']<start+12),
            round_code_nonzero=[sum(r['round_code']) for r in live if start<=r['round']<start+12])
            for start in LIVE_START_ROUNDS}
    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Replay:',json.dumps(output['replay_summary']),flush=True)
    print('Live:',json.dumps(output.get('live_summary')),flush=True)
    print('Saved:',OUTPUT_PATH,flush=True)


if __name__=='__main__':
    main()
