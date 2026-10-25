"""Read-only micro audit: recorded PPO credit and frozen checkpoint comparison."""
from collections import defaultdict
from contextlib import redirect_stdout, ExitStack, contextmanager
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

AI_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "attacker_preplant_objectives_v1"
TEACHER_DATASET = AI_ROOT/"data/attacker_preplant_omoko_site_v1"
BC_CHECKPOINT = AI_ROOT/"data/attacker_preplant_omoko_bc_site_v1/best.pt"
POSTPLANT_CHECKPOINT = AI_ROOT/"data/attacker_postplant_fixed_v1/best.pt"
VALIDATION_MATCHES = 4
MAX_CASES_PER_MATCH = 48
PROBE_SEED = 2031101300
PROBE_OPPONENTS = ("FNC", "GRG", "GC")
REPORT_PATH = AI_ROOT/"logs/validation/gc3_attacker_micro_objectives_audit.json"
TORCH_THREADS = 1

sys.path.insert(0, str(AI_ROOT.parent))
import numpy as np
import torch
from ghost_champions_v3.gc3_rollout import module_digest
from ghost_champions_v3.gc3_io import write_json
from ghost_champions_v3.gc3_observation import ARENA_SHAPE, CELLS, CONTEXT_WIDTH
from grid_paths import distance_map
from ghost_champions_v3.preplant.gc3_data_attacker_preplant import file_hash, load_shard, source_signature
from ghost_champions_v3.preplant.gc3_policy_attacker_history import load_preplant_policy
from ghost_champions_v3.preplant.gc3_rollout_attacker_preplant import play_preplant


def describe(values):
    if not values:
        return dict(n=0)
    array=np.asarray(values,dtype=float)
    return dict(n=len(values),mean=float(array.mean()),median=float(np.median(array)),
                positive_fraction=float((array>0).mean()),minimum=float(array.min()),maximum=float(array.max()))


@contextmanager
def saved_reward_constants(settings):
    # Only this diagnostic process mirrors the saved reward coefficients.
    # Modules are restored afterwards; source/manifests/models are never edited.
    from ghost_champions_v3 import gc3_reward_phase
    from ghost_champions_v3.preplant import gc3_rollout_attacker_preplant, gc3_reward_attacker_objectives
    with ExitStack() as stack:
        for module in (gc3_reward_phase,gc3_rollout_attacker_preplant,gc3_reward_attacker_objectives):
            overrides={name:value for name,value in settings.items()
                       if name.isupper() and isinstance(value,(int,float)) and hasattr(module,name)}
            if overrides:stack.enter_context(patch.multiple(module,**overrides))
        yield


def recorded_credit(data, logs, committed_step):
    metrics=[json.loads(line) for line in (logs/"training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip() and json.loads(line)["step"]<=committed_step]
    batches=defaultdict(list)
    with (logs/"rollout_transitions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row=json.loads(line)
            if row["batch_end_step"]<=committed_step:
                batches[row["batch_end_step"]].append(row)
    report=[]
    for metric in metrics:
        rows=batches[metric["step"]]
        if len(rows)!=metric["transitions"]:
            raise ValueError("Recorded batch length mismatch")
        path=data/"histories"/f"match_{metric['episode']-7:06d}.npz"
        with np.load(path,allow_pickle=False) as archive:
            observations,actions=archive["obs"],archive["action"]
            metadata=json.loads(str(archive["metadata"]))
            indices=[i for i,event in enumerate(metadata["events"])
                     if event["name"]!="round_end" and observations[i,2]<.5]
            if len(indices)!=len(rows):
                raise ValueError("History and recorded PPO decisions do not align")
            advantages=np.asarray([r["advantage"] for r in rows])
            normalized=(advantages-advantages.mean())/(advantages.std()+1e-8)
            groups=defaultdict(lambda:defaultdict(list))
            completion_credit=defaultdict(float)
            for j,i in enumerate(indices):
                row=rows[j];parts=row["reward_components"]
                if not np.isclose(row["reward"],sum(parts.values())):
                    raise ValueError("Reward component sum mismatch")
                labels=[]
                if actions[i,5]>.5:
                    labels.append("plant")
                    labels.append("plant_start" if observations[i,12]==0 else "plant_continue")
                    if parts.get("plant_complete",0)>0:labels.append("plant_completed")
                if parts.get("plant_cancel",0)<0:labels.append("plant_cancel")
                if parts.get("spike_pickup",0)>0:labels.append("spike_pickup")
                if parts.get("retrieval_progress",0)>0:labels.append("retrieval_progress")
                if parts.get("plant_complete",0)>0:
                    completion_credit["plant_mode" if actions[i,5]>.5 else "other_mode"]+=parts["plant_complete"]
                if observations[i,8]>.5 and (actions[i,0]>.5 or actions[i,2]>.5):
                    obs=observations[i]
                    walls=obs[CONTEXT_WIDTH:CONTEXT_WIDTH+CELLS].reshape(ARENA_SHAPE)>.5
                    drop=tuple(np.rint(obs[9:11]*ARENA_SHAPE).astype(int))
                    origin=tuple(np.rint(obs[:2]*ARENA_SHAPE).astype(int))
                    target=tuple(np.rint(actions[i,8:10]*ARENA_SHAPE).astype(int))
                    distances=distance_map(walls.astype(int),drop)
                    if distances[origin]>=0 and distances[target]>=0 and distances[target]<distances[origin]:
                        labels.append("move_command_toward_perceived_drop")
                for label in labels:
                    for field in ("reward","return","value","advantage"):
                        groups[label][field].append(row[field])
                    groups[label]["normalized_advantage"].append(float(normalized[j]))
                    groups[label]["round_outcome"].append(parts.get("round_outcome",0))
            report.append(dict(step=metric["step"],episode=metric["episode"],transitions=len(rows),
                plant_completion_credit=dict(completion_credit),
                groups={key:{field:describe(values) for field,values in fields.items()} for key,fields in groups.items()}))
    return report


def teacher_probabilities(policies):
    manifest=json.loads((TEACHER_DATASET/"manifest.json").read_text(encoding="utf-8"))
    validation=[s for s in manifest["shards"] if s["split"]=="validation"]
    by_map=defaultdict(list)
    for shard in validation:by_map[shard["teacher_map"]].append(shard)
    shards=[]
    for index in range(len(validation)):
        for candidates in by_map.values():
            if index<len(candidates):shards.append(candidates[index])
        if len(shards)>=VALIDATION_MATCHES:break
    shards=shards[:VALIDATION_MATCHES]
    values={name:defaultdict(list) for name in policies}
    summaries=[]
    for shard in shards:
        path=TEACHER_DATASET/shard["file"]
        if file_hash(path)!=shard["sha256"]:
            raise ValueError("Teacher dataset changed")
        rows=load_shard(path)
        eligible=[row for row in rows if row["masks"]["mode"][5]]
        chosen=np.linspace(0,len(eligible)-1,min(len(eligible),MAX_CASES_PER_MATCH),dtype=int)
        batch=[eligible[i] for i in chosen]
        if not batch:continue
        summaries.append(dict(file=shard["file"],teacher_map=shard["teacher_map"],eligible=len(eligible),sampled=len(batch)))
        record=dict(phase="standard",obs=np.stack([r["obs"] for r in batch]),
                    hints=np.stack([r["hints"] for r in batch]),
                    masks={"mode":np.stack([r["masks"]["mode"] for r in batch])})
        with torch.no_grad():
            for name,policy in policies.items():
                context=policy.contexts(batch)
                probability=policy.distribution(context,record,"mode").probs[:,5].tolist()
                for row,p in zip(batch,probability):
                    values[name]["all"].append(p)
                    values[name]["start" if row["obs"][12]==0 else "continue"].append(p)
        print("teacher comparison",shard["file"],len(batch),flush=True)
    return dict(shards=summaries,policies={name:{key:describe(group) for key,group in groups.items()}
                                         for name,groups in values.items()})


def frozen_probe(policy,anchor,rows,stats,config):
    with torch.no_grad():
        outputs=[policy.evaluate_records(rows[start:start+128]) for start in range(0,len(rows),128)]
        logp=torch.cat([x[0] for x in outputs]);values=torch.cat([x[2] for x in outputs])
        logp_error=float((logp-torch.tensor([r["logp"] for r in rows])).abs().max())
        value_error=float((values-torch.tensor([r["value"] for r in rows])).abs().max())
    if logp_error>3e-5 or value_error>3e-5:
        raise ValueError("Sampled and replayed PPO probability/value mismatch")
    selected=[r for r in rows if r["masks"]["mode"][5]]
    # Include legal PLANT cases first, then representative other decisions.
    batch=(selected+rows[::max(1,len(rows)//32)])[:96]
    policy.requires_grad_(True)
    advantages=np.asarray([r["advantage"] for r in rows],dtype=np.float32)
    center,scale=float(advantages.mean()),float(advantages.std())+1e-8
    context=policy.contexts(batch)
    logp,entropy,values=policy.evaluate_contexts(batch,context)
    old=torch.tensor([r["logp"] for r in batch])
    adv=torch.tensor([(r["advantage"]-center)/scale for r in batch])
    ratio=(logp-old).exp()
    clip=config["PPO_CLIP"]
    actor=-torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()
    critic=.5*(values-torch.tensor([r["return"] for r in batch])).square().mean()
    exploration=-config["ENTROPY_WEIGHT"]*entropy.mean()
    with torch.no_grad():anchor_context=anchor.contexts(batch)
    divergence=context.new_zeros(())
    for kind in ("mode","position","watch","target","ally","site"):
        slots=[i for i,r in enumerate(batch) if kind in r["selections"]]
        if not slots:continue
        record=dict(phase="standard",obs=np.stack([batch[i]["obs"] for i in slots]),
            hints=np.stack([batch[i]["hints"] for i in slots]),goal_site=[batch[i].get("goal_site") for i in slots],
            masks={kind:np.stack([batch[i]["masks"][kind] for i in slots])})
        if kind=="site":
            from ghost_champions_v3.preplant.gc3_features_attacker_navigation import navigation_features
            record.update(navigation_features=np.stack([navigation_features(batch[i],batch[i].get("goal_site")) for i in slots]),
                neutral_navigation_features=np.stack([navigation_features(batch[i],None) for i in slots]))
        baseline=anchor.distribution(anchor_context[slots],record,kind)
        current=policy.distribution(context[slots],record,kind)
        divergence+=torch.distributions.kl_divergence(baseline,current).sum()/len(batch)
    imitation=config["BC_KL_WEIGHT"]*divergence
    named=[(name,p) for name,p in policy.named_parameters() if not name.startswith(("critic.","mode.","site.","move.","spatial.","ally."))]
    gradients={}
    for name,loss in dict(actor=actor,critic=critic,entropy=exploration,bc_kl=imitation).items():
        grad=torch.autograd.grad(loss,[p for _,p in named],retain_graph=True,allow_unused=True)
        vector=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(named,grad)])
        gradients[name]=vector
    actor_norm=float(gradients["actor"].norm())
    measurements={name:dict(norm=float(g.norm()),cosine_to_actor=float((torch.dot(g,gradients["actor"])/(g.norm()*actor_norm+1e-12)).clamp(-1,1)))
                  for name,g in gradients.items()}
    policy.requires_grad_(False)
    return dict(round=stats,decisions=len(rows),legal_plant_decisions=len(selected),
        replay_logp_max_error=logp_error,replay_value_max_error=value_error,
        sampled_diagnostic_rows=len(batch),losses=dict(actor=float(actor.detach()),critic=float(critic.detach()),
        entropy=float(exploration.detach()),bc_kl=float(imitation.detach())),shared_encoder_gradients=measurements,
        note="No optimizer step; one frozen round, not the original training minibatch")


def on_policy_comparison(policies):
    cases=[]; round_rows=[]; values={name:defaultdict(list) for name in policies}
    for offset,code in enumerate(PROBE_OPPONENTS):
        with redirect_stdout(io.StringIO()):
            rows,stats=play_preplant(code,PROBE_SEED+offset,policy=policies["latest"],postplant_checkpoint=POSTPLANT_CHECKPOINT)
        selected=[r for r in rows if r["masks"]["mode"][5]]
        round_rows.append((rows,stats))
        cases.append(dict(opponent=code,legal_plant_decisions=len(selected),
                          planted=stats["planted"],success=stats["success"],cancels=stats["plant_cancellations"]))
        for offset in range(0,len(selected),48):
            batch=selected[offset:offset+48]
            record=dict(phase="standard",obs=np.stack([r["obs"] for r in batch]),
                        hints=np.stack([r["hints"] for r in batch]),
                        masks={"mode":np.stack([r["masks"]["mode"] for r in batch])})
            with torch.no_grad():
                for name,model in policies.items():
                    probabilities=model.distribution(model.contexts(batch),record,"mode").probs[:,5].tolist()
                    for row,probability in zip(batch,probabilities):
                        values[name]["all"].append(probability)
                        values[name]["start" if row["obs"][12]==0 else "continue"].append(probability)
        print("on-policy comparison",code,len(selected),flush=True)
    result=dict(rounds=cases,policies={name:{key:describe(group) for key,group in groups.items()} for name,groups in values.items()})
    best=max(round_rows,key=lambda pair:sum(r["masks"]["mode"][5] for r in pair[0]))
    return result,best


def main():
    torch.set_num_threads(TORCH_THREADS)
    data,logs=AI_ROOT/"data"/RUN_NAME,AI_ROOT/"logs"/RUN_NAME
    manifest=json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    config=manifest["config"]
    from ghost_champions_v3.preplant.gc3_rollout_attacker_preplant import preplant_reward_settings
    current_settings=preplant_reward_settings()
    current_source=source_signature()
    paths=dict(bc=BC_CHECKPOINT,best=data/"best.pt",latest=data/"latest.pt")
    hashes={name:file_hash(path) for name,path in paths.items()}
    policies,metadata={},{}
    for name,path in paths.items():
        policy,saved=load_preplant_policy(path)
        policies[name]=policy.requires_grad_(False).eval()
        metadata[name]={key:saved.get(key) for key in ("step","stage","trained","best_rank")}
    digests={name:module_digest([policy]) for name,policy in policies.items()}
    report=dict(run=RUN_NAME,checkpoints=metadata,checkpoint_hashes=hashes,optimizer_updates=0,
                optimizer_settings={key:config[key] for key in ("GAMMA","GAE_LAMBDA","PPO_CLIP","ENTROPY_WEIGHT","BC_KL_WEIGHT")},
                source_check=dict(saved=manifest["source_hash"],current=current_source,
                    resume_compatible=manifest["source_hash"]==current_source,
                    changed_reward_settings={key:dict(saved=value,current=current_settings.get(key))
                        for key,value in manifest["reward_settings"].items() if current_settings.get(key)!=value},
                    frozen_probe_uses="saved manifest reward coefficients"),
                recorded_credit=recorded_credit(data,logs,metadata["latest"]["step"]))
    print("recorded credit checked",len(report["recorded_credit"]),flush=True)
    report["teacher_comparison"]=teacher_probabilities(policies)
    print("frozen rollout and gradient audit",flush=True)
    with saved_reward_constants(manifest["reward_settings"]):
        report["on_policy_comparison"],(rows,stats)=on_policy_comparison(policies)
    with redirect_stdout(io.StringIO()):
        report["frozen_probe"]=frozen_probe(policies["latest"],policies["bc"],rows,stats,config)
        from ghost_champions_v3 import build_team_ai
        factory=build_team_ai().attacker_factory
        report["normal_game_binding"]=dict(controller=factory.func.__name__)
        for key in ("preplant_checkpoint","macro_checkpoint","guard_checkpoint"):
            path=factory.keywords.get(key)
            if path:
                saved=torch.load(path,map_location="cpu",weights_only=True)
                report["normal_game_binding"][key]=dict(path=str(path),step=saved.get("step"))
    if source_signature()!=current_source:
        raise RuntimeError("Source changed while auditing")
    if any(file_hash(paths[name])!=hashes[name] or module_digest([policies[name]])!=digests[name] for name in paths):
        raise RuntimeError("Checkpoint weights changed during audit")
    report["all_weights_unchanged"]=True
    report["complete"]=True
    write_json(REPORT_PATH,report)
    print(json.dumps(dict(path=str(REPORT_PATH),checkpoints=metadata,
        teacher_comparison=report["teacher_comparison"]["policies"],on_policy_comparison=report["on_policy_comparison"],
        frozen_probe={k:v for k,v in report["frozen_probe"].items() if k!="round"})),flush=True)


if __name__=="__main__":main()
