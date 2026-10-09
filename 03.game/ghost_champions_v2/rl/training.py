"""BC recollection, attacker-only PPO/PFSP, periodic six-opponent guardrails."""
import contextlib
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
import torch
import torch.nn.functional as F
from .observation import OBS_DIM,SCHEMA_HASH
from .policy import ResidualPolicy
from .initial_evaluation import REFERENCE_NAME,load_initial_evaluation,evaluation_reference
from .rollout import ROOT,OPPONENTS,play_round,pfsp_probabilities,shaping_coefficient

AI_ROOT=Path(__file__).resolve().parents[1]
EVALUATION_CONFIG_KEYS={"evaluation_series","initial_evaluation_series","final_evaluation_series","tyg_confirmation_series","tyg_periodic_action"}
EVALUATION_SOURCE_PATHS={str(p.resolve()) for p in (
    ROOT/"tools/run_eval.py",AI_ROOT/"rl/training.py",AI_ROOT/"rl/training_config.json",AI_ROOT/"training_attacker.py",
    AI_ROOT/"tools/registry_compatibility.py")}


def validate_evaluation_change(parent,config,frozen,*,runtime_compatibility=None):
    """Permit only evaluation orchestration changes when continuing old data."""
    if parent["schema_hash"]!=SCHEMA_HASH or parent["verification"]:
        raise ValueError("Continuation requires the same observation schema and a production run")
    if ({k:v for k,v in parent["config"].items() if k not in EVALUATION_CONFIG_KEYS}
            !={k:v for k,v in config.items() if k not in EVALUATION_CONFIG_KEYS}):
        raise ValueError("Only evaluation counts and the periodic TYG action may change in this continuation")
    snapshot=Path(os.environ["GC_OPPONENT_SNAPSHOT"]).resolve() if os.environ.get("GC_OPPONENT_SNAPSHOT") else None
    runtime=Path(os.environ["GC_RUNTIME_SNAPSHOT"]).resolve() if os.environ.get("GC_RUNTIME_SNAPSHOT") else None
    def canonicalize(inputs):
        result={}
        for name,digest in inputs.items():
            path=Path(name)
            if snapshot and path.is_relative_to(snapshot):
                path=ROOT/path.relative_to(snapshot)
            if runtime and path==runtime/"run_competition_manager.py":
                path=ROOT/"run_competition_manager.py"
            key=str(path)
            if key in result and result[key]!=digest:
                raise ValueError("Conflicting frozen input copies: "+key)
            result[key]=digest
        return result
    old=canonicalize(parent["frozen_inputs"])
    canonical=canonicalize(frozen)
    marker=str(ROOT/"concon_v1/__init__.py")
    if marker not in old and canonical.get(marker)==hashlib.sha256(b"").hexdigest():
        canonical.pop(marker)
    changed={name for name in old.keys()|canonical.keys() if old.get(name)!=canonical.get(name)}
    registry=str((ROOT/"run_game.py").resolve())
    if registry in changed and runtime_compatibility:
        from ..tools.registry_compatibility import registry_change_allowed
        active={"ghost_champions_v2",*(key for _,key in OPPONENTS.values())}
        if registry_change_allowed(runtime_compatibility,registry,old.get(registry),canonical.get(registry),active):
            changed.remove(registry)
    if changed-EVALUATION_SOURCE_PATHS:
        raise ValueError("Policy/runtime inputs changed; cannot reuse BC: "+", ".join(sorted(changed-EVALUATION_SOURCE_PATHS)))


def load_continuation(data_dir,config,frozen,*,runtime_compatibility=None):
    path=data_dir/"continuation.json"
    if not path.exists():
        return None
    continuation=json.loads(path.read_text(encoding="utf-8"))
    compatibility=runtime_compatibility or continuation.get("runtime_compatibility")
    validate_evaluation_change(continuation["parent_manifest"],config,frozen,runtime_compatibility=compatibility)
    for name,digest in continuation["artifact_hashes"].items():
        artifact=data_dir/name
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest()!=digest:
            raise ValueError("Continuation artifact differs: "+name)
    if "bc_frozen_inputs" in continuation:
        validate_evaluation_change(dict(continuation["parent_manifest"],frozen_inputs=continuation["bc_frozen_inputs"]),config,frozen,
                                   runtime_compatibility=compatibility)
    pending=continuation.get("pending_evaluation")
    if pending:
        reference=pending["reference"]
        if evaluation_reference(reference["folder"])!=reference:
            raise ValueError("Completed PPO evaluation artifacts changed")
        checkpoint=f"evaluated_{pending['step']:07d}.pt"
        if reference["source_checkpoint"]["sha256"]!=continuation["artifact_hashes"].get(checkpoint):
            raise ValueError("Completed PPO evaluation does not match the inherited checkpoint")
    reused=continuation.get("reused_initial_evaluation")
    if reused:
        for name,digest in reused["hashes"].items():
            if hashlib.sha256((Path(reused["folder"])/name).read_bytes()).hexdigest()!=digest:
                raise ValueError("Reused initial evaluation differs: "+name)
    initial=continuation.get("initial_evaluation")
    if initial is None:
        if continuation.get("initial_evaluation_as_baseline"):
            raise ValueError("Initial-baseline recovery requires a completed evaluation")
        return continuation
    for name,digest in initial["hashes"].items():
        if hashlib.sha256((Path(initial["folder"])/name).read_bytes()).hexdigest()!=digest:
            raise ValueError("Inherited initial evaluation differs: "+name)
    initial_manifest=json.loads((Path(initial["folder"])/"manifest.json").read_text(encoding="utf-8"))
    if initial_manifest["residual_checkpoint"]["sha256"]!=continuation["artifact_hashes"]["bc.pt"]:
        raise ValueError("Inherited initial evaluation does not match BC checkpoint")
    return continuation


def evaluation_minimum(series):
    """Small evaluations finish their schedule without masquerading as formal."""
    return 100 if series>=10 else 1


def fingerprint(snapshot):
    paths=set(Path(snapshot)/name for name in
              ("game_core.py","character_stats.py","player_combos.py","awakening_events.py","party_presets.py","map_data.py"))
    paths.update(ROOT/name for name in ("run_game.py","battle_logic.py","abilities_los.py","iq_perception.py",
                                       "iq_controller_adapter.py","public_effects.py","grid_visibility.py",
                                       "ghost_champions_v1.py","ghost_champions_v1_macro.py",
                                       "team_ai.py","controllers.py","roster_utils.py","simulation_runtime.py",
                                       "grid_paths.py","grid_lines.py","combo_awakening.py","defender_setup_phase.py",
                                       "map_data_defender_setup.py","analytics/combat_tracker.py","run_competition_manager.py",
                                       "tools/run_eval.py"))
    manager=Path(snapshot)/"run_competition_manager.py"
    if manager.is_file():
        paths.discard(ROOT/"run_competition_manager.py")
        paths.add(manager)
    for package in ("gc_v1","frc_v1","fnatic_v3","concon_v1","omoko_v1","touyama_v2","attacker_v3","defender_v3"):
        package_root=Path(os.environ["GC_OPPONENT_SNAPSHOT"])/package if package=="concon_v1" and os.environ.get("GC_OPPONENT_SNAPSHOT") else ROOT/package
        paths.update(package_root.glob("*.py"))
    paths.update((AI_ROOT/"rl").glob("*.py"))
    paths.update(AI_ROOT.glob("*.py"))
    paths.add(AI_ROOT/"gc_profiles.json")
    paths.add(AI_ROOT/"tools/registry_compatibility.py")
    if os.environ.get("GC_V2_CONFIG"):
        paths.add(Path(os.environ["GC_V2_CONFIG"]))
    from ghost_champions_v1 import CARRY,ESCORT,RETRIEVE,GUARD,SEARCH,RETAKE
    for candidates in (CARRY,ESCORT,RETRIEVE,GUARD,SEARCH,RETAKE):
        chosen=next((p for p in candidates if p.exists()),None)
        if chosen:
            paths.add(chosen)
    from gc_v1.learning_attacker_macro_gc_runtime import DEFAULT_MODEL_CANDIDATES
    from gc_v1.learning_defender_opening_macro_gc_runtime import find_opening_macro_model
    from gc_v1.learning_defender_setup_gc_runtime import MODEL_PATH
    paths.update(p for p in (next((p for p in DEFAULT_MODEL_CANDIDATES if p.exists()),None),
                            find_opening_macro_model(),MODEL_PATH) if p is not None)
    paths.update((ROOT/p).resolve() for p in ("frc_v1/runs/selfplay_01/A_policy.pt",
                 "frc_v1/runs/tactics_finetune_20260930/D_policy.pt"))
    for package in ("fnatic_v3","concon_v1","omoko_v1","touyama_v2","attacker_v3","defender_v3"):
        package_root=Path(os.environ["GC_OPPONENT_SNAPSHOT"])/package if package=="concon_v1" and os.environ.get("GC_OPPONENT_SNAPSHOT") else ROOT/package
        paths.update(package_root.rglob("*.pt"))
        paths.update(package_root.rglob("*.pth"))
    return {str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths) if path.is_file()}


def assert_frozen(expected):
    changed=[name for name,digest in expected.items() if not Path(name).is_file()
             or hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest]
    if changed:
        raise RuntimeError("Frozen training inputs changed: "+", ".join(changed))


def array_rows(rows):
    return dict(obs=np.stack([r["obs"] for r in rows]),mask=np.stack([r["mask"] for r in rows]),
                teacher=np.array([r["teacher"] for r in rows],dtype=np.int64))


def bc_loss(policy,data,indices,step,config):
    dist,_=policy(torch.from_numpy(data["obs"][indices]),torch.from_numpy(data["mask"][indices]))
    labels=torch.from_numpy(data["teacher"][indices])
    coef=shaping_coefficient(step,config["shaping_decay_steps"])
    weights=torch.where(labels==2,max(config["ability_bc_floor"],coef),coef)
    return -(weights*dist.log_prob(labels)).mean()


def collect(data_dir,log_dir,config,frozen,on_progress=None):
    rows,validation,episodes=[],[],[]
    for code_index,code in enumerate(OPPONENTS):
        for index in range(config["bc_rounds_per_opponent"]):
            assert_frozen(frozen)
            seed=config["seed"]+code_index*10000+index
            with (log_dir/"engine.log").open("a",encoding="utf-8") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                episode,stats=play_round(code,seed)
            rows.extend(episode)
            validation.extend([index==config["bc_rounds_per_opponent"]-1 and index>0]*len(episode))
            episodes.append(stats)
            if on_progress:
                on_progress(bc_collected_rounds=len(episodes),
                    bc_total_rounds=len(OPPONENTS)*config["bc_rounds_per_opponent"],
                    samples=len(rows),opponent=code)
            print(f"BC {code} {index+1}/{config['bc_rounds_per_opponent']}: {len(episode)} decisions",flush=True)
    data=array_rows(rows)
    data["validation"]=np.array(validation,dtype=np.bool_)
    np.savez_compressed(data_dir/"demonstrations.npz",**data)
    (data_dir/"demonstrations.json").write_text(json.dumps(dict(schema_hash=SCHEMA_HASH,
        samples=len(rows),frozen_inputs=frozen,episodes=episodes),indent=2),encoding="utf-8")
    return data


def load_data(data_dir,frozen):
    metadata=json.loads((data_dir/"demonstrations.json").read_text(encoding="utf-8"))
    if metadata["schema_hash"]!=SCHEMA_HASH or metadata["frozen_inputs"]!=frozen:
        raise ValueError("BC schema/runtime differs; recollect demonstrations in a new run")
    with np.load(data_dir/"demonstrations.npz",allow_pickle=False) as arrays:
        return {name:arrays[name] for name in arrays.files}


def accuracy(policy,data,indices):
    if not len(indices):
        return None
    correct=0
    with torch.no_grad():
        for start in range(0,len(indices),512):
            batch=indices[start:start+512]
            dist,_=policy(torch.from_numpy(data["obs"][batch]),torch.from_numpy(data["mask"][batch]))
            correct+=int((dist.logits.argmax(-1)==torch.from_numpy(data["teacher"][batch])).sum())
    return correct/len(indices)


def train_bc(data,config,rng,on_progress=None):
    policy=ResidualPolicy()
    optimizer=torch.optim.Adam(policy.parameters(),lr=config["learning_rate"])
    training=np.flatnonzero(~data["validation"])
    validation=np.flatnonzero(data["validation"])
    for epoch in range(config["bc_epochs"]):
        for indices in np.array_split(rng.permutation(training),max(1,math.ceil(len(training)/config["batch_size"]))):
            loss=bc_loss(policy,data,indices,0,config)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),config["max_gradient_norm"])
            optimizer.step()
        training_accuracy,validation_accuracy=accuracy(policy,data,training),accuracy(policy,data,validation)
        if on_progress:
            on_progress(bc_epoch=epoch+1,bc_total_epochs=config["bc_epochs"],
                        training_accuracy=training_accuracy,validation_accuracy=validation_accuracy)
        print(f"BC epoch {epoch+1}: train={training_accuracy:.3f}, validation={validation_accuracy}",flush=True)
    return policy,dict(training_accuracy=accuracy(policy,data,training),validation_accuracy=accuracy(policy,data,validation),
                       training_samples=len(training),validation_samples=len(validation))


def optimize_ppo(policy,optimizer,rows,demonstrations,step,config,rng):
    obs=torch.from_numpy(np.stack([r["obs"] for r in rows]))
    mask=torch.from_numpy(np.stack([r["mask"] for r in rows]))
    actions=torch.tensor([r["action"] for r in rows])
    old_logp=torch.tensor([r["logp"] for r in rows])
    returns=torch.tensor([r["return"] for r in rows],dtype=torch.float32)
    advantages=torch.tensor([r["advantage"] for r in rows],dtype=torch.float32)
    advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    teacher_indices=np.flatnonzero(~demonstrations["validation"])
    losses=[]
    for _ in range(config["ppo_epochs"]):
        for batch in np.array_split(rng.permutation(len(rows)),max(1,math.ceil(len(rows)/config["batch_size"]))):
            dist,value=policy(obs[batch],mask[batch])
            ratio=(dist.log_prob(actions[batch])-old_logp[batch]).exp()
            clipped=ratio.clamp(1-config["ppo_clip"],1+config["ppo_clip"])
            actor=-torch.minimum(ratio*advantages[batch],clipped*advantages[batch]).mean()
            bc_indices=rng.choice(teacher_indices,min(len(teacher_indices),config["batch_size"]),replace=False)
            loss=(actor+config["value_coefficient"]*F.mse_loss(value,returns[batch])
                  -config["entropy_coefficient"]*dist.entropy().mean()+bc_loss(policy,demonstrations,bc_indices,step,config))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),config["max_gradient_norm"])
            optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


def evaluate(checkpoint,folder,config,snapshot,*,series_count=None,opponents=None,minimum_attack_rounds=100):
    opponents=list(OPPONENTS) if opponents is None else opponents
    env=dict(os.environ,GC_V2_RL_CHECKPOINT=str(checkpoint.resolve()),GC_RUNTIME_SNAPSHOT=str(Path(snapshot).resolve()))
    command=[sys.executable,str(ROOT/"tools/run_eval.py"),"--controller","v2","--series-count",str(series_count or config["evaluation_series"]),
             "--workers",str(config["evaluation_workers"]),"--output",str(folder.resolve()),"--runtime-snapshot",str(snapshot),
             "--minimum-attack-rounds",str(minimum_attack_rounds),"--opponents",*opponents]
    print(f"Evaluating {','.join(opponents)} ({series_count or config['evaluation_series']} BO3 each): {folder}",flush=True)
    subprocess.run(command,cwd=ROOT,env=env,check=True)
    summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    status=json.loads((folder/"status.json").read_text(encoding="utf-8"))
    rates={code:summary[OPPONENTS[code][0]]["attack_rate"] for code in opponents}
    return rates,status


def train(run_name,config,snapshot,*,phase="all",resume=False,verification=False):
    data_dir,log_dir=AI_ROOT/"data"/run_name,AI_ROOT/"logs"/run_name
    data_dir.mkdir(parents=True,exist_ok=True)
    log_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    rng=np.random.default_rng(config["seed"])
    frozen=fingerprint(snapshot)
    continuation=load_continuation(data_dir,config,frozen)
    initial_reference=load_initial_evaluation(data_dir)
    if initial_reference and (verification or continuation):
        raise ValueError("Historical initial evaluation requires a fresh production run, not a continuation")
    manifest=dict(schema_hash=SCHEMA_HASH,config=config,frozen_inputs=frozen,verification=verification)
    if initial_reference:
        manifest["initial_evaluation_reference_sha256"]=hashlib.sha256((data_dir/REFERENCE_NAME).read_bytes()).hexdigest()
    if continuation:
        manifest["continuation_sha256"]=hashlib.sha256((data_dir/"continuation.json").read_bytes()).hexdigest()
    manifest_path=data_dir/"manifest.json"
    if manifest_path.exists():
        if not resume or json.loads(manifest_path.read_text(encoding="utf-8"))!=manifest:
            raise ValueError("Run already exists or inputs differ; use a new run name or matching --resume")
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    status=dict(phase="initializing",requested_phase=phase,process_id=os.getpid(),complete=False,verification=verification,step=0)
    if initial_reference:
        status.update(initial_evaluation_reused=True,initial_evaluation_kind="historical_baseline",
                      initial_evaluation_source=initial_reference["folder"],
                      initial_evaluation_rates=initial_reference["rates"],initial_evaluates_new_bc=False)
    def save_status(**values):
        status.update(values)
        status["updated_at"]=datetime.now(timezone.utc).isoformat()
        status_path=log_dir/"status.json"
        temporary=status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status,indent=2),encoding="utf-8")
        temporary.replace(status_path)
    save_status()
    save_status(phase="collecting_bc",bc_collected_rounds=0,
                bc_total_rounds=len(OPPONENTS)*config["bc_rounds_per_opponent"])
    bc_frozen=continuation.get("bc_frozen_inputs",continuation["parent_manifest"]["frozen_inputs"]) if continuation else frozen
    data=load_data(data_dir,bc_frozen) if (data_dir/"demonstrations.npz").exists() else collect(data_dir,log_dir,config,frozen,on_progress=save_status)
    save_status(samples=len(data["obs"]))
    if phase=="collect":
        save_status(phase="complete",complete=True)
        return
    bc_path=data_dir/"bc.pt"
    save_status(phase="training_bc",opponent=None)
    if bc_path.exists():
        policy,_=ResidualPolicy.load(bc_path)
    else:
        policy,metrics=train_bc(data,config,rng,on_progress=save_status)
        policy.save(bc_path,training_stage="bc",step=0,metrics=metrics,verified=False)
        (log_dir/"bc_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    if phase=="bc":
        save_status(phase="complete",complete=True,bc_checkpoint=str(bc_path))
        return
    rates={code:.5 for code in OPPONENTS}
    if not verification:
        save_status(phase="using_initial_baseline" if initial_reference else "evaluating_bc",
                    bc_checkpoint=str(bc_path),evaluation_output=str(log_dir/"evaluation_bc"))
        if initial_reference:
            rates=dict(initial_reference["rates"])
            sample=json.loads((Path(initial_reference["folder"])/"status.json").read_text(encoding="utf-8"))
            save_status(evaluation_output=initial_reference["folder"],initial_evaluation_reexecuted=False)
            print(f"Reusing completed initial evaluation ({initial_reference['scheduled_series']} series): "
                  f"{initial_reference['folder']}. Historical baseline; new BC has not been evaluated.",flush=True)
        elif continuation and continuation.get("initial_evaluation"):
            initial=Path(continuation["initial_evaluation"]["folder"])
            summary=json.loads((initial/"summary.json").read_text(encoding="utf-8"))
            rates={code:summary[name]["attack_rate"] for code,(name,_) in OPPONENTS.items()}
            sample=json.loads((initial/"status.json").read_text(encoding="utf-8"))
            save_status(evaluation_output=str(initial),inherited_initial_evaluation=True)
            if continuation.get("initial_evaluation_as_baseline"):
                save_status(initial_evaluation_reused=True,initial_evaluation_reexecuted=False,
                            initial_evaluation_as_baseline=True,initial_guardrail_passed=False,
                            initial_evaluation_rates=dict(rates))
                print(f"Reusing completed BC evaluation as starting baseline: {initial}. "
                      "The initial result is not a passed training guardrail.",flush=True)
        else:
            initial_series=config.get("initial_evaluation_series",config["evaluation_series"])
            rates,sample=evaluate(bc_path,log_dir/"evaluation_bc",config,snapshot,
                series_count=initial_series,minimum_attack_rounds=evaluation_minimum(initial_series))
        if not sample["complete"]:
            save_status(phase="stopped",stop_reason="insufficient_evaluation_sample",rates=rates)
            return
        initial_baseline=initial_reference or (continuation and continuation.get("initial_evaluation_as_baseline"))
        if not initial_baseline and config.get("tyg_periodic_action","stop")=="stop" and rates["TYG"]<config["tyg_minimum_win_rate"]:
            save_status(phase="stopped",stop_reason="TYG_regression_after_BC",rates=rates)
            return
    optimizer=torch.optim.Adam(policy.parameters(),lr=config["learning_rate"])
    latest=data_dir/"latest.pt"
    step,episode,last_checked_step=0,0,0
    if resume and latest.exists():
        policy,payload=ResidualPolicy.load(latest)
        optimizer=torch.optim.Adam(policy.parameters(),lr=config["learning_rate"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        step,episode=payload["step"],payload["episode"]
        rng.bit_generator.state=payload["numpy_rng"]
        random.setstate(payload["python_rng"])
        torch.set_rng_state(payload["torch_rng"])
        rates=payload["rates"]
        last_checked_step=payload.get("last_evaluated_step",0)
    next_eval=((last_checked_step//config["evaluate_every_steps"])+1)*config["evaluate_every_steps"]

    def save_latest():
        policy.save(latest,training_stage="ppo",step=step,episode=episode,rates=rates,verified=False,
                    last_evaluated_step=last_checked_step,
                    optimizer_state_dict=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,
                    python_rng=random.getstate(),torch_rng=torch.get_rng_state())

    def check_guardrail():
        nonlocal rates,last_checked_step,next_eval
        evaluated=data_dir/f"evaluated_{step:07d}.pt"
        if evaluated.exists():
            saved,_=ResidualPolicy.load(evaluated)
            if any(not torch.equal(value,saved.state_dict()[name]) for name,value in policy.state_dict().items()):
                raise ValueError("Pending evaluation checkpoint differs from resumed policy")
        else:
            policy.save(evaluated,training_stage="ppo",step=step,verified=False)
        folder=log_dir/f"evaluation_{step:07d}"
        final=step>=config["steps"]
        series=config.get("final_evaluation_series",config["evaluation_series"]) if final else config["evaluation_series"]
        minimum=evaluation_minimum(series)
        save_status(phase="evaluating_ppo",step=step,episode=episode,evaluation_output=str(folder),
                    evaluation_series=series,final_evaluation=final,formal_evaluation=minimum>=100)
        reused=continuation.get("pending_evaluation") if continuation else None
        if reused and reused["step"]==step and last_checked_step<step:
            reference=reused["reference"]
            rates=dict(reference["rates"])
            sample=json.loads((Path(reference["folder"])/"status.json").read_text(encoding="utf-8"))
            save_status(evaluation_reused=True,evaluation_output=reference["folder"])
            print(f"Reusing completed PPO evaluation at {step}: {reference['folder']}",flush=True)
        else:
            save_status(evaluation_reused=False)
            rates,sample=evaluate(evaluated,folder,config,snapshot,series_count=series,
                                  minimum_attack_rounds=minimum)
        warn_only=config.get("tyg_periodic_action","stop")=="warn"
        if sample["complete"] and not final and not warn_only and rates["TYG"]<config["tyg_minimum_win_rate"]:
            confirmation=log_dir/f"evaluation_{step:07d}_tyg_confirmation"
            confirmation_series=config.get("tyg_confirmation_series",config["evaluation_series"])
            save_status(phase="confirming_tyg",evaluation_output=str(confirmation),evaluation_series=confirmation_series)
            confirmation_rates,confirmation_sample=evaluate(evaluated,confirmation,config,snapshot,
                series_count=confirmation_series,opponents=["TYG"],minimum_attack_rounds=evaluation_minimum(confirmation_series))
            rates["TYG"]=confirmation_rates["TYG"]
            sample=dict(sample,complete=confirmation_sample["complete"])
        target_met=rates["TYG"]>=config["tyg_minimum_win_rate"]
        if not sample["complete"] or (not warn_only and not target_met):
            save_status(phase="stopped",stop_reason="TYG_regression" if sample["complete"] else "insufficient_evaluation_sample",rates=rates)
            return False
        formal=bool(final and sample.get("milestone_sample_complete",minimum>=100 and sample["complete"]))
        if target_met:
            policy.save(data_dir/"last_passed_guardrail.pt",training_stage="ppo",step=step,
                        verified=formal,guardrail_passed=True,formal_evaluation=formal,rates=rates)
        elif not final:
            warning=dict(step=step,warning="TYG_below_target",attack_rate=rates["TYG"],
                         target=config["tyg_minimum_win_rate"])
            with (log_dir/"warnings.jsonl").open("a",encoding="utf-8") as log:
                log.write(json.dumps(warning)+"\n")
            print(f"WARNING: TYG {rates['TYG']:.1%} < {config['tyg_minimum_win_rate']:.1%} at {step}; continuing training.",flush=True)
        else:
            print(f"Training finished. Final TYG target not met: {rates['TYG']:.1%} < {config['tyg_minimum_win_rate']:.1%}.",flush=True)
        last_checked_step=step
        next_eval=((step//config["evaluate_every_steps"])+1)*config["evaluate_every_steps"]
        save_latest()
        save_status(phase="ppo",last_evaluated_step=last_checked_step,rates=rates,
                    guardrail_passed=target_met,tyg_below_target=not target_met,
                    warning="TYG_below_target" if not target_met else None)
        if final:
            save_status(training_complete=True,final_target_met=target_met,
                        final_target=config["tyg_minimum_win_rate"])
        return True

    # A crash after saving a rollout must not bypass its pending guardrail.
    if not verification and step and (step>=next_eval or step>=config["steps"]>last_checked_step):
        if not check_guardrail():
            return
    save_status(phase="ppo",step=step,episode=episode,last_evaluated_step=last_checked_step)
    while step<config["steps"]:
        assert_frozen(frozen)
        probabilities=pfsp_probabilities(rates)
        code=str(rng.choice(list(OPPONENTS),p=probabilities))
        with (log_dir/"engine.log").open("a",encoding="utf-8") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            rows,stats=play_round(code,config["seed"]+1000000+episode,policy=policy,mode="stochastic",
                gamma=config["gamma"],step=step,decay_steps=config["shaping_decay_steps"],
                recon_reward=config["recon_new_enemy_reward"],gae_lambda=config["gae_lambda"],
                hazard_shaping_weight=config.get("hazard_shaping_weight",.05),
                hazard_hit_penalty=config.get("hazard_hit_penalty",.03))
        step+=len(rows)
        episode+=1
        loss=optimize_ppo(policy,optimizer,rows,data,step,config,rng)
        rates[code]=.95*rates[code]+.05*float(stats["winner"]=="attacker")
        save_latest()
        with (log_dir/"progress.jsonl").open("a",encoding="utf-8") as log:
            log.write(json.dumps(dict(step=step,episode=episode,loss=loss,shaping=shaping_coefficient(step,config["shaping_decay_steps"]),
                                     pfsp=dict(zip(OPPONENTS,probabilities.tolist())),**stats))+"\n")
        save_status(phase="ppo",step=step,episode=episode,checkpoint=str(latest),rates=rates,opponent=code)
        print(f"PPO {step}/{config['steps']} {code} {stats['winner']} loss={loss:.4f}",flush=True)
        if not verification and (step>=next_eval or step>=config["steps"]):
            if not check_guardrail():
                return
    assert_frozen(frozen)
    if verification:
        save_status(phase="evaluating_verification",evaluation_output=str(log_dir/"evaluation_verification"))
        rates,sample=evaluate(latest,log_dir/"evaluation_verification",config,snapshot)
        save_status(phase="complete",complete=True,verification=True,rates=rates,milestone_sample_complete=sample["complete"])
    else:
        save_status(phase="complete",complete=True,rates=rates)
